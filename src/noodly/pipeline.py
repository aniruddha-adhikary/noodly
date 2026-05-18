"""Pipeline — orchestrates the full ingest → extract → score → project loop."""

from __future__ import annotations

import logging
from pathlib import Path

from noodly.caching.manager import CacheManager
from noodly.config import Settings
from noodly.connectors.local_fs import LocalFSConnector
from noodly.dispatch.dispatcher import EventDispatcher
from noodly.dispatch.handlers import AuditLogHandler, ConflictEscalationHandler
from noodly.extraction.dispatcher import LLMJob, LLMJobDispatcher
from noodly.extraction.extractor import ClaimExtractor
from noodly.graph.brain import Brain
from noodly.models.artifacts import SourceArtifact
from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass
from noodly.parsing.boilerplate import BoilerplateStripper
from noodly.parsing.chunker import chunk_markdown, content_hash
from noodly.parsing.parser import DocumentParser
from noodly.projection.markdown import MarkdownProjector
from noodly.resolution.audit import ResolutionAudit
from noodly.resolution.detector import ConflictDetector
from noodly.resolution.resolver import ConflictResolver
from noodly.resolution.strategies import AutoResolveStrategy
from noodly.scoring.authority import AuthorityRegistry
from noodly.scoring.ledger import FactLedger
from noodly.tracking.changelog import ChangeEvent, ChangeLog, ChangeType
from noodly.tracking.claim_differ import ClaimDiffer
from noodly.tracking.content_differ import ContentDiffer

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end ingest → parse → diff → extract → enrich → project pipeline.

    Phase 3 adds multi-format parsing, section-aware chunking, content diff
    tracking, boilerplate stripping, extraction caching, diff-aware agents,
    and an append-only change log.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._brain = Brain(settings)
        self._connector = LocalFSConnector(settings.watch_dir)
        self._extractor = ClaimExtractor(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        self._llm_dispatcher = LLMJobDispatcher(
            extractor=self._extractor,
            max_concurrent=settings.llm_max_concurrent,
            rate_limit_rpm=settings.llm_rate_limit_rpm,
            retry_max=settings.llm_retry_max,
            request_timeout=settings.llm_request_timeout,
        )
        self._authority = AuthorityRegistry(settings.brain_dir / "authority.json")
        self._ledger = FactLedger(
            settings.brain_dir / "ledger.json",
            authority_registry=self._authority,
        )
        self._projector = MarkdownProjector(settings.brain_dir)

        # Phase 3 modules
        cache_dir = settings.brain_dir / ".cache"
        self._parser = DocumentParser(enable_docling=settings.enable_docling)
        self._boilerplate = BoilerplateStripper()
        self._cache = CacheManager(cache_dir)
        self._content_differ = ContentDiffer(settings.brain_dir / ".content_cache")
        self._claim_differ = ClaimDiffer()
        self._changelog = ChangeLog(settings.brain_dir / "changelog.json")
        self._chunk_size = settings.chunk_size

        # Phase 4: event dispatch
        self._event_dispatcher = EventDispatcher()
        if settings.enable_event_dispatch:
            audit_path = (
                Path(settings.audit_log_path)
                if settings.audit_log_path
                else settings.brain_dir / "audit.jsonl"
            )
            self._event_dispatcher.register(AuditLogHandler(audit_path=audit_path))

        # Phase 4: conflict resolution
        self._resolution_audit = ResolutionAudit(settings.brain_dir / "resolutions.json")
        self._conflict_resolver: ConflictResolver | None = None
        self._conflict_detector: ConflictDetector | None = None
        if settings.enable_conflict_resolution:
            try:
                strategy = AutoResolveStrategy(settings.resolve_strategy)
            except ValueError:
                strategy = AutoResolveStrategy.AUTHORITY_WINS

            gitlab_handler = None
            if settings.enable_gitlab_handler and settings.gitlab_token:
                from noodly.dispatch.gitlab_handler import GitLabConfig, GitLabMRHandler

                gitlab_config = GitLabConfig(
                    url=settings.gitlab_url,
                    token=settings.gitlab_token,
                    project_id=settings.gitlab_project_id,
                    target_branch=settings.gitlab_target_branch,
                    knowledge_path=settings.gitlab_knowledge_path,
                )
                gitlab_handler = GitLabMRHandler(gitlab_config)
                self._event_dispatcher.register(
                    gitlab_handler,
                    event_types=[ChangeType.conflict_detected],
                )

            self._conflict_resolver = ConflictResolver(
                ledger=self._ledger,
                audit=self._resolution_audit,
                changelog=self._changelog,
                auto_threshold=settings.auto_resolve_threshold,
                strategy=strategy,
                similarity_threshold=settings.conflict_similarity_threshold,
                manual_handler=gitlab_handler,
            )
            self._conflict_detector = ConflictDetector(
                similarity_threshold=settings.conflict_similarity_threshold,
            )
            self._event_dispatcher.register(
                ConflictEscalationHandler(resolver=self._conflict_resolver),
                event_types=[ChangeType.conflict_detected],
            )

        # Phase 6: GitLab knowledge projection
        self._gitlab_projector = None
        if settings.enable_gitlab_projection and settings.gitlab_token:
            from noodly.dispatch.gitlab_handler import GitLabConfig
            from noodly.projection.gitlab import GitLabProjector

            gl_config = GitLabConfig(
                url=settings.gitlab_url,
                token=settings.gitlab_token,
                project_id=settings.gitlab_project_id,
                target_branch=settings.gitlab_target_branch,
                knowledge_path=settings.gitlab_knowledge_path,
            )
            self._gitlab_projector = GitLabProjector(gl_config)

        # Phase 4: semantic dedup
        self._semantic_dedup = None
        if settings.enable_semantic_dedup and settings.openai_api_key:
            from noodly.scoring.semantic_dedup import SemanticDeduplicator

            self._semantic_dedup = SemanticDeduplicator(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                threshold=settings.semantic_dedup_threshold,
            )

        # Phase 7: topic classifier
        self._topic_classifier = None
        if settings.enable_topic_clustering and settings.openai_api_key:
            from noodly.scoring.topic_classifier import TopicClassifier

            self._topic_classifier = TopicClassifier(
                api_key=settings.openai_api_key,
                model=settings.topic_model,
                mode=settings.authority_topic_inference,
                cache_path=settings.brain_dir / "topic_cache.json",
            )

    async def initialize(self) -> None:
        """Set up graph indices."""
        await self._brain.initialize()
        logger.info("Pipeline initialized")

    async def run(self) -> dict[str, int]:
        """Run one full cycle: scan → parse → diff → extract → enrich → project."""
        stats = {"artifacts": 0, "claims": 0, "projected": 0, "cached_chunks": 0}

        artifacts = await self._connector.scan()
        stats["artifacts"] = len(artifacts)
        if not artifacts:
            logger.info("No new artifacts found")
            return stats

        all_claims: list[Claim] = []
        for artifact in artifacts:
            claims, cached = await self._process_artifact(artifact)
            all_claims.extend(claims)
            stats["cached_chunks"] += cached

        # Phase 4/5: semantic dedup before storing — actually merge evidence
        if self._semantic_dedup is not None and all_claims:
            existing = self._ledger.list_claims(limit=10000)
            dedup_result = await self._semantic_dedup.deduplicate_and_merge(all_claims, existing)
            if dedup_result.merged_count > 0:
                logger.info(
                    "Semantic dedup: merged %d claims, %d unique",
                    dedup_result.merged_count,
                    dedup_result.unique_count,
                )
                stats["semantic_merged"] = dedup_result.merged_count
                # Auto-promote merged claims and save
                for _, existing_claim in dedup_result.merged:
                    self._ledger._auto_promote(existing_claim)
                self._ledger._save()
                # Only store the unique (unmatched) claims
                all_claims = dedup_result.unique

        stored = self._ledger.add_claims(all_claims)
        stats["claims"] = len(stored)

        self._ledger.apply_decay()

        # Phase 4: conflict detection + resolution
        if self._conflict_detector is not None and all_claims:
            existing_claims = self._ledger.list_claims(limit=10000)
            conflicts = self._conflict_detector.detect(all_claims, existing_claims)
            if conflicts and self._conflict_resolver is not None:
                resolutions = await self._conflict_resolver.resolve_batch(conflicts)
                stats["conflicts_detected"] = len(conflicts)
                stats["conflicts_auto_resolved"] = sum(
                    1 for r in resolutions if r.winner_id is not None
                )
                stats["conflicts_manual_pending"] = sum(
                    1 for r in resolutions if r.winner_id is None
                )

        if self._settings.enable_graph_agent and all_claims:
            await self._run_graph_agent(all_claims)

        # Phase 7: classify topics for projection and authority
        all_ledger_claims = self._ledger.list_claims(limit=10000)
        changed_claim_ids = {str(c.id) for c in all_claims}
        topic_map: dict[str, list[str]] | None = None
        if self._topic_classifier is not None:
            topic_map = await self._topic_classifier.classify(
                all_ledger_claims,
                existing_topics=self._topic_classifier.get_all_topics(),
            )
            stats["topics_classified"] = len(
                {t for ts in topic_map.values() for t in ts}
            )

        force_full = self._settings.emission_mode == "full"
        stats["projected"] = self._projector.project(
            all_ledger_claims,
            topic_map=topic_map,
            changed_claim_ids=changed_claim_ids,
            force_full=force_full,
        )

        # Phase 6: sync to GitLab (incremental — only changed subjects)
        if self._gitlab_projector is not None:
            changed_subjects = {c.subject for c in all_claims}
            try:
                gl_result = await self._gitlab_projector.sync_incremental(
                    all_ledger_claims, changed_subjects
                )
                stats["gitlab_synced"] = gl_result.total_changes
                if gl_result.total_changes > 0:
                    logger.info("GitLab sync: %s", gl_result.summary)
            except Exception:
                logger.exception("GitLab projection sync failed")

        logger.info(
            "Pipeline complete: %d artifacts, %d claims (%d cached), %d projected",
            stats["artifacts"],
            stats["claims"],
            stats["cached_chunks"],
            stats["projected"],
        )
        return stats

    async def _process_artifact(self, artifact: SourceArtifact) -> tuple[list[Claim], int]:
        """Process a single artifact through parse → diff → extract."""
        source_uri = artifact.source_uri or str(artifact.id)
        file_hash = artifact.metadata.get("content_hash", "")

        # 1. Parse (use cache if available)
        markdown = artifact.body
        if artifact.source_uri and Path(artifact.source_uri).exists():
            path = Path(artifact.source_uri)
            if self._parser.can_parse(path):
                cached_doc = self._cache.parse.get(file_hash) if file_hash else None
                if cached_doc is not None:
                    markdown = cached_doc.markdown
                else:
                    parsed = self._parser.parse(path)
                    markdown = parsed.markdown
                    if file_hash:
                        self._cache.parse.put(file_hash, parsed)

        # 2. Strip boilerplate
        markdown = self._boilerplate.strip(markdown)

        # 3. Content diff
        content_diff = self._content_differ.diff(source_uri, markdown, file_hash)

        if content_diff.is_new:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.document_added,
                    source_uri=source_uri,
                    payload={"title": artifact.title, "sections": len(content_diff.added_sections)},
                )
            )
        elif content_diff.change_ratio > 0:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.document_modified,
                    source_uri=source_uri,
                    payload={
                        "change_ratio": round(content_diff.change_ratio, 3),
                        "summary": content_diff.summary,
                    },
                )
            )

        # 4. QA agent (optional)
        if (
            self._settings.enable_qa_agent
            and content_diff.change_ratio >= self._settings.qa_change_threshold
        ):
            await self._run_qa_agent(artifact, markdown, content_diff)

        # 5. Ingest into Graphiti
        try:
            artifact.body = markdown
            await self._brain.ingest_artifact(artifact)
        except Exception:
            logger.exception("Failed to ingest artifact %s", artifact.id)

        # 6. Chunk and extract (with caching + parallel dispatch)
        chunks = chunk_markdown(markdown, max_chars=self._chunk_size)
        all_claims: list[Claim] = []
        cached_count = 0

        source_file = Path(artifact.source_uri).name if artifact.source_uri else artifact.title

        # Separate cached vs uncached chunks
        pending_jobs: list[LLMJob] = []
        pending_hashes: list[str] = []

        for chunk in chunks:
            chunk_hash = content_hash(chunk.content)
            cached_claims_data = self._cache.extraction.get(chunk_hash)

            if cached_claims_data is not None:
                claims = self._deserialize_cached_claims(cached_claims_data, artifact)
                all_claims.extend(claims)
                cached_count += 1
                continue

            chunk_artifact = SourceArtifact(
                source_type=artifact.source_type,
                source_uri=artifact.source_uri,
                title=f"{artifact.title} [{chunk.heading or f'chunk-{chunk.index}'}]",
                body=chunk.content,
                author=artifact.author,
                content_created_at=artifact.content_created_at,
                metadata=artifact.metadata,
                id=artifact.id,
            )
            pending_jobs.append(
                LLMJob(
                    artifact=chunk_artifact,
                    source_filename=source_file,
                    chunk_index=chunk.index,
                )
            )
            pending_hashes.append(chunk_hash)

        # Dispatch uncached chunks in parallel
        if pending_jobs:
            results = await self._llm_dispatcher.submit_batch(pending_jobs)
            for result, chunk_hash in zip(results, pending_hashes):
                if result.error is not None:
                    logger.error(
                        "Chunk extraction failed for %s: %s",
                        result.job_id,
                        result.error,
                    )
                    continue
                all_claims.extend(result.claims)
                self._cache.extraction.put(
                    chunk_hash,
                    [
                        {
                            "subject": c.subject,
                            "predicate": c.predicate,
                            "object": c.object,
                            "natural_language": c.natural_language,
                            "knowledge_class": c.knowledge_class.value,
                            "confidence": c.confidence,
                            "source_span": c.evidence[0].source_span if c.evidence else "",
                            "source_artifact": (
                                c.evidence[0].source_artifact if c.evidence else ""
                            ),
                        }
                        for c in result.claims
                    ],
                )

        for claim in all_claims:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.claim_added,
                    entity_id=claim.subject,
                    source_uri=source_uri,
                    payload={"predicate": claim.predicate, "object": claim.object},
                )
            )

        return all_claims, cached_count

    def _deserialize_cached_claims(
        self, claims_data: list[dict], artifact: SourceArtifact
    ) -> list[Claim]:
        """Reconstruct Claim objects from cached extraction data."""
        claims: list[Claim] = []
        for data in claims_data:
            klass = KnowledgeClass.process
            try:
                klass = KnowledgeClass(data.get("knowledge_class", "process"))
            except ValueError:
                pass

            claim = Claim(
                subject=data["subject"],
                predicate=data["predicate"],
                object=data["object"],
                natural_language=data.get("natural_language", ""),
                confidence=data.get("confidence", 0.5),
                knowledge_class=klass,
                status=ClaimStatus.candidate,
                evidence=[
                    ClaimEvidence(
                        artifact_id=artifact.id,
                        supports=True,
                        source_span=data.get("source_span", ""),
                        source_artifact=data.get("source_artifact", ""),
                        author=artifact.author,
                    )
                ],
            )
            claims.append(claim)
        return claims

    async def _run_qa_agent(self, artifact, markdown, content_diff) -> None:
        """Run QA agent on the parsed output."""
        try:
            from noodly.agents.qa_agent import ExtractionQAAgent
            from noodly.parsing.parser import ParsedDocument

            agent = ExtractionQAAgent(
                api_key=self._settings.openai_api_key,
                model=self._settings.openai_model,
            )
            parsed = ParsedDocument(
                title=artifact.title,
                markdown=markdown,
                source_format=artifact.metadata.get("mime_type", ""),
            )
            result = await agent.review(parsed, content_diff)
            if result.issues:
                logger.warning(
                    "QA found %d issues in %s (quality: %s)",
                    len(result.issues),
                    artifact.title,
                    result.overall_quality,
                )
                for issue in result.issues:
                    logger.warning(
                        "  [%s] %s: %s", issue.severity, issue.section, issue.description
                    )
        except Exception:
            logger.exception("QA agent failed for %s", artifact.title)

    async def _run_graph_agent(self, new_claims) -> None:
        """Run graph population agent on new claims."""
        try:
            from noodly.agents.graph_agent import GraphPopulationAgent
            from noodly.agents.toolkit import AgentToolkit

            toolkit = AgentToolkit(
                ledger=self._ledger,
                changelog=self._changelog,
                cache=self._cache,
            )
            agent = GraphPopulationAgent(
                api_key=self._settings.openai_api_key,
                model=self._settings.openai_model,
                toolkit=toolkit,
                ledger=self._ledger,
                changelog=self._changelog,
            )
            report = await agent.enrich(new_claims=new_claims)
            if report.has_actions:
                logger.info("Graph agent: %s", report.summary)
        except Exception:
            logger.exception("Graph agent failed")

    async def ingest_text(self, title: str, body: str, author: str = "") -> dict[str, int]:
        """Ingest raw text directly (useful for CLI and testing)."""
        from noodly.models.artifacts import SourceType

        artifact = SourceArtifact(
            source_type=SourceType.manual,
            title=title,
            body=body,
            author=author,
        )

        await self._brain.ingest_artifact(artifact)
        claims = await self._extractor.extract(artifact)
        stored = self._ledger.add_claims(claims)

        all_claims = self._ledger.list_claims(limit=10000)
        projected = self._projector.project(all_claims)

        return {"artifacts": 1, "claims": len(stored), "projected": projected}

    async def close(self) -> None:
        """Shut down connections."""
        await self._brain.close()
        if self._gitlab_projector is not None:
            await self._gitlab_projector.close()
