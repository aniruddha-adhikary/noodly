"""Tests for Phase 4 — semantic dedup, conflict resolution, event dispatch, storage, GitLab."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from noodly.models.claims import Claim, ClaimEvidence, KnowledgeClass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    subject: str = "Entity",
    predicate: str = "has property",
    obj: str = "value",
    confidence: float = 0.8,
    authority: float = 0.5,
    knowledge_class: KnowledgeClass = KnowledgeClass.stable,
    evidence_count: int = 1,
    source_authority: float = 0.5,
) -> Claim:
    evidence = [
        ClaimEvidence(
            artifact_id=uuid4(),
            supports=True,
            source_span="test span",
            author=f"author-{i}",
            source_authority=source_authority,
        )
        for i in range(evidence_count)
    ]
    return Claim(
        subject=subject,
        predicate=predicate,
        object=obj,
        natural_language=f"{subject} {predicate} {obj}",
        confidence=confidence,
        authority=authority,
        knowledge_class=knowledge_class,
        evidence=evidence,
        created_at=datetime.now(timezone.utc),
    )


# ===========================================================================
# Conflict Detector
# ===========================================================================

class TestConflictDetector:
    def test_detect_contradictory_values(self):
        from noodly.resolution.detector import ConflictDetector

        detector = ConflictDetector(similarity_threshold=0.8)
        existing = [_make_claim(obj="value A")]
        new = [_make_claim(obj="value B")]

        conflicts = detector.detect(new, existing)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "contradictory_value"

    def test_no_conflict_same_object(self):
        from noodly.resolution.detector import ConflictDetector

        detector = ConflictDetector()
        existing = [_make_claim(obj="value")]
        new = [_make_claim(obj="value")]

        conflicts = detector.detect(new, existing)
        assert len(conflicts) == 0

    def test_no_conflict_similar_objects(self):
        from noodly.resolution.detector import ConflictDetector

        detector = ConflictDetector(similarity_threshold=0.5)
        existing = [_make_claim(obj="the quick brown fox")]
        new = [_make_claim(obj="the quick brown dog")]

        conflicts = detector.detect(new, existing)
        # 3/4 token overlap = 0.75, above 0.5 threshold → not a conflict
        assert len(conflicts) == 0

    def test_detect_within(self):
        from noodly.resolution.detector import ConflictDetector

        detector = ConflictDetector()
        claims = [
            _make_claim(obj="100"),
            _make_claim(obj="200"),
        ]
        conflicts = detector.detect_within(claims)
        assert len(conflicts) == 1

    def test_different_subjects_no_conflict(self):
        from noodly.resolution.detector import ConflictDetector

        detector = ConflictDetector()
        existing = [_make_claim(subject="A", obj="100")]
        new = [_make_claim(subject="B", obj="200")]

        conflicts = detector.detect(new, existing)
        assert len(conflicts) == 0

    def test_score_delta(self):
        from noodly.resolution.detector import ConflictDetector

        detector = ConflictDetector()
        claims = [
            _make_claim(obj="100", confidence=0.9, source_authority=0.9),
            _make_claim(obj="200", confidence=0.3, source_authority=0.3),
        ]
        conflicts = detector.detect_within(claims)
        assert len(conflicts) == 1
        assert conflicts[0].score_delta > 0


# ===========================================================================
# Resolution Strategies
# ===========================================================================

class TestResolutionStrategies:
    def test_authority_wins(self):
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.strategies import AutoResolveStrategy, resolve_by_strategy

        conflict = ConflictPair(
            id=uuid4(),
            claim_a=_make_claim(obj="A", source_authority=0.9),
            claim_b=_make_claim(obj="B", source_authority=0.3),
            conflict_type="contradictory_value",
            detected_at=datetime.now(timezone.utc),
            detected_by="test",
        )
        winner, loser, rationale = resolve_by_strategy(
            conflict, AutoResolveStrategy.AUTHORITY_WINS
        )
        assert winner.object == "A"
        assert loser.object == "B"
        assert "Authority" in rationale

    def test_recency_wins(self):
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.strategies import AutoResolveStrategy, resolve_by_strategy

        old = _make_claim(obj="old")
        old.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        new = _make_claim(obj="new")
        new.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

        conflict = ConflictPair(
            id=uuid4(),
            claim_a=old,
            claim_b=new,
            conflict_type="contradictory_value",
            detected_at=datetime.now(timezone.utc),
            detected_by="test",
        )
        winner, loser, rationale = resolve_by_strategy(
            conflict, AutoResolveStrategy.RECENCY_WINS
        )
        assert winner.object == "new"
        assert "Recency" in rationale

    def test_majority_wins(self):
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.strategies import AutoResolveStrategy, resolve_by_strategy

        conflict = ConflictPair(
            id=uuid4(),
            claim_a=_make_claim(obj="many", evidence_count=5),
            claim_b=_make_claim(obj="few", evidence_count=1),
            conflict_type="contradictory_value",
            detected_at=datetime.now(timezone.utc),
            detected_by="test",
        )
        winner, loser, rationale = resolve_by_strategy(
            conflict, AutoResolveStrategy.MAJORITY_WINS
        )
        assert winner.object == "many"
        assert "Majority" in rationale

    def test_higher_score(self):
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.strategies import AutoResolveStrategy, resolve_by_strategy

        conflict = ConflictPair(
            id=uuid4(),
            claim_a=_make_claim(obj="high", confidence=0.95, source_authority=0.9),
            claim_b=_make_claim(obj="low", confidence=0.2, source_authority=0.2),
            conflict_type="contradictory_value",
            detected_at=datetime.now(timezone.utc),
            detected_by="test",
        )
        winner, loser, rationale = resolve_by_strategy(
            conflict, AutoResolveStrategy.HIGHER_SCORE
        )
        assert winner.object == "high"
        assert "Score" in rationale


# ===========================================================================
# Resolution Audit
# ===========================================================================

class TestResolutionAudit:
    def test_record_and_retrieve(self, tmp_path):
        from noodly.resolution.audit import Resolution, ResolutionAudit

        audit = ResolutionAudit(tmp_path / "resolutions.json")
        conflict_id = uuid4()

        resolution = Resolution(
            id=uuid4(),
            conflict_id=conflict_id,
            winner_id=uuid4(),
            loser_id=uuid4(),
            strategy_used="auto:authority_wins",
            confidence=0.5,
            resolved_by="auto:authority_wins",
            resolved_at=datetime.now(timezone.utc),
            rationale="test",
        )
        audit.record(resolution)

        assert audit.count == 1
        found = audit.get_resolution(str(conflict_id))
        assert found is not None
        assert found.strategy_used == "auto:authority_wins"

    def test_persistence(self, tmp_path):
        from noodly.resolution.audit import Resolution, ResolutionAudit

        path = tmp_path / "resolutions.json"
        audit1 = ResolutionAudit(path)
        audit1.record(Resolution(
            id=uuid4(), conflict_id=uuid4(), winner_id=uuid4(),
            loser_id=uuid4(), strategy_used="test", confidence=0.5,
            resolved_by="test", resolved_at=datetime.now(timezone.utc),
        ))

        audit2 = ResolutionAudit(path)
        assert audit2.count == 1

    def test_pending_count(self, tmp_path):
        from noodly.resolution.audit import Resolution, ResolutionAudit

        audit = ResolutionAudit(tmp_path / "resolutions.json")
        audit.record(Resolution(
            id=uuid4(), conflict_id=uuid4(), winner_id=None,
            loser_id=None, strategy_used="manual:pending", confidence=0.1,
            resolved_by="manual:pending", resolved_at=datetime.now(timezone.utc),
        ))
        audit.record(Resolution(
            id=uuid4(), conflict_id=uuid4(), winner_id=uuid4(),
            loser_id=uuid4(), strategy_used="auto:test", confidence=0.5,
            resolved_by="auto:test", resolved_at=datetime.now(timezone.utc),
        ))

        assert audit.pending_count() == 1
        assert audit.count == 2


# ===========================================================================
# Conflict Resolver
# ===========================================================================

class TestConflictResolver:
    @pytest.mark.asyncio
    async def test_auto_resolve(self, tmp_path):
        from noodly.resolution.audit import ResolutionAudit
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.resolver import ConflictResolver
        from noodly.resolution.strategies import AutoResolveStrategy
        from noodly.scoring.ledger import FactLedger
        from noodly.tracking.changelog import ChangeLog

        ledger = FactLedger(tmp_path / "ledger.json")
        audit = ResolutionAudit(tmp_path / "resolutions.json")
        changelog = ChangeLog(tmp_path / "changelog.json")

        claim_a = _make_claim(obj="winner", confidence=0.9, source_authority=0.9)
        claim_b = _make_claim(obj="loser", confidence=0.2, source_authority=0.2)
        ledger.add_claim(claim_a)
        ledger.add_claim(claim_b)

        conflict = ConflictPair(
            id=uuid4(),
            claim_a=claim_a,
            claim_b=claim_b,
            conflict_type="contradictory_value",
            detected_at=datetime.now(timezone.utc),
            detected_by="test",
        )

        resolver = ConflictResolver(
            ledger=ledger,
            audit=audit,
            changelog=changelog,
            auto_threshold=0.1,
            strategy=AutoResolveStrategy.AUTHORITY_WINS,
        )

        resolution = await resolver.resolve(conflict)
        assert resolution.winner_id is not None
        assert resolution.strategy_used == "auto:authority_wins"
        assert audit.count == 1

    @pytest.mark.asyncio
    async def test_manual_dispatch(self, tmp_path):
        from noodly.resolution.audit import ResolutionAudit
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.resolver import ConflictResolver
        from noodly.scoring.ledger import FactLedger

        ledger = FactLedger(tmp_path / "ledger.json")
        audit = ResolutionAudit(tmp_path / "resolutions.json")

        # Claims with very similar scores → manual
        claim_a = _make_claim(obj="A", confidence=0.5, source_authority=0.5)
        claim_b = _make_claim(obj="B", confidence=0.5, source_authority=0.5)
        ledger.add_claim(claim_a)
        ledger.add_claim(claim_b)

        conflict = ConflictPair(
            id=uuid4(),
            claim_a=claim_a,
            claim_b=claim_b,
            conflict_type="contradictory_value",
            detected_at=datetime.now(timezone.utc),
            detected_by="test",
        )

        resolver = ConflictResolver(
            ledger=ledger,
            audit=audit,
            auto_threshold=0.5,  # High threshold → manual
        )

        resolution = await resolver.resolve(conflict)
        assert resolution.winner_id is None
        assert "manual" in resolution.strategy_used

    @pytest.mark.asyncio
    async def test_batch_resolve(self, tmp_path):
        from noodly.resolution.audit import ResolutionAudit
        from noodly.resolution.detector import ConflictPair
        from noodly.resolution.resolver import ConflictResolver
        from noodly.scoring.ledger import FactLedger

        ledger = FactLedger(tmp_path / "ledger.json")
        audit = ResolutionAudit(tmp_path / "resolutions.json")

        conflicts = []
        for i in range(3):
            a = _make_claim(obj=f"A{i}", confidence=0.9, source_authority=0.9)
            b = _make_claim(obj=f"B{i}", confidence=0.1, source_authority=0.1)
            ledger.add_claim(a)
            ledger.add_claim(b)
            conflicts.append(ConflictPair(
                id=uuid4(), claim_a=a, claim_b=b,
                conflict_type="contradictory_value",
                detected_at=datetime.now(timezone.utc),
                detected_by="test",
            ))

        resolver = ConflictResolver(
            ledger=ledger, audit=audit, auto_threshold=0.1,
        )
        resolutions = await resolver.resolve_batch(conflicts)
        assert len(resolutions) == 3
        assert all(r.winner_id is not None for r in resolutions)


# ===========================================================================
# Event Dispatcher
# ===========================================================================

class TestEventDispatcher:
    @pytest.mark.asyncio
    async def test_dispatch_to_global_handler(self):
        from noodly.dispatch.dispatcher import EventDispatcher, EventHandler, HandlerResult
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        class TestHandler(EventHandler):
            name = "test"
            calls = 0

            async def handle(self, event: ChangeEvent) -> HandlerResult:
                self.calls += 1
                return HandlerResult(
                    handler_name=self.name, event_id=event.id,
                    success=True, action_taken="tested",
                )

        dispatcher = EventDispatcher()
        handler = TestHandler()
        dispatcher.register(handler)

        event = ChangeEvent(change_type=ChangeType.claim_added, entity_id="test")
        results = await dispatcher.dispatch(event)

        assert len(results) == 1
        assert results[0].success
        assert handler.calls == 1

    @pytest.mark.asyncio
    async def test_typed_handler_filters(self):
        from noodly.dispatch.dispatcher import EventDispatcher, EventHandler, HandlerResult
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        class ConflictOnly(EventHandler):
            name = "conflict_only"
            calls = 0

            async def handle(self, event: ChangeEvent) -> HandlerResult:
                self.calls += 1
                return HandlerResult(
                    handler_name=self.name, event_id=event.id,
                    success=True, action_taken="handled",
                )

        dispatcher = EventDispatcher()
        handler = ConflictOnly()
        dispatcher.register(handler, event_types=[ChangeType.conflict_detected])

        # Should not trigger
        event1 = ChangeEvent(change_type=ChangeType.claim_added, entity_id="test")
        results1 = await dispatcher.dispatch(event1)
        assert len(results1) == 0
        assert handler.calls == 0

        # Should trigger
        event2 = ChangeEvent(change_type=ChangeType.conflict_detected, entity_id="test")
        results2 = await dispatcher.dispatch(event2)
        assert len(results2) == 1
        assert handler.calls == 1

    @pytest.mark.asyncio
    async def test_unregister(self):
        from noodly.dispatch.dispatcher import EventDispatcher, EventHandler, HandlerResult
        from noodly.tracking.changelog import ChangeEvent

        class DummyHandler(EventHandler):
            name = "dummy"

            async def handle(self, event: ChangeEvent) -> HandlerResult:
                return HandlerResult(
                    handler_name=self.name, event_id=event.id,
                    success=True, action_taken="ok",
                )

        dispatcher = EventDispatcher()
        dispatcher.register(DummyHandler())
        assert dispatcher.handler_count == 1

        dispatcher.unregister("dummy")
        assert dispatcher.handler_count == 0

    @pytest.mark.asyncio
    async def test_handler_error_captured(self):
        from noodly.dispatch.dispatcher import EventDispatcher, EventHandler, HandlerResult
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        class BrokenHandler(EventHandler):
            name = "broken"

            async def handle(self, event: ChangeEvent) -> HandlerResult:
                raise RuntimeError("intentional failure")

        dispatcher = EventDispatcher()
        dispatcher.register(BrokenHandler())

        event = ChangeEvent(change_type=ChangeType.claim_added, entity_id="test")
        results = await dispatcher.dispatch(event)
        assert len(results) == 1
        assert not results[0].success
        assert results[0].action_taken == "error"


# ===========================================================================
# Audit Log Handler
# ===========================================================================

class TestAuditLogHandler:
    @pytest.mark.asyncio
    async def test_writes_to_file(self, tmp_path):
        from noodly.dispatch.handlers import AuditLogHandler
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        audit_path = tmp_path / "audit.jsonl"
        handler = AuditLogHandler(audit_path=audit_path)

        event = ChangeEvent(
            change_type=ChangeType.claim_added,
            entity_id="TestEntity",
            source_uri="test.md",
            payload={"key": "value"},
        )
        result = await handler.handle(event)
        assert result.success
        assert result.action_taken == "logged"

        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["entity_id"] == "TestEntity"
        assert record["change_type"] == "claim_added"

    @pytest.mark.asyncio
    async def test_append_only(self, tmp_path):
        from noodly.dispatch.handlers import AuditLogHandler
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        audit_path = tmp_path / "audit.jsonl"
        handler = AuditLogHandler(audit_path=audit_path)

        for i in range(3):
            event = ChangeEvent(
                change_type=ChangeType.claim_added,
                entity_id=f"entity-{i}",
            )
            await handler.handle(event)

        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 3


# ===========================================================================
# Storage Backend Protocol
# ===========================================================================

class TestJSONBackend:
    def test_save_and_load(self, tmp_path):
        from noodly.storage.json_backend import JSONBackend

        backend = JSONBackend(tmp_path / "claims.json")
        claim = _make_claim()
        backend.save_claim(claim)

        claims = backend.load_claims()
        assert len(claims) == 1
        loaded = list(claims.values())[0]
        assert loaded.subject == claim.subject

    def test_save_all(self, tmp_path):
        from noodly.storage.json_backend import JSONBackend

        backend = JSONBackend(tmp_path / "claims.json")
        claims_list = [_make_claim(obj=f"val-{i}") for i in range(5)]
        claims_dict = {str(c.id): c for c in claims_list}
        backend.save_all(claims_dict)

        loaded = backend.load_claims()
        assert len(loaded) == 5

    def test_delete_claim(self, tmp_path):
        from noodly.storage.json_backend import JSONBackend

        backend = JSONBackend(tmp_path / "claims.json")
        claim = _make_claim()
        backend.save_claim(claim)
        assert len(backend.load_claims()) == 1

        backend.delete_claim(str(claim.id))
        assert len(backend.load_claims()) == 0

    def test_persistence(self, tmp_path):
        from noodly.storage.json_backend import JSONBackend

        path = tmp_path / "claims.json"
        b1 = JSONBackend(path)
        b1.save_claim(_make_claim())

        b2 = JSONBackend(path)
        assert len(b2.load_claims()) == 1


# ===========================================================================
# Semantic Dedup
# ===========================================================================

class TestSemanticDedup:
    def test_cosine_similarity(self):
        from noodly.scoring.semantic_dedup import SemanticDeduplicator

        # Identical vectors
        assert SemanticDeduplicator._cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
        # Orthogonal vectors
        assert SemanticDeduplicator._cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0
        # Empty vectors
        assert SemanticDeduplicator._cosine_similarity([], []) == 0.0
        # Different lengths
        assert SemanticDeduplicator._cosine_similarity([1, 0], [1, 0, 0]) == 0.0

    def test_claim_text(self):
        from noodly.scoring.semantic_dedup import SemanticDeduplicator

        claim = _make_claim(subject="SLA", predicate="is alias of", obj="Singapore Land Authority")
        text = SemanticDeduplicator._claim_text(claim)
        assert "SLA" in text
        assert "Singapore Land Authority" in text


# ===========================================================================
# LLM Extraction Prompt Improvements
# ===========================================================================

class TestExtractionPrompts:
    def test_schema_includes_valid_from_until(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        item_props = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["properties"]
        assert "valid_from" in item_props
        assert "valid_until" in item_props

    def test_schema_requires_valid_from_until(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        required = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["required"]
        assert "valid_from" in required
        assert "valid_until" in required

    def test_system_prompt_mentions_aliases(self):
        from noodly.extraction.extractor import EXTRACTION_SYSTEM_PROMPT

        assert "alias" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_tables(self):
        from noodly.extraction.extractor import EXTRACTION_SYSTEM_PROMPT

        assert "table" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_valid_from(self):
        from noodly.extraction.extractor import EXTRACTION_SYSTEM_PROMPT

        assert "valid_from" in EXTRACTION_SYSTEM_PROMPT

    def test_parse_date_helper(self):
        from noodly.extraction.extractor import _parse_date

        assert _parse_date("2024-01-15") is not None
        assert _parse_date("2024-01-15").year == 2024
        assert _parse_date(None) is None
        assert _parse_date("") is None
        assert _parse_date("null") is None
        assert _parse_date("not-a-date") is None

    def test_extracted_claim_has_temporal_fields(self):
        from noodly.extraction.extractor import ExtractedClaim

        ec = ExtractedClaim(
            subject="test",
            predicate="pred",
            object="obj",
            natural_language="test",
            knowledge_class="stable",
            confidence=0.9,
            source_span="span",
            valid_from="2024-01-01",
            valid_until="2024-12-31",
        )
        assert ec.valid_from == "2024-01-01"
        assert ec.valid_until == "2024-12-31"


# ===========================================================================
# Config Settings
# ===========================================================================

class TestPhase4Config:
    def test_default_settings(self):
        from noodly.config import Settings

        s = Settings(openai_api_key="test")
        assert s.storage_backend == "json"
        assert s.auto_resolve_threshold == 0.3
        assert s.resolve_strategy == "authority_wins"
        assert s.semantic_dedup_threshold == 0.92
        assert s.embedding_model == "text-embedding-3-large"
        assert s.gitlab_url == "https://gitlab.com"
        assert s.enable_docling is False
        assert s.extraction_mode == "auto"
        assert s.enable_conflict_resolution is False
        assert s.enable_event_dispatch is False
        assert s.enable_gitlab_handler is False

    def test_custom_gitlab_url(self):
        from noodly.config import Settings

        s = Settings(
            openai_api_key="test",
            gitlab_url="https://gitlab.mycompany.com",
            gitlab_token="glpat-xxx",
            gitlab_project_id="42",
        )
        assert s.gitlab_url == "https://gitlab.mycompany.com"
        assert s.gitlab_token == "glpat-xxx"


# ===========================================================================
# GitLab Handler
# ===========================================================================

class TestGitLabConfig:
    def test_api_url(self):
        from noodly.dispatch.gitlab_handler import GitLabConfig

        config = GitLabConfig(url="https://gitlab.mycompany.com/")
        assert config.api_url == "https://gitlab.mycompany.com/api/v4"

    def test_api_url_default(self):
        from noodly.dispatch.gitlab_handler import GitLabConfig

        config = GitLabConfig()
        assert config.api_url == "https://gitlab.com/api/v4"

    def test_custom_enterprise_url(self):
        from noodly.dispatch.gitlab_handler import GitLabConfig

        config = GitLabConfig(
            url="https://git.internal.corp.com",
            token="glpat-xxx",
            project_id="100",
        )
        assert "git.internal.corp.com" in config.api_url


class TestGitLabMRHandler:
    @pytest.mark.asyncio
    async def test_skips_without_token(self):
        from noodly.dispatch.gitlab_handler import GitLabConfig, GitLabMRHandler
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        handler = GitLabMRHandler(GitLabConfig(token=""))
        event = ChangeEvent(
            change_type=ChangeType.conflict_detected,
            entity_id="test",
            payload={"resolution": "manual"},
        )
        result = await handler.handle(event)
        assert not result.success
        assert result.action_taken == "skipped"

    @pytest.mark.asyncio
    async def test_skips_auto_resolved(self):
        from noodly.dispatch.gitlab_handler import GitLabConfig, GitLabMRHandler
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        handler = GitLabMRHandler(GitLabConfig(token="test-token"))
        event = ChangeEvent(
            change_type=ChangeType.conflict_detected,
            entity_id="test",
            payload={"resolution": "auto"},
        )
        result = await handler.handle(event)
        assert result.success
        assert result.action_taken == "skipped"

    def test_accepts_only_conflict_events(self):
        from noodly.dispatch.gitlab_handler import GitLabConfig, GitLabMRHandler
        from noodly.tracking.changelog import ChangeEvent, ChangeType

        handler = GitLabMRHandler(GitLabConfig())
        assert handler.accepts(
            ChangeEvent(change_type=ChangeType.conflict_detected, entity_id="test")
        )
        assert not handler.accepts(
            ChangeEvent(change_type=ChangeType.claim_added, entity_id="test")
        )


# ===========================================================================
# Extraction Orchestrator
# ===========================================================================

class TestExtractionOrchestrator:
    @pytest.mark.asyncio
    async def test_delegates_to_extractor(self):
        from unittest.mock import AsyncMock, MagicMock

        from noodly.extraction.orchestrator import ExtractionOrchestrator
        from noodly.models.artifacts import SourceArtifact, SourceType

        parser = MagicMock()
        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=[_make_claim()])

        orch = ExtractionOrchestrator(
            parser=parser,
            extractor=extractor,
            enable_docling=False,
        )

        artifact = SourceArtifact(
            source_type=SourceType.manual,
            title="test",
            body="test content",
        )
        claims = await orch.extract(artifact)
        assert len(claims) == 1
        extractor.extract.assert_called_once()


# ===========================================================================
# Docling Parser Integration
# ===========================================================================

class TestDoclingParser:
    def test_docling_not_installed_fallback(self, tmp_path):
        from noodly.parsing.parser import DocumentParser

        parser = DocumentParser(enable_docling=True)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        # Should fall back to text parsing even when docling requested
        result = parser.parse(test_file, backend="docling")
        assert "Hello world" in result.markdown

    def test_markitdown_default(self, tmp_path):
        from noodly.parsing.parser import DocumentParser

        parser = DocumentParser(enable_docling=False)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        result = parser.parse(test_file)
        assert "Hello world" in result.markdown
