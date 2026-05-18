"""Tests for Phase 7 — parallel dispatch, topic-aware authority, emission planner,
consolidated projector, and topic classifier."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from noodly.models.claims import Claim, ClaimEvidence
from noodly.scoring.authority import DEFAULT_AUTHORITY, AuthorityRegistry


def _make_claim(subject, predicate, obj, **kwargs):
    defaults = {
        "natural_language": f"{subject} {predicate} {obj}",
        "confidence": 0.8,
        "evidence": [
            ClaimEvidence(
                artifact_id=uuid4(),
                supports=True,
                author="test-source",
                source_artifact="test.pdf",
            )
        ],
    }
    defaults.update(kwargs)
    return Claim(subject=subject, predicate=predicate, object=obj, **defaults)


# ---------------------------------------------------------------------------
# Topic-aware authority
# ---------------------------------------------------------------------------


class TestTopicAwareAuthority:
    def test_flat_weight_unchanged(self, tmp_path):
        path = tmp_path / "authority.json"
        reg = AuthorityRegistry(path)
        reg.set("customs.gov.sg", 0.9)
        assert reg.get("customs.gov.sg") == 0.9

    def test_topic_specific_weight(self, tmp_path):
        path = tmp_path / "authority.json"
        reg = AuthorityRegistry(path)
        reg.set("customs.gov.sg", 0.7)
        reg.set("customs.gov.sg", 0.95, topic="trade")
        assert reg.get("customs.gov.sg", topic="trade") == 0.95
        assert reg.get("customs.gov.sg") == 0.7

    def test_resolution_order(self, tmp_path):
        path = tmp_path / "authority.json"
        reg = AuthorityRegistry(path)
        reg.set("source-a", 0.6)
        reg.set("source-a", 0.9, topic="finance")
        # Exact topic match
        assert reg.get("source-a", topic="finance") == 0.9
        # Unknown topic falls back to source default
        assert reg.get("source-a", topic="sports") == 0.6
        # No topic falls back to source default
        assert reg.get("source-a") == 0.6
        # Unknown source falls back to global default
        assert reg.get("unknown") == DEFAULT_AUTHORITY

    def test_remove_topic(self, tmp_path):
        path = tmp_path / "authority.json"
        reg = AuthorityRegistry(path)
        reg.set("src", 0.5)
        reg.set("src", 0.8, topic="legal")
        assert reg.remove("src", topic="legal")
        assert reg.get("src", topic="legal") == 0.5  # falls back to default

    def test_get_topics(self, tmp_path):
        path = tmp_path / "authority.json"
        reg = AuthorityRegistry(path)
        reg.set("src", 0.5)
        reg.set("src", 0.8, topic="trade")
        reg.set("src", 0.7, topic="finance")
        topics = reg.get_topics("src")
        assert "trade" in topics
        assert "finance" in topics

    def test_persistence(self, tmp_path):
        path = tmp_path / "authority.json"
        reg1 = AuthorityRegistry(path)
        reg1.set("s1", 0.6)
        reg1.set("s1", 0.9, topic="ml")

        reg2 = AuthorityRegistry(path)
        assert reg2.get("s1", topic="ml") == 0.9
        assert reg2.get("s1") == 0.6

    def test_backward_compat_flat_format(self, tmp_path):
        path = tmp_path / "authority.json"
        path.write_text(json.dumps({"old-source": 0.75}))
        reg = AuthorityRegistry(path)
        assert reg.get("old-source") == 0.75
        assert reg.get("old-source", topic="any") == 0.75


# ---------------------------------------------------------------------------
# Emission planner
# ---------------------------------------------------------------------------


class TestEmissionPlanner:
    def test_full_plan_creates_files(self, tmp_path):
        from noodly.projection.planner import EmissionPlanner

        planner = EmissionPlanner(tmp_path)
        rendered = {
            "entities/alpha.md": ("# Alpha\ncontent", ["c1", "c2"]),
            "entities/beta.md": ("# Beta\ncontent", ["c3"]),
        }
        plan = planner.plan(rendered, force_full=True)
        assert len(plan.files_to_create) == 2
        assert len(plan.files_to_update) == 0
        assert len(plan.files_to_delete) == 0

        written = planner.execute(plan)
        assert written == 2
        assert (tmp_path / "entities" / "alpha.md").exists()
        assert (tmp_path / "entities" / "beta.md").exists()

    def test_incremental_detects_changes(self, tmp_path):
        from noodly.projection.planner import EmissionPlanner

        planner = EmissionPlanner(tmp_path)

        # First run — full
        rendered_v1 = {
            "entities/a.md": ("# A\nv1", ["c1"]),
            "entities/b.md": ("# B\nv1", ["c2"]),
        }
        plan1 = planner.plan(rendered_v1, force_full=True)
        planner.execute(plan1)

        # Second run — only a.md changed
        rendered_v2 = {
            "entities/a.md": ("# A\nv2", ["c1"]),
            "entities/b.md": ("# B\nv1", ["c2"]),
        }
        plan2 = planner.plan(rendered_v2, changed_claim_ids={"c1"})
        assert len(plan2.files_to_update) == 1
        assert plan2.files_to_update[0].path == "entities/a.md"
        assert len(plan2.files_to_create) == 0

    def test_incremental_detects_deletions(self, tmp_path):
        from noodly.projection.planner import EmissionPlanner

        planner = EmissionPlanner(tmp_path)

        rendered_v1 = {
            "entities/a.md": ("# A", ["c1"]),
            "entities/b.md": ("# B", ["c2"]),
        }
        plan1 = planner.plan(rendered_v1, force_full=True)
        planner.execute(plan1)

        # b.md no longer in rendered set
        rendered_v2 = {
            "entities/a.md": ("# A", ["c1"]),
        }
        plan2 = planner.plan(rendered_v2, force_full=True)
        assert len(plan2.files_to_delete) == 1
        assert plan2.files_to_delete[0] == "entities/b.md"

    def test_manifest_json_roundtrip(self, tmp_path):
        from noodly.projection.planner import EmissionManifest, EmissionPlan, PlannedFile

        manifest = EmissionManifest(tmp_path / "_manifest.json")
        pf = PlannedFile(path="entities/x.md", content="# X", claim_ids=["c1", "c2"])
        plan = EmissionPlan(files_to_create=[pf])
        manifest.update(plan)

        manifest2 = EmissionManifest(tmp_path / "_manifest.json")
        assert manifest2.get_file_hash("entities/x.md") == pf.content_hash
        affected = manifest2.get_files_for_claims({"c1"})
        assert "entities/x.md" in affected


# ---------------------------------------------------------------------------
# Consolidated projector
# ---------------------------------------------------------------------------


class TestConsolidatedProjector:
    def test_entity_pages_created(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        claims = [
            _make_claim("PortNet", "uses", "Python"),
            _make_claim("PortNet", "deployed_on", "AWS"),
            _make_claim("Jane", "owns", "billing"),
        ]
        written = projector.project(claims)
        assert written >= 3
        assert (tmp_brain / "entities" / "portnet.md").exists()
        assert (tmp_brain / "entities" / "jane.md").exists()
        assert (tmp_brain / "index.md").exists()

    def test_entity_page_contains_all_claims(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        claims = [
            _make_claim("PortNet", "uses", "Python"),
            _make_claim("PortNet", "deployed_on", "AWS"),
        ]
        projector.project(claims)
        content = (tmp_brain / "entities" / "portnet.md").read_text()
        assert "Python" in content
        assert "AWS" in content

    def test_topic_pages_created(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        claims = [
            _make_claim("PortNet", "uses", "Python"),
            _make_claim("Jane", "knows", "Python"),
        ]
        topic_map = {
            str(claims[0].id): ["technology"],
            str(claims[1].id): ["technology"],
        }
        projector.project(claims, topic_map=topic_map)
        assert (tmp_brain / "topics" / "technology.md").exists()
        content = (tmp_brain / "topics" / "technology.md").read_text()
        assert "PortNet" in content or "Jane" in content

    def test_source_pages_created(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        ev = ClaimEvidence(
            artifact_id=uuid4(),
            supports=True,
            author="customs.gov.sg",
            source_artifact="circular-01.pdf",
        )
        claims = [
            _make_claim("GST", "rate_is", "9%", evidence=[ev]),
        ]
        projector.project(claims)
        sources_dir = tmp_brain / "sources"
        assert sources_dir.exists()
        source_files = list(sources_dir.glob("*.md"))
        assert len(source_files) >= 1

    def test_conflicts_page_created(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        c1 = _make_claim("X", "is", "A")
        c2 = _make_claim("X", "is", "B")
        c1.conflicts_with.append(c2.id)
        c2.conflicts_with.append(c1.id)
        projector.project([c1, c2])
        conflicts_dir = tmp_brain / "conflicts"
        assert conflicts_dir.exists()

    def test_file_linking_entity_to_source(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        ev = ClaimEvidence(
            artifact_id=uuid4(),
            supports=True,
            author="arxiv.org",
            source_artifact="attention.pdf",
        )
        claims = [_make_claim("Transformer", "uses", "attention", evidence=[ev])]
        projector.project(claims)
        entity_content = (tmp_brain / "entities" / "transformer.md").read_text()
        # Entity page should link to sources
        assert "../sources/" in entity_content or "sources/" in entity_content

    def test_index_page_summary(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        claims = [
            _make_claim("A", "is", "X"),
            _make_claim("B", "is", "Y"),
        ]
        projector.project(claims)
        index = (tmp_brain / "index.md").read_text()
        assert "Noodly Brain" in index
        assert "entities" in index.lower() or "A" in index

    def test_incremental_update(self, tmp_brain):
        from noodly.projection.markdown import MarkdownProjector

        projector = MarkdownProjector(tmp_brain)
        c1 = _make_claim("Alpha", "is", "first")
        c2 = _make_claim("Beta", "is", "second")
        projector.project([c1, c2], force_full=True)

        # Update only Alpha's claim
        c1_updated = _make_claim("Alpha", "is", "updated-first")
        c1_updated.id = c1.id
        projector.project(
            [c1_updated, c2],
            changed_claim_ids={str(c1.id)},
        )
        content = (tmp_brain / "entities" / "alpha.md").read_text()
        assert "updated-first" in content


# ---------------------------------------------------------------------------
# Parallel dispatcher (unit-level, no real OpenAI calls)
# ---------------------------------------------------------------------------


class TestLLMJobDispatcher:
    def test_dispatch_stats_dataclass(self):
        from noodly.extraction.dispatcher import DispatchStats

        stats = DispatchStats(
            total_jobs=10,
            succeeded=8,
            failed=2,
            total_latency_ms=5000,
        )
        assert stats.avg_latency_ms == 625  # 5000 / 8 succeeded
        assert "10 jobs" in stats.summary
        assert "8 succeeded" in stats.summary

    def test_token_bucket_basic(self):
        from noodly.extraction.dispatcher import _TokenBucket

        bucket = _TokenBucket(rpm=600)
        # Should be able to acquire immediately
        loop = asyncio.new_event_loop()
        loop.run_until_complete(bucket.acquire())
        loop.close()

    def test_llm_job_dataclass(self):
        from noodly.extraction.dispatcher import LLMJob
        from noodly.models.artifacts import SourceArtifact, SourceType

        artifact = SourceArtifact(
            source_type=SourceType.manual,
            title="test",
            body="test body",
        )
        job = LLMJob(artifact=artifact, source_filename="test.md", chunk_index=0)
        assert job.artifact.title == "test"
        assert job.chunk_index == 0
        assert job.id is not None


# ---------------------------------------------------------------------------
# Topic classifier (keyword mode only — no LLM calls)
# ---------------------------------------------------------------------------


class TestTopicClassifierKeyword:
    def test_keyword_classify_trade(self):
        from noodly.scoring.topic_classifier import TopicClassifier

        classifier = TopicClassifier(mode="keyword")
        claim = _make_claim("Singapore", "imposes", "import tariff")
        result = asyncio.get_event_loop().run_until_complete(
            classifier.classify([claim])
        )
        assert str(claim.id) in result
        assert "trade-compliance" in result[str(claim.id)]

    def test_keyword_classify_http(self):
        from noodly.scoring.topic_classifier import TopicClassifier

        classifier = TopicClassifier(mode="keyword")
        claim = _make_claim("HTTP/2", "defines", "binary framing protocol")
        result = asyncio.get_event_loop().run_until_complete(
            classifier.classify([claim])
        )
        assert "http-protocol" in result[str(claim.id)]

    def test_keyword_classify_fallback(self):
        from noodly.scoring.topic_classifier import TopicClassifier

        classifier = TopicClassifier(mode="keyword")
        claim = _make_claim("XYZ", "does", "something random")
        result = asyncio.get_event_loop().run_until_complete(
            classifier.classify([claim])
        )
        assert "general" in result[str(claim.id)]

    def test_cache_persistence(self, tmp_path):
        from noodly.scoring.topic_classifier import TopicClassifier

        cache_path = tmp_path / "topic_cache.json"
        classifier = TopicClassifier(mode="keyword", cache_path=cache_path)
        claim = _make_claim("Customs", "requires", "export permit")
        asyncio.get_event_loop().run_until_complete(
            classifier.classify([claim])
        )
        assert cache_path.exists()

        # Reload from cache
        classifier2 = TopicClassifier(mode="keyword", cache_path=cache_path)
        result = asyncio.get_event_loop().run_until_complete(
            classifier2.classify([claim])
        )
        assert str(claim.id) in result

    def test_get_all_topics(self):
        from noodly.scoring.topic_classifier import TopicClassifier

        classifier = TopicClassifier(mode="keyword")
        claims = [
            _make_claim("A", "imports", "goods with tariff"),
            _make_claim("B", "uses", "http protocol request"),
        ]
        asyncio.get_event_loop().run_until_complete(
            classifier.classify(claims)
        )
        topics = classifier.get_all_topics()
        assert "trade-compliance" in topics
        assert "http-protocol" in topics


# ---------------------------------------------------------------------------
# Config settings for Phase 7
# ---------------------------------------------------------------------------


class TestPhase7Config:
    def test_default_settings(self):
        from noodly.config import Settings

        s = Settings(
            openai_api_key="test",
            watch_dir=Path("/tmp"),
            brain_dir=Path("/tmp/brain"),
        )
        assert s.llm_max_concurrent == 8
        assert s.llm_rate_limit_rpm == 500
        assert s.llm_retry_max == 3
        assert s.llm_request_timeout == 30.0
        assert s.emission_mode == "incremental"
        assert s.enable_topic_clustering is True
        assert s.topic_model == "gpt-4o-mini"
        assert s.authority_topic_inference == "llm"
