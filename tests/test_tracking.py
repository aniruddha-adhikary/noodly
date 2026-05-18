"""Tests for the tracking module (content_differ, claim_differ, changelog)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from noodly.models.claims import Claim, KnowledgeClass
from noodly.tracking.changelog import ChangeEvent, ChangeLog, ChangeType
from noodly.tracking.claim_differ import ClaimDiff, ClaimDiffer, ClaimModification
from noodly.tracking.content_differ import ContentDiff, ContentDiffer, ModifiedSection


# --- ContentDiffer ---


class TestContentDiff:
    def test_is_new(self):
        d = ContentDiff(source_uri="a", old_hash="", new_hash="abc")
        assert d.is_new

    def test_not_new(self):
        d = ContentDiff(source_uri="a", old_hash="old", new_hash="new")
        assert not d.is_new

    def test_change_ratio_all_added(self):
        d = ContentDiff(
            source_uri="a", old_hash="", new_hash="x",
            added_sections=["s1", "s2"],
        )
        assert d.change_ratio == 1.0

    def test_change_ratio_no_changes(self):
        d = ContentDiff(
            source_uri="a", old_hash="x", new_hash="x",
            unchanged_sections=["s1", "s2", "s3"],
        )
        assert d.change_ratio == 0.0

    def test_change_ratio_mixed(self):
        d = ContentDiff(
            source_uri="a", old_hash="x", new_hash="y",
            added_sections=["s1"],
            unchanged_sections=["s2", "s3"],
        )
        assert abs(d.change_ratio - 1 / 3) < 0.01

    def test_change_ratio_empty(self):
        d = ContentDiff(source_uri="a", old_hash="x", new_hash="y")
        assert d.change_ratio == 1.0

    def test_summary(self):
        d = ContentDiff(
            source_uri="a", old_hash="x", new_hash="y",
            added_sections=["s1"],
            removed_sections=["s2"],
            unchanged_sections=["s3"],
        )
        assert "+1 added" in d.summary
        assert "-1 removed" in d.summary
        assert "=1 unchanged" in d.summary


class TestContentDiffer:
    def test_new_document(self, tmp_path):
        differ = ContentDiffer(tmp_path / "cache")
        md = "# Intro\nHello world"
        diff = differ.diff("doc1", md, "hash1")
        assert diff.is_new
        assert len(diff.added_sections) == 1

    def test_unchanged_document(self, tmp_path):
        differ = ContentDiffer(tmp_path / "cache")
        md = "# Intro\nHello world"
        differ.diff("doc1", md, "hash1")
        diff2 = differ.diff("doc1", md, "hash1")
        assert not diff2.is_new
        assert len(diff2.unchanged_sections) == 1
        assert diff2.change_ratio == 0.0

    def test_modified_document(self, tmp_path):
        differ = ContentDiffer(tmp_path / "cache")
        md1 = "# Intro\nHello world\n# Part 2\nOriginal content"
        differ.diff("doc1", md1, "hash1")
        md2 = "# Intro\nHello world\n# Part 2\nUpdated content here"
        diff = differ.diff("doc1", md2, "hash2")
        assert not diff.is_new
        assert len(diff.modified_sections) == 1
        assert len(diff.unchanged_sections) == 1

    def test_added_section(self, tmp_path):
        differ = ContentDiffer(tmp_path / "cache")
        md1 = "# A\nContent A"
        differ.diff("doc1", md1, "h1")
        md2 = "# A\nContent A\n# B\nContent B"
        diff = differ.diff("doc1", md2, "h2")
        assert len(diff.added_sections) == 1
        assert len(diff.unchanged_sections) == 1

    def test_removed_section(self, tmp_path):
        differ = ContentDiffer(tmp_path / "cache")
        md1 = "# A\nContent A\n# B\nContent B"
        differ.diff("doc1", md1, "h1")
        md2 = "# A\nContent A"
        diff = differ.diff("doc1", md2, "h2")
        assert len(diff.removed_sections) == 1

    def test_cache_persists(self, tmp_path):
        cache_dir = tmp_path / "cache"
        differ1 = ContentDiffer(cache_dir)
        differ1.diff("doc1", "# Title\nContent", "h1")
        differ2 = ContentDiffer(cache_dir)
        diff = differ2.diff("doc1", "# Title\nContent", "h1")
        assert diff.change_ratio == 0.0


# --- ClaimDiffer ---


def _make_claim(subject: str, predicate: str, obj: str, **kwargs) -> Claim:
    return Claim(subject=subject, predicate=predicate, object=obj, **kwargs)


class TestClaimDiff:
    def test_has_changes_when_added(self):
        d = ClaimDiff(source_uri="x", added_claims=[_make_claim("a", "b", "c")])
        assert d.has_changes

    def test_has_changes_when_removed(self):
        d = ClaimDiff(source_uri="x", removed_claims=[_make_claim("a", "b", "c")])
        assert d.has_changes

    def test_no_changes(self):
        d = ClaimDiff(source_uri="x", unchanged_claims=[_make_claim("a", "b", "c")])
        assert not d.has_changes

    def test_summary(self):
        d = ClaimDiff(
            source_uri="x",
            added_claims=[_make_claim("a", "b", "c")],
            removed_claims=[_make_claim("d", "e", "f")],
        )
        assert "+1 new" in d.summary
        assert "-1 removed" in d.summary


class TestClaimDiffer:
    def test_all_new(self):
        differ = ClaimDiffer()
        new = [_make_claim("A", "is", "B")]
        diff = differ.diff("src", new, [])
        assert len(diff.added_claims) == 1
        assert len(diff.removed_claims) == 0

    def test_all_removed(self):
        differ = ClaimDiffer()
        old = [_make_claim("A", "is", "B")]
        diff = differ.diff("src", [], old)
        assert len(diff.removed_claims) == 1
        assert len(diff.added_claims) == 0

    def test_unchanged(self):
        differ = ClaimDiffer()
        c = _make_claim("A", "is", "B")
        diff = differ.diff("src", [c], [c])
        assert len(diff.unchanged_claims) == 1
        assert not diff.has_changes

    def test_modified_confidence(self):
        differ = ClaimDiffer()
        old = _make_claim("A", "is", "B", confidence=0.5)
        new = _make_claim("A", "is", "B", confidence=0.9)
        diff = differ.diff("src", [new], [old])
        assert len(diff.modified_claims) == 1
        assert "confidence" in diff.modified_claims[0].changed_fields

    def test_modified_knowledge_class(self):
        differ = ClaimDiffer()
        old = _make_claim("A", "is", "B", knowledge_class=KnowledgeClass.process)
        new = _make_claim("A", "is", "B", knowledge_class=KnowledgeClass.stable)
        diff = differ.diff("src", [new], [old])
        assert len(diff.modified_claims) == 1
        assert "knowledge_class" in diff.modified_claims[0].changed_fields

    def test_case_insensitive_fingerprint(self):
        differ = ClaimDiffer()
        old = _make_claim("Apple", "IS", "Fruit")
        new = _make_claim("apple", "is", "fruit")
        diff = differ.diff("src", [new], [old])
        assert len(diff.unchanged_claims) == 1

    def test_substantive_change(self):
        mod = ClaimModification(
            old_claim=_make_claim("A", "is", "B"),
            new_claim=_make_claim("A", "was", "B"),
            changed_fields=["predicate"],
        )
        assert mod.is_substantive

    def test_non_substantive_change(self):
        mod = ClaimModification(
            old_claim=_make_claim("A", "is", "B"),
            new_claim=_make_claim("A", "is", "B"),
            changed_fields=["confidence"],
        )
        assert not mod.is_substantive


# --- ChangeLog ---


class TestChangeEvent:
    def test_round_trip(self):
        evt = ChangeEvent(
            change_type=ChangeType.claim_added,
            entity_id="Entity1",
            source_uri="file:///test.md",
            payload={"key": "value"},
            agent="qa_agent",
        )
        data = evt.to_dict()
        restored = ChangeEvent.from_dict(data)
        assert restored.change_type == ChangeType.claim_added
        assert restored.entity_id == "Entity1"
        assert restored.source_uri == "file:///test.md"
        assert restored.payload == {"key": "value"}
        assert restored.agent == "qa_agent"
        assert restored.id == evt.id

    def test_triggered_by_round_trip(self):
        parent_id = uuid4()
        evt = ChangeEvent(
            change_type=ChangeType.gap_detected,
            triggered_by=parent_id,
        )
        data = evt.to_dict()
        restored = ChangeEvent.from_dict(data)
        assert restored.triggered_by == parent_id


class TestChangeLog:
    def test_emit_and_count(self, tmp_path):
        log = ChangeLog(tmp_path / "cl.json")
        assert log.count == 0
        log.emit(ChangeEvent(ChangeType.claim_added, entity_id="E1"))
        assert log.count == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "cl.json"
        log1 = ChangeLog(path)
        log1.emit(ChangeEvent(ChangeType.document_added, source_uri="f.md"))
        log2 = ChangeLog(path)
        assert log2.count == 1

    def test_since(self, tmp_path):
        log = ChangeLog(tmp_path / "cl.json")
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        log.emit(ChangeEvent(ChangeType.claim_added, entity_id="E1"))
        events = log.since(old_time)
        assert len(events) == 1
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert len(log.since(future)) == 0

    def test_since_with_types(self, tmp_path):
        log = ChangeLog(tmp_path / "cl.json")
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        log.emit(ChangeEvent(ChangeType.claim_added, entity_id="E1"))
        log.emit(ChangeEvent(ChangeType.document_added, source_uri="f.md"))
        events = log.since(old, types=[ChangeType.claim_added])
        assert len(events) == 1

    def test_for_source(self, tmp_path):
        log = ChangeLog(tmp_path / "cl.json")
        log.emit(ChangeEvent(ChangeType.document_added, source_uri="a.md"))
        log.emit(ChangeEvent(ChangeType.document_added, source_uri="b.md"))
        assert len(log.for_source("a.md")) == 1

    def test_recent(self, tmp_path):
        log = ChangeLog(tmp_path / "cl.json")
        for i in range(5):
            log.emit(ChangeEvent(ChangeType.claim_added, entity_id=f"E{i}"))
        recent = log.recent(limit=3)
        assert len(recent) == 3
        assert recent[0].entity_id == "E4"

    def test_chain(self, tmp_path):
        log = ChangeLog(tmp_path / "cl.json")
        parent = log.emit(ChangeEvent(ChangeType.document_modified, source_uri="f.md"))
        child = log.emit(
            ChangeEvent(ChangeType.claim_modified, triggered_by=parent.id)
        )
        log.emit(
            ChangeEvent(ChangeType.gap_detected, triggered_by=child.id)
        )
        chain = log.chain(parent.id)
        assert len(chain) == 2
