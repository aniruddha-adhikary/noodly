"""Tests for the Markdown projector."""

from __future__ import annotations

from uuid import uuid4

from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass
from noodly.projection.markdown import MarkdownProjector


def _make_claim(subject, predicate, obj, **kwargs):
    defaults = {
        "natural_language": f"{subject} {predicate} {obj}",
        "confidence": 0.8,
        "evidence": [ClaimEvidence(artifact_id=uuid4(), supports=True)],
    }
    defaults.update(kwargs)
    return Claim(subject=subject, predicate=predicate, object=obj, **defaults)


def test_project_creates_files(tmp_brain):
    projector = MarkdownProjector(tmp_brain)
    claims = [
        _make_claim("PortNet", "uses", "Python"),
        _make_claim("PortNet", "deployed_on", "AWS"),
        _make_claim("Jane", "owns", "billing"),
    ]
    written = projector.project(claims)
    assert written >= 4  # 2 entities + 3 claim files + 1 index

    assert (tmp_brain / "index.md").exists()
    assert (tmp_brain / "entities" / "portnet.md").exists()
    assert (tmp_brain / "entities" / "jane.md").exists()


def test_project_index_content(tmp_brain):
    projector = MarkdownProjector(tmp_brain)
    claims = [_make_claim("Widget", "is", "blue")]
    projector.project(claims)

    index = (tmp_brain / "index.md").read_text()
    assert "Noodly Brain" in index
    assert "Widget" in index
    assert "Total active claims" in index


def test_project_entity_page_content(tmp_brain):
    projector = MarkdownProjector(tmp_brain)
    claims = [
        _make_claim("PortNet", "uses", "Python", confidence=0.9),
    ]
    projector.project(claims)

    entity_page = (tmp_brain / "entities" / "portnet.md").read_text()
    assert "# PortNet" in entity_page
    assert "score=" in entity_page
    assert "candidate" in entity_page


def test_project_skips_superseded(tmp_brain):
    projector = MarkdownProjector(tmp_brain)
    claims = [
        _make_claim("X", "is", "old", status=ClaimStatus.superseded),
        _make_claim("X", "is", "new", status=ClaimStatus.candidate),
    ]
    projector.project(claims)

    entity_page = (tmp_brain / "entities" / "x.md").read_text()
    assert "new" in entity_page
    assert "old" not in entity_page


def test_project_entity_pages_by_subject(tmp_brain):
    projector = MarkdownProjector(tmp_brain)
    claims = [
        _make_claim("A", "is", "stable", knowledge_class=KnowledgeClass.stable),
        _make_claim("B", "is", "process", knowledge_class=KnowledgeClass.process),
    ]
    projector.project(claims)

    # Phase 7: consolidated structure uses entities/ directory
    assert (tmp_brain / "entities" / "a.md").exists()
    assert (tmp_brain / "entities" / "b.md").exists()


def test_project_empty_claims(tmp_brain):
    projector = MarkdownProjector(tmp_brain)
    written = projector.project([])
    assert written == 0
