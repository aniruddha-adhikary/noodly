"""Tests for claim and artifact models."""

from __future__ import annotations

from uuid import uuid4

from noodly.models.claims import (
    Claim,
    ClaimEvidence,
    ClaimStatus,
    KnowledgeClass,
)


def test_claim_truth_score_basic():
    claim = Claim(
        subject="PortNet",
        predicate="uses",
        object="Python",
        confidence=0.8,
        authority=0.5,
        evidence=[
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.5),
        ],
    )
    score = claim.truth_score
    assert 0 < score < 1
    # corroboration = min(1.0, 1 * 0.25) = 0.25
    # score = 0.8 * 0.5 * 1.0 * 0.25 * 1.0 = 0.1
    assert abs(score - 0.1) < 0.01


def test_claim_truth_score_multiple_evidence():
    claim = Claim(
        subject="PortNet",
        predicate="uses",
        object="Python",
        confidence=0.8,
        authority=0.5,
        evidence=[
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.5),
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.7),
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.3),
        ],
    )
    # authority should use max supporting = 0.7
    # corroboration = min(1.0, 3 * 0.25) = 0.75
    expected = 0.8 * 0.7 * 1.0 * 0.75 * 1.0
    assert abs(claim.truth_score - expected) < 0.01


def test_claim_truth_score_with_refuting_evidence():
    claim = Claim(
        subject="PortNet",
        predicate="uses",
        object="Python",
        confidence=0.8,
        authority=0.5,
        evidence=[
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.5),
            ClaimEvidence(artifact_id=uuid4(), supports=False, source_authority=0.5),
        ],
    )
    # supporting = 1, refuting = 1
    # authority from supporting = 0.5
    # corroboration = min(1.0, 1 * 0.25) = 0.25
    # conflict_penalty = 0.7^1 = 0.7
    expected = 0.8 * 0.5 * 1.0 * 0.25 * 0.7
    assert abs(claim.truth_score - expected) < 0.01


def test_claim_truth_score_with_conflicts():
    other_id = uuid4()
    claim = Claim(
        subject="PortNet",
        predicate="uses",
        object="Python",
        confidence=0.8,
        authority=0.5,
        conflicts_with=[other_id],
        evidence=[
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.5),
        ],
    )
    # conflict_penalty includes 0.9^1 for conflicts_with
    base = 0.8 * 0.5 * 1.0 * 0.25 * 1.0 * 0.9
    assert abs(claim.truth_score - base) < 0.01


def test_supporting_and_refuting_evidence():
    claim = Claim(
        subject="A",
        predicate="is",
        object="B",
        evidence=[
            ClaimEvidence(artifact_id=uuid4(), supports=True),
            ClaimEvidence(artifact_id=uuid4(), supports=False),
            ClaimEvidence(artifact_id=uuid4(), supports=True),
        ],
    )
    assert len(claim.supporting_evidence) == 2
    assert len(claim.refuting_evidence) == 1


def test_claim_defaults():
    claim = Claim(subject="X", predicate="Y", object="Z")
    assert claim.status == ClaimStatus.candidate
    assert claim.knowledge_class == KnowledgeClass.process
    assert claim.recency == 1.0
    assert claim.superseded_by is None
    assert claim.supersedes is None
    assert claim.conflicts_with == []


def test_claim_serialization_roundtrip():
    claim = Claim(
        subject="A",
        predicate="has",
        object="B",
        confidence=0.9,
        evidence=[ClaimEvidence(artifact_id=uuid4(), supports=True)],
        conflicts_with=[uuid4()],
    )
    data = claim.model_dump(mode="json")
    restored = Claim(**data)
    assert str(restored.id) == str(claim.id)
    assert restored.subject == claim.subject
    assert len(restored.conflicts_with) == 1
    assert len(restored.evidence) == 1
