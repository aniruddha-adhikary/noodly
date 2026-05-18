"""Tests for the fact ledger — dedup, authority, supersession, bi-temporal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass
from noodly.scoring.authority import AuthorityRegistry
from noodly.scoring.ledger import FactLedger, _claim_fingerprint


def _make_claim(subject="A", predicate="is", obj="B", **kwargs):
    return Claim(subject=subject, predicate=predicate, object=obj, **kwargs)


# --- Deduplication ---


def test_dedup_merges_evidence(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    art1, art2 = uuid4(), uuid4()

    c1 = _make_claim(
        evidence=[ClaimEvidence(artifact_id=art1, supports=True)],
        confidence=0.6,
    )
    c2 = _make_claim(
        evidence=[ClaimEvidence(artifact_id=art2, supports=True)],
        confidence=0.8,
    )
    result1 = ledger.add_claim(c1)
    result2 = ledger.add_claim(c2)

    # Should return the same claim (merged)
    assert str(result1.id) == str(result2.id)
    assert len(result2.evidence) == 2
    assert result2.confidence == 0.8  # max of 0.6, 0.8
    assert ledger.count == 1


def test_dedup_same_artifact_not_duplicated(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    art = uuid4()

    c1 = _make_claim(evidence=[ClaimEvidence(artifact_id=art, supports=True)])
    c2 = _make_claim(evidence=[ClaimEvidence(artifact_id=art, supports=True)])
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    claim = ledger.list_claims()[0]
    assert len(claim.evidence) == 1  # same artifact, not duplicated


def test_different_claims_not_deduped(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    c1 = _make_claim(subject="A", predicate="is", obj="B")
    c2 = _make_claim(subject="A", predicate="is", obj="C")
    ledger.add_claim(c1)
    ledger.add_claim(c2)
    assert ledger.count == 2


def test_fingerprint_case_insensitive():
    c1 = _make_claim(subject="PortNet", predicate="Uses", obj="Python")
    c2 = _make_claim(subject="portnet", predicate="uses", obj="python")
    assert _claim_fingerprint(c1) == _claim_fingerprint(c2)


# --- Authority integration ---


def test_authority_stamps_evidence(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    registry.set("jane@co.com", 0.9)

    ledger = FactLedger(tmp_path / "ledger.json", authority_registry=registry)
    claim = _make_claim(
        evidence=[ClaimEvidence(artifact_id=uuid4(), supports=True, author="jane@co.com")],
    )
    result = ledger.add_claim(claim)
    assert result.evidence[0].source_authority == 0.9


def test_authority_default_when_unknown(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    ledger = FactLedger(tmp_path / "ledger.json", authority_registry=registry)
    claim = _make_claim(
        evidence=[ClaimEvidence(artifact_id=uuid4(), supports=True, author="unknown")],
    )
    result = ledger.add_claim(claim)
    assert result.evidence[0].source_authority == 0.5


# --- Supersession ---


def test_supersede_claim(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    c1 = _make_claim(subject="X", predicate="is", obj="old")
    c2 = _make_claim(subject="X", predicate="is", obj="new")
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    assert ledger.supersede_claim(str(c1.id), str(c2.id)) is True

    old = ledger.get_claim(str(c1.id))
    new = ledger.get_claim(str(c2.id))
    assert old is not None and old.status == ClaimStatus.superseded
    assert old.superseded_by == c2.id
    assert new is not None and new.supersedes == c1.id


def test_supersede_missing_claim(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    assert ledger.supersede_claim("missing1", "missing2") is False


# --- Conflicts ---


def test_add_conflict(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    c1 = _make_claim(subject="X", predicate="is", obj="true")
    c2 = _make_claim(subject="X", predicate="is", obj="false")
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    assert ledger.add_conflict(str(c1.id), str(c2.id)) is True

    a = ledger.get_claim(str(c1.id))
    b = ledger.get_claim(str(c2.id))
    assert a is not None and c2.id in a.conflicts_with
    assert b is not None and c1.id in b.conflicts_with


# --- Retraction ---


def test_retract_evidence(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    art = uuid4()
    claim = _make_claim(
        evidence=[ClaimEvidence(artifact_id=art, supports=True)],
        confidence=0.8,
    )
    ledger.add_claim(claim)

    affected = ledger.retract_evidence(str(art))
    assert affected == 1

    updated = ledger.get_claim(str(claim.id))
    assert updated is not None
    assert updated.evidence[0].supports is False
    # truth score should now be lower (refuting evidence)
    assert updated.truth_score < 0.1


# --- Bi-temporal ---


def test_as_of_valid_filter(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    now = datetime.now(timezone.utc)

    c1 = _make_claim(
        subject="Policy",
        predicate="is",
        obj="active",
        valid_from=now - timedelta(days=30),
        valid_until=now + timedelta(days=30),
    )
    c2 = _make_claim(
        subject="OldPolicy",
        predicate="was",
        obj="active",
        valid_from=now - timedelta(days=365),
        valid_until=now - timedelta(days=100),
    )
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    # Query "now" should only return c1
    results = ledger.list_claims(as_of_valid=now)
    assert len(results) == 1
    assert results[0].subject == "Policy"

    # Query 200 days ago should return c2
    past = now - timedelta(days=200)
    results = ledger.list_claims(as_of_valid=past)
    assert len(results) == 1
    assert results[0].subject == "OldPolicy"


def test_as_of_transaction_filter(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    now = datetime.now(timezone.utc)

    c1 = _make_claim(subject="Early", predicate="is", obj="claim")
    c1.created_at = now - timedelta(days=10)
    c2 = _make_claim(subject="Late", predicate="is", obj="claim")
    c2.created_at = now

    ledger.add_claim(c1)
    ledger.add_claim(c2)

    # Query 5 days ago should only return c1
    results = ledger.list_claims(as_of_transaction=now - timedelta(days=5))
    assert len(results) == 1
    assert results[0].subject == "Early"


# --- Persistence ---


def test_ledger_persistence(tmp_path):
    path = tmp_path / "ledger.json"
    l1 = FactLedger(path)
    l1.add_claim(_make_claim(subject="Persistent", predicate="is", obj="yes"))

    l2 = FactLedger(path)
    assert l2.count == 1
    assert l2.list_claims()[0].subject == "Persistent"


# --- Decay ---


def test_apply_decay(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    now = datetime.now(timezone.utc)

    claim = _make_claim(
        knowledge_class=KnowledgeClass.stateful,
    )
    claim.created_at = now - timedelta(days=30)
    ledger.add_claim(claim)

    decayed = ledger.apply_decay()
    assert decayed == 1

    updated = ledger.get_claim(str(claim.id))
    assert updated is not None
    assert updated.recency < 1.0


# --- Auto-promotion ---


def test_auto_promote(tmp_path):
    ledger = FactLedger(tmp_path / "ledger.json")
    claim = _make_claim(
        confidence=0.9,
        authority=0.9,
        evidence=[
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.9),
            ClaimEvidence(artifact_id=uuid4(), supports=True, source_authority=0.9),
        ],
    )
    result = ledger.add_claim(claim)
    # 2 independent supporting sources → corroborated (Phase 8 promotion)
    assert result.status == ClaimStatus.corroborated
