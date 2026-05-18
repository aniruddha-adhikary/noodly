"""Tests for the FactLedger with pluggable backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from noodly.models.claims import Claim, ClaimStatus
from noodly.scoring.ledger import AUTO_PROMOTE_THRESHOLD, FactLedger


def _make_claim(**overrides) -> Claim:
    defaults = dict(
        subject="PortNet",
        predicate="uses",
        object="Python",
        natural_language="PortNet uses Python.",
        confidence=0.8,
        authority=0.7,
    )
    defaults.update(overrides)
    return Claim(**defaults)


class TestFactLedgerWithJSONBackend:
    @pytest.fixture()
    def ledger(self, tmp_path: Path) -> FactLedger:
        return FactLedger(backend=tmp_path / "ledger.json")

    def test_path_constructor_backward_compat(self, tmp_path: Path) -> None:
        ledger = FactLedger(tmp_path / "ledger.json")
        assert ledger.count == 0

    def test_add_and_get_claim(self, ledger: FactLedger) -> None:
        claim = _make_claim()
        ledger.add_claim(claim)
        assert ledger.count == 1

        fetched = ledger.get_claim(str(claim.id))
        assert fetched is not None
        assert fetched.subject == "PortNet"

    def test_auto_promote(self, ledger: FactLedger) -> None:
        from noodly.models.claims import ClaimEvidence

        evidence = ClaimEvidence(
            artifact_id="12345678-1234-1234-1234-123456789abc",
            supports=True,
        )
        claim = _make_claim(
            confidence=0.9,
            authority=0.8,
            status=ClaimStatus.candidate,
            evidence=[evidence, evidence],
        )
        assert claim.truth_score >= AUTO_PROMOTE_THRESHOLD
        result = ledger.add_claim(claim)
        assert result.status == ClaimStatus.unverified

    def test_no_auto_promote_low_score(self, ledger: FactLedger) -> None:
        claim = _make_claim(
            confidence=0.1,
            authority=0.1,
            status=ClaimStatus.candidate,
        )
        assert claim.truth_score < AUTO_PROMOTE_THRESHOLD
        result = ledger.add_claim(claim)
        assert result.status == ClaimStatus.candidate

    def test_list_claims_sorted_by_score(self, ledger: FactLedger) -> None:
        c1 = _make_claim(confidence=0.3, authority=0.3)
        c2 = _make_claim(confidence=0.9, authority=0.9)
        ledger.add_claim(c1)
        ledger.add_claim(c2)

        results = ledger.list_claims()
        assert len(results) == 2
        assert results[0].truth_score >= results[1].truth_score

    def test_list_claims_filter_by_status(self, ledger: FactLedger) -> None:
        c1 = _make_claim(status=ClaimStatus.unverified)
        c2 = _make_claim(status=ClaimStatus.canonical)
        ledger.add_claim(c1)
        ledger.add_claim(c2)

        results = ledger.list_claims(status=ClaimStatus.canonical)
        assert len(results) == 1
        assert results[0].status == ClaimStatus.canonical

    def test_list_claims_filter_by_group(self, ledger: FactLedger) -> None:
        c1 = _make_claim(group_id="team-a")
        c2 = _make_claim(group_id="team-b")
        ledger.add_claim(c1)
        ledger.add_claim(c2)

        results = ledger.list_claims(group_id="team-a")
        assert len(results) == 1

    def test_promote_claim(self, ledger: FactLedger) -> None:
        claim = _make_claim(status=ClaimStatus.unverified)
        ledger.add_claim(claim)

        result = ledger.promote_claim(str(claim.id), ClaimStatus.canonical)
        assert result is not None
        assert result.status == ClaimStatus.canonical
        assert result.last_confirmed_at is not None

    def test_promote_nonexistent(self, ledger: FactLedger) -> None:
        result = ledger.promote_claim("nonexistent", ClaimStatus.canonical)
        assert result is None

    def test_apply_decay(self, ledger: FactLedger) -> None:
        claim = _make_claim(status=ClaimStatus.unverified)
        ledger.add_claim(claim)
        decayed = ledger.apply_decay()
        assert isinstance(decayed, int)

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        ledger1 = FactLedger(backend=path)
        claim = _make_claim()
        ledger1.add_claim(claim)

        ledger2 = FactLedger(backend=path)
        assert ledger2.count == 1

    def test_is_async_backend_false(self, ledger: FactLedger) -> None:
        assert ledger.is_async_backend is False

    def test_add_claims_batch(self, ledger: FactLedger) -> None:
        claims = [
            _make_claim(subject="A", predicate="is", object="B"),
            _make_claim(subject="C", predicate="has", object="D"),
        ]
        results = ledger.add_claims(claims)
        assert len(results) == 2
        assert ledger.count == 2
