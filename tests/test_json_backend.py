"""Tests for the JSON storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from noodly.models.claims import Claim, ClaimEvidence
from noodly.storage.json_backend import JSONBackend


@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger.json"


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


class TestJSONBackend:
    def test_load_empty(self, tmp_ledger: Path) -> None:
        backend = JSONBackend(tmp_ledger)
        assert backend.load_claims() == {}

    def test_save_and_load_roundtrip(self, tmp_ledger: Path) -> None:
        backend = JSONBackend(tmp_ledger)
        claim = _make_claim()

        backend.save_claim(claim)
        loaded = backend.load_claims()

        assert len(loaded) == 1
        loaded_claim = list(loaded.values())[0]
        assert loaded_claim.subject == "PortNet"
        assert loaded_claim.predicate == "uses"
        assert loaded_claim.object == "Python"
        assert loaded_claim.confidence == 0.8

    def test_save_all(self, tmp_ledger: Path) -> None:
        backend = JSONBackend(tmp_ledger)
        c1 = _make_claim(subject="A", predicate="is", object="B")
        c2 = _make_claim(subject="C", predicate="has", object="D")

        claims = {str(c1.id): c1, str(c2.id): c2}
        backend.save_all(claims)

        loaded = backend.load_claims()
        assert len(loaded) == 2

    def test_delete_claim(self, tmp_ledger: Path) -> None:
        backend = JSONBackend(tmp_ledger)
        claim = _make_claim()
        backend.save_claim(claim)

        backend.delete_claim(str(claim.id))
        loaded = backend.load_claims()
        assert len(loaded) == 0

    def test_delete_nonexistent(self, tmp_ledger: Path) -> None:
        backend = JSONBackend(tmp_ledger)
        backend.delete_claim("nonexistent-id")

    def test_evidence_roundtrip(self, tmp_ledger: Path) -> None:
        backend = JSONBackend(tmp_ledger)
        evidence = ClaimEvidence(
            artifact_id="12345678-1234-1234-1234-123456789abc",
            supports=True,
            source_span="some text",
            author="Jane",
        )
        claim = _make_claim(evidence=[evidence])
        backend.save_claim(claim)

        loaded = backend.load_claims()
        loaded_claim = list(loaded.values())[0]
        assert len(loaded_claim.evidence) == 1
        assert loaded_claim.evidence[0].author == "Jane"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "ledger.json"
        backend = JSONBackend(deep_path)
        backend.save_claim(_make_claim())
        assert deep_path.exists()

    def test_corrupt_file_returns_empty(self, tmp_ledger: Path) -> None:
        tmp_ledger.write_text("not valid json{{{")
        backend = JSONBackend(tmp_ledger)
        assert backend.load_claims() == {}
