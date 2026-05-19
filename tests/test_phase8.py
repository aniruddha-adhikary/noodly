"""Tests for Phase 8 — ingestion-time semantic dedup, claim promotion pipeline,
graph node embeddings, and embedding management."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from uuid import uuid4

from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus
from noodly.scoring.ledger import (
    FactLedger,
    _claim_fingerprint,
    _claim_text,
)


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


def _make_embedding(dim: int = 8, seed: float = 1.0) -> list[float]:
    """Generate a deterministic unit-norm embedding vector."""
    raw = [math.sin(seed * (i + 1)) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# Claim embedding field
# ---------------------------------------------------------------------------


class TestClaimEmbedding:
    def test_claim_has_embedding_field(self):
        claim = _make_claim("A", "is", "B")
        assert claim.embedding == []

    def test_claim_with_embedding(self):
        emb = _make_embedding(dim=4)
        claim = _make_claim("A", "is", "B", embedding=emb)
        assert len(claim.embedding) == 4
        assert claim.embedding == emb

    def test_claim_serialization_with_embedding(self):
        emb = _make_embedding(dim=4)
        claim = _make_claim("A", "is", "B", embedding=emb)
        data = claim.model_dump(mode="json")
        assert "embedding" in data
        assert len(data["embedding"]) == 4

    def test_claim_deserialization_with_embedding(self):
        emb = _make_embedding(dim=4)
        claim = _make_claim("A", "is", "B", embedding=emb)
        data = claim.model_dump(mode="json")
        restored = Claim(**data)
        assert restored.embedding == emb


# ---------------------------------------------------------------------------
# Exact fingerprint dedup (regression)
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_exact_match_merges_evidence(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")
        c1 = _make_claim("X", "is", "Y")
        c2 = _make_claim("X", "is", "Y")
        ledger.add_claim(c1)
        result = ledger.add_claim(c2)
        assert result.id == c1.id
        assert len(result.evidence) >= 2
        assert ledger.count == 1

    def test_different_claims_stored_separately(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")
        c1 = _make_claim("X", "is", "Y")
        c2 = _make_claim("A", "is", "B")
        ledger.add_claim(c1)
        ledger.add_claim(c2)
        assert ledger.count == 2


# ---------------------------------------------------------------------------
# Exact dedup still works (semantic dedup now handled by Graphiti natively)
# ---------------------------------------------------------------------------


class TestIngestionDedup:
    def test_no_embedding_still_dedupes_by_fingerprint(self, tmp_path):
        """Claims without embeddings should still dedup by fingerprint."""
        ledger = FactLedger(tmp_path / "ledger.json")

        c1 = _make_claim("A", "is", "B")
        c2 = _make_claim("A", "is", "B")

        ledger.add_claim(c1)
        result = ledger.add_claim(c2)

        assert result.id == c1.id
        assert ledger.count == 1

    def test_different_claims_stored_separately(self, tmp_path):
        """Claims with different fingerprints should not be merged."""
        ledger = FactLedger(tmp_path / "ledger.json")

        c1 = _make_claim("A", "is", "B")
        c2 = _make_claim("A_similar", "is_like", "B_similar")

        ledger.add_claim(c1)
        ledger.add_claim(c2)

        assert ledger.count == 2

    def test_exact_match_takes_priority(self, tmp_path):
        """Exact fingerprint match should happen before any other dedup."""
        ledger = FactLedger(tmp_path / "ledger.json")

        emb1 = _make_embedding(dim=8, seed=1.0)
        emb2 = _make_embedding(dim=8, seed=2.0)

        c1 = _make_claim("X", "is", "Y", embedding=emb1)
        c2 = _make_claim("X", "is", "Y", embedding=emb2)  # same fingerprint

        ledger.add_claim(c1)
        result = ledger.add_claim(c2)

        assert result.id == c1.id


# ---------------------------------------------------------------------------
# Claim promotion pipeline
# ---------------------------------------------------------------------------


class TestClaimPromotion:
    def test_score_based_promotion(self, tmp_path):
        """Claims with truth_score >= promote_threshold should auto-promote to unverified."""
        ledger = FactLedger(tmp_path / "ledger.json", promote_threshold=0.10)
        claim = _make_claim("X", "is", "Y", confidence=0.8)
        result = ledger.add_claim(claim)
        # truth_score should be >= 0.10 for an 0.8 confidence claim
        assert result.status == ClaimStatus.unverified

    def test_low_score_stays_candidate(self, tmp_path):
        """Claims with very low truth_score should stay candidate."""
        ledger = FactLedger(tmp_path / "ledger.json", promote_threshold=0.99)
        claim = _make_claim("X", "is", "Y", confidence=0.1)
        result = ledger.add_claim(claim)
        assert result.status == ClaimStatus.candidate

    def test_corroboration_promotes_to_corroborated(self, tmp_path):
        """Claims with 2+ independent sources should promote to corroborated."""
        ledger = FactLedger(tmp_path / "ledger.json", corroboration_count=2)

        ev1 = ClaimEvidence(artifact_id=uuid4(), supports=True, author="source-a")
        ev2 = ClaimEvidence(artifact_id=uuid4(), supports=True, author="source-b")

        c1 = _make_claim("X", "is", "Y", evidence=[ev1])
        ledger.add_claim(c1)

        c2 = _make_claim("X", "is", "Y", evidence=[ev2])
        result = ledger.add_claim(c2)

        assert result.status == ClaimStatus.corroborated
        assert len(result.evidence) >= 2

    def test_high_authority_promotes_to_unverified(self, tmp_path):
        """Evidence from high-authority sources should auto-promote."""
        from noodly.scoring.authority import AuthorityRegistry

        auth_path = tmp_path / "authority.json"
        registry = AuthorityRegistry(auth_path)
        registry.set("official-source", 0.95)

        ledger = FactLedger(
            tmp_path / "ledger.json",
            authority_registry=registry,
            high_authority_threshold=0.8,
            promote_threshold=0.99,  # very high so score-based won't trigger
        )

        ev = ClaimEvidence(
            artifact_id=uuid4(), supports=True, author="official-source"
        )
        claim = _make_claim("X", "is", "Y", evidence=[ev], confidence=0.3)
        result = ledger.add_claim(claim)

        assert result.status == ClaimStatus.unverified

    def test_auto_promote_all(self, tmp_path):
        """auto_promote_all should promote eligible claims."""
        ledger = FactLedger(tmp_path / "ledger.json", promote_threshold=0.10)

        c1 = _make_claim("A", "is", "X", confidence=0.8)
        c1.status = ClaimStatus.candidate
        ledger._claims[str(c1.id)] = c1
        ledger._fingerprint_index[_claim_fingerprint(c1)] = str(c1.id)

        c2 = _make_claim("B", "is", "Y", confidence=0.01)
        c2.status = ClaimStatus.candidate
        ledger._claims[str(c2.id)] = c2
        ledger._fingerprint_index[_claim_fingerprint(c2)] = str(c2.id)

        promoted = ledger.auto_promote_all()
        assert promoted >= 1
        assert c1.status == ClaimStatus.unverified

    def test_promotion_stats(self, tmp_path):
        """promotion_stats should return accurate counts."""
        ledger = FactLedger(tmp_path / "ledger.json", promote_threshold=0.10)

        ledger.add_claim(_make_claim("A", "is", "X", confidence=0.8))
        ledger.add_claim(_make_claim("B", "is", "Y", confidence=0.01))

        stats = ledger.promotion_stats()
        assert isinstance(stats, dict)
        total = sum(stats.values())
        assert total == 2

    def test_configurable_corroboration_count(self, tmp_path):
        """Corroboration count should be configurable."""
        ledger = FactLedger(tmp_path / "ledger.json", corroboration_count=3)

        # Add claim with 2 sources — should NOT promote (needs 3)
        ev1 = ClaimEvidence(artifact_id=uuid4(), supports=True, author="s1")
        ev2 = ClaimEvidence(artifact_id=uuid4(), supports=True, author="s2")
        c1 = _make_claim("X", "is", "Y", evidence=[ev1])
        c1.status = ClaimStatus.candidate
        ledger.add_claim(c1)
        c2 = _make_claim("X", "is", "Y", evidence=[ev2])
        result = ledger.add_claim(c2)
        assert result.status != ClaimStatus.corroborated

        # Third source should trigger promotion
        ev3 = ClaimEvidence(artifact_id=uuid4(), supports=True, author="s3")
        c3 = _make_claim("X", "is", "Y", evidence=[ev3])
        result = ledger.add_claim(c3)
        assert result.status == ClaimStatus.corroborated


# ---------------------------------------------------------------------------
# Async claim addition
# ---------------------------------------------------------------------------


class TestAsyncClaimAddition:
    def test_add_claims_async_stores_multiple(self, tmp_path):
        """add_claims_async should add multiple claims."""
        ledger = FactLedger(tmp_path / "ledger.json")

        claims = [
            _make_claim("A", "is", "X"),
            _make_claim("B", "is", "Y"),
        ]

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(ledger.add_claims_async(claims))
        loop.close()

        assert len(results) == 2
        assert ledger.count == 2

    def test_add_claim_async_single(self, tmp_path):
        """add_claim_async should add a single claim."""
        ledger = FactLedger(tmp_path / "ledger.json")

        claim = _make_claim("A", "is", "X")
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(ledger.add_claim_async(claim))
        loop.close()

        assert result.id == claim.id
        assert ledger.count == 1


# ---------------------------------------------------------------------------
# Embedding stats and persistence
# ---------------------------------------------------------------------------


class TestEmbeddingPersistence:
    def test_embedded_count(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")
        c1 = _make_claim("A", "is", "X", embedding=_make_embedding(dim=4))
        c2 = _make_claim("B", "is", "Y")
        ledger.add_claim(c1)
        ledger.add_claim(c2)
        assert ledger.embedded_count() == 1

    def test_embeddings_survive_save_load(self, tmp_path):
        """Embeddings should persist through save/load cycle."""
        ledger_path = tmp_path / "ledger.json"
        emb = _make_embedding(dim=4)

        ledger1 = FactLedger(ledger_path)
        c = _make_claim("A", "is", "X", embedding=emb)
        ledger1.add_claim(c)

        # Reload
        ledger2 = FactLedger(ledger_path)
        loaded = ledger2.list_claims(limit=1)[0]
        assert len(loaded.embedding) == 4
        for a, b in zip(loaded.embedding, emb):
            assert abs(a - b) < 1e-6


# ---------------------------------------------------------------------------
# Config settings for Phase 8
# ---------------------------------------------------------------------------


class TestPhase8Config:
    def test_default_settings(self):
        from noodly.config import Settings

        s = Settings(
            openai_api_key="test",
            watch_dir=Path("/tmp"),
            brain_dir=Path("/tmp/brain"),
        )
        assert s.enable_ingestion_embeddings is True
        assert s.embedding_dim == 3072
        assert s.promote_threshold == 0.15
        assert s.high_authority_threshold == 0.8
        assert s.corroboration_count == 2
        assert s.embedding_model == "text-embedding-3-large"

    def test_custom_settings(self):
        from noodly.config import Settings

        s = Settings(
            openai_api_key="test",
            watch_dir=Path("/tmp"),
            brain_dir=Path("/tmp/brain"),
            promote_threshold=0.5,
            high_authority_threshold=0.9,
            corroboration_count=3,
            embedding_dim=1024,
        )
        assert s.promote_threshold == 0.5
        assert s.high_authority_threshold == 0.9
        assert s.corroboration_count == 3
        assert s.embedding_dim == 1024


# ---------------------------------------------------------------------------
# Graph node embeddings (Graphiti config alignment)
# ---------------------------------------------------------------------------


class TestGraphEmbeddingConfig:
    def test_brain_uses_configured_model(self):
        """Brain should pass embedding_model and embedding_dim from settings
        to the Graphiti OpenAIEmbedderConfig."""
        from unittest.mock import patch

        from noodly.config import Settings

        settings = Settings(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-large",
            embedding_dim=3072,
            falkordb_host="localhost",
        )

        with patch("noodly.graph.brain.FalkorDriver"), \
             patch("noodly.graph.brain.OpenAIClient"), \
             patch("noodly.graph.brain.OpenAIRerankerClient"), \
             patch("noodly.graph.brain.OpenAIEmbedder") as mock_embedder, \
             patch("noodly.graph.brain.Graphiti"):
            from noodly.graph.brain import Brain

            Brain(settings)

            # Check that the embedder config used the right model
            call_args = mock_embedder.call_args
            config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][0]
            assert config.embedding_model == "text-embedding-3-large"
            assert config.embedding_dim == 3072


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_claim_text(self):
        claim = _make_claim("Singapore", "imposes", "import tariff")
        text = _claim_text(claim)
        assert "Singapore" in text
        assert "imposes" in text
        assert "import tariff" in text

    def test_claim_fingerprint_normalization(self):
        c1 = _make_claim("  Singapore  ", "IMPOSES", "Import Tariff")
        c2 = _make_claim("singapore", "imposes", "import tariff")
        assert _claim_fingerprint(c1) == _claim_fingerprint(c2)

    def test_merge_into(self, tmp_path):
        """_merge_into should combine evidence and boost confidence."""
        ledger = FactLedger(tmp_path / "ledger.json")
        c1 = _make_claim("A", "is", "X", confidence=0.5)
        c2 = _make_claim("A_v2", "is_v2", "X_v2", confidence=0.9)

        ledger._merge_into(c1, c2)
        assert c1.confidence == 0.9
        assert len(c1.evidence) >= 2
        assert c1.last_confirmed_at is not None
