"""Tests for the pipeline — integration test with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from noodly.config import Settings
from noodly.models.claims import Claim, ClaimEvidence
from noodly.pipeline import Pipeline


def _fake_settings(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "doc.md").write_text("# Test\nJane owns the billing service.")
    brain = tmp_path / "brain"
    brain.mkdir()
    return Settings(
        openai_api_key="test-key",
        brain_dir=brain,
        watch_dir=inbox,
        falkordb_host="localhost",
    )


def _mock_claims(artifact, source_filename=""):
    """Return fake claims as if the LLM extracted them."""
    return [
        Claim(
            subject="Jane",
            predicate="owns",
            object="billing service",
            natural_language="Jane owns the billing service.",
            confidence=0.8,
            evidence=[
                ClaimEvidence(
                    artifact_id=artifact.id,
                    supports=True,
                    source_span="Jane owns the billing service.",
                    author="",
                )
            ],
        )
    ]


@pytest.mark.asyncio
async def test_pipeline_run_with_mock(tmp_path):
    settings = _fake_settings(tmp_path)
    with patch("noodly.pipeline.Brain") as MockBrain:
        mock_brain = MockBrain.return_value
        mock_brain.initialize = AsyncMock()
        mock_brain.ingest_artifact = AsyncMock()
        mock_brain.close = AsyncMock()
        pipeline = Pipeline(settings)

    pipeline._extractor.extract = AsyncMock(side_effect=_mock_claims)

    await pipeline.initialize()
    stats = await pipeline.run()
    await pipeline.close()

    assert stats["artifacts"] == 1
    assert stats["claims"] == 1
    assert stats["projected"] > 0

    # Verify ledger was populated
    assert pipeline._ledger.count == 1

    # Verify projection was created
    assert (settings.brain_dir / "index.md").exists()


@pytest.mark.asyncio
async def test_pipeline_dedup_on_reingest(tmp_path):
    settings = _fake_settings(tmp_path)
    with patch("noodly.pipeline.Brain") as MockBrain:
        mock_brain = MockBrain.return_value
        mock_brain.initialize = AsyncMock()
        mock_brain.ingest_artifact = AsyncMock()
        mock_brain.close = AsyncMock()
        pipeline = Pipeline(settings)

    pipeline._extractor.extract = AsyncMock(side_effect=_mock_claims)

    await pipeline.initialize()

    # First ingest
    stats1 = await pipeline.run()
    assert stats1["claims"] == 1

    # Modify the file to trigger re-scan
    (settings.watch_dir / "doc.md").write_text(
        "# Updated\nJane owns the billing service. Also Bob."
    )

    # Re-ingest — same claim should be deduped
    stats2 = await pipeline.run()
    assert stats2["artifacts"] == 1
    assert pipeline._ledger.count == 1  # Still 1 claim, not 2

    await pipeline.close()


@pytest.mark.asyncio
async def test_pipeline_authority_wiring(tmp_path):
    settings = _fake_settings(tmp_path)
    with patch("noodly.pipeline.Brain"):
        pipeline = Pipeline(settings)

    # Authority should be wired
    assert pipeline._authority is not None
    assert pipeline._ledger._authority is pipeline._authority


@pytest.mark.asyncio
async def test_pipeline_hash_persistence(tmp_path):
    settings = _fake_settings(tmp_path)
    with patch("noodly.pipeline.Brain") as MockBrain:
        mock_brain = MockBrain.return_value
        mock_brain.initialize = AsyncMock()
        mock_brain.ingest_artifact = AsyncMock()
        mock_brain.close = AsyncMock()
        pipeline = Pipeline(settings)

    pipeline._extractor.extract = AsyncMock(return_value=[])

    await pipeline.initialize()
    await pipeline.run()

    # Hash file should exist
    hash_path = settings.watch_dir / ".hashes.json"
    assert hash_path.exists()
    hashes = json.loads(hash_path.read_text())
    assert "doc.md" in hashes

    await pipeline.close()
