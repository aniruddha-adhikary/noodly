"""Pipeline — orchestrates the full ingest → extract → score → project loop."""

from __future__ import annotations

import logging

from noodly.config import Settings
from noodly.connectors.local_fs import LocalFSConnector
from noodly.extraction.extractor import ClaimExtractor
from noodly.graph.brain import Brain
from noodly.models.artifacts import SourceArtifact
from noodly.projection.markdown import MarkdownProjector
from noodly.scoring.ledger import FactLedger

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end ingest → extract → store → project pipeline.

    Usage::

        pipeline = Pipeline(settings)
        await pipeline.initialize()
        stats = await pipeline.run()
        await pipeline.close()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._brain = Brain(settings)
        self._connector = LocalFSConnector(settings.watch_dir)
        self._extractor = ClaimExtractor(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        self._ledger = self._build_ledger(settings)
        self._projector = MarkdownProjector(settings.brain_dir)

    def _build_ledger(self, settings: Settings) -> FactLedger:
        if settings.use_graphiti_backend:
            from noodly.storage.graphiti_backend import GraphitiBackend

            backend = GraphitiBackend(self._brain)
            return FactLedger(backend=backend)
        return FactLedger(backend=settings.brain_dir / "ledger.json")

    async def initialize(self) -> None:
        """Set up graph indices and load async backends."""
        await self._brain.initialize()
        if self._ledger.is_async_backend:
            await self._ledger.load_async()
        logger.info("Pipeline initialized")

    async def run(self) -> dict[str, int]:
        """Run one full cycle: scan → ingest → extract → score → project."""
        stats = {"artifacts": 0, "claims": 0, "projected": 0}

        # 1. Scan for new files
        artifacts = await self._connector.scan()
        stats["artifacts"] = len(artifacts)
        if not artifacts:
            logger.info("No new artifacts found")
            return stats

        all_claims = []
        for artifact in artifacts:
            # 2. Ingest into Graphiti
            try:
                await self._brain.ingest_artifact(artifact)
            except Exception:
                logger.exception("Failed to ingest artifact %s", artifact.id)

            # 3. Extract claims via LLM
            try:
                claims = await self._extractor.extract(artifact)
                all_claims.extend(claims)
            except Exception:
                logger.exception("Failed to extract claims from %s", artifact.id)

        # 4. Store claims in ledger
        if self._ledger.is_async_backend:
            stored = await self._ledger.add_claims_async(all_claims)
        else:
            stored = self._ledger.add_claims(all_claims)
        stats["claims"] = len(stored)

        # 5. Apply decay
        self._ledger.apply_decay()

        # 6. Project to Markdown
        all_ledger_claims = self._ledger.list_claims(limit=10000)
        stats["projected"] = self._projector.project(all_ledger_claims)

        logger.info(
            "Pipeline complete: %d artifacts, %d claims, %d files projected",
            stats["artifacts"],
            stats["claims"],
            stats["projected"],
        )
        return stats

    async def ingest_text(self, title: str, body: str, author: str = "") -> dict[str, int]:
        """Ingest raw text directly (useful for CLI and testing)."""
        from noodly.models.artifacts import SourceType

        artifact = SourceArtifact(
            source_type=SourceType.manual,
            title=title,
            body=body,
            author=author,
        )

        await self._brain.ingest_artifact(artifact)
        claims = await self._extractor.extract(artifact)
        if self._ledger.is_async_backend:
            stored = await self._ledger.add_claims_async(claims)
        else:
            stored = self._ledger.add_claims(claims)

        all_claims = self._ledger.list_claims(limit=10000)
        projected = self._projector.project(all_claims)

        return {"artifacts": 1, "claims": len(stored), "projected": projected}

    async def close(self) -> None:
        """Shut down connections."""
        await self._brain.close()
