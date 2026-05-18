"""Brain — the central Graphiti-backed knowledge graph."""

from __future__ import annotations

import json
import logging

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)

from noodly.config import Settings
from noodly.models.artifacts import SourceArtifact

logger = logging.getLogger(__name__)


class Brain:
    """Wraps Graphiti to provide the Noodly knowledge graph interface.

    Usage::

        brain = Brain(settings)
        await brain.initialize()
        await brain.ingest_artifact(artifact)
        results = await brain.search("who owns the PortNet integration?")
        await brain.close()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        driver = FalkorDriver(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            username=settings.falkordb_username,
            password=settings.falkordb_password,
            database=settings.falkordb_database,
        )
        self._graphiti = Graphiti(graph_driver=driver)

    async def initialize(self) -> None:
        """Set up graph indices. Call once on first run."""
        await self._graphiti.build_indices_and_constraints()
        logger.info("Brain indices initialized")

    async def ingest_artifact(self, artifact: SourceArtifact) -> str:
        """Ingest a source artifact as a Graphiti episode.

        Returns the episode name for reference.
        """
        episode_name = f"{artifact.source_type.value}-{artifact.id}"

        metadata = {
            "source_type": artifact.source_type.value,
            "source_uri": artifact.source_uri,
            "title": artifact.title,
            "author": artifact.author,
            "artifact_id": str(artifact.id),
            **{k: v for k, v in artifact.metadata.items() if v is not None},
        }

        episode_body = json.dumps(
            {"title": artifact.title, "body": artifact.body, "metadata": metadata},
            default=str,
        )

        ref_time = artifact.content_created_at or artifact.created_at

        await self._graphiti.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source=EpisodeType.json,
            source_description=f"{artifact.source_type.value}: {artifact.title}",
            reference_time=ref_time,
            group_id=self._settings.group_id,
        )

        logger.info("Ingested artifact %s as episode %s", artifact.id, episode_name)
        return episode_name

    async def search_nodes(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic + keyword hybrid search over entity nodes."""
        results = await self._graphiti.search(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=[self._settings.group_id],
            num_results=limit,
        )
        return [
            {
                "uuid": r.uuid,
                "name": r.name,
                "summary": r.summary if hasattr(r, "summary") else "",
                "group_id": r.group_id,
                "created_at": str(r.created_at) if hasattr(r, "created_at") else "",
            }
            for r in results
        ]

    async def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic + keyword hybrid search over relationship edges (facts)."""
        results = await self._graphiti.search(
            query=query,
            config=EDGE_HYBRID_SEARCH_RRF,
            group_ids=[self._settings.group_id],
            num_results=limit,
        )
        return [
            {
                "uuid": r.uuid,
                "name": r.name if hasattr(r, "name") else r.fact if hasattr(r, "fact") else "",
                "fact": r.fact if hasattr(r, "fact") else "",
                "created_at": str(r.created_at) if hasattr(r, "created_at") else "",
                "expired_at": str(r.expired_at) if hasattr(r, "expired_at") else "",
            }
            for r in results
        ]

    async def get_episodes(self, last_n: int = 20) -> list[dict]:
        """Return the most recent episodes."""
        results = await self._graphiti.get_episodes(
            group_ids=[self._settings.group_id],
            last_n=last_n,
        )
        return [
            {
                "uuid": ep.uuid,
                "name": ep.name,
                "source_description": ep.source_description,
                "created_at": str(ep.created_at),
                "reference_time": str(getattr(ep, "reference_time", "")),
            }
            for ep in results
        ]

    async def close(self) -> None:
        """Shut down the graph driver."""
        await self._graphiti.close()
