"""Brain — the central Graphiti-backed knowledge graph."""

from __future__ import annotations

import json
import logging

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import (
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
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
        llm_config = LLMConfig(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        llm_client = OpenAIClient(config=llm_config)
        embedder_config = OpenAIEmbedderConfig(api_key=settings.openai_api_key)
        embedder = OpenAIEmbedder(config=embedder_config)
        cross_encoder = OpenAIRerankerClient(config=llm_config)
        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

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
        config = SearchConfig(
            limit=limit,
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.cosine_similarity, NodeSearchMethod.bm25],
            ),
        )
        results = await self._graphiti.search_(
            query=query,
            config=config,
            group_ids=[self._settings.group_id],
        )
        return [
            {
                "uuid": n.uuid,
                "name": n.name,
                "summary": n.summary,
                "group_id": n.group_id,
                "created_at": str(n.created_at),
            }
            for n in results.nodes
        ]

    async def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic + keyword hybrid search over relationship edges (facts)."""
        config = SearchConfig(
            limit=limit,
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.cosine_similarity, EdgeSearchMethod.bm25],
            ),
        )
        results = await self._graphiti.search_(
            query=query,
            config=config,
            group_ids=[self._settings.group_id],
        )
        return [
            {
                "uuid": e.uuid,
                "name": e.name,
                "fact": e.fact,
                "created_at": str(e.created_at),
                "expired_at": str(e.expired_at) if e.expired_at else "",
            }
            for e in results.edges
        ]

    async def get_episodes(self, last_n: int = 20) -> list[dict]:
        """Return the most recent episodes."""
        from datetime import datetime, timezone

        results = await self._graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            group_ids=[self._settings.group_id],
            last_n=last_n,
        )
        return [
            {
                "uuid": ep.uuid,
                "name": ep.name,
                "source_description": ep.source_description,
                "created_at": str(ep.created_at),
            }
            for ep in results
        ]

    async def close(self) -> None:
        """Shut down the graph driver."""
        await self._graphiti.close()
