"""Graphiti-native storage backend for the fact ledger.

Stores claims as Graphiti edges with typed attributes, leveraging
Graphiti's built-in temporal fields, fact embeddings, and hybrid search
instead of reimplementing them in Python.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass

if TYPE_CHECKING:
    from noodly.graph.brain import Brain

logger = logging.getLogger(__name__)

CLAIM_EDGE_NAME = "CLAIM"


def _edge_to_claim(
    edge: Any,
    source_name: str,
    target_name: str,
) -> Claim:
    """Convert a Graphiti EntityEdge to a Claim object."""
    attrs = edge.attributes or {}

    # Status
    status = ClaimStatus.candidate
    raw_status = attrs.get("status")
    if raw_status:
        try:
            status = ClaimStatus(raw_status)
        except ValueError:
            pass

    # Knowledge class
    knowledge_class = KnowledgeClass.process
    raw_kc = attrs.get("knowledge_class")
    if raw_kc:
        try:
            knowledge_class = KnowledgeClass(raw_kc)
        except ValueError:
            pass

    # Evidence
    evidence: list[ClaimEvidence] = []
    evidence_json = attrs.get("evidence_json")
    if evidence_json:
        try:
            evidence_data = (
                json.loads(evidence_json)
                if isinstance(evidence_json, str)
                else evidence_json
            )
            for ev_data in evidence_data:
                evidence.append(ClaimEvidence(**ev_data))
        except (json.JSONDecodeError, Exception):
            logger.warning("Failed to deserialize evidence for edge %s", edge.uuid)

    # Temporal fields — prefer Graphiti native fields, fall back to attributes
    created_at = edge.created_at
    valid_from = edge.valid_at or None
    valid_until = edge.invalid_at or None

    # Parse last_confirmed_at from attributes
    last_confirmed_at = None
    raw_lca = attrs.get("last_confirmed_at")
    if raw_lca:
        try:
            last_confirmed_at = datetime.fromisoformat(str(raw_lca))
        except (ValueError, TypeError):
            pass

    # Conflicts
    conflicts_with: list[UUID] = []
    raw_conflicts = attrs.get("conflicts_with_json")
    if raw_conflicts:
        try:
            conflict_data = (
                json.loads(raw_conflicts)
                if isinstance(raw_conflicts, str)
                else raw_conflicts
            )
            conflicts_with = [UUID(c) for c in conflict_data]
        except (json.JSONDecodeError, ValueError, Exception):
            pass

    # Supersession
    superseded_by = None
    raw_sb = attrs.get("superseded_by")
    if raw_sb:
        try:
            superseded_by = UUID(str(raw_sb))
        except ValueError:
            pass

    supersedes = None
    raw_ss = attrs.get("supersedes")
    if raw_ss:
        try:
            supersedes = UUID(str(raw_ss))
        except ValueError:
            pass

    return Claim(
        id=UUID(edge.uuid),
        subject=source_name,
        predicate=attrs.get("predicate", edge.name),
        object=target_name,
        natural_language=attrs.get("natural_language", edge.fact or ""),
        confidence=float(attrs.get("confidence", 0.5)),
        authority=float(attrs.get("authority", 0.5)),
        recency=float(attrs.get("recency", 1.0)),
        specificity=float(attrs.get("specificity", 0.5)),
        status=status,
        knowledge_class=knowledge_class,
        evidence=evidence,
        created_at=created_at,
        valid_from=valid_from,
        valid_until=valid_until,
        last_confirmed_at=last_confirmed_at,
        group_id=edge.group_id,
        embedding=edge.fact_embedding or [],
        superseded_by=superseded_by,
        supersedes=supersedes,
        conflicts_with=conflicts_with,
    )


def _claim_scoring_attrs(claim: Claim) -> dict[str, Any]:
    """Build edge attributes dict from computed scoring fields.

    These are MERGED into the edge alongside LLM-extracted fields
    (predicate, natural_language, etc.).
    """
    evidence_data = [
        ev.model_dump(mode="json") for ev in claim.evidence
    ]

    attrs: dict[str, Any] = {
        "predicate": claim.predicate,
        "natural_language": claim.natural_language,
        "knowledge_class": claim.knowledge_class.value,
        "confidence": claim.confidence,
        "authority": claim.authority,
        "recency": claim.recency,
        "specificity": claim.specificity,
        "status": claim.status.value,
        "evidence_json": json.dumps(evidence_data, default=str),
    }

    if claim.last_confirmed_at:
        attrs["last_confirmed_at"] = claim.last_confirmed_at.isoformat()

    if claim.superseded_by:
        attrs["superseded_by"] = str(claim.superseded_by)
    if claim.supersedes:
        attrs["supersedes"] = str(claim.supersedes)
    if claim.conflicts_with:
        attrs["conflicts_with_json"] = json.dumps(
            [str(c) for c in claim.conflicts_with]
        )

    return attrs


class GraphitiBackend:
    """Stores claims as native Graphiti edges with typed attributes.

    Uses Graphiti's built-in:
    - Temporal fields (valid_at/invalid_at) instead of custom attributes
    - Fact embeddings instead of custom EmbeddingProvider
    - Hybrid search instead of custom cosine similarity
    """

    def __init__(self, brain: Brain) -> None:
        self._brain = brain
        self._graphiti = brain.get_graphiti()
        self._driver = brain.get_driver()
        self._group_id = brain.group_id

    # -- LedgerBackend protocol (sync wrappers) --

    def load_claims(self) -> dict[str, Claim]:
        raise RuntimeError(
            "GraphitiBackend requires async operations. "
            "Use load_claims_async() instead."
        )

    def save_claim(self, claim: Claim) -> None:
        raise RuntimeError(
            "GraphitiBackend requires async operations. "
            "Use save_claim_async() instead."
        )

    def save_all(self, claims: dict[str, Claim]) -> None:
        raise RuntimeError(
            "GraphitiBackend requires async operations. "
            "Use save_all_async() instead."
        )

    def delete_claim(self, claim_id: str) -> None:
        raise RuntimeError(
            "GraphitiBackend requires async operations. "
            "Use delete_claim_async() instead."
        )

    # -- Async implementations --

    async def load_claims_async(self) -> dict[str, Claim]:
        """Load all CLAIM edges from FalkorDB."""
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError
        from graphiti_core.nodes import EntityNode

        try:
            edges = await EntityEdge.get_by_group_ids(
                self._driver,
                group_ids=[self._group_id],
            )
        except GroupsEdgesNotFoundError:
            return {}

        claim_edges = [
            e for e in edges
            if e.name == CLAIM_EDGE_NAME and e.expired_at is None
        ]

        if not claim_edges:
            return {}

        # Batch-fetch entity nodes for subject/object names
        node_uuids = set()
        for e in claim_edges:
            node_uuids.add(e.source_node_uuid)
            node_uuids.add(e.target_node_uuid)

        nodes = await EntityNode.get_by_uuids(
            self._driver,
            uuids=list(node_uuids),
            group_id=self._group_id,
        )
        node_map = {n.uuid: n.name for n in nodes}

        claims: dict[str, Claim] = {}
        for edge in claim_edges:
            source_name = node_map.get(edge.source_node_uuid, "")
            target_name = node_map.get(edge.target_node_uuid, "")
            try:
                claim = _edge_to_claim(edge, source_name, target_name)
                claims[str(claim.id)] = claim
            except Exception:
                logger.warning("Failed to convert edge %s to claim", edge.uuid)

        return claims

    async def save_claim_async(self, claim: Claim) -> None:
        """Update edge attributes with computed scoring fields."""
        from graphiti_core.edges import EntityEdge

        try:
            edge = await EntityEdge.get_by_uuid(self._driver, str(claim.id))
        except Exception:
            logger.warning(
                "Edge %s not found in Graphiti, cannot save claim", claim.id
            )
            return

        scoring_attrs = _claim_scoring_attrs(claim)
        if edge.attributes is None:
            edge.attributes = {}
        edge.attributes.update(scoring_attrs)

        # Sync temporal fields
        if claim.valid_from:
            edge.valid_at = claim.valid_from
        if claim.valid_until:
            edge.invalid_at = claim.valid_until

        await edge.save(self._driver)

    async def save_all_async(self, claims: dict[str, Claim]) -> None:
        """Persist all claims (batch)."""
        for claim in claims.values():
            await self.save_claim_async(claim)

    async def delete_claim_async(self, claim_id: str) -> None:
        """Remove a claim edge from Graphiti."""
        from graphiti_core.edges import EntityEdge

        try:
            edge = await EntityEdge.get_by_uuid(self._driver, claim_id)
            await edge.delete(self._driver)
        except Exception:
            logger.warning("Could not delete edge %s", claim_id)

    async def search_similar(
        self, text: str, *, limit: int = 5
    ) -> list[Claim]:
        """Semantic search via Graphiti's hybrid search."""
        from graphiti_core.nodes import EntityNode
        from graphiti_core.search.search_config import (
            EdgeSearchConfig,
            EdgeSearchMethod,
            SearchConfig,
        )

        config = SearchConfig(
            limit=limit,
            edge_config=EdgeSearchConfig(
                search_methods=[
                    EdgeSearchMethod.cosine_similarity,
                    EdgeSearchMethod.bm25,
                ],
            ),
        )
        results = await self._graphiti.search_(
            query=text,
            config=config,
            group_ids=[self._group_id],
        )

        claim_edges = [
            e for e in results.edges
            if e.name == CLAIM_EDGE_NAME and e.expired_at is None
        ]

        if not claim_edges:
            return []

        node_uuids = set()
        for e in claim_edges:
            node_uuids.add(e.source_node_uuid)
            node_uuids.add(e.target_node_uuid)

        nodes = await EntityNode.get_by_uuids(
            self._driver,
            uuids=list(node_uuids),
            group_id=self._group_id,
        )
        node_map = {n.uuid: n.name for n in nodes}

        claims = []
        for edge in claim_edges:
            source_name = node_map.get(edge.source_node_uuid, "")
            target_name = node_map.get(edge.target_node_uuid, "")
            try:
                claims.append(_edge_to_claim(edge, source_name, target_name))
            except Exception:
                logger.warning("Failed to convert search result edge %s", edge.uuid)
        return claims
