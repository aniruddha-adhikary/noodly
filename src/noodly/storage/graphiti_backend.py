"""Graphiti-backed storage backend for the fact ledger.

Stores each claim as a Graphiti edge with subject → predicate → object,
leveraging Graphiti's built-in deduplication, embedding, and search
while preserving Noodly's custom truth-scoring logic in the FactLedger.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass

if TYPE_CHECKING:
    from noodly.graph.brain import Brain

logger = logging.getLogger(__name__)

CLAIM_MARKER = "__noodly_claim__"


def _claim_to_edge_attributes(claim: Claim) -> dict:
    """Serialize a Claim's metadata into an edge ``attributes`` dict."""
    evidence_data = [ev.model_dump(mode="json") for ev in claim.evidence]
    return {
        CLAIM_MARKER: True,
        "claim_id": str(claim.id),
        "status": claim.status.value,
        "knowledge_class": claim.knowledge_class.value,
        "confidence": claim.confidence,
        "authority": claim.authority,
        "recency": claim.recency,
        "specificity": claim.specificity,
        "created_at": claim.created_at.isoformat(),
        "valid_from": claim.valid_from.isoformat() if claim.valid_from else "",
        "valid_until": claim.valid_until.isoformat() if claim.valid_until else "",
        "last_confirmed_at": (
            claim.last_confirmed_at.isoformat() if claim.last_confirmed_at else ""
        ),
        "evidence_json": json.dumps(evidence_data, default=str),
        "natural_language": claim.natural_language,
    }


def _parse_optional_dt(value: str) -> datetime | None:
    """Parse an ISO datetime string, returning None for empty strings."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def _edge_to_claim(edge: EntityEdge) -> Claim | None:
    """Reconstruct a Claim from a Graphiti edge's attributes.

    Returns ``None`` if the edge does not carry the noodly claim marker.
    """
    attrs = edge.attributes
    if not attrs or not attrs.get(CLAIM_MARKER):
        return None

    evidence_raw = json.loads(attrs.get("evidence_json", "[]"))
    evidence = [ClaimEvidence(**ev) for ev in evidence_raw]

    # The edge ``name`` stores the predicate; source/target node names
    # are not directly on the edge, so we store subject/object in the
    # ``fact`` field as "subject | object".
    parts = edge.fact.split(" | ", maxsplit=1)
    subject = parts[0] if parts else ""
    obj = parts[1] if len(parts) > 1 else ""

    return Claim(
        id=UUID(attrs["claim_id"]),
        subject=subject,
        predicate=edge.name,
        object=obj,
        natural_language=attrs.get("natural_language", ""),
        confidence=float(attrs.get("confidence", 0.5)),
        authority=float(attrs.get("authority", 0.5)),
        recency=float(attrs.get("recency", 1.0)),
        specificity=float(attrs.get("specificity", 0.5)),
        status=ClaimStatus(attrs.get("status", "candidate")),
        knowledge_class=KnowledgeClass(attrs.get("knowledge_class", "process")),
        evidence=evidence,
        created_at=(
            datetime.fromisoformat(attrs["created_at"])
            if attrs.get("created_at")
            else edge.created_at
        ),
        valid_from=_parse_optional_dt(attrs.get("valid_from", "")),
        valid_until=_parse_optional_dt(attrs.get("valid_until", "")),
        last_confirmed_at=_parse_optional_dt(attrs.get("last_confirmed_at", "")),
        group_id=edge.group_id,
    )


class GraphitiBackend:
    """Persists claims as Graphiti edges with rich metadata.

    Each claim becomes an ``EntityEdge`` connecting a *subject* node to
    an *object* node, with the predicate as the edge name.  All claim
    metadata (scores, evidence, timestamps) are stored in the edge's
    ``attributes`` dict, tagged with a marker so we can distinguish
    claim edges from regular Graphiti-managed edges.
    """

    def __init__(self, brain: Brain) -> None:
        self._brain = brain
        self._graphiti = brain.get_graphiti()
        self._group_id = brain.group_id

    # ------------------------------------------------------------------
    # LedgerBackend interface
    # ------------------------------------------------------------------

    def load_claims(self) -> dict[str, Claim]:
        raise TypeError(
            "GraphitiBackend.load_claims() is async — use load_claims_async() instead"
        )

    async def load_claims_async(self) -> dict[str, Claim]:
        """Query all claim-marked edges from Graphiti."""
        try:
            edges = await EntityEdge.get_by_group_ids(
                self._graphiti.driver,
                group_ids=[self._group_id],
            )
        except Exception:
            logger.debug("No edges found for group %s", self._group_id)
            return {}

        claims: dict[str, Claim] = {}
        for edge in edges:
            claim = _edge_to_claim(edge)
            if claim is not None:
                claims[str(claim.id)] = claim
        return claims

    def save_claim(self, claim: Claim) -> None:
        raise TypeError(
            "GraphitiBackend.save_claim() is async — use save_claim_async() instead"
        )

    async def save_claim_async(self, claim: Claim) -> None:
        """Upsert a single claim as a Graphiti edge via add_triplet."""
        now = datetime.now(timezone.utc)
        attrs = _claim_to_edge_attributes(claim)

        source_node = EntityNode(
            name=claim.subject,
            group_id=claim.group_id,
            created_at=now,
            summary="",
        )
        target_node = EntityNode(
            name=claim.object,
            group_id=claim.group_id,
            created_at=now,
            summary="",
        )

        fact_text = f"{claim.subject} | {claim.object}"
        edge = EntityEdge(
            uuid=str(claim.id),
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name=claim.predicate,
            fact=fact_text,
            group_id=claim.group_id,
            created_at=claim.created_at,
            valid_at=claim.valid_from,
            invalid_at=claim.valid_until,
            episodes=[],
            attributes=attrs,
        )

        await self._graphiti.add_triplet(source_node, edge, target_node)
        logger.debug("Saved claim %s as Graphiti edge", claim.id)

    def save_all(self, claims: dict[str, Claim]) -> None:
        raise TypeError(
            "GraphitiBackend.save_all() is async — use save_all_async() instead"
        )

    async def save_all_async(self, claims: dict[str, Claim]) -> None:
        """Persist all claims (batch). Saves sequentially to respect Graphiti dedup."""
        for claim in claims.values():
            await self.save_claim_async(claim)

    def delete_claim(self, claim_id: str) -> None:
        raise TypeError(
            "GraphitiBackend.delete_claim() is async — use delete_claim_async() instead"
        )

    async def delete_claim_async(self, claim_id: str) -> None:
        """Remove a claim edge by its UUID."""
        try:
            edge = await EntityEdge.get_by_uuid(self._graphiti.driver, claim_id)
            await edge.delete(self._graphiti.driver)
            logger.debug("Deleted claim edge %s", claim_id)
        except Exception:
            logger.warning("Could not delete claim edge %s", claim_id)

    # ------------------------------------------------------------------
    # Graphiti-native search (used by FactLedger for deduplication)
    # ------------------------------------------------------------------

    async def search_similar_claims(
        self, claim: Claim, *, limit: int = 5
    ) -> list[Claim]:
        """Find existing claims semantically similar to the given claim.

        Uses Graphiti's hybrid search (vector + keyword) instead of a
        custom embedding / cosine-similarity implementation.
        """
        query = f"{claim.subject} {claim.predicate} {claim.object}"
        edges = await self._graphiti.search(
            query=query,
            group_ids=[self._group_id],
            num_results=limit,
        )
        results: list[Claim] = []
        for edge in edges:
            parsed = _edge_to_claim(edge)
            if parsed is not None and str(parsed.id) != str(claim.id):
                results.append(parsed)
        return results
