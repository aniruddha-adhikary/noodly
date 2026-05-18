"""MCP server — exposes the Noodly brain to AI agents."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from noodly.config import get_settings
from noodly.graph.brain import Brain
from noodly.scoring.ledger import FactLedger

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "noodly",
    description="Company Brain — query organizational knowledge, facts, and context",
)

_brain: Brain | None = None
_ledger: FactLedger | None = None


async def _get_brain() -> Brain:
    global _brain
    if _brain is None:
        settings = get_settings()
        _brain = Brain(settings)
        await _brain.initialize()
    return _brain


def _get_ledger() -> FactLedger:
    global _ledger
    if _ledger is None:
        settings = get_settings()
        ledger_path = settings.brain_dir / "ledger.json"
        _ledger = FactLedger(ledger_path)
    return _ledger


@mcp.tool()
async def search(query: str, limit: int = 10) -> str:
    """Search the company brain for entities and relationships.

    Args:
        query: Natural language search query (e.g. "who owns the billing service?")
        limit: Maximum number of results to return
    """
    brain = await _get_brain()
    nodes = await brain.search_nodes(query, limit=limit)
    facts = await brain.search_facts(query, limit=limit)
    return json.dumps({"entities": nodes, "facts": facts}, indent=2, default=str)


@mcp.tool()
async def search_claims(query: str, limit: int = 20) -> str:
    """Search extracted claims in the fact ledger.

    Args:
        query: Text to search for in claims
        limit: Maximum number of results
    """
    ledger = _get_ledger()
    all_claims = ledger.list_claims(limit=200)
    query_lower = query.lower()
    matches = [
        c
        for c in all_claims
        if query_lower in c.natural_language.lower()
        or query_lower in c.subject.lower()
        or query_lower in c.object.lower()
    ]
    results = sorted(matches, key=lambda c: c.truth_score, reverse=True)[:limit]
    return json.dumps(
        [
            {
                "id": str(c.id),
                "claim": c.natural_language,
                "subject": c.subject,
                "predicate": c.predicate,
                "object": c.object,
                "truth_score": round(c.truth_score, 3),
                "status": c.status.value,
                "knowledge_class": c.knowledge_class.value,
                "evidence_count": len(c.evidence),
            }
            for c in results
        ],
        indent=2,
    )


@mcp.tool()
async def get_entity(name: str) -> str:
    """Look up a specific entity in the brain by name.

    Args:
        name: Entity name to look up (e.g. "PortNet", "Jane Smith")
    """
    brain = await _get_brain()
    nodes = await brain.search_nodes(name, limit=5)
    facts = await brain.search_facts(name, limit=10)
    return json.dumps({"entity_matches": nodes, "related_facts": facts}, indent=2, default=str)


@mcp.tool()
async def list_recent_episodes(count: int = 20) -> str:
    """List the most recent episodes (ingested documents/messages).

    Args:
        count: Number of recent episodes to return
    """
    brain = await _get_brain()
    episodes = await brain.get_episodes(last_n=count)
    return json.dumps(episodes, indent=2, default=str)


@mcp.tool()
async def list_claims(status: str = "", limit: int = 50) -> str:
    """List claims from the fact ledger, optionally filtered by status.

    Args:
        status: Filter by status (candidate, unverified, corroborated,
            canonical, superseded, rejected). Empty = all.
        limit: Maximum number of claims to return
    """
    from noodly.models.claims import ClaimStatus

    ledger = _get_ledger()
    claim_status = None
    if status:
        try:
            claim_status = ClaimStatus(status)
        except ValueError:
            valid = [s.value for s in ClaimStatus]
            return json.dumps({"error": f"Unknown status: {status}. Valid: {valid}"})

    claims = ledger.list_claims(status=claim_status, limit=limit)
    return json.dumps(
        [
            {
                "id": str(c.id),
                "claim": c.natural_language,
                "subject": c.subject,
                "predicate": c.predicate,
                "object": c.object,
                "truth_score": round(c.truth_score, 3),
                "status": c.status.value,
                "knowledge_class": c.knowledge_class.value,
                "evidence_count": len(c.evidence),
                "created_at": str(c.created_at),
            }
            for c in claims
        ],
        indent=2,
    )


@mcp.tool()
async def brain_stats() -> str:
    """Get summary statistics about the brain's knowledge."""
    ledger = _get_ledger()
    brain = await _get_brain()
    episodes = await brain.get_episodes(last_n=1000)
    all_claims = ledger.list_claims(limit=10000)

    status_counts: dict[str, int] = {}
    for c in all_claims:
        status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1

    return json.dumps(
        {
            "total_episodes": len(episodes),
            "total_claims": len(all_claims),
            "claims_by_status": status_counts,
            "avg_truth_score": round(
                sum(c.truth_score for c in all_claims) / max(len(all_claims), 1), 3
            ),
        },
        indent=2,
    )


def run_server() -> None:
    """Entry point for running the MCP server."""
    mcp.run()
