"""Agent toolkit — tools available to noodly agents via OpenAI function calling."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from noodly.agents.compact_serializer import CompactSerializer
from noodly.caching.manager import CacheManager
from noodly.scoring.ledger import FactLedger
from noodly.tracking.changelog import ChangeLog

logger = logging.getLogger(__name__)


# OpenAI function definitions for agent tool calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "find_claims_by_entity",
            "description": "Get all claims where entity_name is the subject or object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Entity name to search for"},
                },
                "required": ["entity_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_claims",
            "description": (
                "Find claims similar to a given subject-predicate-object triple. "
                "Use this to check for duplicates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_changes",
            "description": "Get all changes in the last N minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "How many minutes back to look",
                    },
                },
                "required": ["minutes"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_history",
            "description": "Get the change history for a specific entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                },
                "required": ["entity_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_coverage",
            "description": (
                "Get statistics about what we know about an entity. "
                "Returns claim count by knowledge class, source count, and confidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                },
                "required": ["entity_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source_section",
            "description": "Read a specific section from a cached source document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_uri": {"type": "string"},
                    "section_heading": {"type": "string"},
                },
                "required": ["source_uri", "section_heading"],
                "additionalProperties": False,
            },
        },
    },
]


class AgentToolkit:
    """Executes tool calls from agents against the noodly knowledge base."""

    def __init__(
        self,
        ledger: FactLedger,
        changelog: ChangeLog,
        cache: CacheManager,
    ) -> None:
        self._ledger = ledger
        self._changelog = changelog
        self._cache = cache

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        try:
            return handler(**arguments)
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return f"Error: {exc}"

    def _tool_find_claims_by_entity(self, entity_name: str) -> str:
        all_claims = self._ledger.list_claims(limit=10000)
        matches = [
            c
            for c in all_claims
            if entity_name.lower() in c.subject.lower()
            or entity_name.lower() in c.object.lower()
        ]
        if not matches:
            return f"No claims found for entity '{entity_name}'"
        return CompactSerializer.claims_block(matches)

    def _tool_find_similar_claims(
        self, subject: str, predicate: str, object: str
    ) -> str:
        all_claims = self._ledger.list_claims(limit=10000)
        matches = [
            c
            for c in all_claims
            if (
                self._fuzzy_match(c.subject, subject) > 0.6
                or self._fuzzy_match(c.object, object) > 0.6
            )
        ]
        if not matches:
            return "No similar claims found"
        return CompactSerializer.claims_block(matches[:20])

    def _tool_get_recent_changes(self, minutes: int) -> str:
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        events = self._changelog.since(since)
        if not events:
            return f"No changes in the last {minutes} minutes"
        lines = [
            f"{e.timestamp.isoformat()}: {e.change_type.value} - {e.entity_id or e.source_uri}"
            for e in events[:50]
        ]
        return "\n".join(lines)

    def _tool_get_entity_history(self, entity_name: str) -> str:
        all_events = self._changelog.since(datetime.min.replace(tzinfo=timezone.utc))
        matches = [
            e
            for e in all_events
            if entity_name.lower() in str(e.payload).lower()
            or entity_name.lower() in e.entity_id.lower()
        ]
        if not matches:
            return f"No history for entity '{entity_name}'"
        lines = [
            f"{e.timestamp.isoformat()}: {e.change_type.value}"
            f" - {json.dumps(e.payload, default=str)[:120]}"
            for e in matches[:30]
        ]
        return "\n".join(lines)

    def _tool_get_claim_coverage(self, entity_name: str) -> str:
        all_claims = self._ledger.list_claims(limit=10000)
        matches = [
            c
            for c in all_claims
            if entity_name.lower() in c.subject.lower()
            or entity_name.lower() in c.object.lower()
        ]
        if not matches:
            return f"No claims found for entity '{entity_name}'"

        from noodly.models.claims import KnowledgeClass

        by_class = {}
        for kc in KnowledgeClass:
            count = len([c for c in matches if c.knowledge_class == kc])
            if count > 0:
                by_class[kc.value] = count

        avg_confidence = sum(c.confidence for c in matches) / len(matches)
        avg_truth = sum(c.truth_score for c in matches) / len(matches)
        sources = len({str(ev.artifact_id) for c in matches for ev in c.evidence})

        return json.dumps(
            {
                "total_claims": len(matches),
                "by_class": by_class,
                "avg_confidence": round(avg_confidence, 3),
                "avg_truth_score": round(avg_truth, 3),
                "source_count": sources,
            },
            indent=2,
        )

    def _tool_read_source_section(
        self, source_uri: str, section_heading: str
    ) -> str:
        # Try to find the source in the parse cache
        # For v1, we just return a placeholder since we'd need the content hash
        return (
            f"Source section lookup not available in v1"
            f" (source={source_uri}, section={section_heading})"
        )

    @staticmethod
    def _fuzzy_match(a: str, b: str) -> float:
        """Simple token overlap ratio for fuzzy matching."""
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        overlap = a_tokens & b_tokens
        return len(overlap) / max(len(a_tokens), len(b_tokens))
