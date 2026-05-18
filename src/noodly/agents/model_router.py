"""Model router — route agent tasks to the most cost-effective model."""

from __future__ import annotations

# Task → model mapping. None means no LLM needed (use heuristics).
_TASK_MODELS: dict[str, str | None] = {
    # Simple classification — cheapest model or no LLM
    "boilerplate_detection": None,
    "table_syntax_check": None,

    # Standard extraction
    "claim_extraction": "gpt-4o-mini",
    "qa_review": "gpt-4o-mini",

    # Complex reasoning
    "entity_resolution": "gpt-4o-mini",
    "relationship_discovery": "gpt-4o-mini",
    "conflict_detection": "gpt-4o-mini",

    # Periodic/batch tasks — can use larger model since infrequent
    "ontology_alignment": "gpt-4o",
    "gap_detection": "gpt-4o",
}


def get_model(task: str, default: str = "gpt-4o-mini") -> str | None:
    """Return model name for a task, or None if no LLM needed."""
    return _TASK_MODELS.get(task, default)
