"""Typed edge schemas for Graphiti's LLM extraction.

When passed to ``add_episode(edge_types={"CLAIM": ClaimEdge})``, Graphiti's
extraction LLM uses the Pydantic model to produce structured edge attributes
directly from document text — eliminating the need for a separate
ClaimExtractor call.

Only fields the LLM can meaningfully extract from text belong here.
Computed scoring fields (status, authority, recency) are added
post-extraction by FactLedger and stored via GraphitiBackend.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

CLAIM_EDGE_TYPE = "CLAIM"


class ClaimEdge(BaseModel):
    """A factual claim extracted from a document.

    Represents a normalized assertion about entities, relationships,
    processes, or constraints.  The source and target entities of the
    Graphiti edge serve as the claim's subject and object respectively.
    """

    predicate: str = Field(
        description=(
            "The relationship verb or phrase between the source and target "
            "entities (e.g. 'owns', 'is responsible for', 'requires')"
        ),
    )
    natural_language: str = Field(
        description="A concise natural-language summary of the claim",
    )
    knowledge_class: str = Field(
        default="process",
        description=(
            "Classification of the claim's expected stability. "
            "One of: stable (permanent facts), process (how things work), "
            "tacit (informal know-how), stateful (current state that changes often)"
        ),
    )
    confidence: float = Field(
        default=0.5,
        description="How clearly the source states this fact (0.0 = vague, 1.0 = explicit)",
    )
    source_span: str = Field(
        default="",
        description="The exact text from the source document that supports this claim",
    )


CLAIM_EXTRACTION_INSTRUCTIONS = """\
Extract concrete, actionable factual claims from the text.

For EACH distinct factual assertion you find:
- Use CLAIM as the relation_type
- Set the `predicate` attribute to the relationship verb/phrase
- Write a concise `natural_language` summary
- Classify `knowledge_class` as one of:
    stable   — permanent facts unlikely to change (e.g. "Python is a programming language")
    process  — how things work, procedures (e.g. "deploys go through staging first")
    tacit    — informal know-how (e.g. "ask Jane for billing questions")
    stateful — current state that changes often (e.g. "the API rate limit is 100 rpm")
- Estimate `confidence` (0.0-1.0) based on how clearly the source states the fact
- Include `source_span` with the exact supporting text from the document

Focus on factual assertions, not opinions, questions, or speculative statements.
Each claim should be independently meaningful as a knowledge graph edge.
"""
