"""Graph Population Agent — diff-aware enrichment of the knowledge graph."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from noodly.agents.base_agent import ToolEquippedAgent
from noodly.agents.compact_serializer import CompactSerializer
from noodly.agents.context_scoper import ContextScoper
from noodly.agents.toolkit import AgentToolkit
from noodly.models.claims import Claim
from noodly.scoring.ledger import FactLedger
from noodly.tracking.changelog import ChangeEvent, ChangeLog, ChangeType
from noodly.tracking.claim_differ import ClaimDiff

logger = logging.getLogger(__name__)

GRAPH_SYSTEM_PROMPT = """\
You are a knowledge graph enrichment agent for a Company Brain system.

Given a set of CHANGES to the knowledge (new, removed, or modified claims),
perform these tasks:

1. ENTITY RESOLUTION: Are any new entity names aliases of existing entities?
   Example: "SLA", "The Authority", "Singapore Land Authority" → same entity.
   Return merge suggestions as {"canonical": "...", "alias": "..."}.

2. RELATIONSHIP DISCOVERY: What implicit relationships exist between entities
   in the new claims and existing entities? Look for shared properties,
   hierarchical relationships, temporal sequences, causal links.

3. CONFLICT DETECTION: Do any new claims contradict existing ones?
   Example: "fee is $5000/month" vs "fee is $6000/month" from different sources.

4. GAP DETECTION: Given what we know about similar entities, are there
   expected facts missing for the entities in the new claims?

Use the available tools to look up existing claims and entity history
before making decisions. Only suggest merges/relationships/conflicts
you are confident about.

Return valid JSON matching the schema.
"""

GRAPH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "alias": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["canonical", "alias", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["subject", "predicate", "object", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_a": {"type": "string"},
                    "claim_b": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["claim_a", "claim_b", "description"],
                "additionalProperties": False,
            },
        },
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "missing_property": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["entity", "missing_property", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entity_merges", "relationships", "conflicts", "gaps"],
    "additionalProperties": False,
}


@dataclass
class EnrichmentReport:
    """Report from graph population agent."""

    entity_merges: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    skipped: bool = False
    summary: str = ""

    @property
    def has_actions(self) -> bool:
        return bool(self.entity_merges or self.relationships or self.conflicts or self.gaps)


class GraphPopulationAgent(ToolEquippedAgent):
    """Diff-aware graph enrichment agent.

    Behavior based on claim diff:
    - added_claims → entity resolution, relationship discovery
    - removed_claims → gap detection (what knowledge was lost?)
    - modified_claims → conflict detection (did the facts change?)
    - unchanged_claims → skip (no work needed)
    """

    system_prompt = GRAPH_SYSTEM_PROMPT

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        toolkit: AgentToolkit | None = None,
        ledger: FactLedger | None = None,
        changelog: ChangeLog | None = None,
    ) -> None:
        if toolkit is not None:
            super().__init__(api_key=api_key, model=model, toolkit=toolkit)
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
            self._model = model
            self._toolkit = None  # type: ignore[assignment]
        self._ledger = ledger
        self._changelog = changelog

    async def enrich(
        self,
        claim_diff: ClaimDiff | None = None,
        new_claims: list[Claim] | None = None,
    ) -> EnrichmentReport:
        """Enrich graph based on what changed, not the full state.

        If claim_diff is provided, uses diff-aware processing.
        Otherwise, processes new_claims as if they're all new.
        """
        if claim_diff is not None:
            return await self._diff_aware_enrich(claim_diff)

        if new_claims:
            return await self._enrich_new_claims(new_claims)

        return EnrichmentReport(skipped=True, summary="No claims to process")

    async def _diff_aware_enrich(self, claim_diff: ClaimDiff) -> EnrichmentReport:
        """Process based on claim diff — only work on what changed."""
        if not claim_diff.has_changes:
            return EnrichmentReport(
                skipped=True,
                summary="No claim changes detected",
            )

        # Build scoped context
        context_parts: list[str] = []

        if claim_diff.added_claims:
            context_parts.append(
                CompactSerializer.diff_summary(
                    added=claim_diff.added_claims,
                    removed=[c for c in claim_diff.removed_claims],
                    modified=[m.new_claim for m in claim_diff.modified_claims],
                )
            )

        # Get related existing claims for context
        if self._ledger is not None:
            scoper = ContextScoper(self._ledger)
            all_changed = (
                claim_diff.added_claims
                + [m.new_claim for m in claim_diff.modified_claims]
            )
            related = scoper.related_claims(all_changed, max_results=30)
            if related:
                context_parts.append(
                    f"\nEXISTING RELATED CLAIMS:\n{CompactSerializer.claims_block(related)}"
                )

        prompt = (
            f"SOURCE: {claim_diff.source_uri}\n"
            f"CHANGES: {claim_diff.summary}\n\n"
            + "\n".join(context_parts)
        )

        return await self._run_enrichment(prompt)

    async def _enrich_new_claims(self, claims: list[Claim]) -> EnrichmentReport:
        """Process a set of new claims (no diff available)."""
        claims_text = CompactSerializer.claims_block(claims)

        context_parts = [f"NEW CLAIMS:\n{claims_text}"]

        if self._ledger is not None:
            scoper = ContextScoper(self._ledger)
            related = scoper.related_claims(claims, max_results=30)
            if related:
                context_parts.append(
                    f"\nEXISTING RELATED CLAIMS:\n{CompactSerializer.claims_block(related)}"
                )

        prompt = "\n".join(context_parts)
        return await self._run_enrichment(prompt)

    async def _run_enrichment(self, prompt: str) -> EnrichmentReport:
        """Run the graph agent."""
        if self._toolkit is not None:
            result = await self._run_with_tools(prompt, GRAPH_RESPONSE_SCHEMA)
        else:
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "graph_response",
                            "strict": True,
                            "schema": GRAPH_RESPONSE_SCHEMA,
                        },
                    },
                    temperature=0.1,
                )
                content = response.choices[0].message.content or "{}"
                result = json.loads(content)
            except Exception:
                logger.exception("Graph agent LLM call failed")
                result = {
                    "entity_merges": [],
                    "relationships": [],
                    "conflicts": [],
                    "gaps": [],
                }

        report = EnrichmentReport(
            entity_merges=result.get("entity_merges", []),
            relationships=result.get("relationships", []),
            conflicts=result.get("conflicts", []),
            gaps=result.get("gaps", []),
        )

        # Emit change events for discovered enrichments
        if self._changelog is not None:
            self._emit_events(report)

        parts: list[str] = []
        if report.entity_merges:
            parts.append(f"{len(report.entity_merges)} merges")
        if report.relationships:
            parts.append(f"{len(report.relationships)} relationships")
        if report.conflicts:
            parts.append(f"{len(report.conflicts)} conflicts")
        if report.gaps:
            parts.append(f"{len(report.gaps)} gaps")
        report.summary = ", ".join(parts) or "No enrichments found"

        return report

    def _emit_events(self, report: EnrichmentReport) -> None:
        """Emit change events for discovered enrichments."""
        if self._changelog is None:
            return

        for merge in report.entity_merges:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.entity_merged,
                    entity_id=merge.get("canonical", ""),
                    payload=merge,
                    agent="graph_population_agent",
                )
            )

        for rel in report.relationships:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.relationship_discovered,
                    entity_id=rel.get("subject", ""),
                    payload=rel,
                    agent="graph_population_agent",
                )
            )

        for conflict in report.conflicts:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.conflict_detected,
                    payload=conflict,
                    agent="graph_population_agent",
                )
            )

        for gap in report.gaps:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.gap_detected,
                    entity_id=gap.get("entity", ""),
                    payload=gap,
                    agent="graph_population_agent",
                )
            )
