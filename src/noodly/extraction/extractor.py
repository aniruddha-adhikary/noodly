"""LLM-powered claim extraction from source artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from noodly.models.artifacts import SourceArtifact
from noodly.models.claims import (
    Claim,
    ClaimEvidence,
    ClaimStatus,
    KnowledgeClass,
)

logger = logging.getLogger(__name__)


def _parse_date(date_str: str) -> datetime | None:
    """Parse an ISO date string into a datetime, with fallback."""
    if not date_str or date_str.lower() in ("null", "none", "n/a", ""):
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug("Could not parse date: %s", date_str)
            return None

EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge extraction engine for a Company Brain system.

Given a document or message, extract **factual claims** — normalized assertions
about entities, relationships, processes, ownership, decisions, or constraints.

Rules:
- Extract concrete, actionable facts — not opinions or questions.
- Each claim must have a subject, predicate, and object.
- Include a natural_language summary of the claim.
- Classify each claim's knowledge_class:
  - "stable": permanent facts (legal entity names, product names, repo ownership)
  - "process": how things work (workflows, procedures, onboarding steps)
  - "tacit": informal know-how (shortcuts, workarounds, "ask X for Y")
  - "stateful": current state that changes often (active incidents, current owners, open deals)
- Estimate confidence (0.0–1.0) based on how clearly the source states the fact.
- Include the source_span — the exact text that supports the claim.
- If the document contains no extractable facts, return an empty list.

Temporal extraction (IMPORTANT — populate dates aggressively):
- For each claim, set valid_from and valid_until using ISO 8601 (YYYY-MM-DD).
- Use the document's publication/issue date as valid_from when the claim describes
  a state of affairs at that time (e.g., benchmark scores, versions, prices).
- Use explicit dates from the text whenever available:
  - "effective from 1 January 2024" → valid_from="2024-01-01"
  - "published June 1999" → valid_from="1999-06-01"
  - "until 31 December 2025" → valid_until="2025-12-31"
  - "RFC 2616 was superseded in June 2014" → valid_until="2014-06-01"
- For laws, regulations, circulars: use the issue/effective date as valid_from.
- For academic papers: use the publication year as valid_from for benchmark claims.
- For versioned documents (v1.0 superseded by v2.0): set valid_until on old version
  claims to the date the new version was published.
- Use null ONLY when the fact is truly timeless (e.g., mathematical definitions,
  physical constants, permanent entity names).

Supersession and document relationships:
- When a document explicitly states it obsoletes, supersedes, replaces, updates,
  or amends another document, extract that relationship:
  - subject="[new doc]", predicate="supersedes"/"obsoletes"/"updates",
    object="[old doc]"
- When a circular/notice updates a previous one, extract:
  - subject="[notice]", predicate="updates", object="[circular]"
- When an RFC obsoletes another: subject="RFC XXXX", predicate="obsoletes",
  object="RFC YYYY"
- Set valid_until on the OLD document's claims to the date the new one took effect.

Entity aliases and cross-references:
- Extract entity aliases when mentioned. E.g., "Singapore Land Authority (SLA)"
  → also emit: subject="SLA", predicate="is alias of",
  object="Singapore Land Authority".
- When a document references another document by name/number, extract that
  reference: subject="[this doc]", predicate="references", object="[other doc]".

Table extraction:
- When extracting from tables, preserve the relationship between columns.
  E.g., "Product X | Price $100 | Q3 2024" → subject="Product X",
  predicate="has price", object="$100", valid_from="2024-07-01",
  valid_until="2024-09-30".
- Each table row should produce at least one claim.

Process/workflow claims:
- Preserve step ordering in the predicate or object.

Return valid JSON matching the schema below.
"""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "natural_language": {"type": "string"},
                    "knowledge_class": {
                        "type": "string",
                        "enum": ["stable", "process", "tacit", "stateful"],
                    },
                    "confidence": {"type": "number"},
                    "source_span": {"type": "string"},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_until": {"type": ["string", "null"]},
                    "entity_aliases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias": {"type": "string"},
                                "canonical": {"type": "string"},
                            },
                            "required": ["alias", "canonical"],
                            "additionalProperties": False,
                        },
                    },
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "subject",
                    "predicate",
                    "object",
                    "natural_language",
                    "knowledge_class",
                    "confidence",
                    "source_span",
                    "valid_from",
                    "valid_until",
                    "entity_aliases",
                    "references",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class EntityAlias(BaseModel):
    """An alias for an entity discovered during extraction."""

    alias: str
    canonical: str


class ExtractedClaim(BaseModel):
    """Raw extraction result before becoming a full Claim."""

    subject: str
    predicate: str
    object: str
    natural_language: str
    knowledge_class: str
    confidence: float
    source_span: str
    valid_from: str | None = None
    valid_until: str | None = None
    entity_aliases: list[EntityAlias] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Batch of extracted claims from one artifact."""

    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClaimExtractor:
    """Extracts structured claims from source artifacts using OpenAI."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def extract(
        self, artifact: SourceArtifact, source_filename: str = ""
    ) -> list[Claim]:
        """Extract claims from a single source artifact."""
        if not artifact.body.strip():
            return []

        truncated_body = artifact.body[:8000]

        # Build context header with source metadata
        source_file = source_filename or artifact.source_uri or ""
        user_prompt = (
            f"Source: {artifact.source_type.value}\n"
            f"Title: {artifact.title}\n"
            f"Author: {artifact.author}\n"
            f"Filename: {source_file}\n"
            f"Date: {artifact.content_created_at or artifact.created_at}\n\n"
            f"Content:\n{truncated_body}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_result",
                        "strict": True,
                        "schema": EXTRACTION_SCHEMA,
                    },
                },
                temperature=0.1,
            )
        except Exception:
            logger.exception("LLM extraction failed for artifact %s", artifact.id)
            return []

        raw = response.choices[0].message.content
        if not raw:
            return []

        try:
            data = json.loads(raw)
            result = ExtractionResult(**data)
        except (json.JSONDecodeError, Exception):
            logger.exception("Failed to parse extraction result for artifact %s", artifact.id)
            return []

        claims: list[Claim] = []
        source_file = source_filename or artifact.source_uri or ""
        for ec in result.claims:
            klass = KnowledgeClass.process
            try:
                klass = KnowledgeClass(ec.knowledge_class)
            except ValueError:
                pass

            valid_from = _parse_date(ec.valid_from) if ec.valid_from else None
            valid_until = _parse_date(ec.valid_until) if ec.valid_until else None

            claim = Claim(
                subject=ec.subject,
                predicate=ec.predicate,
                object=ec.object,
                natural_language=ec.natural_language,
                confidence=ec.confidence,
                knowledge_class=klass,
                status=ClaimStatus.candidate,
                group_id=artifact.metadata.get("group_id", "default") or "default",
                evidence=[
                    ClaimEvidence(
                        artifact_id=artifact.id,
                        supports=True,
                        source_span=ec.source_span,
                        source_artifact=source_file,
                        author=artifact.author,
                    )
                ],
                created_at=datetime.now(timezone.utc),
                valid_from=valid_from,
                valid_until=valid_until,
            )
            claims.append(claim)

            # Generate alias claims from entity_aliases
            for alias in ec.entity_aliases:
                alias_claim = Claim(
                    subject=alias.alias,
                    predicate="is alias of",
                    object=alias.canonical,
                    natural_language=f"{alias.alias} is an alias for {alias.canonical}",
                    confidence=0.95,
                    knowledge_class=KnowledgeClass.stable,
                    status=ClaimStatus.candidate,
                    group_id=artifact.metadata.get("group_id", "default") or "default",
                    evidence=[
                        ClaimEvidence(
                            artifact_id=artifact.id,
                            supports=True,
                            source_span=ec.source_span,
                            source_artifact=source_file,
                            author=artifact.author,
                        )
                    ],
                    created_at=datetime.now(timezone.utc),
                )
                claims.append(alias_claim)

        logger.info(
            "Extracted %d claims from artifact %s (%s)",
            len(claims),
            artifact.id,
            artifact.title,
        )
        return claims
