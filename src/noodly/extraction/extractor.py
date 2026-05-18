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
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_span": {"type": "string"},
                },
                "required": [
                    "subject",
                    "predicate",
                    "object",
                    "natural_language",
                    "knowledge_class",
                    "confidence",
                    "source_span",
                ],
            },
        }
    },
    "required": ["claims"],
}


class ExtractedClaim(BaseModel):
    """Raw extraction result before becoming a full Claim."""

    subject: str
    predicate: str
    object: str
    natural_language: str
    knowledge_class: str
    confidence: float
    source_span: str


class ExtractionResult(BaseModel):
    """Batch of extracted claims from one artifact."""

    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClaimExtractor:
    """Extracts structured claims from source artifacts using OpenAI."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def extract(self, artifact: SourceArtifact) -> list[Claim]:
        """Extract claims from a single source artifact."""
        if not artifact.body.strip():
            return []

        truncated_body = artifact.body[:8000]

        user_prompt = (
            f"Source: {artifact.source_type.value}\n"
            f"Title: {artifact.title}\n"
            f"Author: {artifact.author}\n"
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
        for ec in result.claims:
            klass = KnowledgeClass.process
            try:
                klass = KnowledgeClass(ec.knowledge_class)
            except ValueError:
                pass

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
                        author=artifact.author,
                    )
                ],
                created_at=datetime.now(timezone.utc),
            )
            claims.append(claim)

        logger.info(
            "Extracted %d claims from artifact %s (%s)",
            len(claims),
            artifact.id,
            artifact.title,
        )
        return claims
