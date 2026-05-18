"""LLM-based topic classifier for claims.

Classifies claims into topics for topic-aware authority scoring
and document consolidation. Uses OpenAI to cluster claims into
meaningful topic groups.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

TOPIC_SYSTEM_PROMPT = """\
You are a topic classifier for a knowledge management system.

Given a list of claims (subject-predicate-object triples), assign each claim
to one or more topics. Topics should be broad enough to group related claims
but specific enough to be meaningful.

Rules:
- Use lowercase, hyphenated topic names (e.g., "trade-compliance", "http-protocol")
- Each claim must have at least one topic
- Reuse existing topics when possible — prefer consistency over novelty
- Topics should describe the knowledge domain, not the document type
- Keep topics to 2-4 words maximum

Return valid JSON matching the schema below.
"""

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim_index", "topics"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


class TopicClassifier:
    """Classifies claims into topics using LLM or keyword matching.

    Usage::

        classifier = TopicClassifier(api_key="...", mode="llm")
        topics = await classifier.classify(claims)
        # {"uuid1": ["trade-compliance"], "uuid2": ["http-protocol"]}
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        mode: str = "llm",
        cache_path: Path | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._mode = mode
        self._cache_path = cache_path
        self._cache: dict[str, list[str]] = {}
        self._client: AsyncOpenAI | None = None
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, ValueError):
                pass

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def classify(
        self,
        claims: list,
        existing_topics: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Classify claims into topics.

        Returns a mapping of claim_id → list of topic strings.
        """
        if not claims:
            return {}

        # Check cache for already-classified claims
        uncached = []
        result: dict[str, list[str]] = {}
        for claim in claims:
            claim_id = str(claim.id)
            if claim_id in self._cache:
                result[claim_id] = self._cache[claim_id]
            else:
                uncached.append(claim)

        if not uncached:
            return result

        if self._mode == "keyword":
            for claim in uncached:
                topics = self._keyword_classify(claim)
                claim_id = str(claim.id)
                result[claim_id] = topics
                self._cache[claim_id] = topics
        else:
            llm_results = await self._llm_classify(uncached, existing_topics)
            result.update(llm_results)
            self._cache.update(llm_results)

        self._save_cache()
        return result

    def _keyword_classify(self, claim) -> list[str]:
        """Fast keyword-based topic classification."""
        text = f"{claim.subject} {claim.predicate} {claim.object}".lower()

        topic_keywords = {
            "trade-compliance": ["trade", "customs", "import", "export", "tariff", "duty"],
            "http-protocol": ["http", "rfc", "protocol", "header", "request", "response"],
            "machine-learning": ["model", "training", "neural", "transformer", "bert", "gpt"],
            "legislation": ["act", "law", "regulation", "statute", "section", "amendment"],
            "permits-licensing": ["permit", "license", "approval", "registration"],
            "organizational": ["team", "department", "role", "manager", "employee"],
            "technical-infrastructure": ["server", "database", "api", "deployment", "cloud"],
        }

        matched = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                matched.append(topic)

        return matched or ["general"]

    async def _llm_classify(
        self,
        claims: list,
        existing_topics: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """LLM-based topic classification."""
        client = self._get_client()

        # Build claim descriptions for the prompt
        claim_lines = []
        for i, claim in enumerate(claims):
            claim_lines.append(
                f"{i}. {claim.subject} | {claim.predicate} | {claim.object}"
            )

        topics_hint = ""
        if existing_topics:
            topics_hint = (
                f"\n\nExisting topics to reuse when applicable: "
                f"{', '.join(existing_topics)}"
            )

        # Process in batches of 50 to stay within context limits
        batch_size = 50
        result: dict[str, list[str]] = {}

        for batch_start in range(0, len(claims), batch_size):
            batch = claims[batch_start : batch_start + batch_size]
            batch_lines = []
            for i, claim in enumerate(batch):
                batch_lines.append(
                    f"{i}. {claim.subject} | {claim.predicate} | {claim.object}"
                )

            batch_prompt = (
                f"Classify these claims into topics:{topics_hint}\n\n"
                + "\n".join(batch_lines)
            )

            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
                        {"role": "user", "content": batch_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "topic_classification",
                            "strict": True,
                            "schema": TOPIC_SCHEMA,
                        },
                    },
                    temperature=0.1,
                )

                raw = response.choices[0].message.content
                if raw:
                    data = json.loads(raw)
                    for assignment in data.get("assignments", []):
                        idx = assignment["claim_index"]
                        if 0 <= idx < len(batch):
                            claim_id = str(batch[idx].id)
                            result[claim_id] = assignment["topics"]
            except Exception:
                logger.exception("LLM topic classification failed for batch")
                # Fall back to keyword for this batch
                for claim in batch:
                    claim_id = str(claim.id)
                    if claim_id not in result:
                        result[claim_id] = self._keyword_classify(claim)

        # Ensure all claims have at least one topic
        for claim in claims:
            claim_id = str(claim.id)
            if claim_id not in result:
                result[claim_id] = self._keyword_classify(claim)

        return result

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=2))

    def get_all_topics(self) -> list[str]:
        """Return all known topics from the cache."""
        all_topics: set[str] = set()
        for topics in self._cache.values():
            all_topics.update(topics)
        return sorted(all_topics)
