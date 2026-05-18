"""Semantic claim deduplication — embedding-based similarity matching."""

from __future__ import annotations

import logging
import math

from openai import AsyncOpenAI

from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """Embedding-based claim similarity for deduplication.

    Uses OpenAI text-embedding-3-large to find semantically similar claims
    that exact-match fingerprinting would miss (e.g., "was established" vs
    "is established", or "SLA" vs "Singapore Land Authority").

    Usage::

        dedup = SemanticDeduplicator(api_key="...", threshold=0.92)
        existing = await dedup.find_duplicate(new_claim, all_claims)
        if existing:
            # merge evidence
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        threshold: float = 0.92,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._threshold = threshold
        self._cache: dict[str, list[float]] = {}

    async def find_duplicate(
        self,
        new_claim: Claim,
        existing_claims: list[Claim],
    ) -> Claim | None:
        """Find the most semantically similar existing claim above threshold.

        Returns None if no match above threshold.
        """
        if not existing_claims:
            return None

        new_text = self._claim_text(new_claim)
        new_embedding = await self._embed(new_text)

        best_match: Claim | None = None
        best_score = 0.0

        for existing in existing_claims:
            existing_text = self._claim_text(existing)
            existing_embedding = await self._embed(existing_text)

            similarity = self._cosine_similarity(new_embedding, existing_embedding)
            if similarity > best_score and similarity >= self._threshold:
                best_score = similarity
                best_match = existing

        if best_match:
            logger.info(
                "Semantic dedup: '%.60s' matches '%.60s' (similarity=%.3f)",
                self._claim_text(new_claim),
                self._claim_text(best_match),
                best_score,
            )

        return best_match

    async def find_duplicates_batch(
        self,
        new_claims: list[Claim],
        existing_claims: list[Claim],
    ) -> dict[str, Claim]:
        """Find duplicates for a batch of new claims.

        Returns {new_claim_id: matching_existing_claim}.
        """
        matches: dict[str, Claim] = {}
        for new_claim in new_claims:
            match = await self.find_duplicate(new_claim, existing_claims)
            if match:
                matches[str(new_claim.id)] = match
        return matches

    async def similarity(self, claim_a: Claim, claim_b: Claim) -> float:
        """Calculate semantic similarity between two claims."""
        text_a = self._claim_text(claim_a)
        text_b = self._claim_text(claim_b)
        emb_a = await self._embed(text_a)
        emb_b = await self._embed(text_b)
        return self._cosine_similarity(emb_a, emb_b)

    async def _embed(self, text: str) -> list[float]:
        """Embed text, using cache to avoid redundant API calls."""
        if text in self._cache:
            return self._cache[text]

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            embedding = response.data[0].embedding
            self._cache[text] = embedding
            return embedding
        except Exception:
            logger.exception("Embedding failed for text: %.60s", text)
            return []

    @staticmethod
    def _claim_text(claim: Claim) -> str:
        """Convert claim to text for embedding."""
        return f"{claim.subject} {claim.predicate} {claim.object}"

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
