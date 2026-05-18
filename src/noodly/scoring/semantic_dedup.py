"""Semantic claim deduplication — embedding-based similarity matching."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """Result of semantic deduplication on a batch of claims."""

    merged: list[tuple[Claim, Claim]] = field(default_factory=list)
    unique: list[Claim] = field(default_factory=list)

    @property
    def merged_count(self) -> int:
        return len(self.merged)

    @property
    def unique_count(self) -> int:
        return len(self.unique)


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
        if not existing_claims or not new_claims:
            return {}

        # Pre-embed all existing claims in a single batch call
        await self._embed_batch([self._claim_text(c) for c in existing_claims])
        # Pre-embed all new claims in a single batch call
        await self._embed_batch([self._claim_text(c) for c in new_claims])

        matches: dict[str, Claim] = {}
        for new_claim in new_claims:
            match = await self.find_duplicate(new_claim, existing_claims)
            if match:
                matches[str(new_claim.id)] = match
        return matches

    async def deduplicate_and_merge(
        self,
        new_claims: list[Claim],
        existing_claims: list[Claim],
    ) -> DedupResult:
        """Find semantic duplicates and merge evidence into existing claims.

        For each new claim that matches an existing claim:
        - Merge evidence entries (avoiding duplicates by artifact_id)
        - Boost confidence to max of both
        - Update last_confirmed_at

        Returns a DedupResult with merged pairs and unique (unmatched) claims.
        """
        from datetime import datetime, timezone

        result = DedupResult()
        if not new_claims:
            return result

        matches = await self.find_duplicates_batch(new_claims, existing_claims)

        for new_claim in new_claims:
            existing = matches.get(str(new_claim.id))
            if existing is not None:
                # Merge evidence
                seen_artifacts = {str(ev.artifact_id) for ev in existing.evidence}
                for ev in new_claim.evidence:
                    if str(ev.artifact_id) not in seen_artifacts:
                        existing.evidence.append(ev)
                        seen_artifacts.add(str(ev.artifact_id))
                existing.confidence = max(existing.confidence, new_claim.confidence)
                existing.last_confirmed_at = datetime.now(timezone.utc)
                result.merged.append((new_claim, existing))
                logger.info(
                    "Semantic merge: '%.50s' into existing claim %s (now %d evidence)",
                    self._claim_text(new_claim),
                    existing.id,
                    len(existing.evidence),
                )
            else:
                result.unique.append(new_claim)

        return result

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

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call for efficiency."""
        uncached = [t for t in texts if t not in self._cache]
        if not uncached:
            return [self._cache[t] for t in texts]

        # OpenAI supports up to 2048 inputs per batch call
        batch_size = 2048
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
                for text, data in zip(batch, response.data):
                    self._cache[text] = data.embedding
            except Exception:
                logger.exception(
                    "Batch embedding failed for %d texts", len(batch)
                )
                for text in batch:
                    if text not in self._cache:
                        self._cache[text] = []

        return [self._cache.get(t, []) for t in texts]

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
