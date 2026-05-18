"""Fact ledger — bitemporal claim storage and truth maintenance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from noodly.models.claims import Claim, ClaimStatus, KnowledgeClass
from noodly.storage.json_backend import JSONBackend

if TYPE_CHECKING:
    from noodly.storage import LedgerBackend

logger = logging.getLogger(__name__)

# Decay multipliers per knowledge class (applied per day of staleness)
DECAY_RATES: dict[KnowledgeClass, float] = {
    KnowledgeClass.stable: 0.999,
    KnowledgeClass.process: 0.995,
    KnowledgeClass.tacit: 0.98,
    KnowledgeClass.stateful: 0.95,
}

# Minimum truth_score for auto-promotion to "unverified"
AUTO_PROMOTE_THRESHOLD = 0.3


class FactLedger:
    """Backend-agnostic fact ledger with bitemporal claim storage and truth maintenance.

    Stores claims with bitemporal metadata:
    - ``valid_from`` / ``valid_until`` — when the fact is true in the world
    - ``created_at`` — when the system first learned it (transaction time)

    The storage layer is pluggable via a ``LedgerBackend``.  By default a
    :class:`~noodly.storage.json_backend.JSONBackend` is used for backward
    compatibility; pass a
    :class:`~noodly.storage.graphiti_backend.GraphitiBackend` to store
    claims as Graphiti edges instead.
    """

    def __init__(
        self,
        backend: LedgerBackend | Path | None = None,
    ) -> None:
        if backend is None or isinstance(backend, Path):
            path = backend or Path("ledger.json")
            self._backend: LedgerBackend = JSONBackend(path)
        else:
            self._backend = backend

        self._claims: dict[str, Claim] = {}
        self._load()

    def _load(self) -> None:
        self._claims = self._backend.load_claims()

    async def load_async(self) -> None:
        """Async load for backends that require it (e.g. GraphitiBackend)."""
        from noodly.storage.graphiti_backend import GraphitiBackend

        if isinstance(self._backend, GraphitiBackend):
            self._claims = await self._backend.load_claims_async()
        else:
            self._claims = self._backend.load_claims()

    def _save(self) -> None:
        self._backend.save_all(self._claims)

    async def _save_async(self) -> None:
        from noodly.storage.graphiti_backend import GraphitiBackend

        if isinstance(self._backend, GraphitiBackend):
            await self._backend.save_all_async(self._claims)
        else:
            self._backend.save_all(self._claims)

    def add_claim(self, claim: Claim) -> Claim:
        """Add a new claim to the ledger. Auto-promote if score is high enough."""
        if claim.status == ClaimStatus.candidate and claim.truth_score >= AUTO_PROMOTE_THRESHOLD:
            claim.status = ClaimStatus.unverified

        self._claims[str(claim.id)] = claim
        self._save()
        logger.info(
            "Ledger: added claim %s [%s] score=%.2f",
            claim.id,
            claim.status.value,
            claim.truth_score,
        )
        return claim

    async def add_claim_async(self, claim: Claim) -> Claim:
        """Async variant of add_claim for async backends."""
        if claim.status == ClaimStatus.candidate and claim.truth_score >= AUTO_PROMOTE_THRESHOLD:
            claim.status = ClaimStatus.unverified

        self._claims[str(claim.id)] = claim
        await self._save_claim_single_async(claim)
        logger.info(
            "Ledger: added claim %s [%s] score=%.2f",
            claim.id,
            claim.status.value,
            claim.truth_score,
        )
        return claim

    async def _save_claim_single_async(self, claim: Claim) -> None:
        from noodly.storage.graphiti_backend import GraphitiBackend

        if isinstance(self._backend, GraphitiBackend):
            await self._backend.save_claim_async(claim)
        else:
            self._backend.save_claim(claim)

    def add_claims(self, claims: list[Claim]) -> list[Claim]:
        """Add multiple claims."""
        results = []
        for claim in claims:
            results.append(self.add_claim(claim))
        return results

    async def add_claims_async(self, claims: list[Claim]) -> list[Claim]:
        """Async variant of add_claims for async backends."""
        results = []
        for claim in claims:
            results.append(await self.add_claim_async(claim))
        return results

    def get_claim(self, claim_id: str) -> Claim | None:
        """Look up a claim by ID."""
        return self._claims.get(claim_id)

    def list_claims(
        self,
        status: ClaimStatus | None = None,
        group_id: str | None = None,
        limit: int = 50,
    ) -> list[Claim]:
        """List claims, optionally filtered by status and group."""
        results = list(self._claims.values())
        if status is not None:
            results = [c for c in results if c.status == status]
        if group_id is not None:
            results = [c for c in results if c.group_id == group_id]
        results.sort(key=lambda c: c.truth_score, reverse=True)
        return results[:limit]

    def promote_claim(self, claim_id: str, new_status: ClaimStatus) -> Claim | None:
        """Manually promote or demote a claim's status."""
        claim = self._claims.get(claim_id)
        if claim is None:
            return None
        claim.status = new_status
        if new_status in (ClaimStatus.owner_confirmed, ClaimStatus.canonical):
            claim.last_confirmed_at = datetime.now(timezone.utc)
        self._save()
        return claim

    def apply_decay(self) -> int:
        """Apply time-based decay to recency scores. Returns count of decayed claims."""
        now = datetime.now(timezone.utc)
        decayed = 0
        for claim in self._claims.values():
            if claim.status in (ClaimStatus.superseded, ClaimStatus.rejected):
                continue
            rate = DECAY_RATES.get(claim.knowledge_class, 0.995)
            days_since = (now - claim.created_at).total_seconds() / 86400
            if claim.last_confirmed_at:
                days_since = (now - claim.last_confirmed_at).total_seconds() / 86400
            new_recency = rate**days_since
            if abs(new_recency - claim.recency) > 0.001:
                claim.recency = round(new_recency, 4)
                decayed += 1
        if decayed > 0:
            self._save()
        return decayed

    @property
    def count(self) -> int:
        return len(self._claims)

    @property
    def is_async_backend(self) -> bool:
        """Whether the current backend requires async operations."""
        from noodly.storage.graphiti_backend import GraphitiBackend

        return isinstance(self._backend, GraphitiBackend)
