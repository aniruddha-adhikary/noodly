"""Fact ledger — bitemporal claim storage and truth maintenance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from noodly.models.claims import Claim, ClaimStatus, KnowledgeClass
from noodly.scoring.authority import AuthorityRegistry
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

# Configurable promotion thresholds (overridable via constructor)
DEFAULT_PROMOTE_THRESHOLD = 0.15
DEFAULT_HIGH_AUTHORITY_THRESHOLD = 0.8
DEFAULT_CORROBORATION_COUNT = 2


def _claim_fingerprint(claim: Claim) -> str:
    """Normalize subject+predicate+object into a dedup key."""
    return (
        f"{claim.subject.strip().lower()}"
        f"|{claim.predicate.strip().lower()}"
        f"|{claim.object.strip().lower()}"
    )


def _claim_text(claim: Claim) -> str:
    """Convert claim to text for embedding."""
    return f"{claim.subject} {claim.predicate} {claim.object}"


class FactLedger:
    """Backend-agnostic fact ledger with bitemporal claim storage and truth maintenance.

    Stores claims with bitemporal metadata:
    - ``valid_from`` / ``valid_until`` — when the fact is true in the world
    - ``created_at`` — when the system first learned it (transaction time)

    The storage layer is pluggable via a ``LedgerBackend``.  Pass a ``Path``
    for backward-compatible JSON storage, or a
    :class:`~noodly.storage.graphiti_backend.GraphitiBackend` to store claims
    as Graphiti edges.
    """

    def __init__(
        self,
        backend: LedgerBackend | Path,
        authority_registry: AuthorityRegistry | None = None,
        promote_threshold: float = DEFAULT_PROMOTE_THRESHOLD,
        high_authority_threshold: float = DEFAULT_HIGH_AUTHORITY_THRESHOLD,
        corroboration_count: int = DEFAULT_CORROBORATION_COUNT,
    ) -> None:
        if isinstance(backend, Path):
            self._backend: LedgerBackend = JSONBackend(backend)
        else:
            self._backend = backend

        self._claims: dict[str, Claim] = {}
        self._fingerprint_index: dict[str, str] = {}
        self._authority = authority_registry
        self._promote_threshold = promote_threshold
        self._high_authority_threshold = high_authority_threshold
        self._corroboration_count = corroboration_count

        if not self.is_async_backend:
            self._load()

    @property
    def is_async_backend(self) -> bool:
        """Whether the current backend requires async operations."""
        from noodly.storage.graphiti_backend import GraphitiBackend

        return isinstance(self._backend, GraphitiBackend)

    def _load(self) -> None:
        self._claims = self._backend.load_claims()
        self._fingerprint_index = {}
        for claim_id, claim in self._claims.items():
            fp = _claim_fingerprint(claim)
            self._fingerprint_index[fp] = claim_id

    async def load_async(self) -> None:
        """Async load for backends that require it (e.g. GraphitiBackend)."""
        from noodly.storage.graphiti_backend import GraphitiBackend

        if isinstance(self._backend, GraphitiBackend):
            self._claims = await self._backend.load_claims_async()
            self._fingerprint_index = {}
            for claim_id, claim in self._claims.items():
                fp = _claim_fingerprint(claim)
                self._fingerprint_index[fp] = claim_id

    def _save(self) -> None:
        self._backend.save_all(self._claims)

    async def _save_async(self) -> None:
        """Async save for backends that require it."""
        from noodly.storage.graphiti_backend import GraphitiBackend

        if isinstance(self._backend, GraphitiBackend):
            await self._backend.save_all_async(self._claims)
        else:
            self._backend.save_all(self._claims)

    async def _save_claim_async(self, claim: Claim) -> None:
        """Save a single claim via async backend."""
        from noodly.storage.graphiti_backend import GraphitiBackend

        if isinstance(self._backend, GraphitiBackend):
            await self._backend.save_claim_async(claim)
        else:
            self._backend.save_claim(claim)

    def _apply_authority(self, claim: Claim, topic: str | None = None) -> None:
        """Stamp evidence with authority weights from the registry.

        If ``topic`` is provided, uses topic-aware authority lookup.
        """
        if self._authority is None:
            return
        for ev in claim.evidence:
            if ev.author:
                ev.source_authority = self._authority.get(ev.author, topic=topic)

    def _find_duplicate(self, claim: Claim) -> Claim | None:
        """Check if an exact-match equivalent claim already exists."""
        fp = _claim_fingerprint(claim)
        existing_id = self._fingerprint_index.get(fp)
        if existing_id is None:
            return None
        return self._claims.get(existing_id)

    def _merge_into(self, existing: Claim, new_claim: Claim) -> Claim:
        """Merge new_claim's evidence into existing claim."""
        seen_artifacts = {str(ev.artifact_id) for ev in existing.evidence}
        for ev in new_claim.evidence:
            if str(ev.artifact_id) not in seen_artifacts:
                existing.evidence.append(ev)
                seen_artifacts.add(str(ev.artifact_id))
        existing.confidence = max(existing.confidence, new_claim.confidence)
        if new_claim.last_confirmed_at:
            existing.last_confirmed_at = new_claim.last_confirmed_at
        else:
            existing.last_confirmed_at = datetime.now(timezone.utc)
        self._auto_promote(existing)
        return existing

    def add_claim(self, claim: Claim) -> Claim:
        """Add a claim, deduplicating if an equivalent already exists.

        Dedup: exact fingerprint match (subject|predicate|object).
        Semantic dedup is handled by Graphiti natively when using GraphitiBackend.
        """
        self._apply_authority(claim)

        existing = self._find_duplicate(claim)
        if existing is not None:
            self._merge_into(existing, claim)
            logger.info(
                "Ledger: merged evidence into existing claim %s [%s] score=%.2f",
                existing.id,
                existing.status.value,
                existing.truth_score,
            )
            self._save()
            return existing

        # Insert as new — run promotion check
        self._auto_promote(claim)

        self._claims[str(claim.id)] = claim
        self._fingerprint_index[_claim_fingerprint(claim)] = str(claim.id)
        self._save()
        logger.info(
            "Ledger: added claim %s [%s] score=%.2f",
            claim.id,
            claim.status.value,
            claim.truth_score,
        )
        return claim

    async def add_claim_async(self, claim: Claim) -> Claim:
        """Add a claim using async persistence."""
        self._apply_authority(claim)

        existing = self._find_duplicate(claim)
        if existing is not None:
            self._merge_into(existing, claim)
            logger.info(
                "Ledger: merged evidence into existing claim %s [%s] score=%.2f",
                existing.id,
                existing.status.value,
                existing.truth_score,
            )
            await self._save_claim_async(existing)
            return existing

        self._auto_promote(claim)

        self._claims[str(claim.id)] = claim
        self._fingerprint_index[_claim_fingerprint(claim)] = str(claim.id)
        await self._save_claim_async(claim)
        logger.info(
            "Ledger: added claim %s [%s] score=%.2f",
            claim.id,
            claim.status.value,
            claim.truth_score,
        )
        return claim

    async def add_claims_async(self, claims: list[Claim]) -> list[Claim]:
        """Add multiple claims with async persistence."""
        results = []
        for claim in claims:
            results.append(await self.add_claim_async(claim))
        return results

    def add_claims(self, claims: list[Claim]) -> list[Claim]:
        """Add multiple claims (sync version)."""
        results = []
        for claim in claims:
            results.append(self.add_claim(claim))
        return results

    def get_claim(self, claim_id: str) -> Claim | None:
        """Look up a claim by ID."""
        return self._claims.get(claim_id)

    def list_claims(
        self,
        status: ClaimStatus | None = None,
        group_id: str | None = None,
        limit: int = 50,
        as_of_valid: datetime | None = None,
        as_of_transaction: datetime | None = None,
    ) -> list[Claim]:
        """List claims, optionally filtered by status, group, and time.

        Bi-temporal filters:
        - ``as_of_valid``: only claims whose valid window contains this time
        - ``as_of_transaction``: only claims created on or before this time
        """
        results = list(self._claims.values())

        if status is not None:
            results = [c for c in results if c.status == status]
        if group_id is not None:
            results = [c for c in results if c.group_id == group_id]

        if as_of_valid is not None:
            results = [
                c for c in results
                if (c.valid_from is None or c.valid_from <= as_of_valid)
                and (c.valid_until is None or c.valid_until >= as_of_valid)
            ]

        if as_of_transaction is not None:
            results = [c for c in results if c.created_at <= as_of_transaction]

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

    def supersede_claim(self, old_id: str, new_id: str) -> bool:
        """Mark ``old_id`` as superseded by ``new_id``.

        The old claim is marked ``superseded`` and the new claim records
        what it supersedes.  Returns False if either claim is missing.
        """
        old = self._claims.get(old_id)
        new = self._claims.get(new_id)
        if old is None or new is None:
            return False

        old.status = ClaimStatus.superseded
        old.superseded_by = new.id
        new.supersedes = old.id
        self._save()
        logger.info("Claim %s superseded by %s", old_id, new_id)
        return True

    def add_conflict(self, claim_id_a: str, claim_id_b: str) -> bool:
        """Mark two claims as conflicting with each other."""
        a = self._claims.get(claim_id_a)
        b = self._claims.get(claim_id_b)
        if a is None or b is None:
            return False

        if b.id not in a.conflicts_with:
            a.conflicts_with.append(b.id)
        if a.id not in b.conflicts_with:
            b.conflicts_with.append(a.id)
        self._save()
        logger.info("Conflict recorded: %s <-> %s", claim_id_a, claim_id_b)
        return True

    def retract_evidence(self, artifact_id: str) -> int:
        """Mark all evidence from a given artifact as refuting.

        This implements retraction propagation: when a source is
        invalidated, all claims that relied on it are weakened.
        Returns the number of claims affected.
        """
        affected = 0
        for claim in self._claims.values():
            for ev in claim.evidence:
                if str(ev.artifact_id) == artifact_id and ev.supports:
                    ev.supports = False
                    affected += 1
        if affected > 0:
            self._save()
            logger.info(
                "Retracted evidence from artifact %s (%d claims affected)",
                artifact_id,
                affected,
            )
        return affected

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

    def _auto_promote(self, claim: Claim) -> None:
        """Auto-promote claim status based on evidence, authority, and score.

        Promotion ladder:
        - candidate → unverified: truth_score >= promote_threshold OR
          any evidence has source_authority >= high_authority_threshold
        - candidate/unverified → corroborated: N+ independent supporting sources
        """
        independent_sources = len(
            {str(ev.artifact_id) for ev in claim.evidence if ev.supports}
        )

        # Corroboration check first — N+ independent sources is strong signal
        if (
            claim.status in (ClaimStatus.candidate, ClaimStatus.unverified)
            and independent_sources >= self._corroboration_count
        ):
            claim.status = ClaimStatus.corroborated
            claim.last_confirmed_at = datetime.now(timezone.utc)
            logger.info(
                "Auto-promoted %s to corroborated (%d sources)",
                claim.id,
                independent_sources,
            )
            return

        if claim.status != ClaimStatus.candidate:
            return

        # High-authority source promotes to unverified immediately
        max_authority = max(
            (ev.source_authority for ev in claim.evidence if ev.supports),
            default=0.0,
        )
        if max_authority >= self._high_authority_threshold:
            claim.status = ClaimStatus.unverified
            logger.info(
                "Auto-promoted %s to unverified (high authority=%.2f)",
                claim.id,
                max_authority,
            )
            return

        # Score-based promotion
        if claim.truth_score >= self._promote_threshold:
            claim.status = ClaimStatus.unverified
            logger.info(
                "Auto-promoted %s to unverified (score=%.2f)",
                claim.id,
                claim.truth_score,
            )

    def auto_promote_all(self) -> int:
        """Run auto-promotion on all claims. Returns count of promoted claims."""
        promoted = 0
        for claim in self._claims.values():
            old_status = claim.status
            self._auto_promote(claim)
            if claim.status != old_status:
                promoted += 1
        if promoted > 0:
            self._save()
        return promoted

    def promotion_stats(self) -> dict[str, int]:
        """Return counts of claims by status."""
        stats: dict[str, int] = {}
        for claim in self._claims.values():
            key = claim.status.value
            stats[key] = stats.get(key, 0) + 1
        return stats

    def embedded_count(self) -> int:
        """Return count of claims that have stored embeddings."""
        return sum(1 for c in self._claims.values() if c.embedding)

    @property
    def count(self) -> int:
        return len(self._claims)
