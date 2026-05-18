"""Fact ledger — bitemporal claim storage and truth maintenance."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from noodly.models.claims import Claim, ClaimStatus, KnowledgeClass
from noodly.scoring.authority import AuthorityRegistry

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

# Evidence count thresholds for auto-promotion
CORROBORATION_EVIDENCE_COUNT = 2  # promote to corroborated with 2+ independent sources


def _claim_fingerprint(claim: Claim) -> str:
    """Normalize subject+predicate+object into a dedup key."""
    return (
        f"{claim.subject.strip().lower()}"
        f"|{claim.predicate.strip().lower()}"
        f"|{claim.object.strip().lower()}"
    )


class FactLedger:
    """JSON-file-backed fact ledger (v1 — Postgres later).

    Stores claims with bitemporal metadata:
    - ``valid_from`` / ``valid_until`` — when the fact is true in the world
    - ``created_at`` — when the system first learned it (transaction time)
    """

    def __init__(
        self,
        ledger_path: Path,
        authority_registry: AuthorityRegistry | None = None,
    ) -> None:
        self._path = ledger_path
        self._claims: dict[str, Claim] = {}
        self._fingerprint_index: dict[str, str] = {}
        self._authority = authority_registry
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for item in data:
                    claim = Claim(**item)
                    self._claims[str(claim.id)] = claim
                    fp = _claim_fingerprint(claim)
                    self._fingerprint_index[fp] = str(claim.id)
            except (json.JSONDecodeError, Exception):
                logger.warning("Could not load ledger from %s, starting fresh", self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [claim.model_dump(mode="json") for claim in self._claims.values()]
        self._path.write_text(json.dumps(data, indent=2, default=str))

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
        """Check if a semantically equivalent claim already exists."""
        fp = _claim_fingerprint(claim)
        existing_id = self._fingerprint_index.get(fp)
        if existing_id is None:
            return None
        return self._claims.get(existing_id)

    def add_claim(self, claim: Claim) -> Claim:
        """Add a claim, deduplicating if an equivalent already exists.

        If a duplicate is found (same subject+predicate+object), the new
        evidence is merged into the existing claim instead of creating a
        new entry.
        """
        self._apply_authority(claim)

        existing = self._find_duplicate(claim)
        if existing is not None:
            seen_artifacts = {str(ev.artifact_id) for ev in existing.evidence}
            for ev in claim.evidence:
                if str(ev.artifact_id) not in seen_artifacts:
                    existing.evidence.append(ev)
                    seen_artifacts.add(str(ev.artifact_id))
            existing.confidence = max(existing.confidence, claim.confidence)
            if claim.last_confirmed_at:
                existing.last_confirmed_at = claim.last_confirmed_at
            # Auto-promote based on corroboration
            self._auto_promote(existing)
            logger.info(
                "Ledger: merged evidence into existing claim %s [%s] score=%.2f",
                existing.id,
                existing.status.value,
                existing.truth_score,
            )
            self._save()
            return existing

        if claim.status == ClaimStatus.candidate and claim.truth_score >= AUTO_PROMOTE_THRESHOLD:
            claim.status = ClaimStatus.unverified

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

    def add_claims(self, claims: list[Claim]) -> list[Claim]:
        """Add multiple claims."""
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
        """Auto-promote claim status based on evidence count and score.

        Promotion ladder:
        - candidate → unverified: truth_score >= AUTO_PROMOTE_THRESHOLD
        - candidate/unverified → corroborated: 2+ independent supporting sources
        """
        independent_sources = len(
            {str(ev.artifact_id) for ev in claim.evidence if ev.supports}
        )

        # Corroboration check first — 2+ independent sources is strong signal
        if (
            claim.status in (ClaimStatus.candidate, ClaimStatus.unverified)
            and independent_sources >= CORROBORATION_EVIDENCE_COUNT
        ):
            claim.status = ClaimStatus.corroborated
            claim.last_confirmed_at = datetime.now(timezone.utc)
            logger.info(
                "Auto-promoted %s to corroborated (%d sources)",
                claim.id,
                independent_sources,
            )
            return

        # Score-based promotion for single-source claims
        if (
            claim.status == ClaimStatus.candidate
            and claim.truth_score >= AUTO_PROMOTE_THRESHOLD
        ):
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

    @property
    def count(self) -> int:
        return len(self._claims)
