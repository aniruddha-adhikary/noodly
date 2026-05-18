"""Conflict detector — finds conflicting claims about the same subject+predicate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


@dataclass
class ConflictPair:
    """Two claims that assert different objects for the same subject+predicate."""

    id: UUID
    claim_a: Claim
    claim_b: Claim
    conflict_type: str  # "contradictory_value", "temporal_overlap", "semantic_clash"
    detected_at: datetime
    detected_by: str  # "graph_agent", "ingestion", "detector"
    description: str = ""

    @property
    def score_delta(self) -> float:
        """Absolute difference in truth scores between the two claims."""
        return abs(self.claim_a.truth_score - self.claim_b.truth_score)


class ConflictDetector:
    """Detects conflicts between claims about the same subject+predicate.

    Two claims conflict when they share the same subject and predicate
    but assert different objects.

    Usage::

        detector = ConflictDetector(similarity_threshold=0.8)
        conflicts = detector.detect(new_claims, existing_claims)
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self._similarity_threshold = similarity_threshold

    def detect(
        self,
        new_claims: list[Claim],
        existing_claims: list[Claim],
    ) -> list[ConflictPair]:
        """Find pairs of claims that assert different objects for same subject+predicate."""
        conflicts: list[ConflictPair] = []
        seen_pairs: set[tuple[str, str]] = set()

        existing_by_sp: dict[str, list[Claim]] = {}
        for claim in existing_claims:
            key = self._normalize_key(claim.subject, claim.predicate)
            existing_by_sp.setdefault(key, []).append(claim)

        for new_claim in new_claims:
            key = self._normalize_key(new_claim.subject, new_claim.predicate)
            candidates = existing_by_sp.get(key, [])

            for existing in candidates:
                if str(new_claim.id) == str(existing.id):
                    continue

                pair_key = tuple(sorted([str(new_claim.id), str(existing.id)]))
                if pair_key in seen_pairs:
                    continue

                if self._objects_conflict(new_claim.object, existing.object):
                    seen_pairs.add(pair_key)
                    conflicts.append(
                        ConflictPair(
                            id=uuid4(),
                            claim_a=existing,
                            claim_b=new_claim,
                            conflict_type="contradictory_value",
                            detected_at=datetime.now(timezone.utc),
                            detected_by="detector",
                            description=(
                                f"{new_claim.subject} {new_claim.predicate}: "
                                f'"{existing.object}" vs "{new_claim.object}"'
                            ),
                        )
                    )

        if conflicts:
            logger.info("Detected %d conflicts", len(conflicts))
        return conflicts

    def detect_within(self, claims: list[Claim]) -> list[ConflictPair]:
        """Find conflicts within a single list of claims."""
        conflicts: list[ConflictPair] = []
        seen_pairs: set[tuple[str, str]] = set()

        by_sp: dict[str, list[Claim]] = {}
        for claim in claims:
            key = self._normalize_key(claim.subject, claim.predicate)
            by_sp.setdefault(key, []).append(claim)

        for claims_group in by_sp.values():
            if len(claims_group) < 2:
                continue
            for i, a in enumerate(claims_group):
                for b in claims_group[i + 1 :]:
                    pair_key = tuple(sorted([str(a.id), str(b.id)]))
                    if pair_key in seen_pairs:
                        continue
                    if self._objects_conflict(a.object, b.object):
                        seen_pairs.add(pair_key)
                        conflicts.append(
                            ConflictPair(
                                id=uuid4(),
                                claim_a=a,
                                claim_b=b,
                                conflict_type="contradictory_value",
                                detected_at=datetime.now(timezone.utc),
                                detected_by="detector",
                                description=(
                                    f"{a.subject} {a.predicate}: "
                                    f'"{a.object}" vs "{b.object}"'
                                ),
                            )
                        )

        return conflicts

    def _normalize_key(self, subject: str, predicate: str) -> str:
        """Normalize subject+predicate for grouping."""
        return f"{subject.strip().lower()}|{predicate.strip().lower()}"

    def _objects_conflict(self, obj_a: str, obj_b: str) -> bool:
        """Determine if two objects represent conflicting values."""
        a_norm = obj_a.strip().lower()
        b_norm = obj_b.strip().lower()
        if a_norm == b_norm:
            return False
        similarity = self._token_overlap(a_norm, b_norm)
        if similarity >= self._similarity_threshold:
            return False
        return True

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        """Simple token overlap ratio."""
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        overlap = a_tokens & b_tokens
        return len(overlap) / max(len(a_tokens), len(b_tokens))
