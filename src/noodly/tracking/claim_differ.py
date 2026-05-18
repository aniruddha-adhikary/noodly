"""Claim differ — compute diffs between old and new claims from the same source."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


def _fingerprint(claim: Claim) -> str:
    """Normalize subject+predicate+object into a matching key."""
    return (
        f"{claim.subject.strip().lower()}"
        f"|{claim.predicate.strip().lower()}"
        f"|{claim.object.strip().lower()}"
    )


@dataclass
class ClaimModification:
    """Tracks how a claim changed between versions."""

    old_claim: Claim
    new_claim: Claim
    changed_fields: list[str] = field(default_factory=list)

    @property
    def is_substantive(self) -> bool:
        return bool({"subject", "predicate", "object"} & set(self.changed_fields))


@dataclass
class ClaimDiff:
    """Diff between old and new claims from the same source artifact."""

    source_uri: str
    added_claims: list[Claim] = field(default_factory=list)
    removed_claims: list[Claim] = field(default_factory=list)
    modified_claims: list[ClaimModification] = field(default_factory=list)
    unchanged_claims: list[Claim] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_claims or self.removed_claims or self.modified_claims)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.added_claims:
            parts.append(f"+{len(self.added_claims)} new")
        if self.removed_claims:
            parts.append(f"-{len(self.removed_claims)} removed")
        if self.modified_claims:
            parts.append(f"~{len(self.modified_claims)} modified")
        return ", ".join(parts) or "no changes"


class ClaimDiffer:
    """Computes diffs between old and new claims from the same source."""

    def diff(
        self,
        source_uri: str,
        new_claims: list[Claim],
        old_claims: list[Claim],
    ) -> ClaimDiff:
        """Compare new extraction results against existing claims from same source."""
        old_fps = {_fingerprint(c): c for c in old_claims}
        new_fps = {_fingerprint(c): c for c in new_claims}

        added = [c for fp, c in new_fps.items() if fp not in old_fps]
        removed = [c for fp, c in old_fps.items() if fp not in new_fps]

        modified: list[ClaimModification] = []
        unchanged: list[Claim] = []

        for fp in old_fps.keys() & new_fps.keys():
            changes = self._compare_claims(old_fps[fp], new_fps[fp])
            if changes:
                modified.append(
                    ClaimModification(
                        old_claim=old_fps[fp],
                        new_claim=new_fps[fp],
                        changed_fields=changes,
                    )
                )
            else:
                unchanged.append(old_fps[fp])

        return ClaimDiff(
            source_uri=source_uri,
            added_claims=added,
            removed_claims=removed,
            modified_claims=modified,
            unchanged_claims=unchanged,
        )

    def _compare_claims(self, old: Claim, new: Claim) -> list[str]:
        """Detect which fields changed between two versions of a claim."""
        changed: list[str] = []

        if old.subject.strip().lower() != new.subject.strip().lower():
            changed.append("subject")
        if old.predicate.strip().lower() != new.predicate.strip().lower():
            changed.append("predicate")
        if old.object.strip().lower() != new.object.strip().lower():
            changed.append("object")
        if old.natural_language != new.natural_language:
            changed.append("natural_language")
        if abs(old.confidence - new.confidence) > 0.05:
            changed.append("confidence")
        if old.knowledge_class != new.knowledge_class:
            changed.append("knowledge_class")

        return changed
