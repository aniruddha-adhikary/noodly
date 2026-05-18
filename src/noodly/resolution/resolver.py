"""Conflict resolver — orchestrates auto and manual conflict resolution."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from noodly.resolution.audit import Resolution, ResolutionAudit
from noodly.resolution.detector import ConflictDetector, ConflictPair
from noodly.resolution.strategies import AutoResolveStrategy, resolve_by_strategy
from noodly.scoring.ledger import FactLedger
from noodly.tracking.changelog import ChangeEvent, ChangeLog, ChangeType

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Orchestrates conflict resolution — auto or manual based on confidence delta.

    When the score gap between two conflicting claims exceeds ``auto_threshold``,
    the conflict is auto-resolved using the configured strategy. Otherwise it's
    dispatched for manual resolution (e.g., GitLab MR).

    Usage::

        resolver = ConflictResolver(
            ledger=ledger,
            audit=audit,
            auto_threshold=0.3,
            strategy=AutoResolveStrategy.AUTHORITY_WINS,
        )
        conflicts = resolver.detect_conflicts(new_claims)
        for conflict in conflicts:
            resolution = await resolver.resolve(conflict)
    """

    def __init__(
        self,
        ledger: FactLedger,
        audit: ResolutionAudit,
        changelog: ChangeLog | None = None,
        auto_threshold: float = 0.3,
        strategy: AutoResolveStrategy = AutoResolveStrategy.AUTHORITY_WINS,
        similarity_threshold: float = 0.8,
        manual_handler=None,
    ) -> None:
        self._ledger = ledger
        self._audit = audit
        self._changelog = changelog
        self._auto_threshold = auto_threshold
        self._strategy = strategy
        self._detector = ConflictDetector(similarity_threshold=similarity_threshold)
        self._manual_handler = manual_handler

    def detect_conflicts(self, new_claims: list) -> list[ConflictPair]:
        """Detect conflicts between new claims and existing ones."""
        existing = self._ledger.list_claims(limit=10000)
        return self._detector.detect(new_claims, existing)

    async def resolve(self, conflict: ConflictPair) -> Resolution:
        """Resolve a conflict. Auto if score gap > threshold, else dispatch for manual."""
        delta = conflict.score_delta
        if delta >= self._auto_threshold:
            return await self._auto_resolve(conflict)
        else:
            return await self._manual_resolve(conflict)

    async def resolve_from_event(self, event: ChangeEvent) -> Resolution:
        """Resolve a conflict from a ChangeEvent (used by ConflictEscalationHandler)."""
        claim_a_desc = event.payload.get("claim_a", "")
        claim_b_desc = event.payload.get("claim_b", "")
        description = event.payload.get("description", "")

        resolution = Resolution(
            id=uuid4(),
            conflict_id=event.id,
            winner_id=None,
            loser_id=None,
            strategy_used="manual:pending",
            confidence=0.0,
            resolved_by="manual:pending",
            resolved_at=datetime.now(timezone.utc),
            rationale=f"Conflict from event: {description}",
            audit_trail=[
                {
                    "step": "event_received",
                    "claim_a": claim_a_desc,
                    "claim_b": claim_b_desc,
                    "description": description,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        if self._manual_handler is not None:
            try:
                manual_result = await self._manual_handler.escalate(event)
                resolution.audit_trail.append(
                    {
                        "step": "manual_escalation",
                        "handler": self._manual_handler.name,
                        "result": manual_result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception:
                logger.exception("Manual handler failed")

        self._audit.record(resolution)
        return resolution

    async def resolve_batch(self, conflicts: list[ConflictPair]) -> list[Resolution]:
        """Resolve multiple conflicts."""
        resolutions: list[Resolution] = []
        for conflict in conflicts:
            resolution = await self.resolve(conflict)
            resolutions.append(resolution)
        return resolutions

    async def _auto_resolve(self, conflict: ConflictPair) -> Resolution:
        """Auto-resolve using the configured strategy."""
        winner, loser, rationale = resolve_by_strategy(conflict, self._strategy)

        self._ledger.add_conflict(str(winner.id), str(loser.id))
        self._ledger.supersede_claim(str(loser.id), str(winner.id))

        resolution = Resolution(
            id=uuid4(),
            conflict_id=conflict.id,
            winner_id=winner.id,
            loser_id=loser.id,
            strategy_used=f"auto:{self._strategy.value}",
            confidence=conflict.score_delta,
            resolved_by=f"auto:{self._strategy.value}",
            resolved_at=datetime.now(timezone.utc),
            rationale=rationale,
            audit_trail=[
                {
                    "step": "auto_resolve",
                    "strategy": self._strategy.value,
                    "winner_score": winner.truth_score,
                    "loser_score": loser.truth_score,
                    "score_delta": conflict.score_delta,
                    "threshold": self._auto_threshold,
                    "rationale": rationale,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        self._audit.record(resolution)

        if self._changelog:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.conflict_detected,
                    entity_id=winner.subject,
                    payload={
                        "resolution": "auto",
                        "strategy": self._strategy.value,
                        "winner": str(winner.id),
                        "loser": str(loser.id),
                        "rationale": rationale,
                    },
                    agent="conflict_resolver",
                )
            )

        logger.info(
            "Auto-resolved conflict %s: '%s' wins over '%s' (%s)",
            conflict.id,
            winner.object,
            loser.object,
            self._strategy.value,
        )
        return resolution

    async def _manual_resolve(self, conflict: ConflictPair) -> Resolution:
        """Dispatch for manual resolution."""
        resolution = Resolution(
            id=uuid4(),
            conflict_id=conflict.id,
            winner_id=None,
            loser_id=None,
            strategy_used="manual:pending",
            confidence=conflict.score_delta,
            resolved_by="manual:pending",
            resolved_at=datetime.now(timezone.utc),
            rationale=(
                f"Score delta {conflict.score_delta:.3f} below auto-threshold "
                f"{self._auto_threshold}. Requires manual review."
            ),
            audit_trail=[
                {
                    "step": "manual_dispatch",
                    "score_delta": conflict.score_delta,
                    "threshold": self._auto_threshold,
                    "claim_a_score": conflict.claim_a.truth_score,
                    "claim_b_score": conflict.claim_b.truth_score,
                    "description": conflict.description,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        if self._manual_handler is not None:
            try:
                result = await self._manual_handler.escalate_conflict(conflict)
                resolution.audit_trail.append(
                    {
                        "step": "handler_escalation",
                        "handler": self._manual_handler.name,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                resolution.resolved_by = f"manual:{self._manual_handler.name}"
            except Exception:
                logger.exception("Manual handler failed for conflict %s", conflict.id)

        self._audit.record(resolution)

        if self._changelog:
            self._changelog.emit(
                ChangeEvent(
                    change_type=ChangeType.conflict_detected,
                    entity_id=conflict.claim_a.subject,
                    payload={
                        "resolution": "manual_pending",
                        "description": conflict.description,
                        "claim_a": str(conflict.claim_a.id),
                        "claim_b": str(conflict.claim_b.id),
                    },
                    agent="conflict_resolver",
                )
            )

        logger.info(
            "Conflict %s dispatched for manual resolution: %s",
            conflict.id,
            conflict.description,
        )
        return resolution
