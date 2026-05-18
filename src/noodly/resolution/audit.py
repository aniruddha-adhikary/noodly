"""Resolution audit trail — immutable record of all conflict resolutions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class Resolution:
    """Record of a conflict resolution decision."""

    id: UUID
    conflict_id: UUID
    winner_id: UUID | None
    loser_id: UUID | None
    strategy_used: str
    confidence: float
    resolved_by: str  # "auto:authority_wins", "manual:user", "manual:gitlab_mr"
    resolved_at: datetime
    rationale: str = ""
    audit_trail: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "conflict_id": str(self.conflict_id),
            "winner_id": str(self.winner_id) if self.winner_id else None,
            "loser_id": str(self.loser_id) if self.loser_id else None,
            "strategy_used": self.strategy_used,
            "confidence": self.confidence,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat(),
            "rationale": self.rationale,
            "audit_trail": self.audit_trail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Resolution:
        return cls(
            id=UUID(data["id"]),
            conflict_id=UUID(data["conflict_id"]),
            winner_id=UUID(data["winner_id"]) if data.get("winner_id") else None,
            loser_id=UUID(data["loser_id"]) if data.get("loser_id") else None,
            strategy_used=data["strategy_used"],
            confidence=data["confidence"],
            resolved_by=data["resolved_by"],
            resolved_at=datetime.fromisoformat(data["resolved_at"]),
            rationale=data.get("rationale", ""),
            audit_trail=data.get("audit_trail", []),
        )


class ResolutionAudit:
    """Immutable audit trail for all conflict resolutions. JSON-backed.

    Records are never deleted — this is a permanent audit log.

    Usage::

        audit = ResolutionAudit(Path("brain/resolutions.json"))
        audit.record(resolution)
        recent = audit.list_resolutions(limit=10)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._resolutions: list[Resolution] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._resolutions = [Resolution.from_dict(item) for item in data]
            except (json.JSONDecodeError, Exception):
                logger.warning("Could not load resolutions from %s", self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._resolutions]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def record(self, resolution: Resolution) -> Resolution:
        """Record a resolution (append-only)."""
        self._resolutions.append(resolution)
        self._save()
        logger.info(
            "Resolution recorded: %s → winner=%s (strategy=%s)",
            resolution.conflict_id,
            resolution.winner_id,
            resolution.strategy_used,
        )
        return resolution

    def get_resolution(self, conflict_id: str) -> Resolution | None:
        """Get the resolution for a specific conflict."""
        for r in reversed(self._resolutions):
            if str(r.conflict_id) == conflict_id:
                return r
        return None

    def list_resolutions(
        self,
        strategy: str | None = None,
        resolved_by: str | None = None,
        limit: int = 50,
    ) -> list[Resolution]:
        """List resolutions, optionally filtered."""
        results = list(self._resolutions)
        if strategy:
            results = [r for r in results if r.strategy_used == strategy]
        if resolved_by:
            results = [r for r in results if resolved_by in r.resolved_by]
        results.sort(key=lambda r: r.resolved_at, reverse=True)
        return results[:limit]

    @property
    def count(self) -> int:
        return len(self._resolutions)

    def pending_count(self) -> int:
        """Count resolutions where no winner was picked (pending manual)."""
        return sum(1 for r in self._resolutions if r.winner_id is None)
