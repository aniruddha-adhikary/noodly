"""Built-in event handlers — audit log, conflict escalation, file sync."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from noodly.dispatch.dispatcher import EventHandler, HandlerResult
from noodly.tracking.changelog import ChangeEvent, ChangeType

logger = logging.getLogger(__name__)


class AuditLogHandler(EventHandler):
    """Writes all events to an immutable audit log (JSON file or PostgreSQL).

    Every event is recorded with full payload and timestamp.
    Records are never deleted (append-only).
    """

    name = "audit_log"

    def __init__(self, audit_path: Path | None = None, pg_backend=None) -> None:
        self._path = audit_path
        self._pg = pg_backend

    async def handle(self, event: ChangeEvent) -> HandlerResult:
        record = {
            "event_id": str(event.id),
            "change_type": event.change_type.value,
            "timestamp": event.timestamp.isoformat(),
            "entity_id": event.entity_id,
            "source_uri": event.source_uri,
            "payload": event.payload,
            "triggered_by": str(event.triggered_by) if event.triggered_by else None,
            "agent": event.agent,
            "audit_recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._pg is not None:
            try:
                await self._pg.record_audit(
                    event_type=event.change_type.value,
                    entity_id=event.entity_id,
                    source_uri=event.source_uri,
                    payload=event.payload,
                    agent=event.agent,
                )
            except Exception:
                logger.exception("Failed to write audit to PostgreSQL")

        if self._path is not None:
            self._append_to_file(record)

        return HandlerResult(
            handler_name=self.name,
            event_id=event.id,
            success=True,
            action_taken="logged",
            details=record,
        )

    def _append_to_file(self, record: dict) -> None:
        """Append a record to the JSON-lines audit file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


class ConflictEscalationHandler(EventHandler):
    """Routes conflict_detected events to the conflict resolution system.

    When a conflict is detected (by the graph agent or during ingestion),
    this handler triggers the resolution workflow.
    """

    name = "conflict_escalation"

    def __init__(self, resolver=None) -> None:
        self._resolver = resolver

    def accepts(self, event: ChangeEvent) -> bool:
        return event.change_type == ChangeType.conflict_detected

    async def handle(self, event: ChangeEvent) -> HandlerResult:
        if self._resolver is None:
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=True,
                action_taken="skipped",
                details={"reason": "no resolver configured"},
            )

        try:
            resolution = await self._resolver.resolve_from_event(event)
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=True,
                action_taken="resolved" if resolution.winner else "escalated",
                details={
                    "resolution_id": str(resolution.id),
                    "strategy": resolution.strategy_used,
                    "resolved_by": resolution.resolved_by,
                },
            )
        except Exception:
            logger.exception("Conflict resolution failed for event %s", event.id)
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=False,
                action_taken="error",
            )
