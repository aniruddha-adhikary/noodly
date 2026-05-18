"""Change log — append-only event log for all knowledge mutations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of changes in the knowledge system."""

    # Content events
    document_added = "document_added"
    document_modified = "document_modified"
    document_removed = "document_removed"

    # Claim events
    claim_added = "claim_added"
    claim_merged = "claim_merged"
    claim_modified = "claim_modified"
    claim_retracted = "claim_retracted"
    claim_superseded = "claim_superseded"
    claim_promoted = "claim_promoted"

    # Graph events
    entity_merged = "entity_merged"
    conflict_detected = "conflict_detected"
    relationship_discovered = "relationship_discovered"
    gap_detected = "gap_detected"


class ChangeEvent:
    """A single change in the knowledge system."""

    def __init__(
        self,
        change_type: ChangeType,
        entity_id: str = "",
        source_uri: str = "",
        payload: dict | None = None,
        triggered_by: UUID | None = None,
        agent: str = "",
        event_id: UUID | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.id = event_id or uuid4()
        self.change_type = change_type
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.entity_id = entity_id
        self.source_uri = source_uri
        self.payload = payload or {}
        self.triggered_by = triggered_by
        self.agent = agent

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "entity_id": self.entity_id,
            "source_uri": self.source_uri,
            "payload": self.payload,
            "triggered_by": str(self.triggered_by) if self.triggered_by else None,
            "agent": self.agent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChangeEvent:
        triggered_by = None
        if data.get("triggered_by"):
            triggered_by = UUID(data["triggered_by"])
        return cls(
            change_type=ChangeType(data["change_type"]),
            entity_id=data.get("entity_id", ""),
            source_uri=data.get("source_uri", ""),
            payload=data.get("payload", {}),
            triggered_by=triggered_by,
            agent=data.get("agent", ""),
            event_id=UUID(data["id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def __repr__(self) -> str:
        return f"ChangeEvent({self.change_type.value}, {self.entity_id or self.source_uri})"


class ChangeLog:
    """Append-only log of all changes. JSON-backed (v1).

    Usage::

        log = ChangeLog(Path("brain/changelog.json"))
        event = log.emit(ChangeEvent(ChangeType.claim_added, ...))
        recent = log.since(some_datetime)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._events: list[ChangeEvent] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._events = [ChangeEvent.from_dict(item) for item in data]
            except (json.JSONDecodeError, Exception):
                logger.warning("Could not load changelog from %s, starting fresh", self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [event.to_dict() for event in self._events]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def emit(self, event: ChangeEvent) -> ChangeEvent:
        """Record an event."""
        self._events.append(event)
        self._save()
        logger.debug("ChangeLog: %s", event)
        return event

    def since(
        self,
        timestamp: datetime,
        types: list[ChangeType] | None = None,
    ) -> list[ChangeEvent]:
        """Get all events since a given time, optionally filtered by type."""
        events = [e for e in self._events if e.timestamp > timestamp]
        if types:
            events = [e for e in events if e.change_type in types]
        return events

    def for_source(self, source_uri: str) -> list[ChangeEvent]:
        """Get all events related to a specific source document."""
        return [e for e in self._events if e.source_uri == source_uri]

    def chain(self, event_id: UUID) -> list[ChangeEvent]:
        """Follow the causal chain from an event (what it triggered)."""
        children = [e for e in self._events if e.triggered_by == event_id]
        result: list[ChangeEvent] = []
        for child in children:
            result.append(child)
            result.extend(self.chain(child.id))
        return result

    def recent(self, limit: int = 50) -> list[ChangeEvent]:
        """Get the most recent events."""
        return list(reversed(self._events[-limit:]))

    @property
    def count(self) -> int:
        return len(self._events)
