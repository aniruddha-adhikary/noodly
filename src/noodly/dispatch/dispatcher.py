"""Event dispatcher — routes ChangeEvents to registered handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from noodly.tracking.changelog import ChangeEvent, ChangeType

logger = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """Result of a handler processing an event."""

    handler_name: str
    event_id: UUID
    success: bool
    action_taken: str  # "logged", "mr_created", "resolved", "skipped"
    details: dict = field(default_factory=dict)


class EventHandler:
    """Base class for event handlers.

    Subclass and override ``handle`` and optionally ``accepts``.
    """

    name: str = "base"

    async def handle(self, event: ChangeEvent) -> HandlerResult:
        """Process an event. Override in subclasses."""
        return HandlerResult(
            handler_name=self.name,
            event_id=event.id,
            success=True,
            action_taken="skipped",
        )

    def accepts(self, event: ChangeEvent) -> bool:
        """Whether this handler should receive a given event."""
        return True


class EventDispatcher:
    """Routes ChangeEvents to registered handlers based on event type.

    Usage::

        dispatcher = EventDispatcher()
        dispatcher.register(AuditLogHandler())
        dispatcher.register(ConflictHandler(), event_types=[ChangeType.conflict_detected])
        results = await dispatcher.dispatch(event)
    """

    def __init__(self) -> None:
        self._typed_handlers: dict[ChangeType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def register(
        self,
        handler: EventHandler,
        event_types: list[ChangeType] | None = None,
    ) -> None:
        """Register a handler. If event_types is None, receives all events."""
        if event_types is None:
            self._global_handlers.append(handler)
        else:
            for et in event_types:
                self._typed_handlers.setdefault(et, []).append(handler)
        logger.debug(
            "Registered handler %s for %s",
            handler.name,
            [e.value for e in event_types] if event_types else "ALL",
        )

    def unregister(self, handler_name: str) -> int:
        """Remove all handlers with the given name. Returns count removed."""
        removed = 0
        self._global_handlers = [
            h for h in self._global_handlers if h.name != handler_name
        ]
        for et in list(self._typed_handlers.keys()):
            before = len(self._typed_handlers[et])
            self._typed_handlers[et] = [
                h for h in self._typed_handlers[et] if h.name != handler_name
            ]
            removed += before - len(self._typed_handlers[et])
        return removed

    async def dispatch(self, event: ChangeEvent) -> list[HandlerResult]:
        """Dispatch an event to all matching handlers."""
        results: list[HandlerResult] = []

        handlers = list(self._global_handlers)
        handlers.extend(self._typed_handlers.get(event.change_type, []))

        for handler in handlers:
            if not handler.accepts(event):
                continue
            try:
                result = await handler.handle(event)
                results.append(result)
            except Exception:
                logger.exception("Handler %s failed for event %s", handler.name, event.id)
                results.append(
                    HandlerResult(
                        handler_name=handler.name,
                        event_id=event.id,
                        success=False,
                        action_taken="error",
                    )
                )

        return results

    async def dispatch_batch(self, events: list[ChangeEvent]) -> list[HandlerResult]:
        """Dispatch multiple events."""
        all_results: list[HandlerResult] = []
        for event in events:
            results = await self.dispatch(event)
            all_results.extend(results)
        return all_results

    @property
    def handler_count(self) -> int:
        typed = sum(len(hs) for hs in self._typed_handlers.values())
        return len(self._global_handlers) + typed
