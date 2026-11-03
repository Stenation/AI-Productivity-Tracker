"""Async in-process publish/subscribe EventBus with deduplication and error isolation."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal

from jarvis.core.schema import EventType, PlatformEvent

logger = logging.getLogger(__name__)

WILDCARD_TOPIC: Literal["*"] = "*"
EventTopic = EventType | Literal["*"] | str
EventCallback = Callable[[PlatformEvent], Awaitable[Any] | Any]


class EventBus:
    """Async pub/sub dispatcher with bounded dedupe_key cache and error isolation."""

    def __init__(
        self,
        *,
        max_dedupe_keys: int = 1024,
        max_recent_events: int = 200,
    ) -> None:
        if max_dedupe_keys < 1:
            raise ValueError("max_dedupe_keys must be at least 1")
        if max_recent_events < 1:
            raise ValueError("max_recent_events must be at least 1")

        self._max_dedupe_keys = max_dedupe_keys
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(list)
        self._wildcard_subscribers: list[EventCallback] = []
        self._seen_dedupe_keys: OrderedDict[str, datetime] = OrderedDict()
        self._recent_events: deque[PlatformEvent] = deque(maxlen=max_recent_events)

    def subscribe(self, event_type_or_wildcard: EventTopic, callback: EventCallback) -> None:
        """Subscribe a callback to a specific EventType or '*' for all events."""
        if not callable(callback):
            raise TypeError(f"Subscriber callback must be callable, got {callback!r}")

        if event_type_or_wildcard == WILDCARD_TOPIC:
            if callback not in self._wildcard_subscribers:
                self._wildcard_subscribers.append(callback)
            return

        bucket = self._subscribers[event_type_or_wildcard]
        if callback not in bucket:
            bucket.append(callback)

    def unsubscribe(
        self,
        event_type_or_callback: EventTopic | EventCallback,
        callback: EventCallback | None = None,
    ) -> bool:
        """Unsubscribe a callback from one topic, or from all topics if topic is omitted."""
        if callable(event_type_or_callback) and callback is None:
            target = event_type_or_callback
            removed = False
            if target in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(target)
                removed = True
            for bucket in list(self._subscribers.values()):
                if target in bucket:
                    bucket.remove(target)
                    removed = True
            return removed

        if not isinstance(event_type_or_callback, str) or callback is None:
            raise TypeError(
                "unsubscribe expects (event_type_or_wildcard, callback) or (callback,)"
            )

        if event_type_or_callback == WILDCARD_TOPIC:
            if callback in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(callback)
                return True
            return False

        bucket = self._subscribers.get(event_type_or_callback)
        if bucket and callback in bucket:
            bucket.remove(callback)
            return True
        return False

    async def publish(self, event: PlatformEvent) -> bool:
        """Publish a PlatformEvent to matching and wildcard subscribers.

        Returns True when the event is dispatched, or False when suppressed by dedupe_key.
        """
        if event.dedupe_key:
            if event.dedupe_key in self._seen_dedupe_keys:
                logger.debug(
                    "Suppressing duplicate event type=%s source=%s dedupe_key=%s",
                    event.type,
                    event.source,
                    event.dedupe_key,
                )
                return False
            self._seen_dedupe_keys[event.dedupe_key] = event.timestamp
            while len(self._seen_dedupe_keys) > self._max_dedupe_keys:
                self._seen_dedupe_keys.popitem(last=False)

        self._recent_events.append(event)

        targets: list[EventCallback] = []
        for cb in self._subscribers.get(event.type, ()):
            if cb not in targets:
                targets.append(cb)
        for cb in self._wildcard_subscribers:
            if cb not in targets:
                targets.append(cb)

        for callback in targets:
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Subscriber %r raised an unhandled exception for event type=%s source=%s",
                    callback,
                    event.type,
                    event.source,
                )

        return True

    def recent_events(self, limit: int = 50) -> list[PlatformEvent]:
        """Return up to `limit` most recently published non-duplicate events (newest last)."""
        if limit <= 0:
            return []
        events = list(self._recent_events)
        return events[-limit:]
