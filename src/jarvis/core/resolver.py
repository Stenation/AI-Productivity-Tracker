"""StateResolver converting manual and sensor reports into effective activity spans."""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import (
    ActivityState,
    EffectiveActivityState,
    PlatformEvent,
    ensure_aware,
    now_local,
)
from jarvis.core.settings import Settings

logger = logging.getLogger(__name__)

VALID_ACTIVITIES: frozenset[str] = frozenset(
    {"sitting", "standing", "walking", "away", "unknown"}
)
VALID_MANUAL_ACTIVITIES: frozenset[str] = frozenset(
    {"sitting", "standing", "walking", "away"}
)
MANUAL_SOURCES: frozenset[str] = frozenset(
    {"manual", "manual_control", "waybar", "dashboard", "cli", "api"}
)


def _sensor_priority(source: str) -> int:
    lowered = source.lower()
    if "swalk" in lowered or "ble" in lowered:
        return 30
    if "s24" in lowered or "phone" in lowered:
        return 20
    if "webcam" in lowered:
        return 10
    return 15


@dataclass(frozen=True, slots=True)
class _SensorSample:
    activity: ActivityState
    source: str
    confidence: float
    timestamp: datetime


class StateResolver:
    """Resolve raw activity reports into persisted effective activity states."""

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        settings: Settings,
        *,
        min_sensor_confidence: float = 0.5,
        sensor_priority_ttl_seconds: float = 15.0,
    ) -> None:
        self._bus = bus
        self._db = db
        self._settings = settings
        self._min_sensor_confidence = min_sensor_confidence
        self._sensor_priority_ttl_seconds = sensor_priority_ttl_seconds

        self._activity: ActivityState = "unknown"
        self._source: str = "system"
        self._confidence: float = 1.0
        self._reason: str = "startup"
        self._started_at: datetime = now_local()
        self._override_expires_at: datetime | None = None
        self._has_open_span: bool = False

        self._privacy_paused: bool = bool(settings.privacy.global_pause)
        self._dnd_mode: bool = bool(settings.privacy.dnd_mode)

        window_size = max(
            1,
            int(settings.posture.sensor_consensus_window_samples),
            int(settings.posture.sensor_consensus_samples),
        )
        self._sensor_window: deque[_SensorSample] = deque(maxlen=window_size)
        self._last_sensor_by_source: dict[str, _SensorSample] = {}
        self._started: bool = False

        self._subscribe()

    def _subscribe(self) -> None:
        self._bus.subscribe("activity_reported", self._on_activity_reported)
        self._bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)

    async def start(self) -> EffectiveActivityState:
        """Subscribe to bus events and recover any open activity span from SQLite."""
        self._subscribe()
        self._privacy_paused = bool(self._settings.privacy.global_pause)
        self._dnd_mode = bool(self._settings.privacy.dnd_mode)

        now = now_local()
        current_span = await self._db.get_current_activity_span()
        if current_span is not None:
            raw_activity = str(current_span.get("activity", "unknown"))
            activity: ActivityState = (
                cast(ActivityState, raw_activity)
                if raw_activity in VALID_ACTIVITIES
                else "unknown"
            )
            started_at = ensure_aware(
                datetime.fromisoformat(str(current_span["started_at"]))
            )
            reason = str(current_span.get("reason", "recovered"))
            metadata = current_span.get("metadata") or {}

            expires_at: datetime | None = None
            if isinstance(metadata, dict):
                raw_expires = metadata.get("override_expires_at")
                raw_ttl = metadata.get("ttl_minutes")
                if isinstance(raw_expires, str) and raw_expires:
                    candidate = ensure_aware(datetime.fromisoformat(raw_expires))
                    if candidate > now:
                        expires_at = candidate
                elif reason == "manual_override" and raw_ttl is not None:
                    candidate = started_at + timedelta(minutes=float(raw_ttl))
                    if candidate > now:
                        expires_at = candidate

            self._activity = activity
            self._source = str(current_span.get("source", "system"))
            self._confidence = float(current_span.get("confidence", 1.0))
            self._reason = reason
            self._started_at = started_at
            self._override_expires_at = expires_at
            self._has_open_span = True
        else:
            self._activity = "unknown"
            self._source = "system"
            self._confidence = 1.0
            self._reason = "startup"
            self._started_at = now
            self._override_expires_at = None
            self._has_open_span = False

        self._started = True
        return self.get_effective_state(now=now)

    async def stop(self) -> None:
        """Unsubscribe from EventBus topics cleanly."""
        self._bus.unsubscribe("activity_reported", self._on_activity_reported)
        self._bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self._started = False

    def get_effective_state(self, now: datetime | None = None) -> EffectiveActivityState:
        """Return the current effective activity, elapsed duration, and override expiry."""
        ref_now = ensure_aware(now) if now is not None else now_local()

        effective_override = self._override_expires_at
        if effective_override is not None and ref_now >= effective_override:
            effective_override = None
            if now is None:
                self._override_expires_at = None

        elapsed = max(0, int((ref_now - self._started_at).total_seconds()))
        return EffectiveActivityState(
            activity=self._activity,
            started_at=self._started_at,
            elapsed_seconds=elapsed,
            source=self._source,
            confidence=self._confidence,
            reason=self._reason,
            override_expires_at=effective_override,
            privacy_paused=self._privacy_paused,
        )

    async def _on_privacy_mode_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        for key in ("global_pause", "privacy_paused", "paused"):
            if key in payload and payload[key] is not None:
                self._privacy_paused = bool(payload[key])
                self._settings.privacy.global_pause = self._privacy_paused
                break

        if "dnd_mode" in payload and payload["dnd_mode"] is not None:
            self._dnd_mode = bool(payload["dnd_mode"])
            self._settings.privacy.dnd_mode = self._dnd_mode

        if self._privacy_paused:
            self._sensor_window.clear()
            self._last_sensor_by_source.clear()

    async def _on_activity_reported(self, event: PlatformEvent) -> None:
        # Priority 1: Global privacy pause suppresses incoming activity tracking.
        if self._privacy_paused:
            logger.debug(
                "Ignoring activity_reported from %s while privacy pause is active",
                event.source,
            )
            return

        raw_activity = event.payload.get("activity")
        if not isinstance(raw_activity, str) or raw_activity not in VALID_ACTIVITIES:
            logger.warning(
                "Ignoring invalid activity %r from source %s",
                raw_activity,
                event.source,
            )
            return

        activity = cast(ActivityState, raw_activity)
        event_time = ensure_aware(event.timestamp)

        is_manual = (
            event.source in MANUAL_SOURCES
            or bool(event.payload.get("manual"))
            or event.payload.get("reason") == "manual_override"
        )

        if is_manual:
            await self._apply_manual_override(
                activity=activity,
                source=event.source,
                payload=event.payload,
                event_time=event_time,
            )
            return

        # Priority 2: Active manual override takes precedence until its TTL expires.
        if self._override_expires_at is not None:
            if event_time < self._override_expires_at:
                logger.debug(
                    "Ignoring sensor report (%s=%s) during active manual override until %s",
                    event.source,
                    activity,
                    self._override_expires_at.isoformat(),
                )
                return
            self._override_expires_at = None

        is_discrete = (
            bool(event.payload.get("discrete"))
            or event.source in ("hyprland_idle", "idle_monitor")
            or event.payload.get("reason") in ("idle_timeout", "input_resumed")
        )

        incoming_reason = str(
            event.payload.get("reason")
            or ("idle_timeout" if activity == "away" else "sensor_consensus")
        )

        await self._evaluate_sensor_report(
            activity=activity,
            source=event.source,
            confidence=event.confidence,
            event_time=event_time,
            discrete=is_discrete,
            reason=incoming_reason,
        )

    async def _apply_manual_override(
        self,
        *,
        activity: ActivityState,
        source: str,
        payload: dict[str, Any],
        event_time: datetime,
    ) -> None:
        if activity not in VALID_MANUAL_ACTIVITIES:
            logger.warning(
                "Ignoring unsupported manual override state %r from %s",
                activity,
                source,
            )
            return

        raw_ttl = payload.get("ttl_minutes")
        ttl_minutes = (
            float(raw_ttl)
            if raw_ttl is not None
            else float(self._settings.posture.manual_override_ttl_minutes)
        )
        expires_at = (
            event_time + timedelta(minutes=ttl_minutes) if ttl_minutes > 0 else None
        )

        self._sensor_window.clear()
        self._last_sensor_by_source.clear()

        if activity == self._activity and self._has_open_span:
            self._source = source
            self._confidence = 1.0
            self._reason = "manual_override"
            self._override_expires_at = expires_at
            return

        metadata: dict[str, Any] = {
            "ttl_minutes": ttl_minutes,
            "override_expires_at": expires_at.isoformat() if expires_at is not None else None,
        }
        await self._transition_to(
            activity=activity,
            source=source,
            confidence=1.0,
            reason="manual_override",
            started_at=event_time,
            override_expires_at=expires_at,
            metadata=metadata,
        )

    async def _evaluate_sensor_report(
        self,
        *,
        activity: ActivityState,
        source: str,
        confidence: float,
        event_time: datetime,
        discrete: bool = False,
        reason: str = "sensor_consensus",
    ) -> None:
        incoming_priority = _sensor_priority(source)

        # Priority 4-6: Do not let a lower-priority sensor override fresh higher-priority evidence.
        if confidence >= self._min_sensor_confidence and activity != "unknown":
            for other_source, other_sample in list(self._last_sensor_by_source.items()):
                if _sensor_priority(other_source) <= incoming_priority:
                    continue
                age_seconds = (event_time - other_sample.timestamp).total_seconds()
                if (
                    0.0 <= age_seconds < self._sensor_priority_ttl_seconds
                    and other_sample.activity != activity
                ):
                    logger.debug(
                        "Ignoring lower-priority sensor %s (%s) due to fresh %s (%s)",
                        source,
                        activity,
                        other_source,
                        other_sample.activity,
                    )
                    return

            self._last_sensor_by_source[source] = _SensorSample(
                activity=activity,
                source=source,
                confidence=confidence,
                timestamp=event_time,
            )

        required_samples = max(1, int(self._settings.posture.sensor_consensus_samples))
        configured_window = max(
            required_samples,
            int(self._settings.posture.sensor_consensus_window_samples),
        )
        if self._sensor_window.maxlen != configured_window:
            self._sensor_window = deque(self._sensor_window, maxlen=configured_window)

        sample = _SensorSample(
            activity=activity,
            source=source,
            confidence=confidence,
            timestamp=event_time,
        )
        if discrete:
            self._sensor_window.clear()
            self._sensor_window.extend([sample] * required_samples)
        else:
            self._sensor_window.append(sample)

        strong_samples = [
            s
            for s in self._sensor_window
            if s.confidence >= self._min_sensor_confidence and s.activity != "unknown"
        ]
        if len(strong_samples) < required_samples:
            return

        counts = Counter(s.activity for s in strong_samples)
        winning_activity, agreeing_count = counts.most_common(1)[0]
        if agreeing_count < required_samples:
            return

        if winning_activity == self._activity and self._has_open_span:
            agreeing = [s for s in strong_samples if s.activity == winning_activity]
            self._confidence = round(
                sum(s.confidence for s in agreeing) / len(agreeing), 4
            )
            self._source = agreeing[-1].source
            return

        agreeing = [s for s in strong_samples if s.activity == winning_activity]
        avg_confidence = round(
            min(1.0, max(0.0, sum(s.confidence for s in agreeing) / len(agreeing))),
            4,
        )
        winning_source = agreeing[-1].source

        await self._transition_to(
            activity=winning_activity,
            source=winning_source,
            confidence=avg_confidence,
            reason=reason,
            started_at=event_time,
            override_expires_at=None,
            metadata={
                "samples_agreeing": agreeing_count,
                "window_samples": len(self._sensor_window),
            },
        )

    async def _transition_to(
        self,
        *,
        activity: ActivityState,
        source: str,
        confidence: float,
        reason: str,
        started_at: datetime,
        override_expires_at: datetime | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        previous_activity = self._activity
        span = await self._db.open_activity_span(
            activity=activity,
            source=source,
            confidence=confidence,
            reason=reason,
            started_at=started_at,
            metadata=metadata,
        )

        self._activity = activity
        self._source = source
        self._confidence = confidence
        self._reason = reason
        self._started_at = started_at
        self._override_expires_at = override_expires_at
        self._has_open_span = True

        await self._bus.publish(
            PlatformEvent(
                type="activity_state_changed",
                source="resolver",
                confidence=confidence,
                timestamp=started_at,
                payload={
                    "previous_activity": previous_activity,
                    "activity": activity,
                    "activity_source": source,
                    "reason": reason,
                    "started_at": started_at.isoformat(),
                    "override_expires_at": (
                        override_expires_at.isoformat()
                        if override_expires_at is not None
                        else None
                    ),
                    "span_id": span["id"],
                },
            )
        )
