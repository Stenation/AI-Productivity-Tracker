"""Manual posture/activity override input module."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import (
    MANUAL_ACTIVITY_CYCLE,
    ActivityState,
    ManualActivityState,
    PlatformEvent,
    now_local,
)
from jarvis.modules.base import BaseModule

DEFAULT_MANUAL_OVERRIDE_TTL_MINUTES = 30


class ManualControlModule(BaseModule):
    """Input module emitting manual activity overrides and cycling states."""

    name: str = "manual_control"

    def __init__(self, bus: EventBus, db: Database, config: dict[str, Any]) -> None:
        super().__init__(bus=bus, db=db, config=config)
        self._last_effective_activity: ActivityState = "unknown"
        self._last_non_away_activity: ManualActivityState = "sitting"
        self._last_event_at: datetime | None = None
        self._running: bool = False

    async def start(self) -> None:
        """Subscribe to state transitions to keep effective activity context fresh."""
        self.bus.subscribe("activity_state_changed", self._on_activity_state_changed)
        if getattr(self.db, "_conn", None) is not None:
            current_span = await self.db.get_current_activity_span()
            if current_span is not None:
                raw_act = str(current_span.get("activity", "unknown"))
                if raw_act in (*MANUAL_ACTIVITY_CYCLE, "unknown"):
                    self._last_effective_activity = cast(ActivityState, raw_act)
                    if raw_act in ("sitting", "standing", "walking"):
                        self._last_non_away_activity = cast(ManualActivityState, raw_act)
            if self._last_non_away_activity == "sitting":
                recent = await self.db.list_activity_spans(limit=10)
                for s in recent:
                    act = str(s.get("activity", ""))
                    if act in ("sitting", "standing", "walking"):
                        self._last_non_away_activity = cast(ManualActivityState, act)
                        break
        self._running = True

    async def stop(self) -> None:
        """Unsubscribe from EventBus cleanly."""
        self.bus.unsubscribe("activity_state_changed", self._on_activity_state_changed)
        self._running = False

    async def health(self) -> dict[str, Any]:
        """Return current module health payload."""
        return {
            "module": self.name,
            "status": "ok" if self._running else "disabled",
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
        }

    def default_ttl_minutes(self) -> int | float:
        """Return default manual override TTL from module/platform config."""
        direct = self.config.get("manual_override_ttl_minutes")
        if direct is not None:
            return self._normalize_ttl(direct)

        posture_cfg = self.config.get("posture")
        if isinstance(posture_cfg, dict):
            nested = posture_cfg.get("manual_override_ttl_minutes")
            if nested is not None:
                return self._normalize_ttl(nested)

        return DEFAULT_MANUAL_OVERRIDE_TTL_MINUTES

    @staticmethod
    def next_activity_in_cycle(
        current_activity: ActivityState | str | None,
    ) -> ManualActivityState:
        """Return the next manual state in `sitting -> standing -> walking -> away`."""
        if current_activity in MANUAL_ACTIVITY_CYCLE:
            idx = MANUAL_ACTIVITY_CYCLE.index(cast(ActivityState, current_activity))
            nxt = MANUAL_ACTIVITY_CYCLE[(idx + 1) % len(MANUAL_ACTIVITY_CYCLE)]
            return cast(ManualActivityState, nxt)
        return cast(ManualActivityState, MANUAL_ACTIVITY_CYCLE[0])

    async def set_activity(
        self,
        activity: ManualActivityState | str,
        ttl_minutes: int | float | None = None,
    ) -> PlatformEvent:
        """Publish a manual override `activity_reported` event with confidence=1.0."""
        if activity not in MANUAL_ACTIVITY_CYCLE:
            raise ValueError(
                f"Invalid manual activity {activity!r}; expected one of "
                f"{list(MANUAL_ACTIVITY_CYCLE)}"
            )

        resolved_ttl = (
            self._normalize_ttl(ttl_minutes)
            if ttl_minutes is not None
            else self.default_ttl_minutes()
        )
        now = now_local()
        event = PlatformEvent(
            type="activity_reported",
            source=self.name,
            confidence=1.0,
            timestamp=now,
            payload={
                "activity": str(activity),
                "ttl_minutes": resolved_ttl,
                "reason": "manual_override",
            },
        )
        await self.bus.publish(event)
        self._last_event_at = now
        return event

    async def cycle_activity(
        self,
        current_activity: ActivityState | str | None = None,
        ttl_minutes: int | float | None = None,
    ) -> PlatformEvent:
        """Cycle `sitting -> standing -> walking -> away` from current effective state."""
        effective_current = current_activity
        if effective_current is None:
            if getattr(self.db, "_conn", None) is not None:
                current_span = await self.db.get_current_activity_span()
                if current_span is not None:
                    effective_current = str(current_span.get("activity", "unknown"))
            if effective_current is None:
                effective_current = self._last_effective_activity

        next_activity = self.next_activity_in_cycle(effective_current)
        return await self.set_activity(next_activity, ttl_minutes=ttl_minutes)

    async def toggle_away(
        self,
        current_activity: ActivityState | str | None = None,
        ttl_minutes: int | float | None = None,
    ) -> PlatformEvent:
        """Toggle between 'away' and the last non-away state (sitting/standing/walking)."""
        effective_current = current_activity
        if effective_current is None:
            if getattr(self.db, "_conn", None) is not None:
                current_span = await self.db.get_current_activity_span()
                if current_span is not None:
                    effective_current = str(current_span.get("activity", "unknown"))
            if effective_current is None:
                effective_current = self._last_effective_activity

        if effective_current == "away":
            next_activity = self._last_non_away_activity
        else:
            if effective_current in ("sitting", "standing", "walking"):
                self._last_non_away_activity = cast(ManualActivityState, effective_current)
            next_activity = "away"

        return await self.set_activity(next_activity, ttl_minutes=ttl_minutes)

    def _on_activity_state_changed(self, event: PlatformEvent) -> None:
        raw_activity = event.payload.get("activity")
        if isinstance(raw_activity, str) and raw_activity in (
            *MANUAL_ACTIVITY_CYCLE,
            "unknown",
        ):
            self._last_effective_activity = cast(ActivityState, raw_activity)
            if raw_activity in ("sitting", "standing", "walking"):
                self._last_non_away_activity = cast(ManualActivityState, raw_activity)

    @staticmethod
    def _normalize_ttl(value: int | float | str) -> int | float:
        numeric = float(value)
        if numeric <= 0:
            raise ValueError("ttl_minutes must be greater than 0")
        return int(numeric) if numeric.is_integer() else numeric
