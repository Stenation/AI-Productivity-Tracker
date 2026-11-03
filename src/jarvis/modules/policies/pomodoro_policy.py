"""Deterministic Pomodoro selector and calendar-triggered timer recommendation policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, MutableMapping
from datetime import datetime, timedelta
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.database import VALID_RECOMMENDATION_RESOLUTIONS, Database
from jarvis.core.schema import (
    PlatformEvent,
    Recommendation,
    ensure_aware,
    now_local,
)
from jarvis.core.settings import PomodoroSettings, Settings
from jarvis.modules.base import BaseModule

DEFAULT_POMODORO_MAP: dict[str, str | None] = {
    "deep_work": "50_10",
    "coding": "50_10",
    "reading": "50_10",
    "admin": "25_5",
    "email": "25_5",
    "planning": "25_5",
    "meeting": None,
    "break": None,
    "unknown": None,
}

WALKING_SUITABLE_TASK_TYPES: frozenset[str] = frozenset(
    {"admin", "email", "planning", "break"}
)

DEFAULT_COOLDOWN_MINUTES = 30
DEFAULT_IMMINENT_MINUTES = 10


def select_pomodoro(
    task_type: str | None,
    settings: Settings | PomodoroSettings | Mapping[str, Any] | None = None,
) -> str | None:
    """Return the recommended Pomodoro preset ('50_10', '25_5', or None) for `task_type`."""
    normalized = str(task_type).strip().lower() if task_type is not None else "unknown"
    if not normalized:
        normalized = "unknown"

    if settings is None:
        return DEFAULT_POMODORO_MAP.get(normalized)

    if isinstance(settings, Settings):
        return settings.pomodoro.get(normalized)

    if isinstance(settings, PomodoroSettings):
        return settings.get(normalized)

    if isinstance(settings, Mapping):
        if normalized == "break":
            if "break" in settings:
                val = settings["break"]
                return str(val) if val is not None else None
            if "break_" in settings:
                val = settings["break_"]
                return str(val) if val is not None else None
        elif normalized in settings:
            val = settings[normalized]
            return str(val) if val is not None else None

        nested = settings.get("pomodoro")
        if isinstance(nested, PomodoroSettings):
            return nested.get(normalized)
        if isinstance(nested, Mapping):
            return select_pomodoro(normalized, nested)

    return DEFAULT_POMODORO_MAP.get(normalized)


def is_walking_suitable(task_type: str | None) -> bool:
    """Return True when `task_type` is suitable for walking by default (`SPEC.MD` §3)."""
    if task_type is None:
        return False
    normalized = str(task_type).strip().lower()
    return normalized in WALKING_SUITABLE_TASK_TYPES


def _parse_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value:
        return ensure_aware(datetime.fromisoformat(value))
    return None


class PomodoroPolicyModule(BaseModule):
    """Recommend non-destructive Pomodoro cycles when productive calendar blocks start."""

    name: str = "pomodoro_policy"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any] | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=dict(config) if config else {})
        self._custom_clock = clock is not None
        self._clock: Callable[[], datetime] = clock if clock is not None else now_local

        self._privacy_paused: bool = self._extract_privacy_paused(self.config)
        self._dnd_mode: bool = self._extract_dnd_mode(self.config)

        self._cached_current_event: dict[str, Any] | None = None
        self._cached_next_event: dict[str, Any] | None = None

        self._eval_lock = asyncio.Lock()
        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Subscribe to calendar, resolution, and privacy events."""
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected() and not self._privacy_paused:
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if persisted_pause is True:
                self._privacy_paused = True

        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.subscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self._running = True
        self._last_error = None

    async def stop(self) -> None:
        """Unsubscribe from EventBus topics cleanly."""
        self._running = False
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.unsubscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Refresh configuration and runtime privacy flags."""
        await super().reload_config(dict(config))
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

    async def health(self) -> dict[str, Any]:
        """Return current module health status."""
        status = (
            "degraded"
            if self._last_error is not None
            else ("ok" if self._running else "disabled")
        )
        return {
            "module": self.name,
            "status": status,
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
            "last_error": self._last_error,
        }

    def cooldown_minutes(self) -> float:
        """Return configured recommendation cooldown window in minutes."""
        raw = self._get_pomodoro_value("recommendation_cooldown_minutes")
        return float(raw) if raw is not None else float(DEFAULT_COOLDOWN_MINUTES)

    def imminent_minutes(self) -> float:
        """Return lookahead horizon (in minutes) for imminent calendar blocks."""
        for key in ("imminent_minutes", "imminent_window_minutes"):
            raw = self._get_pomodoro_value(key)
            if raw is not None:
                return float(raw)
        return float(DEFAULT_IMMINENT_MINUTES)

    async def evaluate(self, now: datetime | None = None) -> dict[str, Any] | None:
        """Evaluate current or imminent calendar block and emit a non-destructive recommendation."""
        async with self._eval_lock:
            if self._privacy_paused or not self._db_connected():
                return None

            ref_now = ensure_aware(now) if now is not None else self._now()
            current_event, next_event = await self._resolve_calendar_context(ref_now)

            candidates: list[tuple[dict[str, Any], bool]] = []
            if current_event is not None:
                candidates.append((current_event, True))
            if next_event is not None and self._is_imminent(next_event, ref_now):
                candidates.append((next_event, False))

            active_timer = await self.db.get_active_timer()
            active_timer_event_id = (
                str(active_timer["calendar_event_id"])
                if active_timer is not None
                and active_timer.get("calendar_event_id") is not None
                else None
            )

            for target_event, is_active_block in candidates:
                event_id_raw = target_event.get("event_id")
                if not event_id_raw:
                    continue
                event_id = str(event_id_raw)
                if active_timer_event_id == event_id:
                    continue

                task_type = str(target_event.get("task_type", "unknown")).strip().lower()
                kind = select_pomodoro(task_type, self.config)
                if kind is None:
                    continue

                dedupe_key = f"pomodoro:event:{event_id}"
                if await self._is_suppressed(dedupe_key, now=ref_now):
                    continue

                explicit_walking = target_event.get("suitable_for_walking")
                walking_suitable = (
                    bool(explicit_walking)
                    if explicit_walking is not None
                    else is_walking_suitable(task_type)
                )

                message = self._build_message(
                    target_event=target_event,
                    kind=kind,
                    walking_suitable=walking_suitable,
                    is_active_block=is_active_block,
                    now=ref_now,
                )
                action_payload: dict[str, Any] = {
                    "kind": kind,
                    "walking_suitable": walking_suitable,
                    "calendar_event_id": event_id,
                    "task_type": task_type,
                }
                # Milestone 1: never auto-start timers, even if auto_start is set in config.
                return await self._create_and_publish(
                    message=message,
                    suggested_action="start_timer",
                    action_payload=action_payload,
                    dedupe_key=dedupe_key,
                    now=ref_now,
                )

            return None

    async def _is_suppressed(self, dedupe_key: str, *, now: datetime) -> bool:
        pending_rows = await self.db.get_pending_recommendations()
        if any(row.get("dedupe_key") == dedupe_key for row in pending_rows):
            return True

        latest = await self.db.find_recent_recommendation(dedupe_key)
        if latest is None:
            return False
        if latest.get("status") == "pending":
            return True

        effective_cooldown = self.cooldown_minutes()
        if latest.get("status") == "snoozed":
            metadata = latest.get("resolution_metadata")
            if isinstance(metadata, dict) and metadata.get("snooze_minutes") is not None:
                effective_cooldown = float(metadata["snooze_minutes"])

        if effective_cooldown <= 0:
            return False

        since = now - timedelta(minutes=effective_cooldown) + timedelta(microseconds=1)
        recent = await self.db.find_recent_recommendation(dedupe_key, since=since)
        return recent is not None

    async def _create_and_publish(
        self,
        *,
        message: str,
        suggested_action: str,
        action_payload: dict[str, Any],
        dedupe_key: str,
        now: datetime,
    ) -> dict[str, Any]:
        rec_row = await self.db.create_recommendation(
            policy=self.name,
            message=message,
            suggested_action=suggested_action,
            action_payload=action_payload,
            dedupe_key=dedupe_key,
            created_at=now,
        )
        serialized = Recommendation.model_validate(rec_row).model_dump(mode="json")

        seen_keys = getattr(self.bus, "_seen_dedupe_keys", None)
        if isinstance(seen_keys, MutableMapping):
            seen_keys.pop(dedupe_key, None)

        await self.bus.publish(
            PlatformEvent(
                type="recommendation_created",
                source=self.name,
                confidence=1.0,
                timestamp=now,
                dedupe_key=dedupe_key,
                payload={
                    "recommendation_id": serialized["id"],
                    "id": serialized["id"],
                    "policy": serialized["policy"],
                    "message": serialized["message"],
                    "suggested_action": serialized["suggested_action"],
                    "action_payload": serialized["action_payload"],
                    "dedupe_key": serialized["dedupe_key"],
                    "status": serialized["status"],
                    "created_at": serialized["created_at"],
                },
            )
        )
        self._last_event_at = now
        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "ok",
                last_event_at=now,
            )
        return rec_row

    async def _on_calendar_context_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        self._update_cached_calendar_from_payload(event.payload, event_time)
        await self.evaluate(now=event_time)

    async def _on_recommendation_resolved(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time

        if not self._db_connected():
            return

        payload = event.payload
        rec_id = payload.get("recommendation_id") or payload.get("id")
        status = payload.get("status")
        if (
            rec_id is not None
            and isinstance(status, str)
            and status in VALID_RECOMMENDATION_RESOLUTIONS
        ):
            existing = await self.db.get_recommendation(int(rec_id))
            if existing is not None and existing.get("status") == "pending":
                metadata = payload.get("resolution_metadata")
                resolved_at = _parse_dt(payload.get("resolved_at")) or event_time
                await self.db.resolve_recommendation(
                    int(rec_id),
                    status,
                    metadata=metadata if isinstance(metadata, dict) else None,
                    resolved_at=resolved_at,
                )

    def _on_privacy_mode_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        for key in ("global_pause", "privacy_paused", "paused"):
            if key in payload and payload[key] is not None:
                self._privacy_paused = bool(payload[key])
                self._sync_privacy_in_config("global_pause", self._privacy_paused)
                break

        if "dnd_mode" in payload and payload["dnd_mode"] is not None:
            self._dnd_mode = bool(payload["dnd_mode"])
            self._sync_privacy_in_config("dnd_mode", self._dnd_mode)

    async def _resolve_calendar_context(
        self,
        now: datetime,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        db_cur: dict[str, Any] | None = None
        db_next: dict[str, Any] | None = None
        if self._db_connected():
            db_cur, db_next = await self.db.get_current_and_next_events(now=now)

        cur_evt = db_cur or self._validate_cached_current(self._cached_current_event, now)
        next_evt = db_next or self._validate_cached_next(self._cached_next_event, now)
        return cur_evt, next_evt

    def _update_cached_calendar_from_payload(
        self,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        if "current_event" in payload:
            cur = payload.get("current_event")
            self._cached_current_event = cur if isinstance(cur, dict) else None
        if "next_event" in payload:
            nxt = payload.get("next_event")
            self._cached_next_event = nxt if isinstance(nxt, dict) else None

        single = payload.get("event")
        candidate: dict[str, Any] | None = None
        if isinstance(single, dict):
            candidate = single
        elif "event_id" in payload:
            candidate = dict(payload)

        if candidate is not None:
            start_dt = _parse_dt(candidate.get("start_time"))
            if start_dt is not None and start_dt > now:
                self._cached_next_event = candidate
            else:
                self._cached_current_event = candidate

    @staticmethod
    def _validate_cached_current(
        candidate: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        if candidate is None:
            return None
        start_dt = _parse_dt(candidate.get("start_time"))
        end_dt = _parse_dt(candidate.get("end_time"))
        if start_dt is not None and end_dt is not None:
            return candidate if start_dt <= now < end_dt else None
        return candidate

    @staticmethod
    def _validate_cached_next(
        candidate: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        if candidate is None:
            return None
        start_dt = _parse_dt(candidate.get("start_time"))
        if start_dt is not None:
            return candidate if start_dt > now else None
        return candidate

    def _is_imminent(self, event_row: dict[str, Any], now: datetime) -> bool:
        start_dt = _parse_dt(event_row.get("start_time"))
        if start_dt is None:
            return True
        delta = start_dt - now
        return timedelta(0) < delta <= timedelta(minutes=self.imminent_minutes())

    @staticmethod
    def _build_message(
        *,
        target_event: dict[str, Any],
        kind: str,
        walking_suitable: bool,
        is_active_block: bool,
        now: datetime,
    ) -> str:
        title = str(target_event.get("title") or "Focus block")
        display_kind = kind.replace("_", "/")
        walk_note = " (walking-compatible)" if walking_suitable else ""

        if is_active_block:
            return f"'{title}' is active{walk_note}. Suggested Pomodoro timer: {display_kind}."

        start_dt = _parse_dt(target_event.get("start_time"))
        if start_dt is not None:
            mins = max(1, int(round((start_dt - now).total_seconds() / 60)))
            return (
                f"'{title}' starts in {mins} minutes{walk_note}. "
                f"Suggested Pomodoro timer: {display_kind}."
            )
        return f"'{title}' is starting soon{walk_note}. Suggested Pomodoro timer: {display_kind}."

    def _now(self) -> datetime:
        return ensure_aware(self._clock())

    def _event_now(self, event: PlatformEvent) -> datetime:
        if self._custom_clock:
            return self._now()
        return ensure_aware(event.timestamp)

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    def _get_pomodoro_value(self, key: str) -> Any:
        direct = self.config.get(key)
        if direct is not None:
            return direct
        nested = self.config.get("pomodoro")
        if isinstance(nested, dict):
            return nested.get(key)
        return None

    @staticmethod
    def _extract_privacy_paused(config: dict[str, Any]) -> bool:
        for key in ("global_pause", "privacy_paused"):
            if key in config and config[key] is not None:
                return bool(config[key])
        privacy_cfg = config.get("privacy")
        if isinstance(privacy_cfg, dict) and privacy_cfg.get("global_pause") is not None:
            return bool(privacy_cfg["global_pause"])
        return False

    @staticmethod
    def _extract_dnd_mode(config: dict[str, Any]) -> bool:
        if "dnd_mode" in config and config["dnd_mode"] is not None:
            return bool(config["dnd_mode"])
        privacy_cfg = config.get("privacy")
        if isinstance(privacy_cfg, dict) and privacy_cfg.get("dnd_mode") is not None:
            return bool(privacy_cfg["dnd_mode"])
        return False

    def _sync_privacy_in_config(self, key: str, value: bool) -> None:
        privacy_cfg = self.config.get("privacy")
        if isinstance(privacy_cfg, dict):
            privacy_cfg[key] = value
        if key in self.config:
            self.config[key] = value
