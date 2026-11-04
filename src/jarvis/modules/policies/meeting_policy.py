"""Deterministic meeting guardrail policy warning walking/away users before meetings."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, MutableMapping
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any, cast

from jarvis.core.bus import EventBus
from jarvis.core.classification import is_call_window
from jarvis.core.database import VALID_RECOMMENDATION_RESOLUTIONS, Database
from jarvis.core.schema import (
    ActivityState,
    PlatformEvent,
    Recommendation,
    ensure_aware,
    now_local,
)
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_PRE_MEETING_WARNING_MINUTES = 10
DEFAULT_COOLDOWN_MINUTES = 30
DEFAULT_CHECK_INTERVAL_SECONDS = 60.0

_VALID_ACTIVITIES: frozenset[str] = frozenset(
    {"sitting", "standing", "walking", "away", "unknown"}
)
_GUARDRAIL_ACTIVITIES: frozenset[str] = frozenset({"walking", "away"})


def _parse_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value:
        return ensure_aware(datetime.fromisoformat(value))
    return None


class MeetingPolicyModule(BaseModule):
    """Warn users who are walking or away when an upcoming meeting is imminent."""

    name: str = "meeting_policy"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any] | None = None,
        *,
        check_interval_seconds: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=dict(config) if config else {})
        if check_interval_seconds is not None and check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")

        self._injected_interval = check_interval_seconds
        self._custom_clock = clock is not None
        self._clock: Callable[[], datetime] = clock if clock is not None else now_local

        self._privacy_paused: bool = self._extract_privacy_paused(self.config)
        self._dnd_mode: bool = self._extract_dnd_mode(self.config)

        self._current_activity: ActivityState | None = None
        self._activity_started_at: datetime | None = None
        self._cached_next_event: dict[str, Any] | None = None

        self._is_idle: bool = False
        self._is_call_window: bool = False

        self._eval_lock = asyncio.Lock()
        self._loop_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Subscribe to events and start periodic evaluation."""
        await self._cancel_loop()

        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected() and not self._privacy_paused:
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if persisted_pause is True:
                self._privacy_paused = True

        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.subscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.subscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.subscribe("window_focus_changed", self._on_window_focus_changed)
        self.bus.subscribe("input_idle_changed", self._on_input_idle_changed)

        self._running = True
        self._last_error = None
        self._loop_task = asyncio.create_task(
            self._periodic_loop(),
            name=f"{self.name}_periodic_loop",
        )

    async def stop(self) -> None:
        """Unsubscribe from EventBus topics and cancel background evaluation task."""
        self._running = False
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.unsubscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.unsubscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.unsubscribe("window_focus_changed", self._on_window_focus_changed)
        self.bus.unsubscribe("input_idle_changed", self._on_input_idle_changed)
        await self._cancel_loop()

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

    def check_interval_seconds(self) -> float:
        """Return periodic check interval in seconds."""
        if self._injected_interval is not None:
            return float(self._injected_interval)
        raw = self._get_posture_value("check_interval_seconds")
        if raw is not None:
            val = float(raw)
            if val > 0:
                return val
        return DEFAULT_CHECK_INTERVAL_SECONDS

    def pre_meeting_warning_minutes(self) -> float:
        """Return configured pre-meeting warning window in minutes."""
        raw = self._get_posture_value("pre_meeting_warning_minutes")
        return (
            float(raw) if raw is not None else float(DEFAULT_PRE_MEETING_WARNING_MINUTES)
        )

    def cooldown_minutes(self) -> float:
        """Return configured recommendation cooldown window in minutes."""
        raw = self._get_posture_value("recommendation_cooldown_minutes")
        return float(raw) if raw is not None else float(DEFAULT_COOLDOWN_MINUTES)

    async def evaluate(self, now: datetime | None = None) -> dict[str, Any] | None:
        """Check whether an upcoming meeting starts within warning window while walking/away."""
        async with self._eval_lock:
            if self._privacy_paused or not self._db_connected():
                return None

            # Suppress redundant warning if the user has already returned to a desk call window
            # (is_call_window=True with recent input activity, i.e., not idle).
            if not self._is_idle and await self._is_in_call_window():
                return None

            ref_now = ensure_aware(now) if now is not None else self._now()
            activity = await self._resolve_activity()
            if activity not in _GUARDRAIL_ACTIVITIES:
                return None

            upcoming_meeting = await self._find_imminent_meeting(ref_now)
            if upcoming_meeting is None:
                return None

            event_id_raw = upcoming_meeting.get("event_id")
            if not event_id_raw:
                return None
            event_id = str(event_id_raw)
            dedupe_key = f"meeting:warn:{event_id}"

            if await self._is_suppressed(dedupe_key, now=ref_now):
                return None

            title = str(upcoming_meeting.get("title") or "Meeting")
            start_dt = _parse_dt(upcoming_meeting.get("start_time"))
            if start_dt is not None:
                seconds_left = max(1.0, (start_dt - ref_now).total_seconds())
                minutes_until = max(1, int(math.ceil(seconds_left / 60.0)))
            else:
                minutes_until = int(self.pre_meeting_warning_minutes())

            if activity == "walking":
                message = (
                    f"'{title}' starts in {minutes_until} minutes. "
                    "Stop walking and return to your desk."
                )
            else:
                message = (
                    f"'{title}' starts in {minutes_until} minutes. "
                    "Return to your desk."
                )

            action_payload: dict[str, Any] = {
                "calendar_event_id": event_id,
                "title": title,
                "minutes_until_start": minutes_until,
                "activity": activity,
                "recommended_activity": "sitting",
            }
            return await self._create_and_publish(
                message=message,
                suggested_action="return_to_desk",
                action_payload=action_payload,
                dedupe_key=dedupe_key,
                now=ref_now,
            )

    async def _find_imminent_meeting(self, now: datetime) -> dict[str, Any] | None:
        warning_delta = timedelta(minutes=self.pre_meeting_warning_minutes())

        candidates: list[dict[str, Any]] = []
        if self._db_connected():
            _, next_evt = await self.db.get_current_and_next_events(now=now)
            if next_evt is not None:
                candidates.append(next_evt)
                # Look past a short intervening non-meeting block if it ends inside the window.
                if str(next_evt.get("task_type")) != "meeting":
                    next_end = _parse_dt(next_evt.get("end_time"))
                    if next_end is not None and timedelta(0) < (next_end - now) <= warning_delta:
                        _, after_evt = await self.db.get_current_and_next_events(now=next_end)
                        if after_evt is not None:
                            candidates.append(after_evt)

        if self._cached_next_event is not None:
            candidates.append(self._cached_next_event)

        for candidate in candidates:
            if str(candidate.get("task_type", "")).strip().lower() != "meeting":
                continue
            start_dt = _parse_dt(candidate.get("start_time"))
            if start_dt is None:
                return candidate
            delta = start_dt - now
            if timedelta(0) < delta <= warning_delta:
                return candidate

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

    async def _on_activity_state_changed(self, event: PlatformEvent) -> None:
        raw_activity = event.payload.get("activity")
        event_time = self._event_now(event)
        self._last_event_at = event_time

        if isinstance(raw_activity, str) and raw_activity in _VALID_ACTIVITIES:
            self._current_activity = cast(ActivityState, raw_activity)
            parsed_start = _parse_dt(event.payload.get("started_at"))
            self._activity_started_at = parsed_start or ensure_aware(event.timestamp)

        await self.evaluate(now=event_time)

    async def _on_calendar_context_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        self._update_cached_next_event(event.payload, event_time)
        await self.evaluate(now=event_time)

    async def _on_window_focus_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        payload = event.payload
        app_class = str(payload.get("app_class") or "")
        window_title = str(payload.get("window_title") or payload.get("title") or "")

        if payload.get("is_call_window") is not None:
            self._is_call_window = bool(payload["is_call_window"])
        elif isinstance(payload.get("metadata"), dict) and "is_call_window" in payload["metadata"]:
            self._is_call_window = bool(payload["metadata"]["is_call_window"])
        elif app_class or window_title:
            self._is_call_window = is_call_window(app_class, window_title)
        else:
            self._is_call_window = False

        await self.evaluate(now=event_time)

    async def _on_input_idle_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        payload = event.payload
        raw_idle = payload.get("idle")
        if raw_idle is None:
            raw_idle = payload.get("is_idle")
        if raw_idle is not None:
            self._is_idle = bool(raw_idle)
        await self.evaluate(now=event_time)

    async def _is_in_call_window(self) -> bool:
        if self._is_call_window:
            return True
        if self._db_connected():
            cur_win = await self.db.get_current_window_span()
            if cur_win is not None:
                meta = cur_win.get("metadata")
                if isinstance(meta, dict) and meta.get("is_call_window") is True:
                    return True
                if is_call_window(
                    str(cur_win.get("app_class") or ""),
                    str(cur_win.get("window_title") or ""),
                ):
                    return True
        return False

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

        if self._privacy_paused:
            self._is_call_window = False
            self._is_idle = False

    async def _periodic_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.check_interval_seconds())
                if not self._running:
                    break
                try:
                    await self.evaluate()
                    self._last_error = None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    logger.exception("Error during %s periodic evaluation", self.name)
        except asyncio.CancelledError:
            raise

    async def _cancel_loop(self) -> None:
        task = self._loop_task
        self._loop_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _resolve_activity(self) -> ActivityState | None:
        db_span = (
            await self.db.get_current_activity_span() if self._db_connected() else None
        )
        if db_span is not None:
            raw_act = str(db_span.get("activity", "unknown"))
            db_act: ActivityState = (
                cast(ActivityState, raw_act)
                if raw_act in _VALID_ACTIVITIES
                else "unknown"
            )
            db_start = _parse_dt(db_span.get("started_at"))
            if (
                self._activity_started_at is not None
                and self._current_activity is not None
                and db_start is not None
                and self._activity_started_at > db_start
            ):
                return self._current_activity
            return db_act

        return self._current_activity

    def _update_cached_next_event(
        self,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        if "next_event" in payload:
            nxt = payload.get("next_event")
            self._cached_next_event = nxt if isinstance(nxt, dict) else None
            return

        single = payload.get("event")
        candidate: dict[str, Any] | None = None
        if isinstance(single, dict):
            candidate = single
        elif "event_id" in payload and "task_type" in payload:
            candidate = dict(payload)

        if candidate is not None:
            start_dt = _parse_dt(candidate.get("start_time"))
            if start_dt is None or start_dt > now:
                self._cached_next_event = candidate

    def _now(self) -> datetime:
        return ensure_aware(self._clock())

    def _event_now(self, event: PlatformEvent) -> datetime:
        if self._custom_clock:
            return self._now()
        return ensure_aware(event.timestamp)

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    def _get_posture_value(self, key: str) -> Any:
        direct = self.config.get(key)
        if direct is not None:
            return direct
        nested = self.config.get("posture")
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
