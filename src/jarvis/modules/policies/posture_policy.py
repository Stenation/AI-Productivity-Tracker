"""Deterministic posture policy recommending standing/walking or seated breaks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, MutableMapping
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any, cast

from jarvis.core.bus import EventBus
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

DEFAULT_MAX_SITTING_MINUTES = 50
DEFAULT_MAX_WALKING_MINUTES = 45
DEFAULT_PRE_MEETING_WARNING_MINUTES = 10
DEFAULT_COOLDOWN_MINUTES = 30
DEFAULT_CHECK_INTERVAL_SECONDS = 60.0
DEFAULT_UPCOMING_WINDOW_MINUTES = 30

SITTING_DEDUPE_KEY = "posture:sitting"
WALKING_DEDUPE_KEY = "posture:walking"

_VALID_ACTIVITIES: frozenset[str] = frozenset(
    {"sitting", "standing", "walking", "away", "unknown"}
)
_WALKING_SUITABLE_TASKS: frozenset[str] = frozenset(
    {"admin", "email", "planning", "break"}
)
_NON_WALKING_TASKS: frozenset[str] = frozenset(
    {"deep_work", "coding", "reading", "meeting"}
)


def _parse_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value:
        return ensure_aware(datetime.fromisoformat(value))
    return None


class PosturePolicyModule(BaseModule):
    """Evaluate continuous sitting/walking durations and emit posture recommendations."""

    name: str = "posture_policy"

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
        self._cached_current_event: dict[str, Any] | None = None
        self._cached_next_event: dict[str, Any] | None = None

        self._eval_lock = asyncio.Lock()
        self._loop_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Subscribe to state/calendar/privacy events and start periodic evaluation."""
        await self._cancel_loop()

        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected() and not self._privacy_paused:
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if persisted_pause is True:
                self._privacy_paused = True

        self.bus.subscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.subscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)

        self._running = True
        self._last_error = None
        self._loop_task = asyncio.create_task(
            self._periodic_loop(),
            name=f"{self.name}_periodic_loop",
        )

    async def stop(self) -> None:
        """Unsubscribe from EventBus topics and cancel background evaluation task."""
        self._running = False
        self.bus.unsubscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.unsubscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)
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

    def max_sitting_minutes(self) -> float:
        """Return configured continuous sitting threshold in minutes."""
        raw = self._get_posture_value("max_sitting_minutes")
        return float(raw) if raw is not None else float(DEFAULT_MAX_SITTING_MINUTES)

    def max_walking_minutes(self) -> float:
        """Return configured continuous walking threshold in minutes."""
        raw = self._get_posture_value("max_walking_minutes")
        return float(raw) if raw is not None else float(DEFAULT_MAX_WALKING_MINUTES)

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

    def upcoming_window_minutes(self) -> float:
        """Return lookahead horizon for upcoming calendar blocks in minutes."""
        raw = self._get_posture_value("upcoming_window_minutes")
        return float(raw) if raw is not None else float(DEFAULT_UPCOMING_WINDOW_MINUTES)

    async def evaluate(self, now: datetime | None = None) -> dict[str, Any] | None:
        """Evaluate current posture duration and calendar context once."""
        async with self._eval_lock:
            if self._privacy_paused or not self._db_connected():
                return None

            ref_now = ensure_aware(now) if now is not None else self._now()
            activity, started_at = await self._resolve_activity_context()
            if activity not in ("sitting", "walking") or started_at is None:
                return None

            elapsed_seconds = max(0.0, (ref_now - started_at).total_seconds())
            elapsed_minutes = int(elapsed_seconds // 60)

            if activity == "sitting":
                max_sit = self.max_sitting_minutes()
                if elapsed_seconds < max_sit * 60.0:
                    return None
                return await self._evaluate_sitting(
                    now=ref_now,
                    started_at=started_at,
                    elapsed_minutes=max(int(max_sit), elapsed_minutes),
                    max_sit_minutes=max_sit,
                )

            max_walk = self.max_walking_minutes()
            if elapsed_seconds < max_walk * 60.0:
                return None
            return await self._evaluate_walking(
                now=ref_now,
                started_at=started_at,
                elapsed_minutes=max(int(max_walk), elapsed_minutes),
                max_walk_minutes=max_walk,
            )

    async def _evaluate_sitting(
        self,
        *,
        now: datetime,
        started_at: datetime,
        elapsed_minutes: int,
        max_sit_minutes: float,
    ) -> dict[str, Any] | None:
        current_event, next_event = await self._resolve_calendar_context(now)

        # Suppress posture interruption while currently in a meeting.
        if current_event is not None and str(current_event.get("task_type")) == "meeting":
            return None

        if await self._is_suppressed(
            SITTING_DEDUPE_KEY,
            now=now,
            activity_started_at=started_at,
        ):
            return None

        upcoming_event = self._select_upcoming_event(next_event, now)
        imminent_meeting = self._is_imminent_meeting(next_event, now)

        target_event = current_event if current_event is not None else upcoming_event
        suitability = self._classify_event_walking_suitability(target_event)
        if imminent_meeting:
            suitability = False

        if suitability is True:
            suggested_action = "walk"
            recommended_activity = "walking"
        elif suitability is False:
            suggested_action = "stand"
            recommended_activity = "standing"
        else:
            suggested_action = "stand_or_walk"
            recommended_activity = "standing_or_walking"

        message = self._build_sitting_message(
            elapsed_minutes=elapsed_minutes,
            suggested_action=suggested_action,
            current_event=current_event,
            upcoming_event=upcoming_event,
            now=now,
        )
        action_payload: dict[str, Any] = {
            "activity": "sitting",
            "elapsed_minutes": elapsed_minutes,
            "max_sitting_minutes": (
                int(max_sit_minutes)
                if max_sit_minutes.is_integer()
                else max_sit_minutes
            ),
            "recommended_activity": recommended_activity,
            "walking_suitable": suitability,
            "calendar_event_id": (
                target_event.get("event_id") if target_event is not None else None
            ),
        }
        return await self._create_and_publish(
            message=message,
            suggested_action=suggested_action,
            action_payload=action_payload,
            dedupe_key=SITTING_DEDUPE_KEY,
            now=now,
        )

    async def _evaluate_walking(
        self,
        *,
        now: datetime,
        started_at: datetime,
        elapsed_minutes: int,
        max_walk_minutes: float,
    ) -> dict[str, Any] | None:
        if await self._is_suppressed(
            WALKING_DEDUPE_KEY,
            now=now,
            activity_started_at=started_at,
        ):
            return None

        message = (
            f"You have been walking for {elapsed_minutes} minutes. "
            "Consider taking a seated break."
        )
        action_payload: dict[str, Any] = {
            "activity": "walking",
            "elapsed_minutes": elapsed_minutes,
            "max_walking_minutes": (
                int(max_walk_minutes)
                if max_walk_minutes.is_integer()
                else max_walk_minutes
            ),
            "recommended_activity": "sitting",
        }
        return await self._create_and_publish(
            message=message,
            suggested_action="take_seated_break",
            action_payload=action_payload,
            dedupe_key=WALKING_DEDUPE_KEY,
            now=now,
        )

    async def _is_suppressed(
        self,
        dedupe_key: str,
        *,
        now: datetime,
        activity_started_at: datetime | None = None,
    ) -> bool:
        pending_rows = await self.db.get_pending_recommendations()
        if any(row.get("dedupe_key") == dedupe_key for row in pending_rows):
            return True

        latest = await self.db.find_recent_recommendation(dedupe_key)
        if latest is None:
            return False
        if latest.get("status") == "pending":
            return True

        created_dt = _parse_dt(latest.get("created_at"))
        resolved_dt = _parse_dt(latest.get("resolved_at"))
        last_lifecycle_dt = resolved_dt or created_dt
        if (
            created_dt is not None
            and resolved_dt is not None
            and created_dt > resolved_dt
        ):
            last_lifecycle_dt = created_dt

        # Fresh activity span started after prior recommendation was resolved resets cooldown.
        if (
            activity_started_at is not None
            and last_lifecycle_dt is not None
            and activity_started_at > last_lifecycle_dt
        ):
            return False

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

    async def _resolve_activity_context(
        self,
    ) -> tuple[ActivityState | None, datetime | None]:
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
                return self._current_activity, self._activity_started_at
            return db_act, db_start

        return self._current_activity, self._activity_started_at

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
        cur = payload.get("current_event")
        nxt = payload.get("next_event")
        single = payload.get("event")

        if "current_event" in payload:
            self._cached_current_event = cur if isinstance(cur, dict) else None
        if "next_event" in payload:
            self._cached_next_event = nxt if isinstance(nxt, dict) else None

        candidate: dict[str, Any] | None = None
        if isinstance(single, dict):
            candidate = single
        elif "event_id" in payload and "task_type" in payload:
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

    def _select_upcoming_event(
        self,
        next_event: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        if next_event is None:
            return None
        start_dt = _parse_dt(next_event.get("start_time"))
        if start_dt is None:
            return next_event
        delta = start_dt - now
        if timedelta(0) < delta <= timedelta(minutes=self.upcoming_window_minutes()):
            return next_event
        return None

    def _is_imminent_meeting(
        self,
        next_event: dict[str, Any] | None,
        now: datetime,
    ) -> bool:
        if next_event is None or str(next_event.get("task_type")) != "meeting":
            return False
        start_dt = _parse_dt(next_event.get("start_time"))
        if start_dt is None:
            return True
        delta = start_dt - now
        return timedelta(0) < delta <= timedelta(minutes=self.pre_meeting_warning_minutes())

    @staticmethod
    def _classify_event_walking_suitability(
        event: dict[str, Any] | None,
    ) -> bool | None:
        if event is None:
            return None
        explicit = event.get("suitable_for_walking")
        if explicit is not None:
            return bool(explicit)
        task_type = str(event.get("task_type", "unknown")).strip().lower()
        if task_type in _WALKING_SUITABLE_TASKS:
            return True
        if task_type in _NON_WALKING_TASKS:
            return False
        return None

    @staticmethod
    def _build_sitting_message(
        *,
        elapsed_minutes: int,
        suggested_action: str,
        current_event: dict[str, Any] | None,
        upcoming_event: dict[str, Any] | None,
        now: datetime,
    ) -> str:
        base = f"You have been sitting for {elapsed_minutes} minutes."
        if current_event is None and upcoming_event is not None:
            title = str(upcoming_event.get("title") or "Next block")
            start_dt = _parse_dt(upcoming_event.get("start_time"))
            if start_dt is not None:
                mins = max(1, int(round((start_dt - now).total_seconds() / 60)))
                prefix = f"{title} starts in {mins} minutes. {base}"
            else:
                prefix = f"{title} is starting soon. {base}"
        elif current_event is not None and current_event.get("title"):
            prefix = f"{base} Current block: {current_event['title']}."
        else:
            prefix = base

        if suggested_action == "walk":
            return f"{prefix} Consider walking for this block."
        if suggested_action == "stand":
            return f"{prefix} Consider standing up."
        return f"{prefix} Consider standing or walking."

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
