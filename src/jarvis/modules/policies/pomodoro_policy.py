"""Deterministic Pomodoro selector and calendar-triggered timer recommendation policy."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, MutableMapping
from datetime import datetime, timedelta
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.classification import is_call_window
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


FLOW_OR_NO_TIMER_REGEX: re.Pattern[str] = re.compile(
    r"(?i)(?:^|[\s#\[(])(?:flow|uninterrupted|notimer|no[-_]timer|flow\s*state)(?:$|[\s#\])])"
)

SHORT_POMO_FALLBACK_REGEX: re.Pattern[str] = re.compile(
    r"(?i)(?:^|[\s\[(])(#shortpomo|\[shortpomo\]|#25m|\[25m\]|shortpomo)(?=\s|$|[.,!?;\])])"
)

HIGH_MENTAL_EFFORT_REGEX: re.Pattern[str] = re.compile(
    r"(?i)(?:^|[\s\[(])(#intense|\[intense\]|#demanding|\[demanding\]|#challenging|"
    r"#sprint|\[sprint\]|#debugging|debugging|#dense|#complex|"
    r"#25m|\[25m\])(?=\s|$|[.,!?;\])])"
)

HIGH_RESTART_COST_REGEX: re.Pattern[str] = re.compile(
    r"(?i)(?:^|[\s\[(])(#deep|\[deep\]|#immersive|\[immersive\]|#architecture|"
    r"\[architecture\]|#longform|#50m|\[50m\])(?=\s|$|[.,!?;\])])"
)

MIN_AUTOMATIC_MINUTES_50_10 = 120.0  # At least 2 hours for 50/10
MIN_AUTOMATIC_MINUTES_SHORTER = 60.0  # At least 1 hour for shorter preset (25/5)

DEFAULT_SHORTER_PRESET_TASKS: frozenset[str] = frozenset(
    {"admin", "email", "planning"}
)


def normalize_event_preset(value: Any) -> str | None:
    """Normalize explicit preset value or alias to '25_5', '50_10', 'none', or None."""
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    if clean in {"25_5", "25", "25m", "25/5"}:
        return "25_5"
    if clean in {"50_10", "50", "50m", "50/10"}:
        return "50_10"
    if clean in {"none", "off", "no_timer", "notimer"}:
        return "none"
    return None


def parse_cognitive_factors(
    target_event: Mapping[str, Any] | None,
) -> tuple[bool, bool]:
    """Extract (has_high_mental_effort, has_high_restart_cost) heuristics from event context.

    Note: These are configurable user-preference heuristics, not clinical or universal facts:
    - Mental effort / difficulty sustaining attention: suggests shorter 25/5 intervals.
    - Need for uninterrupted concentration / high restart cost: suggests 50/10 or flow.
    """
    if not target_event:
        return False, False

    high_mental_effort = False
    high_restart_cost = False

    # Mental effort / difficulty sustaining attention
    effort_val = str(
        target_event.get("mental_effort") or target_event.get("difficulty") or ""
    ).strip().lower()
    if effort_val in {"high", "very_high", "demanding", "hard"} or (
        target_event.get("difficulty_sustaining_attention") is True
    ):
        high_mental_effort = True

    # Need for uninterrupted concentration / high restart cost
    restart_val = str(
        target_event.get("restart_cost") or target_event.get("concentration_need") or ""
    ).strip().lower()
    if restart_val in {"high", "very_high", "deep"} or target_event.get("uninterrupted") is True:
        high_restart_cost = True

    title = str(target_event.get("title") or "")
    desc = str(target_event.get("description") or "")
    combined = f"{title} {desc}".strip()
    if combined:
        if not high_mental_effort and HIGH_MENTAL_EFFORT_REGEX.search(combined):
            high_mental_effort = True
        if not high_restart_cost and HIGH_RESTART_COST_REGEX.search(combined):
            high_restart_cost = True

    return high_mental_effort, high_restart_cost


def select_task_preset(
    target_event: Mapping[str, Any] | None,
    task_type: str | None,
    config: Any = None,
) -> str | None:
    """Select the appropriate Pomodoro preset ('50_10', '25_5', or None) for this task context.

    Considers cognitive intensity and restart cost heuristics rather than rigid task mappings:
    - Mental effort / difficulty sustaining attention: heuristics favor '25_5'.
    - Need for uninterrupted concentration / high restart cost: heuristics favor '50_10'.
    - Explicit user preferences and event tags take precedence.
    - Uses conservative, configurable defaults when context is limited.
    """
    if target_event is None and task_type is None:
        return None

    normalized_task = str(task_type).strip().lower() if task_type is not None else "unknown"

    # 1. Explicit preset in event metadata or tags
    if target_event:
        explicit_norm = normalize_event_preset(
            target_event.get("preferred_preset") or target_event.get("preset")
        )
        if explicit_norm in {"25_5", "50_10"}:
            return explicit_norm
        if explicit_norm == "none":
            return None

        title = str(target_event.get("title") or "")
        desc = str(target_event.get("description") or "")
        combined = f"{title} {desc}".strip()
        if combined:
            if re.search(
                r"(?i)(?:^|[\s\[(])(#25m|\[25m\]|#25_5)(?=\s|$|[.,!?;\])])",
                combined,
            ):
                return "25_5"
            if re.search(
                r"(?i)(?:^|[\s\[(])(#50m|\[50m\]|#50_10)(?=\s|$|[.,!?;\])])",
                combined,
            ):
                return "50_10"

    # 2. Cognitive factor evaluation: mental effort vs. restart cost
    high_mental_effort, high_restart_cost = parse_cognitive_factors(target_event)

    # High mental effort / difficulty sustaining attention favors 25/5
    if high_mental_effort and not high_restart_cost:
        return "25_5"

    # High restart cost favors 50/10
    if high_restart_cost and not high_mental_effort:
        return "50_10"

    # 3. Conservative, configurable defaults when context is limited
    return select_pomodoro(normalized_task, config)


def task_suits_shorter_preset(
    task_type: str | None,
    target_event: Mapping[str, Any] | None = None,
    config: Any = None,
) -> bool:
    """Return True if the task or event context also suits the shorter 25/5 preset.

    Tasks with high mental effort or difficulty sustaining attention may suit shorter
    intervals based on user preference heuristics.
    Tasks with high restart cost avoid downshifting to 25/5.
    """
    if task_type is None and target_event is None:
        return False

    normalized = str(task_type).strip().lower() if task_type is not None else ""
    high_mental_effort, high_restart_cost = parse_cognitive_factors(target_event)

    # Strictly high restart cost suppresses downshifting
    if high_restart_cost:
        return False

    # Check event indicators for shorter interval suitability
    if target_event:
        if target_event.get("allow_shorter_preset") is True:
            return True
        if high_mental_effort:
            return True
        title = str(target_event.get("title") or "")
        desc = str(target_event.get("description") or "")
        combined = f"{title} {desc}".strip()
        if combined and SHORT_POMO_FALLBACK_REGEX.search(combined):
            return True

    # Routine / admin / planning tasks
    if normalized in DEFAULT_SHORTER_PRESET_TASKS:
        return True

    # Check config for allowed fallback tasks or global flag
    if isinstance(config, Mapping):
        pomo_cfg = config.get("pomodoro") if isinstance(config.get("pomodoro"), Mapping) else config
        allowed_list = pomo_cfg.get("allow_shorter_fallback_tasks")
        if isinstance(allowed_list, (list, tuple, set, frozenset)):
            if normalized in {str(t).lower() for t in allowed_list}:
                return True
        if pomo_cfg.get("allow_shorter_preset_fallback") is True:
            return True

    return False


def is_pomodoro_appropriate(
    target_event: Mapping[str, Any] | None,
    task_type: str | None,
    config: Any = None,
) -> bool:
    """Return True only if a Pomodoro recommendation is appropriate and useful for this context.

    'No recommendation' is a valid, intentional outcome when the activity does not
    benefit from timeboxing, the user prefers uninterrupted flow, or context is insufficient.
    """
    if target_event is None and task_type is None:
        return False

    normalized_task = str(task_type).strip().lower() if task_type is not None else ""
    if normalized_task in {"unknown", "meeting", "break"}:
        return False

    # Explicit user opt-outs in event
    if target_event:
        if target_event.get("pomodoro_enabled") is False or target_event.get("timeboxing") is False:
            return False
        if target_event.get("uninterrupted") is True or target_event.get("flow") is True:
            return False
        raw_pref = target_event.get("preferred_preset") or target_event.get("preset")
        if normalize_event_preset(raw_pref) == "none":
            return False

        title = str(target_event.get("title") or "")
        desc = str(target_event.get("description") or "")
        combined = f"{title} {desc}".strip()
        if combined and FLOW_OR_NO_TIMER_REGEX.search(combined):
            return False

    # Check config: if user mapped this task type to None, timeboxing is opted out by default
    base_preset = select_pomodoro(normalized_task, config)
    if base_preset is None:
        raw_pref = (
            target_event.get("preferred_preset") or target_event.get("preset")
            if target_event
            else None
        )
        if normalize_event_preset(raw_pref) not in {"25_5", "50_10"}:
            return False

    return True


def fit_pomodoro_preset(
    base_kind: str,
    available_minutes: float | None,
    *,
    suits_shorter: bool = False,
) -> tuple[str | None, bool]:
    """Fit a Pomodoro preset against remaining available minutes.

    Requirements:
    - 50/10: at least two hours (120 minutes) of available time.
    - Shorter preset (25/5): at least one hour (60 minutes) of available time.
    - Under one hour (< 60 minutes): no automatic Pomodoro suggestion.
    - Having enough time must NOT automatically upgrade to a longer preset.
    - If the preferred preset does not fit, suggest the shorter one ONLY if
      it also suits the task; otherwise suggest nothing.

    Note on available_minutes=None:
        An unconstrained schedule window (e.g. all-day event or open-ended block)
        has no known calendar boundary cutting the session short. It is treated as
        unconstrained for timer fitting, not as proof that the user is actually free.

    Returns:
        (fitted_kind, is_downshifted)
        Returns (None, False) if no preset fits or is appropriate.
    """
    if available_minutes is None:
        return base_kind, False

    # Under one hour: no automatic Pomodoro suggestion
    if available_minutes < MIN_AUTOMATIC_MINUTES_SHORTER:
        return None, False

    if base_kind == "50_10":
        if available_minutes >= MIN_AUTOMATIC_MINUTES_50_10:
            return "50_10", False
        # 50/10 does not fit (< 120m). Suggest shorter preset ONLY if task also suits it
        if available_minutes >= MIN_AUTOMATIC_MINUTES_SHORTER and suits_shorter:
            return "25_5", True
        return None, False

    if base_kind == "25_5":
        if available_minutes >= MIN_AUTOMATIC_MINUTES_SHORTER:
            return "25_5", False
        return None, False

    return None, False


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

        self._current_activity: str | None = None
        self._current_app_class: str | None = None
        self._current_window_title: str | None = None
        self._is_call_window: bool = False

        self._eval_lock = asyncio.Lock()
        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Subscribe to calendar, activity, window, resolution, and privacy events."""
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected() and not self._privacy_paused:
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if persisted_pause is True:
                self._privacy_paused = True

        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.subscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.subscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.subscribe("window_focus_changed", self._on_window_focus_changed)
        self._running = True
        self._last_error = None

    async def stop(self) -> None:
        """Unsubscribe from EventBus topics cleanly."""
        self._running = False
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.unsubscribe("recommendation_resolved", self._on_recommendation_resolved)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.unsubscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.unsubscribe("window_focus_changed", self._on_window_focus_changed)

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

            # Skip if an ad-hoc call window is currently active
            if await self._is_in_call_window():
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

                # Stage 1: Appropriateness Gate
                if not is_pomodoro_appropriate(target_event, task_type, self.config):
                    continue

                base_kind = select_task_preset(target_event, task_type, self.config)
                if base_kind is None:
                    continue

                # Stage 2: Calculate Available Schedule Window
                start_dt = _parse_dt(target_event.get("start_time"))
                end_dt = _parse_dt(target_event.get("end_time"))
                is_all_day = bool(target_event.get("is_all_day"))

                available_mins: float | None = None
                if not is_all_day:
                    if is_active_block:
                        if end_dt is not None:
                            available_mins = (end_dt - ref_now).total_seconds() / 60.0
                            if available_mins <= 0:
                                continue
                    else:
                        if start_dt is not None and end_dt is not None:
                            available_mins = (end_dt - start_dt).total_seconds() / 60.0
                            if available_mins <= 0:
                                continue

                if is_active_block and next_event is not None:
                    next_start = _parse_dt(next_event.get("start_time"))
                    if next_start is not None and next_start > ref_now:
                        time_until_next = (next_start - ref_now).total_seconds() / 60.0
                        if available_mins is None:
                            available_mins = time_until_next
                        else:
                            available_mins = min(available_mins, time_until_next)

                suits_shorter = task_suits_shorter_preset(task_type, target_event, self.config)
                kind, is_downshifted = fit_pomodoro_preset(
                    base_kind,
                    available_mins,
                    suits_shorter=suits_shorter,
                )
                if kind is None:
                    continue

                dedupe_key = f"pomodoro:event:{event_id}"
                if await self._is_suppressed(dedupe_key, now=ref_now, event_end=end_dt):
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
                    is_downshifted=is_downshifted,
                )
                action_payload: dict[str, Any] = {
                    "kind": kind,
                    "walking_suitable": walking_suitable,
                    "calendar_event_id": event_id,
                    "task_type": task_type,
                    "downshifted": is_downshifted,
                    "original_kind": base_kind,
                }
                if self._current_activity:
                    action_payload["current_activity"] = self._current_activity
                # Milestone 1: never auto-start timers, even if auto_start is set in config.
                return await self._create_and_publish(
                    message=message,
                    suggested_action="start_timer",
                    action_payload=action_payload,
                    dedupe_key=dedupe_key,
                    now=ref_now,
                )

            return None

    async def _is_suppressed(
        self,
        dedupe_key: str,
        *,
        now: datetime,
        event_end: datetime | None = None,
    ) -> bool:
        pending_rows = await self.db.get_pending_recommendations()
        if any(row.get("dedupe_key") == dedupe_key for row in pending_rows):
            return True

        recent = await self.db.get_recent_recommendations_for_dedupe(dedupe_key, limit=5)
        if not recent:
            return False

        latest = recent[0]
        if latest.get("status") == "pending":
            return True

        metadata = latest.get("resolution_metadata")
        if isinstance(metadata, dict):
            cooldown_until_raw = metadata.get("cooldown_until") or metadata.get("snoozed_until")
            if cooldown_until_raw:
                cooldown_until = _parse_dt(cooldown_until_raw)
                if cooldown_until is not None and now < cooldown_until:
                    return True

        latest_status = latest.get("status")
        base_cooldown = self.cooldown_minutes()

        if latest_status == "snoozed":
            snooze_mins = (
                float(metadata["snooze_minutes"])
                if isinstance(metadata, dict) and metadata.get("snooze_minutes") is not None
                else base_cooldown
            )
            if snooze_mins <= 0:
                return False
            raw_resolved = latest.get("resolved_at") or latest.get("created_at")
            resolved_at = _parse_dt(raw_resolved) or now
            return now < (resolved_at + timedelta(minutes=snooze_mins))

        if latest_status == "dismissed":
            consecutive_dismissals = 0
            for r in recent:
                if r.get("status") == "dismissed":
                    consecutive_dismissals += 1
                elif r.get("status") == "accepted":
                    break
                else:
                    break

            if consecutive_dismissals >= 3:
                # Suppress for the remainder of this calendar block!
                if event_end is not None and now < event_end:
                    return True
                effective_mins = base_cooldown * 4
            elif consecutive_dismissals == 2:
                effective_mins = base_cooldown * 2
            else:
                effective_mins = base_cooldown

            if effective_mins <= 0:
                return False
            raw_resolved = latest.get("resolved_at") or latest.get("created_at")
            resolved_at = _parse_dt(raw_resolved) or now
            return now < (resolved_at + timedelta(minutes=effective_mins))

        if latest_status == "accepted":
            if base_cooldown <= 0:
                return False
            raw_resolved = latest.get("resolved_at") or latest.get("created_at")
            resolved_at = _parse_dt(raw_resolved) or now
            return now < (resolved_at + timedelta(minutes=base_cooldown))

        return False

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

    async def _on_activity_state_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        act = event.payload.get("activity")
        if isinstance(act, str):
            self._current_activity = act
        await self.evaluate(now=event_time)

    async def _on_window_focus_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        payload = event.payload
        app_class = str(payload.get("app_class") or "")
        window_title = str(payload.get("window_title") or payload.get("title") or "")
        self._current_app_class = app_class
        self._current_window_title = window_title

        if payload.get("is_call_window") is not None:
            self._is_call_window = bool(payload["is_call_window"])
        elif isinstance(payload.get("metadata"), dict) and "is_call_window" in payload["metadata"]:
            self._is_call_window = bool(payload["metadata"]["is_call_window"])
        elif app_class or window_title:
            self._is_call_window = is_call_window(app_class, window_title)
        else:
            self._is_call_window = False

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
            self._current_app_class = None
            self._current_window_title = None

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
        is_downshifted: bool = False,
    ) -> str:
        title = str(target_event.get("title") or "Focus block")

        if is_downshifted:
            timer_label = "25/5 block (adjusted to fit remaining time)"
        elif kind == "25_5":
            if walking_suitable:
                timer_label = "walking-suitable 25/5 block"
            else:
                timer_label = "25/5 block"
        elif kind == "50_10":
            timer_label = "focused 50/10 block"
        else:
            display_kind = kind.replace("_", "/")
            timer_label = f"{display_kind} block"

        if is_active_block:
            return f"'{title}' is active. Suggested Pomodoro timer: {timer_label}."

        start_dt = _parse_dt(target_event.get("start_time"))
        if start_dt is not None:
            mins = max(1, int(round((start_dt - now).total_seconds() / 60)))
            return (
                f"'{title}' starts in {mins} minutes. "
                f"Suggested Pomodoro timer: {timer_label}."
            )
        return f"'{title}' is starting soon. Suggested Pomodoro timer: {timer_label}."

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
