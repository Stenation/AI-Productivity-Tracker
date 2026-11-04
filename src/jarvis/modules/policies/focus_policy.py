"""Observe-only focus context policy consuming window, idle, calendar, and privacy events."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from typing import Any, Literal

from jarvis.core.bus import EventBus
from jarvis.core.classification import classify_window, is_call_window
from jarvis.core.database import Database
from jarvis.core.schema import FocusState, PlatformEvent, ensure_aware, now_local
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_DISTRACTION_GRACE_SECONDS = 180
DEFAULT_WORK_APP_CLASSES: tuple[str, ...] = (
    "code",
    "Alacritty",
    "kitty",
    "obsidian",
    "Zotero",
)
DEFAULT_DISTRACTION_TITLE_PATTERNS: tuple[str, ...] = (
    "YouTube",
    "Reddit",
    "Twitch",
    "Twitter",
    "X",
)
DEFAULT_CALENDAR_TASK_FOCUS_MAP: dict[str, FocusState] = {
    "meeting": "neutral",
    "break": "neutral",
}

_PRODUCTIVITY_TASK_TYPES: frozenset[str] = frozenset(
    {"deep_work", "coding", "reading", "admin", "email", "planning"}
)
_FOCUS_CATEGORY_ALIASES: dict[str, FocusState] = {
    "focused": "focused",
    "work": "focused",
    "productive": "focused",
    "neutral": "neutral",
    "distracted": "distracted",
    "distraction": "distracted",
    "unknown": "unknown",
}


def _parse_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value.strip():
        with suppress(ValueError):
            return ensure_aware(datetime.fromisoformat(value))
    return None


def _title_matches_pattern(window_title: str, pattern: str) -> bool:
    cleaned = pattern.strip()
    if not cleaned or not window_title:
        return False
    if len(cleaned) <= 2:
        token_re = rf"(?<![A-Za-z0-9]){re.escape(cleaned)}(?![A-Za-z0-9])"
        return bool(re.search(token_re, window_title, re.IGNORECASE))
    return cleaned.lower() in window_title.lower()


def classify_focus_context(
    app_class: str,
    window_title: str,
    *,
    work_app_classes: Sequence[str] = DEFAULT_WORK_APP_CLASSES,
    distraction_title_patterns: Sequence[str] = DEFAULT_DISTRACTION_TITLE_PATTERNS,
    calendar_task_type: str | None = None,
) -> tuple[FocusState, str]:
    """Deterministically classify `(app_class, window_title)` per SPEC.MD FR-09.1."""
    clean_class = app_class.strip()
    clean_title = window_title.strip()
    if not clean_class and not clean_title:
        return ("neutral", "desktop")

    lowered_class = clean_class.lower()
    if lowered_class and any(
        lowered_class == str(item).strip().lower()
        for item in work_app_classes
        if str(item).strip()
    ):
        return ("focused", "app_class")

    if clean_title and any(
        _title_matches_pattern(clean_title, str(pattern))
        for pattern in distraction_title_patterns
    ):
        return ("distracted", "title_pattern")

    if calendar_task_type:
        norm_task = calendar_task_type.strip().lower()
        mapped = DEFAULT_CALENDAR_TASK_FOCUS_MAP.get(norm_task)
        if mapped is not None:
            return (mapped, "calendar_task")

        # Richer calendar task-type correlation (e.g. docs during coding/reading)
        correlated = classify_window(
            clean_class,
            clean_title,
            active_task_type=norm_task,
            work_app_classes=work_app_classes,
            distraction_title_patterns=distraction_title_patterns,
        )
        if (
            correlated.category != "unknown"
            and correlated.classifier == "calendar_correlation"
        ):
            return (correlated.category, "calendar_correlation")

    return ("unknown", "default")


class FocusPolicyModule(BaseModule):
    """Milestone 2 observe-only focus policy (logs context without issuing nudges)."""

    name: str = "focus_policy"

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

        self._focus_state: FocusState = "unknown"
        self._app_class: str | None = None
        self._window_title: str | None = None
        self._classifier: str | None = None
        self._window_started_at: datetime | None = None
        self._is_call_window: bool = False

        self._idle: bool = False
        self._idle_since: datetime | None = None

        self._cached_current_event: dict[str, Any] | None = None
        self._cached_next_event: dict[str, Any] | None = None

        self._observed_window_events: int = 0
        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Load persisted privacy settings and subscribe to focus/idle/calendar topics."""
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected():
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if isinstance(persisted_pause, bool):
                self._privacy_paused = self._privacy_paused or persisted_pause
            persisted_dnd = await self.db.get_setting("privacy.dnd_mode")
            if isinstance(persisted_dnd, bool):
                self._dnd_mode = self._dnd_mode or persisted_dnd

            if not self._privacy_paused:
                cur_win = await self.db.get_current_window_span()
                if cur_win is not None:
                    raw_cat = str(cur_win.get("category") or "unknown").strip().lower()
                    self._focus_state = _FOCUS_CATEGORY_ALIASES.get(raw_cat, "unknown")
                    self._app_class = str(cur_win.get("app_class") or "")
                    self._window_title = str(cur_win.get("window_title") or "")
                    self._classifier = str(cur_win.get("classifier") or "recovered")
                    self._window_started_at = _parse_dt(cur_win.get("started_at"))
                    meta = cur_win.get("metadata")
                    if isinstance(meta, dict) and meta.get("is_call_window") is not None:
                        self._is_call_window = bool(meta["is_call_window"])
                    elif self._app_class or self._window_title:
                        self._is_call_window = is_call_window(
                            self._app_class, self._window_title
                        )
                    else:
                        self._is_call_window = False

        if self.configured_mode() != "observe_only":
            logger.info(
                "%s configured with hyprland_ipc_mode=%r; enforcing 'observe_only' in Milestone 2",
                self.name,
                self.configured_mode(),
            )

        self.bus.subscribe("window_focus_changed", self._on_window_focus_changed)
        self.bus.subscribe("input_idle_changed", self._on_input_idle_changed)
        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)

        self._running = True
        self._last_error = None

    async def stop(self) -> None:
        """Unsubscribe cleanly from EventBus topics."""
        self._running = False
        self.bus.unsubscribe("window_focus_changed", self._on_window_focus_changed)
        self.bus.unsubscribe("input_idle_changed", self._on_input_idle_changed)
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Reload configuration and apply privacy pause overrides."""
        await super().reload_config(dict(config))
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)
        if self._privacy_paused:
            self._clear_observed_context()

    async def health(self) -> dict[str, Any]:
        """Return current module health and observe-only status."""
        status = (
            "degraded"
            if self._last_error is not None
            else ("ok" if self._running else "disabled")
        )
        return {
            "module": self.name,
            "status": status,
            "mode": self.effective_mode(),
            "configured_mode": self.configured_mode(),
            "focus_state": self._focus_state,
            "idle": self._idle,
            "observed_window_events": self._observed_window_events,
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
            "last_error": self._last_error,
        }

    def configured_mode(self) -> str:
        """Return configured `hyprland_ipc_mode` from privacy/module config."""
        direct = self.config.get("hyprland_ipc_mode")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        privacy_cfg = self.config.get("privacy")
        if isinstance(privacy_cfg, Mapping):
            nested = privacy_cfg.get("hyprland_ipc_mode")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return "observe_only"

    @staticmethod
    def effective_mode() -> Literal["observe_only"]:
        """Return effective Milestone 2 focus policy mode (strictly `observe_only`)."""
        return "observe_only"

    def distraction_grace_seconds(self) -> int:
        """Return configured distraction grace duration in seconds."""
        raw = self._get_focus_value("distraction_grace_seconds")
        if raw is not None:
            return max(0, int(raw))
        return DEFAULT_DISTRACTION_GRACE_SECONDS

    def work_app_classes(self) -> list[str]:
        """Return configured productive application class names."""
        raw = self._get_focus_value("work_app_classes")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [str(item) for item in raw]
        return list(DEFAULT_WORK_APP_CLASSES)

    def distraction_title_patterns(self) -> list[str]:
        """Return configured distraction window title patterns."""
        raw = self._get_focus_value("distraction_title_patterns")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [str(item) for item in raw]
        return list(DEFAULT_DISTRACTION_TITLE_PATTERNS)

    async def evaluate(self, now: datetime | None = None) -> None:
        """Observe current focus context without issuing any alerts or recommendations."""
        if self._privacy_paused:
            return None
        snapshot = await self.get_context_snapshot(now=now)
        logger.debug(
            "Focus observe-only snapshot: state=%s app=%r idle=%s productivity_block=%s "
            "elapsed=%ss grace_exceeded=%s",
            snapshot["focus_state"],
            snapshot["app_class"],
            snapshot["idle"],
            snapshot["is_productivity_block"],
            snapshot["window_elapsed_seconds"],
            snapshot["distraction_grace_exceeded"],
        )
        return None

    async def get_context_snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        """Return structured snapshot of observed window, idle, and calendar context."""
        ref_now = ensure_aware(now) if now is not None else self._now()
        current_event = await self._resolve_current_calendar_event(ref_now)
        task_type = (
            str(current_event["task_type"]).strip().lower()
            if current_event is not None and current_event.get("task_type")
            else None
        )
        event_id = (
            str(current_event["event_id"])
            if current_event is not None and current_event.get("event_id")
            else None
        )
        is_prod = task_type in _PRODUCTIVITY_TASK_TYPES if task_type else False

        elapsed = 0
        if self._window_started_at is not None and not self._privacy_paused:
            elapsed = max(0, int((ref_now - self._window_started_at).total_seconds()))

        grace_exceeded = (
            not self._privacy_paused
            and self._focus_state == "distracted"
            and elapsed >= self.distraction_grace_seconds()
        )

        return {
            "mode": self.effective_mode(),
            "configured_mode": self.configured_mode(),
            "focus_state": "unknown" if self._privacy_paused else self._focus_state,
            "app_class": None if self._privacy_paused else self._app_class,
            "window_title": None if self._privacy_paused else self._window_title,
            "classifier": None if self._privacy_paused else self._classifier,
            "window_started_at": (
                self._window_started_at.isoformat()
                if self._window_started_at is not None and not self._privacy_paused
                else None
            ),
            "window_elapsed_seconds": elapsed,
            "distraction_grace_seconds": self.distraction_grace_seconds(),
            "distraction_grace_exceeded": grace_exceeded,
            "idle": False if self._privacy_paused else self._idle,
            "idle_since": (
                self._idle_since.isoformat()
                if self._idle_since is not None and not self._privacy_paused
                else None
            ),
            "calendar_event_id": event_id,
            "calendar_task_type": task_type,
            "is_productivity_block": is_prod,
            "is_call_window": False if self._privacy_paused else self._is_call_window,
            "privacy_paused": self._privacy_paused,
            "dnd_mode": self._dnd_mode,
            "observed_window_events": self._observed_window_events,
        }

    async def get_focus_summary(self, now: datetime | None = None) -> dict[str, Any]:
        """Return structured focus summary with call window, app, and durations."""
        ref_now = ensure_aware(now) if now is not None else self._now()
        cur_evt = await self._resolve_current_calendar_event(ref_now)
        cal_task_type = (
            str(cur_evt["task_type"]).strip().lower()
            if cur_evt is not None and cur_evt.get("task_type")
            else None
        )

        per_category_durations: dict[str, float] = {
            "focused": 0.0,
            "neutral": 0.0,
            "distracted": 0.0,
            "unknown": 0.0,
        }

        if self._db_connected() and not self._privacy_paused:
            spans = await self.db.list_window_spans(limit=500)
            for span in spans:
                cat = str(span.get("category") or "unknown").strip().lower()
                normalized_cat = _FOCUS_CATEGORY_ALIASES.get(cat, "unknown")
                start_dt = _parse_dt(span.get("started_at"))
                end_dt = _parse_dt(span.get("ended_at"))
                if start_dt is not None:
                    effective_end = end_dt if end_dt is not None else ref_now
                    duration = max(0.0, (effective_end - start_dt).total_seconds())
                    per_category_durations[normalized_cat] = (
                        per_category_durations.get(normalized_cat, 0.0) + duration
                    )

        focus_state = "unknown" if self._privacy_paused else self._focus_state
        app_class = None if self._privacy_paused else self._app_class
        is_call = False if self._privacy_paused else self._is_call_window

        return {
            "focus_state": focus_state,
            "current_focus_state": focus_state,
            "is_call_window": is_call,
            "app_class": app_class,
            "active_app_class": app_class,
            "window_title": None if self._privacy_paused else self._window_title,
            "per_category_durations": per_category_durations,
            "category_durations": per_category_durations,
            "calendar_task_type": cal_task_type,
            "idle": False if self._privacy_paused else self._idle,
            "mode": self.effective_mode(),
        }

    async def _on_window_focus_changed(self, event: PlatformEvent) -> None:
        if self._privacy_paused:
            return

        payload = event.payload
        event_time = self._event_now(event)
        self._last_event_at = event_time

        app_class = str(payload.get("app_class") or "")
        window_title = str(
            payload.get("window_title")
            if payload.get("window_title") is not None
            else (payload.get("title") or "")
        )

        current_event = await self._resolve_current_calendar_event(event_time)
        task_type = (
            str(current_event["task_type"])
            if current_event is not None and current_event.get("task_type")
            else None
        )

        raw_category = payload.get("category") or payload.get("focus")
        normalized_cat = (
            _FOCUS_CATEGORY_ALIASES.get(str(raw_category).strip().lower())
            if isinstance(raw_category, str) and raw_category.strip()
            else None
        )

        computed_cat, computed_classifier = classify_focus_context(
            app_class,
            window_title,
            work_app_classes=self.work_app_classes(),
            distraction_title_patterns=self.distraction_title_patterns(),
            calendar_task_type=task_type,
        )

        if normalized_cat is not None and normalized_cat != "unknown":
            self._focus_state = normalized_cat
            self._classifier = str(payload.get("classifier") or computed_classifier)
        else:
            self._focus_state = computed_cat
            self._classifier = str(payload.get("classifier") or computed_classifier)

        self._app_class = app_class
        self._window_title = window_title
        self._window_started_at = event_time
        self._observed_window_events += 1

        if payload.get("is_call_window") is not None:
            self._is_call_window = bool(payload["is_call_window"])
        elif isinstance(payload.get("metadata"), dict) and "is_call_window" in payload["metadata"]:
            self._is_call_window = bool(payload["metadata"]["is_call_window"])
        elif app_class or window_title:
            self._is_call_window = is_call_window(app_class, window_title)
        else:
            self._is_call_window = False

        logger.debug(
            "Observed window focus [mode=observe_only]: app_class=%r category=%s classifier=%s",
            self._app_class,
            self._focus_state,
            self._classifier,
        )
        await self._touch_health(event_time)

    async def _on_input_idle_changed(self, event: PlatformEvent) -> None:
        if self._privacy_paused:
            return

        payload = event.payload
        raw_idle = payload.get("idle")
        if raw_idle is None:
            raw_idle = payload.get("is_idle")
        if raw_idle is None:
            return

        event_time = self._event_now(event)
        self._last_event_at = event_time

        if bool(raw_idle):
            parsed_since = _parse_dt(payload.get("idle_since"))
            self._idle = True
            self._idle_since = parsed_since or self._idle_since or event_time
        else:
            self._idle = False
            self._idle_since = None

        logger.debug(
            "Observed input idle [mode=observe_only]: idle=%s idle_since=%s",
            self._idle,
            self._idle_since.isoformat() if self._idle_since else None,
        )
        await self._touch_health(event_time)

    async def _on_calendar_context_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._last_event_at = event_time
        self._update_cached_calendar(event.payload, event_time)

        if (
            not self._privacy_paused
            and self._app_class is not None
            and self._window_title is not None
            and self._classifier in ("calendar_task", "calendar_correlation", "default")
        ):
            cur_evt = await self._resolve_current_calendar_event(event_time)
            task_type = (
                str(cur_evt["task_type"])
                if cur_evt is not None and cur_evt.get("task_type")
                else None
            )
            re_cat, re_cls = classify_focus_context(
                self._app_class,
                self._window_title,
                work_app_classes=self.work_app_classes(),
                distraction_title_patterns=self.distraction_title_patterns(),
                calendar_task_type=task_type,
            )
            self._focus_state = re_cat
            self._classifier = re_cls

        await self._touch_health(event_time)

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
            self._clear_observed_context()

    def _clear_observed_context(self) -> None:
        self._focus_state = "unknown"
        self._app_class = None
        self._window_title = None
        self._classifier = None
        self._window_started_at = None
        self._is_call_window = False
        self._idle = False
        self._idle_since = None

    async def _touch_health(self, event_time: datetime) -> None:
        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "ok",
                last_event_at=event_time,
                last_error=None,
            )

    async def _resolve_current_calendar_event(
        self,
        now: datetime,
    ) -> dict[str, Any] | None:
        if self._db_connected():
            cur_evt, _ = await self.db.get_current_and_next_events(now=now)
            if cur_evt is not None:
                return cur_evt

        candidate = self._cached_current_event
        if candidate is None:
            return None
        start_dt = _parse_dt(candidate.get("start_time"))
        end_dt = _parse_dt(candidate.get("end_time"))
        if start_dt is not None and end_dt is not None:
            return candidate if start_dt <= now < end_dt else None
        return candidate

    def _update_cached_calendar(
        self,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> None:
        if "current_event" in payload:
            cur = payload.get("current_event")
            self._cached_current_event = cur if isinstance(cur, dict) else None
        if "next_event" in payload:
            nxt = payload.get("next_event")
            self._cached_next_event = nxt if isinstance(nxt, dict) else None

        candidate: dict[str, Any] | None = None
        if isinstance(payload.get("event"), dict):
            candidate = payload["event"]
        elif "event_id" in payload and "task_type" in payload:
            candidate = dict(payload)

        if candidate is not None and "current_event" not in payload:
            start_dt = _parse_dt(candidate.get("start_time"))
            if start_dt is not None and start_dt > now:
                self._cached_next_event = candidate
            else:
                self._cached_current_event = candidate

    def _now(self) -> datetime:
        return ensure_aware(self._clock())

    def _event_now(self, event: PlatformEvent) -> datetime:
        if self._custom_clock and "timestamp" not in event.model_fields_set:
            return self._now()
        return ensure_aware(event.timestamp)

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    def _get_focus_value(self, key: str) -> Any:
        direct = self.config.get(key)
        if direct is not None:
            return direct
        nested = self.config.get("focus")
        if isinstance(nested, Mapping) and nested.get(key) is not None:
            return nested[key]
        return None

    @staticmethod
    def _extract_privacy_paused(config: Mapping[str, Any]) -> bool:
        for key in ("global_pause", "privacy_paused"):
            if key in config and config[key] is not None:
                return bool(config[key])
        privacy_cfg = config.get("privacy")
        if isinstance(privacy_cfg, Mapping) and privacy_cfg.get("global_pause") is not None:
            return bool(privacy_cfg["global_pause"])
        return False

    @staticmethod
    def _extract_dnd_mode(config: Mapping[str, Any]) -> bool:
        if "dnd_mode" in config and config["dnd_mode"] is not None:
            return bool(config["dnd_mode"])
        privacy_cfg = config.get("privacy")
        if isinstance(privacy_cfg, Mapping) and privacy_cfg.get("dnd_mode") is not None:
            return bool(privacy_cfg["dnd_mode"])
        return False

    def _sync_privacy_in_config(self, key: str, value: bool) -> None:
        privacy_cfg = self.config.get("privacy")
        if isinstance(privacy_cfg, dict):
            privacy_cfg[key] = value
        if key in self.config:
            self.config[key] = value
