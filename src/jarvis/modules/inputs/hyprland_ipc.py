"""Hyprland IPC active-window and input-idle observer module."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import FocusState, PlatformEvent, ensure_aware, now_local
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_IDLE_THRESHOLD_SECONDS = 300.0
DEFAULT_IDLE_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RECONNECT_INITIAL_SECONDS = 1.0
DEFAULT_RECONNECT_MAX_SECONDS = 30.0
DEFAULT_RECONNECT_MULTIPLIER = 2.0
DEFAULT_HYPRCTL_BINARY = "hyprctl"
DEFAULT_HYPRCTL_TIMEOUT_SECONDS = 3.0

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

_VALID_FOCUS_STATES: frozenset[str] = frozenset(
    {"focused", "neutral", "distracted", "unknown"}
)

HyprctlRunnerResult = (
    dict[str, Any]
    | str
    | bytes
    | tuple[int, str | bytes]
    | tuple[int, str | bytes, str | bytes]
    | None
)
HyprctlRunner = Callable[..., Awaitable[HyprctlRunnerResult] | HyprctlRunnerResult]
SocketConnectionResult = (
    tuple[asyncio.StreamReader, asyncio.StreamWriter | None]
    | asyncio.StreamReader
    | AsyncIterable[str | bytes]
)
SocketConnector = Callable[[], Awaitable[SocketConnectionResult] | SocketConnectionResult]
IdleMonitorSample = dict[str, Any] | tuple[bool, float] | float | bool | None
IdleMonitor = (
    Callable[[], Awaitable[IdleMonitorSample] | IdleMonitorSample]
    | AsyncIterable[IdleMonitorSample]
)


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


def resolve_hyprland_socket2_path(
    *,
    socket_path: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve Hyprland `.socket2.sock` path from overrides, XDG_RUNTIME_DIR, or /tmp."""
    if socket_path is not None and str(socket_path).strip():
        return Path(str(socket_path)).expanduser()

    if config is not None:
        direct = config.get("socket_path")
        if direct is not None and str(direct).strip():
            return Path(str(direct)).expanduser()
        for section_key in ("hyprland_ipc", "focus"):
            nested = config.get(section_key)
            if isinstance(nested, Mapping):
                nested_path = nested.get("socket_path")
                if nested_path is not None and str(nested_path).strip():
                    return Path(str(nested_path)).expanduser()

    active_env = env if env is not None else os.environ
    signature = active_env.get("HYPRLAND_INSTANCE_SIGNATURE", "").strip()
    if not signature:
        return None

    candidates: list[Path] = []
    xdg_runtime = active_env.get("XDG_RUNTIME_DIR", "").strip()
    if xdg_runtime:
        candidates.append(Path(xdg_runtime) / "hypr" / signature / ".socket2.sock")
    candidates.append(Path("/tmp") / "hypr" / signature / ".socket2.sock")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_socket2_line(raw_line: str | bytes) -> tuple[str, str] | None:
    """Parse a Hyprland `.socket2.sock` line, returning `(app_class, window_title)`."""
    decoded = (
        raw_line.decode("utf-8", errors="replace")
        if isinstance(raw_line, bytes)
        else raw_line
    )
    line = decoded.rstrip("\r\n")
    if ">>" not in line:
        return None

    event_name, raw_payload = line.split(">>", 1)
    if event_name != "activewindow":
        return None

    if raw_payload == "":
        return ("", "")
    if "," in raw_payload:
        app_class, window_title = raw_payload.split(",", 1)
        return (app_class, window_title)
    return (raw_payload, "")


def classify_window(
    app_class: str,
    window_title: str,
    *,
    work_app_classes: Sequence[str] = DEFAULT_WORK_APP_CLASSES,
    distraction_title_patterns: Sequence[str] = DEFAULT_DISTRACTION_TITLE_PATTERNS,
    distraction_app_classes: Sequence[str] = (),
    neutral_app_classes: Sequence[str] = (),
    work_title_patterns: Sequence[str] = (),
    neutral_title_patterns: Sequence[str] = (),
    calendar_task_type: str | None = None,
    calendar_task_focus_map: Mapping[str, FocusState] | None = None,
) -> tuple[FocusState, str]:
    """Classify a window into `(FocusState, classifier)` per SPEC.MD FR-09.1 precedence."""
    clean_class = app_class.strip()
    clean_title = window_title.strip()
    if not clean_class and not clean_title:
        return ("neutral", "desktop")

    # 1. App-class rules
    lowered_class = clean_class.lower()
    if lowered_class:
        if any(lowered_class == str(c).strip().lower() for c in work_app_classes if str(c).strip()):
            return ("focused", "app_class")
        if any(
            lowered_class == str(c).strip().lower()
            for c in distraction_app_classes
            if str(c).strip()
        ):
            return ("distracted", "app_class")
        if any(
            lowered_class == str(c).strip().lower()
            for c in neutral_app_classes
            if str(c).strip()
        ):
            return ("neutral", "app_class")

    # 2. Title rules
    if clean_title:
        if any(_title_matches_pattern(clean_title, str(p)) for p in distraction_title_patterns):
            return ("distracted", "title_pattern")
        if any(_title_matches_pattern(clean_title, str(p)) for p in work_title_patterns):
            return ("focused", "title_pattern")
        if any(_title_matches_pattern(clean_title, str(p)) for p in neutral_title_patterns):
            return ("neutral", "title_pattern")

    # 3. Calendar-task mappings
    task_map = (
        calendar_task_focus_map
        if calendar_task_focus_map is not None
        else DEFAULT_CALENDAR_TASK_FOCUS_MAP
    )
    if calendar_task_type:
        mapped = task_map.get(calendar_task_type.strip().lower())
        if mapped in _VALID_FOCUS_STATES:
            return (mapped, "calendar_task")

    # 4. Fallback
    return ("unknown", "default")


async def _iter_stream_reader(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while True:
        chunk = await reader.readline()
        if not chunk:
            break
        yield chunk


class HyprlandIpcModule(BaseModule):
    """Observe Hyprland active windows via `.socket2.sock` and idle state via `hypridle`."""

    name: str = "hyprland_ipc"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any] | None = None,
        *,
        socket_path: Path | str | None = None,
        stream_reader: asyncio.StreamReader | AsyncIterable[str | bytes] | None = None,
        connect_fn: SocketConnector | None = None,
        hyprctl_binary: str | None = None,
        hyprctl_runner: HyprctlRunner | None = None,
        seed_active_window: bool | None = None,
        idle_threshold_seconds: float | None = None,
        idle_monitor: IdleMonitor | None = None,
        idle_poll_interval_seconds: float = DEFAULT_IDLE_POLL_INTERVAL_SECONDS,
        reconnect_initial_seconds: float | None = None,
        reconnect_max_seconds: float | None = None,
        reconnect_multiplier: float = DEFAULT_RECONNECT_MULTIPLIER,
        subprocess_timeout_seconds: float = DEFAULT_HYPRCTL_TIMEOUT_SECONDS,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=dict(config) if config else {})
        if idle_threshold_seconds is not None and idle_threshold_seconds < 0:
            raise ValueError("idle_threshold_seconds must be non-negative")
        if idle_poll_interval_seconds <= 0:
            raise ValueError("idle_poll_interval_seconds must be positive")
        if reconnect_initial_seconds is not None and reconnect_initial_seconds <= 0:
            raise ValueError("reconnect_initial_seconds must be positive")
        if reconnect_max_seconds is not None and reconnect_max_seconds <= 0:
            raise ValueError("reconnect_max_seconds must be positive")
        if reconnect_multiplier < 1.0:
            raise ValueError("reconnect_multiplier must be >= 1.0")
        if subprocess_timeout_seconds <= 0:
            raise ValueError("subprocess_timeout_seconds must be positive")

        self._injected_socket_path = socket_path
        self._injected_stream_reader = stream_reader
        self._injected_connect_fn = connect_fn
        self._injected_hyprctl_binary = hyprctl_binary
        self._injected_hyprctl_runner = hyprctl_runner
        self._injected_seed_active_window = seed_active_window
        self._injected_idle_threshold = idle_threshold_seconds
        self._idle_monitor = idle_monitor
        self._idle_poll_interval_seconds = float(idle_poll_interval_seconds)
        self._injected_reconnect_initial = reconnect_initial_seconds
        self._injected_reconnect_max = reconnect_max_seconds
        self._reconnect_multiplier = float(reconnect_multiplier)
        self._subprocess_timeout_seconds = float(subprocess_timeout_seconds)
        self._env = env

        self._custom_clock = clock is not None
        self._clock: Callable[[], datetime] = clock if clock is not None else now_local

        self._privacy_paused: bool = self._extract_privacy_paused(self.config)
        self._dnd_mode: bool = self._extract_dnd_mode(self.config)

        self._last_window: tuple[str, str] | None = None
        self._cached_current_event: dict[str, Any] | None = None

        self._idle: bool = False
        self._idle_since: datetime | None = None
        self._away_emitted: bool = False

        self._window_lock = asyncio.Lock()
        self._active_writer: asyncio.StreamWriter | None = None
        self._socket_task: asyncio.Task[None] | None = None
        self._idle_timer_task: asyncio.Task[None] | None = None
        self._idle_monitor_task: asyncio.Task[None] | None = None

        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None
        self._degraded_reported: bool = False

    @property
    def is_idle(self) -> bool:
        """Return True when input is currently tracked as idle."""
        return self._idle

    @property
    def idle_since(self) -> datetime | None:
        """Return timestamp when current idle period started, or None if active."""
        return self._idle_since

    async def start(self) -> None:
        """Subscribe to bus events, optionally seed active window, and start IPC listener."""
        await self._cancel_background_tasks()
        await self._close_active_writer()

        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected():
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if isinstance(persisted_pause, bool):
                self._privacy_paused = self._privacy_paused or persisted_pause
            persisted_dnd = await self.db.get_setting("privacy.dnd_mode")
            if isinstance(persisted_dnd, bool):
                self._dnd_mode = self._dnd_mode or persisted_dnd

        self.bus.subscribe("input_idle_changed", self._on_input_idle_changed)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)

        self._running = True
        self._last_error = None
        self._degraded_reported = False
        self._last_window = None
        self._idle = False
        self._idle_since = None
        self._away_emitted = False

        now = self._now()
        initial_conn: tuple[AsyncIterable[str | bytes], asyncio.StreamWriter | None, bool] | None
        try:
            initial_conn = await self._open_stream_once()
        except Exception as exc:
            initial_conn = None
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Hyprland IPC unavailable on start: %s", error_msg)
            await self._record_failure(error_msg, event_time=now)

        if initial_conn is not None:
            self._active_writer = initial_conn[1]
            if self._should_seed_active_window() and not self._privacy_paused:
                await self._seed_from_hyprctl(now=now)
            if self._last_error is None:
                await self._record_success(event_time=now)

        self._socket_task = asyncio.create_task(
            self._socket_loop(initial_conn),
            name=f"{self.name}_socket_loop",
        )

        if self._idle_monitor is not None:
            self._idle_monitor_task = asyncio.create_task(
                self._run_idle_monitor(),
                name=f"{self.name}_idle_monitor",
            )

    async def stop(self) -> None:
        """Stop IPC/idle background tasks, unsubscribe, and close any open window span."""
        self._running = False
        self.bus.unsubscribe("input_idle_changed", self._on_input_idle_changed)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)

        await self._cancel_background_tasks()
        await self._close_active_writer()

        async with self._window_lock:
            self._last_window = None
            if self._db_connected():
                await self.db.close_current_window_span(ended_at=self._now())

        self._idle = False
        self._idle_since = None
        self._away_emitted = False

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Reload module configuration and apply updated privacy pause state."""
        await super().reload_config(dict(config))
        was_paused = self._privacy_paused
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)
        if self._privacy_paused and not was_paused:
            await self._apply_privacy_pause(event_time=self._now())

    async def health(self) -> dict[str, Any]:
        """Return current module health payload."""
        status = (
            "degraded"
            if self._last_error is not None
            else ("ok" if self._running else "disabled")
        )
        resolved_socket = self.resolve_socket_path()
        return {
            "module": self.name,
            "status": status,
            "socket_path": str(resolved_socket) if resolved_socket is not None else None,
            "idle": self._idle,
            "idle_since": (
                self._idle_since.isoformat() if self._idle_since is not None else None
            ),
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
            "last_error": self._last_error,
        }

    def resolve_socket_path(self) -> Path | None:
        """Return resolved Hyprland `.socket2.sock` path or None if unavailable."""
        return resolve_hyprland_socket2_path(
            socket_path=self._injected_socket_path,
            config=self.config,
            env=self._env,
        )

    def hyprctl_binary(self) -> str:
        """Return configured `hyprctl` executable name or path."""
        if self._injected_hyprctl_binary is not None:
            return self._injected_hyprctl_binary
        val = self._get_config_value("hyprctl_binary")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return DEFAULT_HYPRCTL_BINARY

    def idle_threshold_seconds(self) -> float:
        """Return idle duration threshold (seconds) before reporting `away`."""
        if self._injected_idle_threshold is not None:
            return float(self._injected_idle_threshold)
        raw = self._get_config_value("idle_threshold_seconds")
        if raw is not None:
            return max(0.0, float(raw))
        return DEFAULT_IDLE_THRESHOLD_SECONDS

    def reconnect_initial_seconds(self) -> float:
        """Return initial reconnect backoff delay in seconds."""
        if self._injected_reconnect_initial is not None:
            return float(self._injected_reconnect_initial)
        raw = self._get_config_value("reconnect_initial_seconds")
        if raw is not None and float(raw) > 0:
            return float(raw)
        return DEFAULT_RECONNECT_INITIAL_SECONDS

    def reconnect_max_seconds(self) -> float:
        """Return maximum bounded reconnect backoff delay in seconds."""
        if self._injected_reconnect_max is not None:
            return float(self._injected_reconnect_max)
        raw = self._get_config_value("reconnect_max_seconds")
        if raw is not None and float(raw) > 0:
            return float(raw)
        return DEFAULT_RECONNECT_MAX_SECONDS

    def work_app_classes(self) -> list[str]:
        """Return configured productive application class names."""
        raw = self._get_config_value("work_app_classes")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [str(item) for item in raw]
        return list(DEFAULT_WORK_APP_CLASSES)

    def distraction_title_patterns(self) -> list[str]:
        """Return configured distraction window title patterns."""
        raw = self._get_config_value("distraction_title_patterns")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [str(item) for item in raw]
        return list(DEFAULT_DISTRACTION_TITLE_PATTERNS)

    def classify(
        self,
        app_class: str,
        window_title: str,
        *,
        calendar_task_type: str | None = None,
    ) -> tuple[FocusState, str]:
        """Classify `(app_class, window_title)` using configured FocusSettings rules."""
        distraction_classes = self._get_str_list("distraction_app_classes")
        neutral_classes = self._get_str_list("neutral_app_classes")
        work_titles = self._get_str_list("work_title_patterns")
        neutral_titles = self._get_str_list("neutral_title_patterns")

        raw_task_map = self._get_config_value("calendar_task_focus_map")
        task_map: dict[str, FocusState] | None = None
        if isinstance(raw_task_map, Mapping):
            task_map = {
                str(k).strip().lower(): v
                for k, v in raw_task_map.items()
                if v in _VALID_FOCUS_STATES
            }

        return classify_window(
            app_class,
            window_title,
            work_app_classes=self.work_app_classes(),
            distraction_title_patterns=self.distraction_title_patterns(),
            distraction_app_classes=distraction_classes,
            neutral_app_classes=neutral_classes,
            work_title_patterns=work_titles,
            neutral_title_patterns=neutral_titles,
            calendar_task_type=calendar_task_type,
            calendar_task_focus_map=task_map,
        )

    async def handle_raw_line(
        self,
        raw_line: str | bytes,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Parse and process a single `.socket2.sock` line."""
        parsed = parse_socket2_line(raw_line)
        if parsed is None:
            return None
        app_class, window_title = parsed
        event_time = ensure_aware(now) if now is not None else self._now()
        return await self._process_active_window(
            app_class=app_class,
            window_title=window_title,
            event_time=event_time,
        )

    async def evaluate_idle(self, now: datetime | None = None) -> PlatformEvent | None:
        """Check whether current `idle=True` state has exceeded threshold and emit `away`."""
        if self._privacy_paused or not self._idle or self._idle_since is None:
            return None

        ref_now = ensure_aware(now) if now is not None else self._now()
        elapsed_idle = max(0.0, (ref_now - self._idle_since).total_seconds())
        if elapsed_idle < self.idle_threshold_seconds() or self._away_emitted:
            return None

        await self._cancel_idle_timer()
        return await self._emit_away_activity(
            now=ref_now,
            idle_seconds=elapsed_idle,
            idle_since=self._idle_since,
        )

    async def _process_active_window(
        self,
        *,
        app_class: str,
        window_title: str,
        event_time: datetime,
    ) -> dict[str, Any] | None:
        async with self._window_lock:
            if self._privacy_paused:
                return None

            pair = (app_class, window_title)
            if self._last_window == pair:
                return None

            self._last_window = pair
            self._last_event_at = event_time

            current_event = await self._resolve_current_calendar_event(event_time)
            calendar_event_id = (
                str(current_event["event_id"])
                if current_event is not None and current_event.get("event_id")
                else None
            )
            calendar_task_type = (
                str(current_event["task_type"])
                if current_event is not None and current_event.get("task_type")
                else None
            )

            # Empty / desktop focus: close open window span without inserting a blank row
            if not app_class.strip() and not window_title.strip():
                if self._db_connected():
                    await self.db.close_current_window_span(ended_at=event_time)
                await self.bus.publish(
                    PlatformEvent(
                        type="window_focus_changed",
                        source=self.name,
                        confidence=1.0,
                        timestamp=event_time,
                        payload={
                            "app_class": "",
                            "title": "",
                            "window_title": "",
                            "category": "neutral",
                            "focus": "neutral",
                            "classifier": "desktop",
                            "calendar_event_id": calendar_event_id,
                            "span_id": None,
                        },
                    )
                )
                return None

            category, classifier = self.classify(
                app_class,
                window_title,
                calendar_task_type=calendar_task_type,
            )

            span_row: dict[str, Any] | None = None
            span_id: int | None = None
            if self._db_connected():
                span_row = await self.db.open_window_span(
                    app_class=app_class,
                    window_title=window_title,
                    category=category,
                    classifier=classifier,
                    calendar_event_id=calendar_event_id,
                    started_at=event_time,
                )
                span_id = int(span_row["id"])
                if not self._degraded_reported:
                    await self.db.upsert_module_health(
                        self.name,
                        "ok",
                        last_event_at=event_time,
                        last_error=None,
                    )

            await self.bus.publish(
                PlatformEvent(
                    type="window_focus_changed",
                    source=self.name,
                    confidence=1.0,
                    timestamp=event_time,
                    payload={
                        "app_class": app_class,
                        "title": window_title,
                        "window_title": window_title,
                        "category": category,
                        "focus": category,
                        "classifier": classifier,
                        "calendar_event_id": calendar_event_id,
                        "span_id": span_id,
                    },
                )
            )
            return span_row

    async def _on_input_idle_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        await self._handle_idle_payload(event.payload, event_time=event_time)

    async def _handle_idle_payload(
        self,
        payload: Mapping[str, Any],
        *,
        event_time: datetime,
    ) -> PlatformEvent | None:
        if self._privacy_paused:
            self._idle = False
            self._idle_since = None
            self._away_emitted = False
            await self._cancel_idle_timer()
            return None

        raw_idle = payload.get("idle")
        if raw_idle is None:
            raw_idle = payload.get("is_idle")
        if raw_idle is None:
            return None

        self._last_event_at = event_time

        if not bool(raw_idle):
            self._idle = False
            self._idle_since = None
            self._away_emitted = False
            await self._cancel_idle_timer()
            return None

        idle_since_dt = _parse_dt(payload.get("idle_since"))
        if idle_since_dt is None:
            for dur_key in (
                "idle_seconds",
                "idle_duration_seconds",
                "duration_seconds",
                "elapsed_seconds",
            ):
                raw_dur = payload.get(dur_key)
                if raw_dur is not None:
                    idle_since_dt = event_time - timedelta(
                        seconds=max(0.0, float(raw_dur))
                    )
                    break

        if idle_since_dt is None:
            if self._idle and self._idle_since is not None:
                idle_since_dt = self._idle_since
            else:
                idle_since_dt = event_time

        self._idle = True
        self._idle_since = idle_since_dt

        elapsed_idle = max(0.0, (event_time - idle_since_dt).total_seconds())
        threshold = self.idle_threshold_seconds()

        if elapsed_idle >= threshold:
            await self._cancel_idle_timer()
            return await self._emit_away_activity(
                now=event_time,
                idle_seconds=elapsed_idle,
                idle_since=idle_since_dt,
            )

        remaining = threshold - elapsed_idle
        await self._schedule_idle_timer(remaining)
        return None

    async def _emit_away_activity(
        self,
        *,
        now: datetime,
        idle_seconds: float,
        idle_since: datetime,
    ) -> PlatformEvent:
        self._away_emitted = True
        event = PlatformEvent(
            type="activity_reported",
            source="hyprland_idle",
            confidence=0.7,
            timestamp=now,
            payload={
                "activity": "away",
                "idle_seconds": round(idle_seconds, 3),
                "idle_since": idle_since.isoformat(),
            },
        )
        await self.bus.publish(event)
        return event

    async def _schedule_idle_timer(self, delay_seconds: float) -> None:
        await self._cancel_idle_timer()
        self._idle_timer_task = asyncio.create_task(
            self._idle_timer_wait(max(0.0, delay_seconds)),
            name=f"{self.name}_idle_threshold_timer",
        )

    async def _idle_timer_wait(self, delay_seconds: float) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            if (
                not self._running
                or self._privacy_paused
                or not self._idle
                or self._idle_since is None
                or self._away_emitted
            ):
                return
            now = self._now()
            elapsed_idle = max(
                self.idle_threshold_seconds(),
                (now - self._idle_since).total_seconds(),
            )
            await self._emit_away_activity(
                now=now,
                idle_seconds=elapsed_idle,
                idle_since=self._idle_since,
            )
        except asyncio.CancelledError:
            raise

    async def _run_idle_monitor(self) -> None:
        monitor = self._idle_monitor
        if monitor is None:
            return
        try:
            if isinstance(monitor, AsyncIterable):
                async for sample in monitor:
                    if not self._running:
                        break
                    await self._ingest_idle_monitor_sample(sample)
                return

            while self._running:
                sample_res = monitor()
                if inspect.isawaitable(sample_res):
                    sample_res = await sample_res
                await self._ingest_idle_monitor_sample(sample_res)
                await asyncio.sleep(self._idle_poll_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_msg = f"IdleMonitor {type(exc).__name__}: {exc}"
            logger.warning("Hyprland idle monitor failed: %s", error_msg)
            await self._record_failure(error_msg, event_time=self._now())

    async def _ingest_idle_monitor_sample(self, sample: IdleMonitorSample) -> None:
        if sample is None:
            return
        now = self._now()
        if isinstance(sample, Mapping):
            await self._handle_idle_payload(sample, event_time=now)
            return
        if isinstance(sample, tuple) and len(sample) == 2:
            is_idle, secs = sample
            await self._handle_idle_payload(
                {"idle": bool(is_idle), "idle_seconds": float(secs)},
                event_time=now,
            )
            return
        if isinstance(sample, bool):
            await self._handle_idle_payload({"idle": sample}, event_time=now)
            return
        if isinstance(sample, (int, float)):
            secs_val = float(sample)
            is_idle = secs_val > 0
            await self._handle_idle_payload(
                {"idle": is_idle, "idle_seconds": max(0.0, secs_val)},
                event_time=now,
            )

    async def _on_privacy_mode_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        event_time = self._event_now(event)
        was_paused = self._privacy_paused

        for key in ("global_pause", "privacy_paused", "paused"):
            if key in payload and payload[key] is not None:
                self._privacy_paused = bool(payload[key])
                self._sync_privacy_in_config("global_pause", self._privacy_paused)
                break

        if "dnd_mode" in payload and payload["dnd_mode"] is not None:
            self._dnd_mode = bool(payload["dnd_mode"])
            self._sync_privacy_in_config("dnd_mode", self._dnd_mode)

        if self._privacy_paused and not was_paused:
            await self._apply_privacy_pause(event_time=event_time)
        elif self._privacy_paused:
            self._last_window = None

    async def _apply_privacy_pause(self, *, event_time: datetime) -> None:
        async with self._window_lock:
            self._last_window = None
            if self._db_connected():
                await self.db.close_current_window_span(ended_at=event_time)

        self._idle = False
        self._idle_since = None
        self._away_emitted = False
        await self._cancel_idle_timer()

    def _on_calendar_context_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        now = self._event_now(event)
        if "current_event" in payload:
            cur = payload.get("current_event")
            self._cached_current_event = cur if isinstance(cur, dict) else None
            return

        candidate: dict[str, Any] | None = None
        if isinstance(payload.get("event"), dict):
            candidate = payload["event"]
        elif "event_id" in payload and "task_type" in payload:
            candidate = dict(payload)

        if candidate is not None:
            start_dt = _parse_dt(candidate.get("start_time"))
            end_dt = _parse_dt(candidate.get("end_time"))
            if start_dt is None or (end_dt is not None and start_dt <= now < end_dt):
                self._cached_current_event = candidate

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

    def _should_seed_active_window(self) -> bool:
        if self._injected_seed_active_window is not None:
            return bool(self._injected_seed_active_window)
        cfg_flag = self._get_config_value("seed_active_window")
        if cfg_flag is not None:
            return bool(cfg_flag)
        if (
            self._injected_hyprctl_runner is not None
            or self.config.get("hyprctl_runner") is not None
            or self._injected_hyprctl_binary is not None
            or self._get_config_value("hyprctl_binary") is not None
        ):
            return True
        # When an explicit test socket/stream is injected without a custom hyprctl runner,
        # avoid invoking host `/usr/bin/hyprctl` implicitly.
        if (
            self._injected_stream_reader is not None
            or self._injected_connect_fn is not None
            or self._injected_socket_path is not None
            or self.config.get("socket_path") is not None
        ):
            return False
        return True

    async def _seed_from_hyprctl(self, *, now: datetime) -> None:
        try:
            window_data = await self._run_hyprctl_activewindow()
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Failed to seed active window via hyprctl: %s", error_msg)
            await self._record_failure(error_msg, event_time=now)
            return

        if not window_data:
            return

        app_class = str(
            window_data.get("class") or window_data.get("initialClass") or ""
        )
        window_title = str(
            window_data.get("title") or window_data.get("initialTitle") or ""
        )
        if not app_class.strip() and not window_title.strip():
            return

        await self._process_active_window(
            app_class=app_class,
            window_title=window_title,
            event_time=now,
        )

    async def _run_hyprctl_activewindow(self) -> dict[str, Any]:
        argv = [self.hyprctl_binary(), "activewindow", "-j"]
        runner = self._injected_hyprctl_runner or self._get_config_value("hyprctl_runner")
        if callable(runner):
            sig = inspect.signature(runner)
            accepts_args = any(
                p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
                for p in sig.parameters.values()
            )
            raw_res = runner(argv) if accepts_args else runner()
            if inspect.isawaitable(raw_res):
                raw_res = await raw_res
            return self._parse_hyprctl_output(raw_res)

        proc: asyncio.subprocess.Process | None = None
        proc_env = dict(self._env) if self._env is not None else None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._subprocess_timeout_seconds,
            )
        except TimeoutError as exc:
            if proc is not None and proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(Exception):
                    await proc.wait()
            raise TimeoutError(
                f"{argv[0]} timed out after {self._subprocess_timeout_seconds}s"
            ) from exc

        rc = int(proc.returncode if proc.returncode is not None else 1)
        if rc != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            detail = f": {stderr_text}" if stderr_text else ""
            raise RuntimeError(f"{argv[0]} exited with status {rc}{detail}")

        return self._parse_hyprctl_output(stdout_bytes)

    @staticmethod
    def _parse_hyprctl_output(raw: HyprctlRunnerResult) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, tuple):
            rc = int(raw[0])
            stdout_part = raw[1] if len(raw) >= 2 else ""
            stderr_part = raw[2] if len(raw) >= 3 else ""
            if rc != 0:
                err_str = (
                    stderr_part.decode("utf-8", errors="replace").strip()
                    if isinstance(stderr_part, bytes)
                    else str(stderr_part).strip()
                )
                detail = f": {err_str}" if err_str else ""
                raise RuntimeError(f"hyprctl exited with status {rc}{detail}")
            raw = stdout_part

        text = (
            raw.decode("utf-8", errors="replace").strip()
            if isinstance(raw, bytes)
            else str(raw).strip()
        )
        if not text or text.lower() == "invalid":
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    async def _open_stream_once(
        self,
    ) -> tuple[AsyncIterable[str | bytes], asyncio.StreamWriter | None, bool]:
        if self._injected_stream_reader is not None:
            stream = self._injected_stream_reader
            if isinstance(stream, asyncio.StreamReader):
                return (_iter_stream_reader(stream), None, True)
            return (stream, None, True)

        connector = self._injected_connect_fn or self._get_config_value("connect_fn")
        if callable(connector):
            conn_res = connector()
            if inspect.isawaitable(conn_res):
                conn_res = await conn_res
            if isinstance(conn_res, tuple):
                reader, writer = conn_res
                return (_iter_stream_reader(reader), writer, False)
            if isinstance(conn_res, asyncio.StreamReader):
                return (_iter_stream_reader(conn_res), None, False)
            if isinstance(conn_res, AsyncIterable):
                return (conn_res, None, False)
            raise TypeError(f"Unsupported connect_fn return value: {type(conn_res)!r}")

        sock_path = self.resolve_socket_path()
        if sock_path is None:
            raise FileNotFoundError(
                "HYPRLAND_INSTANCE_SIGNATURE is not set and no socket_path was configured"
            )
        if not sock_path.exists():
            raise FileNotFoundError(f"Hyprland event socket not found: {sock_path}")

        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        return (_iter_stream_reader(reader), writer, False)

    async def _socket_loop(
        self,
        initial_conn: (
            tuple[AsyncIterable[str | bytes], asyncio.StreamWriter | None, bool] | None
        ),
    ) -> None:
        try:
            # Yield once so if ModuleRegistry.enable_module() wrote status='ok' right after
            # start(), any startup degradation is immediately re-persisted to SQLite.
            await asyncio.sleep(0)
            if (
                self._degraded_reported
                and self._last_error is not None
                and self._db_connected()
            ):
                await self.db.upsert_module_health(
                    self.name,
                    "degraded",
                    last_event_at=self._last_event_at,
                    last_error=self._last_error,
                )

            delay = self.reconnect_initial_seconds()
            conn = initial_conn

            while self._running:
                if conn is None:
                    await asyncio.sleep(delay)
                    if not self._running:
                        break
                    delay = min(
                        self.reconnect_max_seconds(),
                        delay * self._reconnect_multiplier,
                    )
                    try:
                        conn = await self._open_stream_once()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        logger.debug("Hyprland IPC reconnect failed: %s", error_msg)
                        await self._record_failure(error_msg, event_time=self._now())
                        continue

                    self._active_writer = conn[1]
                    now = self._now()
                    await self._record_success(event_time=now)
                    if self._should_seed_active_window() and not self._privacy_paused:
                        await self._seed_from_hyprctl(now=now)

                line_iter, writer, is_static_stream = conn
                conn = None
                self._active_writer = writer
                delay = self.reconnect_initial_seconds()

                try:
                    async for raw_line in line_iter:
                        if not self._running:
                            break
                        await self.handle_raw_line(raw_line)
                        if self._degraded_reported:
                            await self._record_success(event_time=self._now())
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    logger.warning("Hyprland IPC stream error: %s", error_msg)
                    await self._close_active_writer()
                    if self._running:
                        await self._record_failure(error_msg, event_time=self._now())
                    continue

                await self._close_active_writer()
                if not self._running or is_static_stream:
                    break

                await self._record_failure(
                    "Hyprland IPC socket disconnected (EOF)",
                    event_time=self._now(),
                )
        except asyncio.CancelledError:
            raise
        finally:
            await self._close_active_writer()

    async def _record_failure(self, error_msg: str, *, event_time: datetime) -> None:
        previously_same = self._degraded_reported and self._last_error == error_msg
        self._last_error = error_msg
        self._degraded_reported = True

        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "degraded",
                last_event_at=self._last_event_at,
                last_error=error_msg,
            )
        if not previously_same:
            await self.bus.publish(
                PlatformEvent(
                    type="module_health_changed",
                    source=self.name,
                    timestamp=event_time,
                    payload={
                        "module": self.name,
                        "category": "inputs",
                        "enabled": True,
                        "status": "degraded",
                        "last_error": error_msg,
                    },
                )
            )

    async def _record_success(self, *, event_time: datetime) -> None:
        was_degraded = self._degraded_reported or self._last_error is not None
        self._last_error = None
        self._degraded_reported = False

        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "ok",
                last_event_at=self._last_event_at,
                last_error=None,
            )
        if was_degraded:
            await self.bus.publish(
                PlatformEvent(
                    type="module_health_changed",
                    source=self.name,
                    timestamp=event_time,
                    payload={
                        "module": self.name,
                        "category": "inputs",
                        "enabled": True,
                        "status": "ok",
                        "last_error": None,
                    },
                )
            )

    async def _cancel_background_tasks(self) -> None:
        await self._cancel_idle_timer()

        for attr in ("_idle_monitor_task", "_socket_task"):
            task: asyncio.Task[None] | None = getattr(self, attr)
            setattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _cancel_idle_timer(self) -> None:
        task = self._idle_timer_task
        self._idle_timer_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _close_active_writer(self) -> None:
        writer = self._active_writer
        self._active_writer = None
        if writer is not None:
            with suppress(Exception):
                writer.close()
                await writer.wait_closed()

    def _now(self) -> datetime:
        return ensure_aware(self._clock())

    def _event_now(self, event: PlatformEvent) -> datetime:
        if self._custom_clock and "timestamp" not in event.model_fields_set:
            return self._now()
        return ensure_aware(event.timestamp)

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    def _get_config_value(self, key: str) -> Any:
        direct = self.config.get(key)
        if direct is not None:
            return direct
        for section_key in ("hyprland_ipc", "focus"):
            nested = self.config.get(section_key)
            if isinstance(nested, Mapping) and nested.get(key) is not None:
                return nested[key]
        return None

    def _get_str_list(self, key: str) -> list[str]:
        raw = self._get_config_value(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [str(item) for item in raw]
        return []

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
