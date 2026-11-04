"""Waybar status output module writing atomic JSON to /dev/shm and signaling Waybar."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import ActivityState, PlatformEvent, ensure_aware, now_local
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_WAYBAR_STATUS_FILE = Path("/dev/shm/posture_waybar.json")
DEFAULT_REFRESH_INTERVAL_SECONDS = 15.0
DEFAULT_WAYBAR_SIGNAL = 8
WAYBAR_FILE_ENV_VAR = "POSTURE_WAYBAR_FILE"

_VALID_ACTIVITIES: frozenset[str] = frozenset(
    {"sitting", "standing", "walking", "away", "unknown"}
)

_ACTIVITY_DISPLAY: dict[ActivityState, tuple[str, str, str]] = {
    "sitting": ("󰔄", "SIT", "Sitting"),
    "standing": ("󰋑", "STAND", "Standing"),
    "walking": ("󰗉", "WALK", "Walking"),
    "away": ("󰍃", "AWAY", "Away"),
    "unknown": ("󰘥", "UNKNOWN", "Unknown"),
}

_PAUSE_BADGE = "󰏤 PAUSED"
_DND_BADGE = "󰂛 DND"
_PRESET_RE = re.compile(r"^(\d+)_(\d+)$")

SignalRunner = Callable[[Sequence[str]], Awaitable[Any] | Any]


def _parse_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value:
        with suppress(ValueError):
            return ensure_aware(datetime.fromisoformat(value))
    return None


def _format_timer_kind(kind: str) -> str:
    cleaned = kind.strip()
    if not cleaned:
        return "TIMER"
    match = _PRESET_RE.match(cleaned)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return cleaned.upper()


def _write_json_atomic(target_path: Path, payload: dict[str, str]) -> None:
    """Atomically write single-line UTF-8 JSON via same-directory temp file + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n"

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(serialized)
            tmp_file.flush()
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path is not None:
            with suppress(OSError):
                tmp_path.unlink()
        raise


class WaybarStatusModule(BaseModule):
    """Render live posture, Pomodoro, calendar, and privacy state for Waybar."""

    name: str = "waybar_status"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any] | None = None,
        *,
        status_file: Path | str | None = None,
        refresh_interval_seconds: float | None = None,
        signal_waybar: bool | None = None,
        waybar_signal: int | None = None,
        pkill_binary: str = "pkill",
        waybar_process_name: str = "waybar",
        signal_runner: SignalRunner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=dict(config) if config else {})
        if refresh_interval_seconds is not None and refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be positive")

        self._injected_status_file = (
            Path(status_file).expanduser() if status_file is not None else None
        )
        self._injected_interval = refresh_interval_seconds
        self._injected_signal_waybar = signal_waybar
        self._injected_waybar_signal = waybar_signal
        self._pkill_binary = pkill_binary
        self._waybar_process_name = waybar_process_name
        self._signal_runner = signal_runner

        self._custom_clock = clock is not None
        self._clock: Callable[[], datetime] = clock if clock is not None else now_local

        self._privacy_paused: bool = self._extract_privacy_paused(self.config)
        self._dnd_mode: bool = self._extract_dnd_mode(self.config)

        self._activity: ActivityState = "unknown"
        self._activity_source: str = "system"
        self._activity_reason: str = "startup"
        self._activity_started_at: datetime | None = None
        self._override_expires_at: datetime | None = None

        self._timer_active: bool = False
        self._timer_kind: str | None = None
        self._timer_phase: Literal["work", "break"] = "work"
        self._timer_started_at: datetime | None = None
        self._timer_planned_seconds: int | None = None
        self._timer_remaining_anchor: int | None = None
        self._timer_anchor_at: datetime | None = None

        self._cached_current_event: dict[str, Any] | None = None
        self._cached_next_event: dict[str, Any] | None = None
        self._pending_rec_ids: set[int | str] = set()

        self._write_lock = asyncio.Lock()
        self._loop_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Recover initial state, subscribe to EventBus topics, write JSON, and start loop."""
        await self._cancel_loop()

        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)
        await self._recover_initial_state()

        self.bus.subscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.subscribe("timer_state_changed", self._on_timer_state_changed)
        self.bus.subscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.subscribe("recommendation_created", self._on_recommendation_created)
        self.bus.subscribe("recommendation_resolved", self._on_recommendation_resolved)

        self._running = True
        self._last_error = None
        await self.refresh()

        self._loop_task = asyncio.create_task(
            self._periodic_loop(),
            name=f"{self.name}_refresh_loop",
        )

    async def stop(self) -> None:
        """Unsubscribe from EventBus topics and cancel periodic refresh loop."""
        self._running = False
        self.bus.unsubscribe("activity_state_changed", self._on_activity_state_changed)
        self.bus.unsubscribe("timer_state_changed", self._on_timer_state_changed)
        self.bus.unsubscribe("calendar_context_changed", self._on_calendar_context_changed)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)
        self.bus.unsubscribe("recommendation_created", self._on_recommendation_created)
        self.bus.unsubscribe("recommendation_resolved", self._on_recommendation_resolved)
        await self._cancel_loop()

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Reload module configuration and re-render Waybar status."""
        await super().reload_config(dict(config))
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)
        if self._running:
            await self.refresh()

    async def health(self) -> dict[str, Any]:
        """Return current module health payload."""
        status = (
            "degraded"
            if self._last_error is not None
            else ("ok" if self._running else "disabled")
        )
        return {
            "module": self.name,
            "status": status,
            "status_file": str(self.status_file()),
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
            "last_error": self._last_error,
        }

    def status_file(self) -> Path:
        """Return resolved target path for atomic Waybar JSON output."""
        if self._injected_status_file is not None:
            return self._injected_status_file

        env_override = os.environ.get(WAYBAR_FILE_ENV_VAR, "").strip()
        if env_override:
            return Path(env_override).expanduser()

        for key in ("status_file", "waybar_file", "output_path"):
            direct = self.config.get(key)
            if direct:
                return Path(str(direct)).expanduser()

        for section_key in ("waybar_status", "waybar"):
            section = self.config.get(section_key)
            if isinstance(section, dict):
                for key in ("status_file", "path", "output_path"):
                    nested = section.get(key)
                    if nested:
                        return Path(str(nested)).expanduser()

        return DEFAULT_WAYBAR_STATUS_FILE

    def refresh_interval_seconds(self) -> float:
        """Return periodic Waybar JSON refresh interval in seconds."""
        if self._injected_interval is not None:
            return float(self._injected_interval)
        raw = self._get_waybar_value("refresh_interval_seconds")
        if raw is not None:
            val = float(raw)
            if val > 0:
                return val
        return DEFAULT_REFRESH_INTERVAL_SECONDS

    def signal_waybar_enabled(self) -> bool:
        """Return True when `pkill -RTMIN+<signal> waybar` should run after writes."""
        if self._injected_signal_waybar is not None:
            return bool(self._injected_signal_waybar)
        raw = self._get_waybar_value("signal_waybar")
        if raw is not None:
            return bool(raw)
        return True

    def waybar_signal(self) -> int:
        """Return realtime signal offset configured for Waybar (`SIGRTMIN+N`)."""
        if self._injected_waybar_signal is not None:
            return int(self._injected_waybar_signal)
        raw = self._get_waybar_value("waybar_signal")
        if raw is None:
            raw = self._get_waybar_value("signal")
        return int(raw) if raw is not None else DEFAULT_WAYBAR_SIGNAL

    async def build_payload(self, now: datetime | None = None) -> dict[str, str]:
        """Build the Waybar JSON dictionary (`text`, `tooltip`, `class`) for `now`."""
        ref_now = ensure_aware(now) if now is not None else self._now()

        (
            activity,
            started_at,
            source,
            override_expires_at,
        ) = await self._resolve_activity_context(ref_now)
        icon, short_label, title_label = _ACTIVITY_DISPLAY[activity]

        elapsed_minutes = 0
        if started_at is not None:
            elapsed_seconds = max(0, int((ref_now - started_at).total_seconds()))
            elapsed_minutes = elapsed_seconds // 60

        text_parts: list[str] = [f"{icon} {short_label} {elapsed_minutes}m"]
        tooltip_lines: list[str] = [f"{title_label} ({elapsed_minutes}m)"]

        timer_info = await self._resolve_timer_context(ref_now)
        if timer_info is not None:
            timer_kind, timer_phase, remaining_seconds = timer_info
            kind_label = _format_timer_kind(timer_kind)
            remaining_minutes = (
                max(1, math.ceil(remaining_seconds / 60)) if remaining_seconds > 0 else 0
            )
            if timer_phase == "break":
                text_parts.append(f"{kind_label} BREAK {remaining_minutes}m")
            else:
                text_parts.append(f"{kind_label} {remaining_minutes}m")
            tooltip_lines.append(
                f"Timer: {kind_label} ({timer_phase}, {remaining_minutes}m left)"
            )

        if self._privacy_paused:
            text_parts.append(_PAUSE_BADGE)
        if self._dnd_mode:
            text_parts.append(_DND_BADGE)

        current_event, next_event = await self._resolve_calendar_context(ref_now)
        if current_event is not None and current_event.get("title"):
            tooltip_lines.append(f"Current: {current_event['title']}")
        else:
            tooltip_lines.append("Current: none")

        if next_event is not None and next_event.get("title"):
            next_title = str(next_event["title"])
            next_start = _parse_dt(next_event.get("start_time"))
            if next_start is not None:
                mins_until = max(0, int(round((next_start - ref_now).total_seconds() / 60)))
                tooltip_lines.append(f"Next: {next_title} in {mins_until}m")
            else:
                tooltip_lines.append(f"Next: {next_title}")
        else:
            tooltip_lines.append("Next: none")

        if override_expires_at is not None and override_expires_at > ref_now:
            mins_left = max(
                1, math.ceil((override_expires_at - ref_now).total_seconds() / 60)
            )
            exp_local = override_expires_at.astimezone().strftime("%H:%M")
            tooltip_lines.append(
                f"Manual override: until {exp_local} ({mins_left}m, source: {source})"
            )
        else:
            tooltip_lines.append("Manual override: none")

        pending_count = await self._resolve_pending_recommendation_count()
        tooltip_lines.append(f"Pending recommendations: {pending_count}")

        if self._privacy_paused and self._dnd_mode:
            tooltip_lines.append("Privacy: PAUSED · DND enabled")
        elif self._privacy_paused:
            tooltip_lines.append("Privacy: PAUSED")
        elif self._dnd_mode:
            tooltip_lines.append("Mode: DND enabled")

        if self._privacy_paused:
            css_class = "paused"
        elif self._dnd_mode:
            css_class = "dnd"
        else:
            css_class = activity

        return {
            "text": " · ".join(text_parts),
            "tooltip": "\n".join(tooltip_lines),
            "class": css_class,
        }

    async def refresh(self, now: datetime | None = None) -> dict[str, str]:
        """Compute Waybar JSON, write it atomically, and optionally signal Waybar."""
        async with self._write_lock:
            ref_now = ensure_aware(now) if now is not None else self._now()
            payload = await self.build_payload(now=ref_now)
            target_path = self.status_file()
            try:
                _write_json_atomic(target_path, payload)
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to write Waybar status JSON to %s", target_path)
                if self._db_connected():
                    await self.db.upsert_module_health(
                        self.name,
                        "degraded",
                        last_event_at=ref_now,
                        last_error=self._last_error,
                    )
                raise

            await self._signal_waybar_safe()
            self._last_event_at = ref_now
            if self._db_connected():
                await self.db.upsert_module_health(
                    self.name,
                    "ok",
                    last_event_at=ref_now,
                    last_error=None,
                )
            return payload

    async def _signal_waybar_safe(self) -> None:
        if not self.signal_waybar_enabled():
            return

        argv = [
            self._pkill_binary,
            f"-RTMIN+{self.waybar_signal()}",
            self._waybar_process_name,
        ]
        if self._signal_runner is not None:
            try:
                result = self._signal_runner(argv)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("Custom Waybar signal runner failed", exc_info=True)
            return

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except TimeoutError:
            if proc is not None and proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(Exception):
                    await proc.wait()
            logger.debug("Timed out signaling Waybar with %s", argv)
        except OSError:
            logger.debug("Unable to invoke %s to signal Waybar", argv, exc_info=True)

    async def _recover_initial_state(self) -> None:
        now = self._now()
        if not self._db_connected():
            if self._activity_started_at is None:
                self._activity_started_at = now
            return

        persisted_pause = await self.db.get_setting("privacy.global_pause")
        if isinstance(persisted_pause, bool):
            self._privacy_paused = self._privacy_paused or persisted_pause

        persisted_dnd = await self.db.get_setting("privacy.dnd_mode")
        if isinstance(persisted_dnd, bool):
            self._dnd_mode = self._dnd_mode or persisted_dnd

        current_span = await self.db.get_current_activity_span()
        if current_span is not None:
            self._apply_db_activity_span(current_span, now)
        elif self._activity_started_at is None:
            self._activity_started_at = now

        active_timer = await self.db.get_active_timer()
        if active_timer is not None:
            raw_kind = str(active_timer.get("kind") or "custom")
            is_break = raw_kind.endswith("_break") or raw_kind == "break"
            base_kind = raw_kind.removesuffix("_break") if is_break else raw_kind
            started_dt = _parse_dt(active_timer.get("started_at")) or now
            planned_raw = active_timer.get("planned_duration_seconds")
            planned_sec = int(planned_raw) if planned_raw is not None else None
            self._timer_active = True
            self._timer_kind = base_kind
            self._timer_phase = "break" if is_break else "work"
            self._timer_started_at = started_dt
            self._timer_planned_seconds = planned_sec
            self._timer_remaining_anchor = None
            self._timer_anchor_at = None

        pending_rows = await self.db.get_pending_recommendations()
        self._pending_rec_ids = {int(row["id"]) for row in pending_rows if "id" in row}

    def _apply_db_activity_span(
        self,
        span: dict[str, Any],
        now: datetime,
    ) -> None:
        raw_act = str(span.get("activity", "unknown"))
        self._activity = (
            cast(ActivityState, raw_act) if raw_act in _VALID_ACTIVITIES else "unknown"
        )
        self._activity_source = str(span.get("source", "system"))
        self._activity_reason = str(span.get("reason", "state_change"))
        started_dt = _parse_dt(span.get("started_at")) or now
        self._activity_started_at = started_dt

        expires_at = self._extract_span_override_expiry(span, started_dt, now)
        self._override_expires_at = expires_at

    @staticmethod
    def _extract_span_override_expiry(
        span: dict[str, Any],
        started_at: datetime,
        now: datetime,
    ) -> datetime | None:
        metadata = span.get("metadata")
        reason = str(span.get("reason", ""))
        if not isinstance(metadata, dict):
            return None
        raw_expires = metadata.get("override_expires_at")
        if isinstance(raw_expires, str) and raw_expires:
            parsed = _parse_dt(raw_expires)
            if parsed is not None and parsed > now:
                return parsed
        raw_ttl = metadata.get("ttl_minutes")
        if reason == "manual_override" and raw_ttl is not None:
            candidate = started_at + timedelta(minutes=float(raw_ttl))
            if candidate > now:
                return candidate
        return None

    async def _resolve_activity_context(
        self,
        now: datetime,
    ) -> tuple[ActivityState, datetime | None, str, datetime | None]:
        if self._db_connected():
            db_span = await self.db.get_current_activity_span()
            if db_span is not None:
                db_start = _parse_dt(db_span.get("started_at"))
                if (
                    self._activity_started_at is not None
                    and db_start is not None
                    and self._activity_started_at > db_start
                ):
                    exp = (
                        self._override_expires_at
                        if self._override_expires_at is not None
                        and self._override_expires_at > now
                        else None
                    )
                    return (
                        self._activity,
                        self._activity_started_at,
                        self._activity_source,
                        exp,
                    )

                raw_act = str(db_span.get("activity", "unknown"))
                act: ActivityState = (
                    cast(ActivityState, raw_act)
                    if raw_act in _VALID_ACTIVITIES
                    else "unknown"
                )
                src = str(db_span.get("source", "system"))
                effective_start = db_start or now
                db_exp = self._extract_span_override_expiry(db_span, effective_start, now)
                if (
                    db_exp is None
                    and self._override_expires_at is not None
                    and self._override_expires_at > now
                    and act == self._activity
                ):
                    db_exp = self._override_expires_at
                return act, effective_start, src, db_exp

        exp = (
            self._override_expires_at
            if self._override_expires_at is not None and self._override_expires_at > now
            else None
        )
        return (
            self._activity,
            self._activity_started_at or now,
            self._activity_source,
            exp,
        )

    async def _resolve_timer_context(
        self,
        now: datetime,
    ) -> tuple[str, Literal["work", "break"], int] | None:
        if self._timer_active and self._timer_kind is not None:
            remaining = self._compute_timer_remaining(now)
            return self._timer_kind, self._timer_phase, remaining

        if self._db_connected():
            db_timer = await self.db.get_active_timer()
            if db_timer is not None:
                raw_kind = str(db_timer.get("kind") or "custom")
                is_break = raw_kind.endswith("_break") or raw_kind == "break"
                base_kind = raw_kind.removesuffix("_break") if is_break else raw_kind
                phase: Literal["work", "break"] = "break" if is_break else "work"
                started_dt = _parse_dt(db_timer.get("started_at")) or now
                planned_raw = db_timer.get("planned_duration_seconds")
                if planned_raw is not None:
                    elapsed = max(0.0, (now - started_dt).total_seconds())
                    remaining = max(0, math.ceil(int(planned_raw) - elapsed))
                else:
                    remaining = 0
                return base_kind, phase, remaining

        return None

    def _compute_timer_remaining(self, now: datetime) -> int:
        if (
            self._timer_remaining_anchor is not None
            and self._timer_anchor_at is not None
        ):
            elapsed = max(0.0, (now - self._timer_anchor_at).total_seconds())
            return max(0, math.ceil(self._timer_remaining_anchor - elapsed))
        if (
            self._timer_planned_seconds is not None
            and self._timer_started_at is not None
        ):
            elapsed = max(0.0, (now - self._timer_started_at).total_seconds())
            return max(0, math.ceil(self._timer_planned_seconds - elapsed))
        if self._timer_remaining_anchor is not None:
            return max(0, int(self._timer_remaining_anchor))
        return 0

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

    async def _resolve_pending_recommendation_count(self) -> int:
        if not self._db_connected():
            return len(self._pending_rec_ids)

        db_pending = await self.db.get_pending_recommendations()
        pending_ids: set[int | str] = {
            int(row["id"]) for row in db_pending if row.get("id") is not None
        }

        for tracked_id in list(self._pending_rec_ids):
            if tracked_id in pending_ids:
                continue
            if isinstance(tracked_id, int):
                existing = await self.db.get_recommendation(tracked_id)
                if existing is not None:
                    if existing.get("status") != "pending":
                        self._pending_rec_ids.discard(tracked_id)
                    continue
            pending_ids.add(tracked_id)

        return len(pending_ids)

    async def _on_activity_state_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        event_time = self._event_now(event)
        raw_activity = payload.get("activity")
        if isinstance(raw_activity, str) and raw_activity in _VALID_ACTIVITIES:
            self._activity = cast(ActivityState, raw_activity)
        self._activity_source = str(
            payload.get("activity_source") or payload.get("source") or event.source
        )
        self._activity_reason = str(payload.get("reason") or "state_change")
        self._activity_started_at = _parse_dt(payload.get("started_at")) or ensure_aware(
            event.timestamp
        )
        self._override_expires_at = _parse_dt(payload.get("override_expires_at"))
        await self._refresh_safe(now=event_time)

    async def _on_timer_state_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        event_time = self._event_now(event)
        active = bool(payload.get("active"))
        phase = str(payload.get("phase") or "idle")
        transition = str(payload.get("transition") or "")

        if active and phase in ("work", "break") and transition not in (
            "completed",
            "stopped",
        ):
            raw_kind = str(payload.get("kind") or "custom")
            self._timer_active = True
            self._timer_kind = (
                raw_kind.removesuffix("_break")
                if raw_kind.endswith("_break")
                else raw_kind
            )
            self._timer_phase = "break" if phase == "break" else "work"
            self._timer_started_at = _parse_dt(payload.get("started_at")) or event_time
            planned = payload.get("planned_duration_seconds")
            self._timer_planned_seconds = int(planned) if planned is not None else None
            rem = payload.get("remaining_seconds")
            self._timer_remaining_anchor = int(rem) if rem is not None else None
            self._timer_anchor_at = event_time
        else:
            self._timer_active = False
            self._timer_kind = None
            self._timer_started_at = None
            self._timer_planned_seconds = None
            self._timer_remaining_anchor = None
            self._timer_anchor_at = None

        await self._refresh_safe(now=event_time)

    async def _on_calendar_context_changed(self, event: PlatformEvent) -> None:
        event_time = self._event_now(event)
        self._update_cached_calendar_from_payload(event.payload, event_time)
        await self._refresh_safe(now=event_time)

    async def _on_privacy_mode_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        event_time = self._event_now(event)
        for key in ("global_pause", "privacy_paused", "paused"):
            if key in payload and payload[key] is not None:
                self._privacy_paused = bool(payload[key])
                self._sync_privacy_in_config("global_pause", self._privacy_paused)
                break

        if "dnd_mode" in payload and payload["dnd_mode"] is not None:
            self._dnd_mode = bool(payload["dnd_mode"])
            self._sync_privacy_in_config("dnd_mode", self._dnd_mode)

        await self._refresh_safe(now=event_time)

    async def _on_recommendation_created(self, event: PlatformEvent) -> None:
        payload = event.payload
        event_time = self._event_now(event)
        rec_key = self._extract_recommendation_key(payload, fallback=f"evt:{id(event)}")
        if payload.get("status", "pending") == "pending" and rec_key is not None:
            self._pending_rec_ids.add(rec_key)
        await self._refresh_safe(now=event_time)

    async def _on_recommendation_resolved(self, event: PlatformEvent) -> None:
        payload = event.payload
        event_time = self._event_now(event)
        rec_key = self._extract_recommendation_key(payload)
        if rec_key is not None:
            self._pending_rec_ids.discard(rec_key)
        elif self._pending_rec_ids:
            self._pending_rec_ids.pop()
        await self._refresh_safe(now=event_time)

    async def _refresh_safe(self, now: datetime | None = None) -> None:
        with suppress(Exception):
            await self.refresh(now=now)

    async def _periodic_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.refresh_interval_seconds())
                if not self._running:
                    break
                await self._refresh_safe()
        except asyncio.CancelledError:
            raise

    async def _cancel_loop(self) -> None:
        task = self._loop_task
        self._loop_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

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
        elif "event_id" in payload or "title" in payload:
            candidate = dict(payload)

        if candidate is not None and "current_event" not in payload and "next_event" not in payload:
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

    @staticmethod
    def _extract_recommendation_key(
        payload: dict[str, Any],
        *,
        fallback: str | None = None,
    ) -> int | str | None:
        raw_id = payload.get("recommendation_id")
        if raw_id is None:
            raw_id = payload.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool):
            return raw_id
        if isinstance(raw_id, str) and raw_id.strip():
            with suppress(ValueError):
                return int(raw_id)
            return raw_id
        dedupe = payload.get("dedupe_key")
        if isinstance(dedupe, str) and dedupe.strip():
            return dedupe
        return fallback

    def _now(self) -> datetime:
        return ensure_aware(self._clock())

    def _event_now(self, event: PlatformEvent) -> datetime:
        if self._custom_clock:
            return self._now()
        return ensure_aware(event.timestamp)

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    def _get_waybar_value(self, key: str) -> Any:
        direct = self.config.get(key)
        if direct is not None:
            return direct
        for section_key in ("waybar_status", "waybar"):
            nested = self.config.get(section_key)
            if isinstance(nested, dict) and nested.get(key) is not None:
                return nested[key]
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
