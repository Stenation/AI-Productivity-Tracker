"""Desktop notification output module dispatching via notify-send (mako/dunst)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import datetime
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, ensure_aware, now_local
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_NOTIFY_BINARY = "notify-send"
DEFAULT_APP_NAME = "Posture Platform"
DEFAULT_URGENCY = "normal"
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 5.0
DEFAULT_NOTIFY_TRANSITIONS: frozenset[str] = frozenset({"break_started", "completed"})

_POLICY_TITLES: dict[str, str] = {
    "posture_policy": "Posture Recommendation",
    "pomodoro_policy": "Pomodoro Recommendation",
    "meeting_policy": "Upcoming Meeting",
    "focus_policy": "Focus Nudge",
}

_VALID_URGENCIES: frozenset[str] = frozenset({"low", "normal", "critical"})
_PRESET_RE = re.compile(r"^(\d+)_(\d+)$")

NotifyRunnerResult = int | tuple[int, str | bytes, str | bytes]
NotifyRunner = Callable[[Sequence[str]], Awaitable[NotifyRunnerResult] | NotifyRunnerResult]


def _format_timer_kind(kind: str | None) -> str:
    if not kind or not kind.strip():
        return "Custom"
    cleaned = kind.strip()
    if cleaned.endswith("_break"):
        cleaned = cleaned.removesuffix("_break")
    match = _PRESET_RE.match(cleaned)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return cleaned.replace("_", " ").title()


class DesktopNotifyModule(BaseModule):
    """Send desktop notifications via notify-send while strictly honoring privacy/DND."""

    name: str = "desktop_notify"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any] | None = None,
        *,
        notify_binary: str | None = None,
        runner: NotifyRunner | None = None,
        app_name: str | None = None,
        urgency: str | None = None,
        expire_time_ms: int | None = None,
        subprocess_timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=dict(config) if config else {})
        if subprocess_timeout_seconds <= 0:
            raise ValueError("subprocess_timeout_seconds must be positive")

        self._injected_binary = notify_binary
        self._injected_runner = runner
        self._injected_app_name = app_name
        self._injected_urgency = urgency
        self._injected_expire_ms = expire_time_ms
        self._subprocess_timeout_seconds = float(subprocess_timeout_seconds)

        self._custom_clock = clock is not None
        self._clock: Callable[[], datetime] = clock if clock is not None else now_local

        self._privacy_paused: bool = self._extract_privacy_paused(self.config)
        self._dnd_mode: bool = self._extract_dnd_mode(self.config)

        self._running: bool = False
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None
        self._degraded_reported: bool = False

    async def start(self) -> None:
        """Load persisted privacy/DND state and subscribe to notification topics."""
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if self._db_connected():
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if isinstance(persisted_pause, bool):
                self._privacy_paused = self._privacy_paused or persisted_pause
            persisted_dnd = await self.db.get_setting("privacy.dnd_mode")
            if isinstance(persisted_dnd, bool):
                self._dnd_mode = self._dnd_mode or persisted_dnd

        self.bus.subscribe("recommendation_created", self._on_recommendation_created)
        self.bus.subscribe("timer_state_changed", self._on_timer_state_changed)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)

        self._running = True
        self._last_error = None
        self._degraded_reported = False

    async def stop(self) -> None:
        """Unsubscribe cleanly from EventBus topics."""
        self._running = False
        self.bus.unsubscribe("recommendation_created", self._on_recommendation_created)
        self.bus.unsubscribe("timer_state_changed", self._on_timer_state_changed)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Refresh configuration and runtime privacy/DND flags."""
        await super().reload_config(dict(config))
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

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
            "notify_binary": self.notify_binary(),
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
            "last_error": self._last_error,
        }

    def notify_binary(self) -> str:
        """Return configured notify-send executable name or path."""
        if self._injected_binary is not None:
            return self._injected_binary
        for key in ("notify_binary", "binary"):
            val = self._get_notify_value(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return DEFAULT_NOTIFY_BINARY

    def app_name(self) -> str:
        """Return notification application name passed to notify-send."""
        if self._injected_app_name is not None:
            return self._injected_app_name
        val = self._get_notify_value("app_name")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return DEFAULT_APP_NAME

    def urgency(self) -> str:
        """Return default libnotify urgency level ('low', 'normal', or 'critical')."""
        raw = self._injected_urgency or self._get_notify_value("urgency")
        if isinstance(raw, str) and raw.lower() in _VALID_URGENCIES:
            return raw.lower()
        return DEFAULT_URGENCY

    def expire_time_ms(self) -> int | None:
        """Return optional notification expiration timeout in milliseconds."""
        if self._injected_expire_ms is not None:
            return int(self._injected_expire_ms)
        for key in ("expire_time_ms", "timeout_ms"):
            raw = self._get_notify_value(key)
            if raw is not None:
                return int(raw)
        return None

    def is_suppressed(self) -> bool:
        """Return True when global privacy pause or DND mode suppresses notifications."""
        return bool(self._privacy_paused or self._dnd_mode)

    async def send_notification(
        self,
        summary: str,
        body: str = "",
        *,
        urgency: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Dispatch a desktop notification via notify-send if privacy/DND permit."""
        if self.is_suppressed():
            logger.debug(
                "Suppressing desktop notification (%r) due to privacy_pause=%s dnd_mode=%s",
                summary,
                self._privacy_paused,
                self._dnd_mode,
            )
            return False

        clean_summary = summary.strip()
        clean_body = body.strip()
        if not clean_summary and not clean_body:
            return False
        if not clean_summary:
            clean_summary = self.app_name()

        resolved_urgency = (
            urgency.lower()
            if isinstance(urgency, str) and urgency.lower() in _VALID_URGENCIES
            else self.urgency()
        )

        argv: list[str] = [
            self.notify_binary(),
            "--app-name",
            self.app_name(),
            "--urgency",
            resolved_urgency,
        ]
        expire_ms = self.expire_time_ms()
        if expire_ms is not None and expire_ms > 0:
            argv.extend(["--expire-time", str(expire_ms)])
        argv.append("--")
        argv.append(clean_summary)
        if clean_body:
            argv.append(clean_body)

        event_time = ensure_aware(now) if now is not None else self._now()
        try:
            returncode, stderr_text = await self._execute(argv)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Desktop notification dispatch failed: %s", error_msg)
            await self._record_failure(error_msg, event_time=event_time)
            return False

        if returncode != 0:
            detail = f": {stderr_text}" if stderr_text else ""
            error_msg = (
                f"{self.notify_binary()} exited with status {returncode}{detail}"
            )
            logger.warning("Desktop notification returned non-zero: %s", error_msg)
            await self._record_failure(error_msg, event_time=event_time)
            return False

        await self._record_success(event_time=event_time)
        return True

    async def _execute(self, argv: list[str]) -> tuple[int, str]:
        runner = self._injected_runner or self._get_notify_value("runner")
        if callable(runner):
            raw_res = runner(argv)
            if inspect.isawaitable(raw_res):
                raw_res = await raw_res
            if isinstance(raw_res, tuple):
                rc = int(raw_res[0])
                stderr_part = raw_res[2] if len(raw_res) >= 3 else ""
                if isinstance(stderr_part, bytes):
                    stderr_str = stderr_part.decode("utf-8", errors="replace").strip()
                else:
                    stderr_str = str(stderr_part).strip()
                return rc, stderr_str
            return int(raw_res), ""

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_bytes = await asyncio.wait_for(
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
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        return rc, stderr_text

    async def _record_failure(self, error_msg: str, *, event_time: datetime) -> None:
        self._last_error = error_msg
        self._last_event_at = event_time
        self._degraded_reported = True

        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "degraded",
                last_event_at=event_time,
                last_error=error_msg,
            )
        await self.bus.publish(
            PlatformEvent(
                type="module_health_changed",
                source=self.name,
                timestamp=event_time,
                payload={
                    "module": self.name,
                    "category": "outputs",
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
        self._last_event_at = event_time

        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "ok",
                last_event_at=event_time,
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
                        "category": "outputs",
                        "enabled": True,
                        "status": "ok",
                        "last_error": None,
                    },
                )
            )

    async def _on_recommendation_created(self, event: PlatformEvent) -> None:
        if self.is_suppressed():
            return

        payload = event.payload
        message = str(payload.get("message") or "").strip()
        if not message:
            return

        policy = str(payload.get("policy") or event.source or "").strip()
        summary = _POLICY_TITLES.get(
            policy,
            policy.replace("_", " ").title() if policy else self.app_name(),
        )
        await self.send_notification(
            summary=summary,
            body=message,
            now=self._event_now(event),
        )

    async def _on_timer_state_changed(self, event: PlatformEvent) -> None:
        if self.is_suppressed():
            return

        payload = event.payload
        transition = str(payload.get("transition") or "").strip()
        if not transition:
            status = str(payload.get("status") or "").strip()
            phase = str(payload.get("phase") or "").strip()
            prev = str(payload.get("previous_phase") or "").strip()
            if status == "completed":
                transition = "completed"
            elif phase == "break" and prev == "work":
                transition = "break_started"

        allowed_transitions = self._allowed_transitions()
        if transition not in allowed_transitions:
            return

        kind_label = _format_timer_kind(
            str(payload.get("kind")) if payload.get("kind") is not None else None
        )
        summary, body = self._format_timer_notification(
            transition=transition,
            kind_label=kind_label,
            payload=payload,
        )
        await self.send_notification(
            summary=summary,
            body=body,
            now=self._event_now(event),
        )

    @staticmethod
    def _format_timer_notification(
        *,
        transition: str,
        kind_label: str,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        if transition == "break_started":
            raw_sec = payload.get("remaining_seconds")
            if raw_sec is None:
                raw_sec = payload.get("planned_duration_seconds")
            if raw_sec is not None and int(raw_sec) > 0:
                mins = max(1, math.ceil(int(raw_sec) / 60))
                return (
                    "Timer — Break Started",
                    f"Work block complete ({kind_label}). Take a {mins}m break.",
                )
            return (
                "Timer — Break Started",
                f"Work block complete ({kind_label}). Time for a break.",
            )

        if transition == "completed":
            prev_phase = str(payload.get("previous_phase") or "")
            if prev_phase == "break":
                return (
                    "Timer — Completed",
                    f"{kind_label} timer cycle complete. Break finished.",
                )
            return ("Timer — Completed", f"{kind_label} timer complete.")

        if transition == "started":
            return ("Timer — Started", f"Started {kind_label} timer.")

        return ("Timer — Stopped", f"Stopped {kind_label} timer.")

    def _allowed_transitions(self) -> frozenset[str]:
        configured = self._get_notify_value("notify_transitions")
        if isinstance(configured, (list, tuple, set, frozenset)):
            return frozenset(str(item) for item in configured)
        return DEFAULT_NOTIFY_TRANSITIONS

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

    def _now(self) -> datetime:
        return ensure_aware(self._clock())

    def _event_now(self, event: PlatformEvent) -> datetime:
        if self._custom_clock:
            return self._now()
        return ensure_aware(event.timestamp)

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    def _get_notify_value(self, key: str) -> Any:
        direct = self.config.get(key)
        if direct is not None:
            return direct
        for section_key in ("desktop_notify", "notifications"):
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
