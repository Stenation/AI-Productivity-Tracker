"""Async local Pomodoro and custom timer manager with SQLite span persistence."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, ensure_aware, now_local
from jarvis.core.settings import Settings

TimerPhase = Literal["idle", "work", "break"]

TIMER_PRESETS: dict[str, tuple[int, int]] = {
    "25_5": (25 * 60, 5 * 60),
    "50_10": (50 * 60, 10 * 60),
}


@dataclass(slots=True)
class _ActiveTimerState:
    generation: int
    span_id: int
    kind: str
    phase: Literal["work", "break"]
    phase_started_at: datetime
    planned_duration_seconds: int
    remaining_seconds: int
    work_duration_seconds: int
    break_duration_seconds: int
    calendar_event_id: str | None


class TimerManager:
    """Manage local work/break countdown timers, persistence, and EventBus transitions."""

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        settings: Settings | None = None,
        *,
        tick_interval_seconds: float = 1.0,
        clock: Callable[[], datetime] = now_local,
    ) -> None:
        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be positive")

        self._bus = bus
        self._db = db
        self._settings = settings
        self._tick_interval_seconds = float(tick_interval_seconds)
        self._clock = clock

        self._lock = asyncio.Lock()
        self._state: _ActiveTimerState | None = None
        self._tick_task: asyncio.Task[None] | None = None
        self._generation: int = 0
        self._started: bool = False

    async def start(self) -> int:
        """Inspect SQLite for leftover running spans, mark them interrupted, and start clean.

        Leftover running timers from a previous daemon run are never silently resumed.
        """
        old_task: asyncio.Task[None] | None = None
        async with self._lock:
            old_task = self._tick_task
            self._tick_task = None
            self._state = None
            startup_time = ensure_aware(self._clock())
            interrupted = await self._db.mark_running_timers_interrupted(
                ended_at=startup_time
            )
            self._started = True

        await self._cancel_task(old_task)
        return interrupted

    async def stop(self) -> None:
        """Cancel the background tick loop cleanly on daemon shutdown."""
        async with self._lock:
            task = self._tick_task
            self._tick_task = None
            self._state = None
            self._started = False

        await self._cancel_task(task)

    async def start_timer(
        self,
        kind: str | None = None,
        *,
        work_minutes: int | float | None = None,
        break_minutes: int | float | None = None,
        work_seconds: int | None = None,
        break_seconds: int | None = None,
        calendar_event_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a preset ('25_5', '50_10') or custom timer cycle."""
        resolved_kind, resolved_work_sec, resolved_break_sec = self._resolve_durations(
            kind=kind,
            work_minutes=work_minutes,
            break_minutes=break_minutes,
            work_seconds=work_seconds,
            break_seconds=break_seconds,
        )

        old_task: asyncio.Task[None] | None = None
        replaced_event: PlatformEvent | None = None

        async with self._lock:
            now = ensure_aware(self._clock())
            if self._state is not None:
                old_task, replaced_event, _ = await self._stop_active_locked(
                    reason="stopped",
                    ended_at=now,
                )

            span = await self._db.start_timer_span(
                kind=resolved_kind,
                planned_duration_seconds=resolved_work_sec,
                calendar_event_id=calendar_event_id,
                started_at=now,
            )
            self._generation += 1
            gen = self._generation
            self._state = _ActiveTimerState(
                generation=gen,
                span_id=int(span["id"]),
                kind=resolved_kind,
                phase="work",
                phase_started_at=now,
                planned_duration_seconds=resolved_work_sec,
                remaining_seconds=resolved_work_sec,
                work_duration_seconds=resolved_work_sec,
                break_duration_seconds=resolved_break_sec,
                calendar_event_id=calendar_event_id,
            )
            self._tick_task = asyncio.create_task(
                self._tick_loop(gen),
                name=f"jarvis-timer-{gen}",
            )
            status_snapshot = self._build_status(self._state, now=now)
            started_event = PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=now,
                payload={
                    **status_snapshot,
                    "status": "running",
                    "transition": "started",
                },
            )

        await self._cancel_task(old_task)
        if replaced_event is not None:
            await self._bus.publish(replaced_event)
        await self._bus.publish(started_event)
        return status_snapshot

    async def stop_timer(self, reason: str = "stopped") -> dict[str, Any]:
        """Stop the currently active timer (if any), persist its end, and emit an event."""
        stop_reason = reason.strip() if isinstance(reason, str) and reason.strip() else "stopped"

        async with self._lock:
            now = ensure_aware(self._clock())
            if self._state is None:
                return {
                    **self._build_status(None, now=now),
                    "stopped": False,
                    "reason": None,
                }
            old_task, stopped_event, ended_span = await self._stop_active_locked(
                reason=stop_reason,
                ended_at=now,
            )
            idle_status = self._build_status(None, now=now)

        await self._cancel_task(old_task)
        if stopped_event is not None:
            await self._bus.publish(stopped_event)

        return {
            **idle_status,
            "stopped": True,
            "reason": stop_reason,
            "ended_span": ended_span,
        }

    def get_status(self) -> dict[str, Any]:
        """Return current timer status snapshot."""
        now = ensure_aware(self._clock())
        return self._build_status(self._state, now=now)

    @staticmethod
    def _resolve_durations(
        *,
        kind: str | None,
        work_minutes: int | float | None,
        break_minutes: int | float | None,
        work_seconds: int | None,
        break_seconds: int | None,
    ) -> tuple[str, int, int]:
        has_custom_work = work_minutes is not None or work_seconds is not None

        if kind is None or not str(kind).strip():
            resolved_kind = "custom" if has_custom_work else "25_5"
        else:
            resolved_kind = str(kind).strip()

        if has_custom_work:
            if work_seconds is not None:
                w_sec = int(work_seconds)
            else:
                assert work_minutes is not None
                w_sec = round(float(work_minutes) * 60)

            if break_seconds is not None:
                b_sec = int(break_seconds)
            elif break_minutes is not None:
                b_sec = round(float(break_minutes) * 60)
            elif resolved_kind in TIMER_PRESETS:
                b_sec = TIMER_PRESETS[resolved_kind][1]
            else:
                b_sec = 0
        else:
            if resolved_kind not in TIMER_PRESETS:
                raise ValueError(
                    f"Unsupported timer preset {resolved_kind!r}; expected one of "
                    f"{sorted(TIMER_PRESETS)} or explicit work_minutes"
                )
            w_sec, preset_break = TIMER_PRESETS[resolved_kind]
            if break_seconds is not None:
                b_sec = int(break_seconds)
            elif break_minutes is not None:
                b_sec = round(float(break_minutes) * 60)
            else:
                b_sec = preset_break

        if w_sec <= 0:
            raise ValueError("Timer work duration must be greater than 0 seconds")
        if b_sec < 0:
            raise ValueError("Timer break duration cannot be negative")

        return resolved_kind, w_sec, b_sec

    @staticmethod
    def _build_status(
        state: _ActiveTimerState | None,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        if state is None:
            return {
                "active": False,
                "timer_id": None,
                "kind": None,
                "phase": "idle",
                "remaining_seconds": 0,
                "planned_duration_seconds": None,
                "elapsed_seconds": 0,
                "work_duration_seconds": None,
                "break_duration_seconds": None,
                "calendar_event_id": None,
                "started_at": None,
            }

        elapsed_float = max(0.0, (now - state.phase_started_at).total_seconds())
        remaining = max(0, math.ceil(state.planned_duration_seconds - elapsed_float))
        elapsed = min(state.planned_duration_seconds, max(0, int(elapsed_float)))
        return {
            "active": True,
            "timer_id": state.span_id,
            "kind": state.kind,
            "phase": state.phase,
            "remaining_seconds": remaining,
            "planned_duration_seconds": state.planned_duration_seconds,
            "elapsed_seconds": elapsed,
            "work_duration_seconds": state.work_duration_seconds,
            "break_duration_seconds": state.break_duration_seconds,
            "calendar_event_id": state.calendar_event_id,
            "started_at": state.phase_started_at.isoformat(),
        }

    async def _stop_active_locked(
        self,
        *,
        reason: str,
        ended_at: datetime,
    ) -> tuple[asyncio.Task[None] | None, PlatformEvent | None, dict[str, Any] | None]:
        state = self._state
        if state is None:
            return None, None, None

        old_task = self._tick_task
        self._tick_task = None
        self._state = None

        ended_span = await self._db.end_timer_span(
            state.span_id,
            status=reason,
            ended_at=ended_at,
        )
        event = PlatformEvent(
            type="timer_state_changed",
            source="timer_manager",
            timestamp=ended_at,
            payload={
                "active": False,
                "timer_id": state.span_id,
                "kind": state.kind,
                "phase": "idle",
                "previous_phase": state.phase,
                "remaining_seconds": 0,
                "planned_duration_seconds": state.planned_duration_seconds,
                "started_at": state.phase_started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "calendar_event_id": state.calendar_event_id,
                "status": reason,
                "reason": reason,
                "transition": "stopped",
            },
        )
        return old_task, event, ended_span

    async def _tick_loop(self, generation: int) -> None:
        while True:
            await asyncio.sleep(self._tick_interval_seconds)
            event_to_publish, keep_running = await self._advance_locked(generation)
            if event_to_publish is not None:
                await self._bus.publish(event_to_publish)
            if not keep_running:
                return

    async def _advance_locked(
        self,
        generation: int,
    ) -> tuple[PlatformEvent | None, bool]:
        async with self._lock:
            state = self._state
            if state is None or state.generation != generation:
                return None, False

            now = ensure_aware(self._clock())
            elapsed_float = max(0.0, (now - state.phase_started_at).total_seconds())
            remaining = max(0, math.ceil(state.planned_duration_seconds - elapsed_float))
            state.remaining_seconds = remaining

            if elapsed_float < state.planned_duration_seconds:
                return None, True

            await self._db.end_timer_span(
                state.span_id,
                status="completed",
                ended_at=now,
            )

            if state.phase == "work" and state.break_duration_seconds > 0:
                break_span = await self._db.start_timer_span(
                    kind=f"{state.kind}_break",
                    planned_duration_seconds=state.break_duration_seconds,
                    calendar_event_id=state.calendar_event_id,
                    started_at=now,
                )
                state.span_id = int(break_span["id"])
                state.phase = "break"
                state.phase_started_at = now
                state.planned_duration_seconds = state.break_duration_seconds
                state.remaining_seconds = state.break_duration_seconds

                status_snapshot = self._build_status(state, now=now)
                event = PlatformEvent(
                    type="timer_state_changed",
                    source="timer_manager",
                    timestamp=now,
                    payload={
                        **status_snapshot,
                        "previous_phase": "work",
                        "status": "running",
                        "transition": "break_started",
                    },
                )
                return event, True

            finished_phase = state.phase
            finished_kind = state.kind
            finished_span_id = state.span_id
            finished_planned = state.planned_duration_seconds
            finished_started_at = state.phase_started_at
            finished_cal_id = state.calendar_event_id

            self._state = None
            self._tick_task = None

            event = PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=now,
                payload={
                    "active": False,
                    "timer_id": finished_span_id,
                    "kind": finished_kind,
                    "phase": "idle",
                    "previous_phase": finished_phase,
                    "remaining_seconds": 0,
                    "planned_duration_seconds": finished_planned,
                    "started_at": finished_started_at.isoformat(),
                    "ended_at": now.isoformat(),
                    "calendar_event_id": finished_cal_id,
                    "status": "completed",
                    "transition": "completed",
                },
            )
            return event, False

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
