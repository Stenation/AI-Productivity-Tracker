"""Unit tests for PomodoroPolicyModule helpers, calendar triggers, cooldowns, and privacy."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import PomodoroSettings, Settings
from jarvis.modules.policies.pomodoro_policy import (
    PomodoroPolicyModule,
    is_walking_suitable,
    select_pomodoro,
)


def _pomodoro_config(settings: Settings, **overrides: Any) -> dict[str, Any]:
    full = settings.model_dump()
    merged = {**full["pomodoro"], **full}
    merged.update(overrides)
    return merged


def test_select_pomodoro_and_is_walking_suitable_match_spec_table(
    test_settings: Settings,
) -> None:
    # SPEC.MD §3 default Pomodoro presets
    assert select_pomodoro("deep_work", test_settings) == "50_10"
    assert select_pomodoro("coding", test_settings) == "50_10"
    assert select_pomodoro("reading", test_settings) == "50_10"
    assert select_pomodoro("admin", test_settings) == "25_5"
    assert select_pomodoro("email", test_settings) == "25_5"
    assert select_pomodoro("planning", test_settings) == "25_5"
    assert select_pomodoro("meeting", test_settings) is None
    assert select_pomodoro("break", test_settings) is None
    assert select_pomodoro("unknown", test_settings) is None
    assert select_pomodoro(None, test_settings) is None

    # Supports PomodoroSettings, nested/flat dicts, and None fallback
    custom_pomo = PomodoroSettings(coding="25_5", unknown="25_5")
    assert select_pomodoro("coding", custom_pomo) == "25_5"
    assert select_pomodoro("unknown", custom_pomo) == "25_5"
    assert select_pomodoro("deep_work", {"pomodoro": {"deep_work": "25_5"}}) == "25_5"
    assert select_pomodoro("admin", {"admin": "50_10"}) == "50_10"
    assert select_pomodoro("deep_work") == "50_10"

    # SPEC.MD §3 walking suitability
    assert is_walking_suitable("admin") is True
    assert is_walking_suitable("email") is True
    assert is_walking_suitable("planning") is True
    assert is_walking_suitable("break") is True
    assert is_walking_suitable("deep_work") is False
    assert is_walking_suitable("coding") is False
    assert is_walking_suitable("reading") is False
    assert is_walking_suitable("meeting") is False
    assert is_walking_suitable("unknown") is False
    assert is_walking_suitable(None) is False


async def test_calendar_context_changed_active_imminent_and_non_productive_blocks(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 9, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    timer_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)
    bus.subscribe("timer_state_changed", timer_events.append)

    # Even if auto_start is set True in config, Milestone 1 MUST NOT auto-start timers
    cfg = _pomodoro_config(test_settings, auto_start=True)
    policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    await policy.start()
    try:
        # 1. Active walking-compatible admin block (UC-03) -> 25_5 + walking_suitable=True
        await db.upsert_calendar_event(
            event_id="evt_email_1",
            title="Email triage",
            start_time=t0 - timedelta(minutes=2),
            end_time=t0 + timedelta(minutes=28),
            task_type="admin",
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )

        pending = await db.get_pending_recommendations()
        assert len(pending) == 1
        rec1 = pending[0]
        assert rec1["policy"] == "pomodoro_policy"
        assert rec1["suggested_action"] == "start_timer"
        assert rec1["dedupe_key"] == "pomodoro:event:evt_email_1"
        assert rec1["action_payload"]["kind"] == "25_5"
        assert rec1["action_payload"]["walking_suitable"] is True
        assert len(created_events) == 1
        assert created_events[0].dedupe_key == "pomodoro:event:evt_email_1"

        # Milestone 1 non-destructive invariant: no timer span started!
        assert await db.get_active_timer() is None
        assert timer_events == []

        # 2. Move past evt_email_1; coding block 25 minutes away (not imminent) -> None
        await db.resolve_recommendation(rec1["id"], "accepted", resolved_at=now)
        now = t0 + timedelta(minutes=35)
        await db.upsert_calendar_event(
            event_id="evt_code_2",
            title="Core Engine Refactor",
            start_time=now + timedelta(minutes=25),
            end_time=now + timedelta(minutes=85),
            task_type="coding",
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 1

        # 3. Advance to 5 minutes before evt_code_2 (imminent!) -> 50_10 + walking_suitable=False
        now = now + timedelta(minutes=20)
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 2
        assert created_events[1].dedupe_key == "pomodoro:event:evt_code_2"
        assert created_events[1].payload["action_payload"]["kind"] == "50_10"
        assert created_events[1].payload["action_payload"]["walking_suitable"] is False

        # 4. Meeting, break, and unknown blocks do NOT create Pomodoro recommendations
        now = t0 + timedelta(hours=3)
        for idx, non_pomo_type in enumerate(("meeting", "break", "unknown")):
            await db.upsert_calendar_event(
                event_id=f"evt_skip_{idx}",
                title=f"Block {non_pomo_type}",
                start_time=now - timedelta(minutes=1),
                end_time=now + timedelta(minutes=20),
                task_type=non_pomo_type,
            )
            await bus.publish(
                PlatformEvent(
                    type="calendar_context_changed",
                    source="google_calendar",
                    timestamp=now,
                )
            )
            now = now + timedelta(minutes=25)

        assert len(created_events) == 2

        # 5. Explicit suitable_for_walking=True on a reading event overrides default False
        await db.upsert_calendar_event(
            event_id="evt_light_reading",
            title="Literature survey",
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=49),
            task_type="reading",
            suitable_for_walking=True,
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 3
        assert created_events[2].dedupe_key == "pomodoro:event:evt_light_reading"
        assert created_events[2].payload["action_payload"]["kind"] == "50_10"
        assert created_events[2].payload["action_payload"]["walking_suitable"] is True
    finally:
        await policy.stop()


@pytest.mark.parametrize("resolution_status", ["accepted", "snoozed", "dismissed"])
async def test_pomodoro_dedupe_cooldown_across_states_and_distinct_events(
    db: Database,
    test_settings: Settings,
    resolution_status: str,
) -> None:
    t0 = datetime(2026, 11, 3, 14, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PomodoroPolicyModule(
        bus=bus,
        db=db,
        config=_pomodoro_config(test_settings, recommendation_cooldown_minutes=30),
        clock=lambda: now,
    )
    await policy.start()
    try:
        # 90-minute deep_work block
        await db.upsert_calendar_event(
            event_id="evt_deep_1",
            title="Deep Work: Policies",
            start_time=t0,
            end_time=t0 + timedelta(minutes=90),
            task_type="deep_work",
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 1
        rec1_id = created_events[0].payload["id"]

        # Duplicate calendar sync while pending -> suppressed
        now = t0 + timedelta(minutes=10)
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 1

        # Resolve recommendation at t0 + 10m
        await db.resolve_recommendation(rec1_id, resolution_status, resolved_at=now)

        # Within 30m cooldown (at t0 + 39m) -> suppressed
        now = t0 + timedelta(minutes=39)
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 1

        # After 30m cooldown expires (at t0 + 40m, still inside 90m block) -> re-emits
        now = t0 + timedelta(minutes=40)
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 2
        assert created_events[1].dedupe_key == "pomodoro:event:evt_deep_1"

        # New calendar event (distinct event_id) immediately emits even inside evt_deep_1 cooldown
        now = t0 + timedelta(minutes=95)
        await db.upsert_calendar_event(
            event_id="evt_plan_2",
            title="Sprint Planning Prep",
            start_time=now,
            end_time=now + timedelta(minutes=30),
            task_type="planning",
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 3
        assert created_events[2].dedupe_key == "pomodoro:event:evt_plan_2"
        assert created_events[2].payload["action_payload"]["kind"] == "25_5"
        assert created_events[2].payload["action_payload"]["walking_suitable"] is True
    finally:
        await policy.stop()


async def test_pomodoro_global_pause_active_timer_skip_and_registry(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 16, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PomodoroPolicyModule(
        bus=bus,
        db=db,
        config=_pomodoro_config(test_settings),
        clock=lambda: now,
    )
    await policy.start()
    try:
        await db.upsert_calendar_event(
            event_id="evt_focus_pause",
            title="Deep Focus",
            start_time=t0,
            end_time=t0 + timedelta(minutes=60),
            task_type="deep_work",
        )

        # Pause privacy at runtime -> suppresses recommendation
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=now,
                payload={"global_pause": True},
            )
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert created_events == []

        # Resume privacy, but if timer is already running for this event -> skipped
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=now,
                payload={"global_pause": False},
            )
        )
        timer_row = await db.start_timer_span(
            kind="50_10",
            planned_duration_seconds=3000,
            calendar_event_id="evt_focus_pause",
            started_at=now,
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert created_events == []

        # Once active timer ends, calendar evaluation emits recommendation
        await db.end_timer_span(timer_row["id"], status="stopped", ended_at=now)
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert len(created_events) == 1
        assert created_events[0].dedupe_key == "pomodoro:event:evt_focus_pause"
    finally:
        await policy.stop()

    reg_bus = EventBus()
    registry = ModuleRegistry(bus=reg_bus, db=db, settings=test_settings)
    assert await registry.enable_module("pomodoro_policy") is True
    assert registry.is_enabled("pomodoro_policy") is True
    assert await registry.disable_module("pomodoro_policy") is True
    assert registry.is_enabled("pomodoro_policy") is False
