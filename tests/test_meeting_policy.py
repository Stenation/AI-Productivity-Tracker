"""Unit tests for MeetingPolicyModule pre-meeting guardrail, dedupe cooldowns, and privacy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import Settings
from jarvis.modules.policies.meeting_policy import MeetingPolicyModule


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.0,
    interval: float = 0.005,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for async condition")
        await asyncio.sleep(interval)


def _meeting_config(settings: Settings, **overrides: Any) -> dict[str, Any]:
    full = settings.model_dump()
    merged = {**full["posture"], **full}
    merged.update(overrides)
    return merged


async def test_pre_meeting_warning_window_and_activity_filtering(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 10, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = MeetingPolicyModule(
        bus=bus,
        db=db,
        config=_meeting_config(test_settings, pre_meeting_warning_minutes=10),
        clock=lambda: now,
    )
    await policy.start()
    try:
        # Upcoming meeting at t0 + 15m (outside 10m warning window) while walking -> no warning
        meeting_start = t0 + timedelta(minutes=15)
        await db.open_activity_span("walking", "swalk_ble", started_at=t0 - timedelta(minutes=20))
        await db.upsert_calendar_event(
            event_id="evt_mtg_1",
            title="Team Standup",
            start_time=meeting_start,
            end_time=meeting_start + timedelta(minutes=30),
            task_type="meeting",
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=now,
            )
        )
        assert created_events == []

        # Non-meeting upcoming in 5 minutes while walking -> no return_to_desk warning
        await db.upsert_calendar_event(
            event_id="evt_admin_soon",
            title="Quick Inbox Check",
            start_time=t0 + timedelta(minutes=1),
            end_time=t0 + timedelta(minutes=4),
            task_type="admin",
        )
        assert await policy.evaluate() is None
        assert created_events == []

        # At t0 + 5m (10 minutes before meeting), if user is sitting or standing -> no warning
        now = t0 + timedelta(minutes=5)
        for desk_state in ("sitting", "standing", "unknown"):
            await db.open_activity_span(desk_state, "manual_control", started_at=now)
            assert await policy.evaluate() is None
        assert created_events == []

        # User transitions to walking 10 minutes before meeting -> warns immediately!
        await db.open_activity_span("walking", "manual_control", started_at=now)
        await bus.publish(
            PlatformEvent(
                type="activity_state_changed",
                source="resolver",
                timestamp=now,
                payload={"activity": "walking", "started_at": now.isoformat()},
            )
        )
        assert len(created_events) == 1
        ev1 = created_events[0]
        assert ev1.dedupe_key == "meeting:warn:evt_mtg_1"
        assert ev1.payload["policy"] == "meeting_policy"
        assert ev1.payload["suggested_action"] == "return_to_desk"
        assert ev1.payload["action_payload"]["minutes_until_start"] == 10
        assert "Team Standup" in ev1.payload["message"]

        # Second meeting while user is "away" -> also warns with return_to_desk
        now = t0 + timedelta(minutes=50)
        mtg2_start = now + timedelta(minutes=6)
        await db.open_activity_span("away", "manual_control", started_at=now)
        await db.upsert_calendar_event(
            event_id="evt_mtg_2",
            title="Design Sync",
            start_time=mtg2_start,
            end_time=mtg2_start + timedelta(minutes=30),
            task_type="meeting",
        )
        rec2 = await policy.evaluate()
        assert rec2 is not None
        assert rec2["dedupe_key"] == "meeting:warn:evt_mtg_2"
        assert rec2["suggested_action"] == "return_to_desk"
        assert rec2["action_payload"]["activity"] == "away"
        assert len(created_events) == 2

        # Once meeting has already started (no upcoming meeting), pre-meeting warning does not fire
        now = mtg2_start + timedelta(minutes=2)
        await db.upsert_calendar_event(
            event_id="evt_mtg_already_live",
            title="Live Call",
            start_time=now - timedelta(minutes=2),
            end_time=now + timedelta(minutes=25),
            task_type="meeting",
        )
        assert await policy.evaluate() is None
        assert len(created_events) == 2
    finally:
        await policy.stop()


@pytest.mark.parametrize("resolution_status", ["accepted", "snoozed", "dismissed"])
async def test_meeting_dedupe_and_cooldown_across_states(
    db: Database,
    test_settings: Settings,
    resolution_status: str,
) -> None:
    t0 = datetime(2026, 11, 3, 14, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = MeetingPolicyModule(
        bus=bus,
        db=db,
        config=_meeting_config(
            test_settings,
            pre_meeting_warning_minutes=15,
            recommendation_cooldown_minutes=10,
        ),
        clock=lambda: now,
    )
    await policy.start()
    try:
        meeting_start = t0 + timedelta(minutes=14)
        await db.open_activity_span("walking", "swalk_ble", started_at=t0 - timedelta(minutes=10))
        await db.upsert_calendar_event(
            event_id="evt_dedupe_mtg",
            title="Product Review",
            start_time=meeting_start,
            end_time=meeting_start + timedelta(minutes=30),
            task_type="meeting",
        )

        rec1 = await policy.evaluate()
        assert rec1 is not None
        assert len(created_events) == 1

        # Repeated evaluations while pending -> suppressed
        now = t0 + timedelta(minutes=1)
        assert await policy.evaluate() is None
        assert len(created_events) == 1

        # Resolve at t0 + 1m (with snooze_minutes=5 when snoozed, else 10m cooldown)
        meta = {"snooze_minutes": 5} if resolution_status == "snoozed" else None
        expected_wait = 5 if resolution_status == "snoozed" else 10
        await db.resolve_recommendation(
            rec1["id"],
            resolution_status,
            metadata=meta,
            resolved_at=now,
        )

        # Inside cooldown/snooze window -> suppressed
        now = t0 + timedelta(minutes=1 + expected_wait - 1)
        assert await policy.evaluate() is None
        assert len(created_events) == 1

        # Once cooldown/snooze expires while meeting is still upcoming -> re-emits
        now = t0 + timedelta(minutes=1 + expected_wait)
        rec2 = await policy.evaluate()
        assert rec2 is not None
        assert rec2["id"] != rec1["id"]
        assert rec2["dedupe_key"] == "meeting:warn:evt_dedupe_mtg"
        assert len(created_events) == 2
        assert created_events[1].dedupe_key == "meeting:warn:evt_dedupe_mtg"
    finally:
        await policy.stop()


async def test_meeting_periodic_evaluation_global_pause_and_registry(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 16, 0, 0).astimezone()
    now = t0

    await db.open_activity_span("walking", "swalk_ble", started_at=t0 - timedelta(minutes=15))
    await db.upsert_calendar_event(
        event_id="evt_periodic_mtg",
        title="Weekly Retro",
        start_time=t0 + timedelta(minutes=12),
        end_time=t0 + timedelta(minutes=42),
        task_type="meeting",
    )

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = MeetingPolicyModule(
        bus=bus,
        db=db,
        config=_meeting_config(test_settings, pre_meeting_warning_minutes=10),
        check_interval_seconds=0.015,
        clock=lambda: now,
    )
    await policy.start()
    try:
        # At t0 (12m before meeting > 10m window) -> periodic loop does not warn yet
        await asyncio.sleep(0.04)
        assert created_events == []

        # Enable global_pause BEFORE entering 10m window -> suppressed even inside window!
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=now,
                payload={"global_pause": True},
            )
        )
        now = t0 + timedelta(minutes=4)  # 8m before meeting
        await asyncio.sleep(0.04)
        assert created_events == []

        # Resume global_pause -> periodic loop automatically emits single warning!
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=now,
                payload={"global_pause": False},
            )
        )
        await _wait_until(lambda: len(created_events) == 1)
        assert created_events[0].dedupe_key == "meeting:warn:evt_periodic_mtg"

        # Subsequent periodic ticks do NOT emit duplicates
        await asyncio.sleep(0.05)
        assert len(created_events) == 1
        assert len(await db.get_pending_recommendations()) == 1
    finally:
        await policy.stop()

    reg_bus = EventBus()
    registry = ModuleRegistry(bus=reg_bus, db=db, settings=test_settings)
    assert await registry.enable_module("meeting_policy") is True
    assert registry.is_enabled("meeting_policy") is True
    assert await registry.disable_module("meeting_policy") is True
    assert registry.is_enabled("meeting_policy") is False


async def test_meeting_warning_suppressed_when_user_returned_to_call_window_with_input(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 4, 13, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = MeetingPolicyModule(
        bus=bus,
        db=db,
        config=_meeting_config(test_settings, pre_meeting_warning_minutes=10),
        clock=lambda: now,
    )
    await policy.start()
    try:
        # User is marked as walking
        await db.open_activity_span("walking", "manual_control", started_at=t0)

        # Upcoming meeting in 8 minutes
        meeting_start = t0 + timedelta(minutes=8)
        await db.upsert_calendar_event(
            event_id="evt_desk_return_check",
            title="Sprint Retro",
            start_time=meeting_start,
            end_time=meeting_start + timedelta(minutes=30),
            task_type="meeting",
        )

        # 1. User has already opened Zoom call window with active input -> warning suppressed!
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=now,
                payload={
                    "app_class": "zoom",
                    "title": "Zoom Meeting",
                    "is_call_window": True,
                },
            )
        )
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=now,
                payload={"idle": False},
            )
        )
        assert await policy.evaluate() is None
        assert created_events == []

        # 2. If input becomes idle (e.g. left Zoom open but walked away) -> warning triggers!
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=now,
                payload={"idle": True, "idle_seconds": 120},
            )
        )
        assert len(created_events) == 1
        ev = created_events[0]
        assert ev.dedupe_key == "meeting:warn:evt_desk_return_check"
        assert ev.payload["suggested_action"] == "return_to_desk"
        assert "Sprint Retro" in ev.payload["message"]
    finally:
        await policy.stop()
