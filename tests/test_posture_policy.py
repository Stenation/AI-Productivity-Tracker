"""Unit tests for PosturePolicyModule thresholds, calendar suitability, cooldowns, and privacy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import Settings
from jarvis.modules.policies.posture_policy import (
    SITTING_DEDUPE_KEY,
    WALKING_DEDUPE_KEY,
    PosturePolicyModule,
)


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


def _policy_config(settings: Settings, **overrides: Any) -> dict[str, Any]:
    full = settings.model_dump()
    merged = {**full["posture"], **full}
    merged.update(overrides)
    return merged


async def test_sitting_and_walking_thresholds_and_non_triggering_states(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 9, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(test_settings),
        check_interval_seconds=60.0,
        clock=lambda: now,
    )
    await policy.start()
    try:
        # Standing or away for 2 hours never triggers sitting/walking nudges
        await db.open_activity_span("standing", "manual_control", started_at=t0)
        now = t0 + timedelta(minutes=120)
        assert await policy.evaluate() is None
        assert created_events == []

        # Open sitting span at t1
        t1 = t0 + timedelta(hours=2)
        await db.open_activity_span("sitting", "manual_control", started_at=t1)

        # 49 minutes (< default 50m threshold) -> no recommendation
        now = t1 + timedelta(minutes=49, seconds=59)
        assert await policy.evaluate() is None
        assert await db.get_pending_recommendations() == []
        assert created_events == []

        # Exact 50-minute threshold with no calendar event -> stand_or_walk
        now = t1 + timedelta(minutes=50)
        sit_rec = await policy.evaluate()
        assert sit_rec is not None
        assert sit_rec["policy"] == "posture_policy"
        assert sit_rec["suggested_action"] == "stand_or_walk"
        assert sit_rec["dedupe_key"] == SITTING_DEDUPE_KEY
        assert sit_rec["status"] == "pending"
        assert sit_rec["action_payload"]["elapsed_minutes"] == 50
        assert len(created_events) == 1
        assert created_events[0].dedupe_key == SITTING_DEDUPE_KEY
        assert created_events[0].payload["suggested_action"] == "stand_or_walk"

        # Transition to walking at t2
        t2 = t1 + timedelta(hours=1)
        await db.open_activity_span("walking", "swalk_ble", started_at=t2)

        # 44 minutes (< default 45m threshold) -> no walking recommendation
        now = t2 + timedelta(minutes=44, seconds=59)
        assert await policy.evaluate() is None
        assert len(created_events) == 1

        # Exact 45-minute threshold -> take_seated_break with dedupe_key="posture:walking"
        now = t2 + timedelta(minutes=45)
        walk_rec = await policy.evaluate()
        assert walk_rec is not None
        assert walk_rec["policy"] == "posture_policy"
        assert walk_rec["suggested_action"] == "take_seated_break"
        assert walk_rec["dedupe_key"] == WALKING_DEDUPE_KEY
        assert walk_rec["action_payload"]["recommended_activity"] == "sitting"
        assert len(created_events) == 2
        assert created_events[1].dedupe_key == WALKING_DEDUPE_KEY
        assert created_events[1].payload["suggested_action"] == "take_seated_break"
    finally:
        await policy.stop()


async def test_sitting_calendar_walking_suitability_and_meeting_suppression(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 10, 0, 0).astimezone()
    now = t0 + timedelta(minutes=60)

    bus = EventBus()
    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(test_settings, recommendation_cooldown_minutes=0),
        clock=lambda: now,
    )
    await policy.start()
    try:
        await db.open_activity_span("sitting", "manual_control", started_at=t0)

        # 1. Currently in a meeting -> suppressed completely
        await db.upsert_calendar_event(
            event_id="evt_meeting_active",
            title="Architecture Review",
            start_time=now - timedelta(minutes=15),
            end_time=now + timedelta(minutes=15),
            task_type="meeting",
        )
        assert await policy.evaluate() is None
        assert await db.get_pending_recommendations() == []

        # 2. Move past meeting; upcoming coding block in 5 minutes (UC-02) -> recommends "stand"
        now = t0 + timedelta(minutes=80)
        await db.upsert_calendar_event(
            event_id="evt_coding_next",
            title="Backend refactor",
            start_time=now + timedelta(minutes=5),
            end_time=now + timedelta(minutes=65),
            task_type="coding",
        )
        rec_stand = await policy.evaluate()
        assert rec_stand is not None
        assert rec_stand["suggested_action"] == "stand"
        assert rec_stand["action_payload"]["walking_suitable"] is False
        assert rec_stand["action_payload"]["calendar_event_id"] == "evt_coding_next"
        assert "Backend refactor" in rec_stand["message"]
        await db.resolve_recommendation(rec_stand["id"], "dismissed", resolved_at=now)

        # 3. Active walking-suitable admin block -> recommends "walk"
        now = t0 + timedelta(minutes=150)
        await db.upsert_calendar_event(
            event_id="evt_admin_active",
            title="Email triage",
            start_time=now - timedelta(minutes=5),
            end_time=now + timedelta(minutes=25),
            task_type="admin",
        )
        rec_walk = await policy.evaluate()
        assert rec_walk is not None
        assert rec_walk["suggested_action"] == "walk"
        assert rec_walk["action_payload"]["walking_suitable"] is True
        assert rec_walk["action_payload"]["calendar_event_id"] == "evt_admin_active"
        await db.resolve_recommendation(rec_walk["id"], "dismissed", resolved_at=now)

        # 4. Active admin block BUT imminent meeting in 5 minutes -> downgrades to "stand"
        now = now + timedelta(minutes=20)
        await db.upsert_calendar_event(
            event_id="evt_imminent_mtg",
            title="1:1 Sync",
            start_time=now + timedelta(minutes=5),
            end_time=now + timedelta(minutes=35),
            task_type="meeting",
        )
        rec_before_mtg = await policy.evaluate()
        assert rec_before_mtg is not None
        assert rec_before_mtg["suggested_action"] == "stand"
        assert rec_before_mtg["action_payload"]["walking_suitable"] is False
    finally:
        await policy.stop()


@pytest.mark.parametrize("resolution_status", ["accepted", "snoozed", "dismissed"])
async def test_cooldown_suppression_across_pending_and_resolved_states(
    db: Database,
    test_settings: Settings,
    resolution_status: str,
) -> None:
    t0 = datetime(2026, 11, 3, 11, 0, 0).astimezone()
    now = t0 + timedelta(minutes=50)

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(test_settings, recommendation_cooldown_minutes=30),
        clock=lambda: now,
    )
    await policy.start()
    try:
        await db.open_activity_span("sitting", "manual_control", started_at=t0)

        # Initial trigger at 50m
        rec1 = await policy.evaluate()
        assert rec1 is not None
        assert len(created_events) == 1

        # While still pending, never re-emits even 2 hours later
        now = t0 + timedelta(minutes=170)
        assert await policy.evaluate() is None
        assert len(created_events) == 1
        assert len(await db.get_pending_recommendations()) == 1

        # Resolve recommendation via recommendation_resolved event
        resolved_at = now
        await bus.publish(
            PlatformEvent(
                type="recommendation_resolved",
                source="api",
                timestamp=resolved_at,
                payload={
                    "recommendation_id": rec1["id"],
                    "policy": "posture_policy",
                    "status": resolution_status,
                },
            )
        )
        assert await db.get_pending_recommendations() == []

        # Within 30-minute cooldown (29m59s after resolution) -> suppressed
        now = resolved_at + timedelta(minutes=29, seconds=59)
        assert await policy.evaluate() is None
        assert len(created_events) == 1

        # At 30-minute cooldown boundary -> re-emits both in SQLite and on EventBus
        now = resolved_at + timedelta(minutes=30)
        rec2 = await policy.evaluate()
        assert rec2 is not None
        assert rec2["id"] != rec1["id"]
        assert rec2["dedupe_key"] == SITTING_DEDUPE_KEY
        assert len(created_events) == 2
        assert created_events[1].dedupe_key == SITTING_DEDUPE_KEY
        assert created_events[1].payload["id"] == rec2["id"]
    finally:
        await policy.stop()


async def test_custom_snooze_minutes_and_fresh_activity_transition_reset(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 13, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    resolver = StateResolver(bus=bus, db=db, settings=test_settings)
    await resolver.start()

    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(
            test_settings,
            max_sitting_minutes=20,
            recommendation_cooldown_minutes=60,
        ),
        clock=lambda: now,
    )
    await policy.start()
    try:
        # Start sitting via Resolver
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="manual_control",
                timestamp=t0,
                payload={"activity": "sitting", "ttl_minutes": 120},
            )
        )

        now = t0 + timedelta(minutes=20)
        rec1 = await policy.evaluate()
        assert rec1 is not None
        assert len(created_events) == 1

        # Snooze with explicit snooze_minutes=10 (< 60m default cooldown)
        snoozed_at = now + timedelta(minutes=1)
        now = snoozed_at
        await db.resolve_recommendation(
            rec1["id"],
            "snoozed",
            metadata={"snooze_minutes": 10},
            resolved_at=snoozed_at,
        )

        # 9 minutes after snooze -> still suppressed
        now = snoozed_at + timedelta(minutes=9)
        assert await policy.evaluate() is None

        # 10 minutes after snooze -> re-emits despite 60m default cooldown
        now = snoozed_at + timedelta(minutes=10)
        rec2 = await policy.evaluate()
        assert rec2 is not None
        assert len(created_events) == 2

        # Accept rec2, stand up for 5 minutes (resetting posture), then sit again
        accepted_at = now + timedelta(minutes=1)
        await db.resolve_recommendation(rec2["id"], "accepted", resolved_at=accepted_at)

        stand_at = accepted_at + timedelta(minutes=1)
        now = stand_at
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="manual_control",
                timestamp=stand_at,
                payload={"activity": "standing", "ttl_minutes": 30},
            )
        )

        new_sit_at = stand_at + timedelta(minutes=5)
        now = new_sit_at
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="manual_control",
                timestamp=new_sit_at,
                payload={"activity": "sitting", "ttl_minutes": 60},
            )
        )

        # 19 minutes into new sitting span -> below 20m threshold
        now = new_sit_at + timedelta(minutes=19)
        assert await policy.evaluate() is None

        # 20 minutes into new sitting span (< 60m since rec2 accepted, but fresh span!) -> emits!
        now = new_sit_at + timedelta(minutes=20)
        rec3 = await policy.evaluate()
        assert rec3 is not None
        assert rec3["id"] not in {rec1["id"], rec2["id"]}
        assert len(created_events) == 3

        # Snooze longer than default cooldown (90m > 60m) extends suppression
        long_snooze_at = now + timedelta(minutes=1)
        await db.resolve_recommendation(
            rec3["id"],
            "snoozed",
            metadata={"snooze_minutes": 90},
            resolved_at=long_snooze_at,
        )
        now = long_snooze_at + timedelta(minutes=75)
        assert await policy.evaluate() is None
        now = long_snooze_at + timedelta(minutes=90)
        rec4 = await policy.evaluate()
        assert rec4 is not None
        assert len(created_events) == 4
    finally:
        await policy.stop()
        await resolver.stop()


async def test_periodic_loop_privacy_global_pause_and_registry_wiring(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 3, 15, 0, 0).astimezone()
    now = t0 + timedelta(minutes=46)

    await db.open_activity_span("walking", "swalk_ble", started_at=t0)

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    # Start with global_pause=True -> periodic ticks must NOT emit
    cfg = _policy_config(test_settings)
    cfg["privacy"]["global_pause"] = True

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=cfg,
        check_interval_seconds=0.015,
        clock=lambda: now,
    )
    await policy.start()
    try:
        await asyncio.sleep(0.05)
        assert created_events == []
        assert await db.get_pending_recommendations() == []

        # Resume privacy at runtime via privacy_mode_changed -> next periodic tick emits!
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=now,
                payload={"global_pause": False, "privacy_paused": False, "dnd_mode": False},
            )
        )
        await _wait_until(lambda: len(created_events) == 1)
        assert created_events[0].dedupe_key == WALKING_DEDUPE_KEY

        # Pause again -> further evaluations suppressed
        await db.resolve_recommendation(
            created_events[0].payload["id"],
            "dismissed",
            resolved_at=now,
        )
        now = now + timedelta(minutes=60)
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=now,
                payload={"global_pause": True},
            )
        )
        assert await policy.evaluate() is None
        assert len(created_events) == 1
    finally:
        await policy.stop()

    # Verify ModuleRegistry can load, start, and stop posture_policy cleanly
    reg_bus = EventBus()
    registry = ModuleRegistry(bus=reg_bus, db=db, settings=test_settings)
    assert await registry.enable_module("posture_policy") is True
    assert registry.is_enabled("posture_policy") is True
    assert await registry.disable_module("posture_policy") is True
    assert registry.is_enabled("posture_policy") is False


async def test_deep_work_sitting_55m_recommends_stand_and_50_10_timer(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 4, 10, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(test_settings, max_sitting_minutes=50),
        clock=lambda: now,
    )
    await policy.start()
    try:
        await db.open_activity_span("sitting", "manual_control", started_at=t0)
        await db.upsert_calendar_event(
            event_id="evt_deep_work",
            title="Deep Work: Distributed Systems",
            start_time=t0,
            end_time=t0 + timedelta(hours=2),
            task_type="deep_work",
        )

        # Sitting for 55m during Deep Work without active timer
        now = t0 + timedelta(minutes=55)
        rec = await policy.evaluate()
        assert rec is not None
        assert rec["policy"] == "posture_policy"
        assert rec["suggested_action"] == "stand"
        assert rec["dedupe_key"] == SITTING_DEDUPE_KEY
        assert rec["action_payload"]["recommended_pomodoro"] == "50_10"
        assert rec["action_payload"]["walking_suitable"] is False
        assert "50/10" in rec["message"]
        assert "Deep Work" in rec["message"]
        assert len(created_events) == 1
        assert created_events[0].payload["action_payload"]["recommended_pomodoro"] == "50_10"
    finally:
        await policy.stop()


async def test_admin_block_sitting_20m_triggers_walking_block_nudge(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 4, 14, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(
            test_settings,
            max_sitting_minutes=50,
            walking_block_sit_threshold_minutes=20,
            recommendation_cooldown_minutes=30,
        ),
        clock=lambda: now,
    )
    await policy.start()
    try:
        await db.open_activity_span("sitting", "manual_control", started_at=t0)
        await db.upsert_calendar_event(
            event_id="evt_admin_triage",
            title="Email & Invoices",
            start_time=t0,
            end_time=t0 + timedelta(hours=2),
            task_type="admin",
        )

        # 19 minutes (< 20m threshold) -> no recommendation
        now = t0 + timedelta(minutes=19, seconds=50)
        assert await policy.evaluate() is None
        assert created_events == []

        # 20 minutes -> walking-suitable block recommendation
        now = t0 + timedelta(minutes=20)
        rec = await policy.evaluate()
        assert rec is not None
        assert rec["suggested_action"] == "walk"
        assert rec["dedupe_key"] == "posture:walk_block:evt_admin_triage"
        assert rec["action_payload"]["recommended_pomodoro"] == "25_5"
        assert rec["action_payload"]["walking_suitable"] is True
        assert rec["action_payload"]["calendar_event_id"] == "evt_admin_triage"
        assert len(created_events) == 1

        # While pending, no duplicate or contradictory stand recommendation is created
        now = t0 + timedelta(minutes=50)
        assert await policy.evaluate() is None
        assert len(created_events) == 1

        # Resolve walking block recommendation
        await db.resolve_recommendation(rec["id"], "dismissed", resolved_at=now)

        # Still within 30m cooldown -> suppressed, no contradictory generic stand
        now = now + timedelta(minutes=20)
        assert await policy.evaluate() is None
        assert len(created_events) == 1
    finally:
        await policy.stop()


async def test_posture_nudges_suppressed_by_ad_hoc_call_and_input_idle(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 4, 11, 0, 0).astimezone()
    now = t0

    bus = EventBus()
    created_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_created", created_events.append)

    policy = PosturePolicyModule(
        bus=bus,
        db=db,
        config=_policy_config(test_settings, max_sitting_minutes=50),
        clock=lambda: now,
    )
    await policy.start()
    try:
        await db.open_activity_span("sitting", "manual_control", started_at=t0)
        now = t0 + timedelta(minutes=60)

        # 1. Ad-hoc call window active -> posture nudge suppressed
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
        assert await policy.evaluate() is None
        assert created_events == []

        # Switch to regular coding window -> nudge fires via event handler
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=now,
                payload={
                    "app_class": "code",
                    "title": "main.py",
                    "is_call_window": False,
                },
            )
        )
        assert len(created_events) == 1
        rec1 = created_events[0].payload
        assert rec1["suggested_action"] == "stand_or_walk"

        # Resolve recommendation
        await db.resolve_recommendation(rec1["id"], "dismissed", resolved_at=now)

        # Reset sitting span at t1
        t1 = now + timedelta(minutes=35)
        now = t1 + timedelta(minutes=60)
        await db.open_activity_span("sitting", "manual_control", started_at=t1)

        # 2. Input idle active -> posture nudge suppressed
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=now,
                payload={"idle": True, "idle_seconds": 300},
            )
        )
        assert await policy.evaluate() is None
        assert len(created_events) == 1

        # Input active again -> nudge fires via event handler
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=now,
                payload={"idle": False},
            )
        )
        assert len(created_events) == 2
        rec2 = created_events[1].payload
        assert rec2["dedupe_key"] == SITTING_DEDUPE_KEY
    finally:
        await policy.stop()
