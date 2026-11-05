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
    fit_pomodoro_preset,
    is_pomodoro_appropriate,
    is_walking_suitable,
    normalize_event_preset,
    parse_cognitive_factors,
    select_pomodoro,
    select_task_preset,
    task_suits_shorter_preset,
)
from jarvis.modules.policies.posture_policy import PosturePolicyModule


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
            end_time=t0 + timedelta(minutes=68),
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
            end_time=now + timedelta(minutes=155),
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
                end_time=now + timedelta(minutes=150),
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
            end_time=now + timedelta(minutes=130),
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
        # 4-hour deep_work block
        await db.upsert_calendar_event(
            event_id="evt_deep_1",
            title="Deep Work: Policies",
            start_time=t0,
            end_time=t0 + timedelta(hours=4),
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
            end_time=now + timedelta(minutes=75),
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
            end_time=t0 + timedelta(minutes=150),
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


async def test_pomodoro_highlights_walking_suitable_and_focused_blocks(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 4, 9, 0, 0).astimezone()
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
        # 1. Admin block highlights walking-suitable 25/5 block
        await db.upsert_calendar_event(
            event_id="evt_admin_highlight",
            title="Inbox Cleanup",
            start_time=t0,
            end_time=t0 + timedelta(minutes=75),
            task_type="admin",
        )
        rec_admin = await policy.evaluate()
        assert rec_admin is not None
        assert "walking-suitable 25/5 block" in rec_admin["message"]
        assert rec_admin["action_payload"]["kind"] == "25_5"
        assert rec_admin["action_payload"]["walking_suitable"] is True
        await db.resolve_recommendation(rec_admin["id"], "dismissed", resolved_at=now)

        # 2. Deep work block highlights focused 50/10 block
        now = t0 + timedelta(minutes=35)
        await db.upsert_calendar_event(
            event_id="evt_deep_highlight",
            title="Kernel Optimization",
            start_time=now,
            end_time=now + timedelta(hours=2, minutes=15),
            task_type="deep_work",
        )
        rec_deep = await policy.evaluate()
        assert rec_deep is not None
        assert "focused 50/10 block" in rec_deep["message"]
        assert rec_deep["action_payload"]["kind"] == "50_10"
        assert rec_deep["action_payload"]["walking_suitable"] is False
    finally:
        await policy.stop()


async def test_pomodoro_skips_when_call_window_active_or_task_unknown(
    db: Database,
    test_settings: Settings,
) -> None:
    t0 = datetime(2026, 11, 4, 11, 0, 0).astimezone()
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
        # 1. Active coding block + active call window -> skips Pomodoro
        await db.upsert_calendar_event(
            event_id="evt_coding_call",
            title="Frontend Dev",
            start_time=t0,
            end_time=t0 + timedelta(hours=2, minutes=30),
            task_type="coding",
        )
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=now,
                payload={
                    "app_class": "teams",
                    "title": "Microsoft Teams",
                    "is_call_window": True,
                },
            )
        )
        assert await policy.evaluate() is None
        assert created_events == []

        # Switch away from call window -> evaluates and creates recommendation
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=now,
                payload={
                    "app_class": "code",
                    "title": "app.tsx",
                    "is_call_window": False,
                },
            )
        )
        assert len(created_events) == 1
        assert created_events[0].dedupe_key == "pomodoro:event:evt_coding_call"
        await db.resolve_recommendation(
            created_events[0].payload["id"],
            "dismissed",
            resolved_at=now,
        )

        # 2. Block with task_type == "unknown" -> skipped
        now = t0 + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_unknown_block",
            title="Random Busy Time",
            start_time=now,
            end_time=now + timedelta(hours=1),
            task_type="unknown",
        )
        assert await policy.evaluate() is None
        assert len(created_events) == 1
    finally:
        await policy.stop()


async def test_pomodoro_appropriateness_flow_and_opt_outs(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify Stage 1 appropriateness gate correctly emits 'no recommendation' for flow states."""
    t0 = datetime(2026, 11, 4, 10, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(test_settings)
    policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)

    # 1. Helper function checks
    assert is_pomodoro_appropriate({"title": "Coding sprint"}, "coding", cfg) is True
    assert is_pomodoro_appropriate({"title": "Deep Work #flow"}, "deep_work", cfg) is False
    assert is_pomodoro_appropriate({"title": "Writing [notimer]"}, "deep_work", cfg) is False
    assert is_pomodoro_appropriate({"title": "Thesis [uninterrupted]"}, "deep_work", cfg) is False
    assert (
        is_pomodoro_appropriate({"title": "Coding", "pomodoro_enabled": False}, "coding", cfg)
        is False
    )
    assert is_pomodoro_appropriate({"title": "Quick sync"}, "meeting", cfg) is False
    assert is_pomodoro_appropriate({"title": "Coffee"}, "break", cfg) is False
    assert is_pomodoro_appropriate({"title": "Misc"}, "unknown", cfg) is False

    # Config opt-out check: when pomodoro.deep_work is None
    no_deep_work_cfg = _pomodoro_config(test_settings, deep_work=None)
    assert is_pomodoro_appropriate({"title": "Deep Work"}, "deep_work", no_deep_work_cfg) is False

    # 2. Integration check: evaluate() skips #flow events
    await policy.start()
    try:
        await db.upsert_calendar_event(
            event_id="evt_flow_1",
            title="Creative Writing #flow",
            start_time=now,
            end_time=now + timedelta(hours=2),
            task_type="deep_work",
        )
        rec = await policy.evaluate(now=now)
        assert rec is None
        assert await db.get_pending_recommendations() == []
    finally:
        await policy.stop()


async def test_pomodoro_duration_fitting_and_downshifting(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify task-driven feasibility: >=2h for 50/10, >=1h for 25/5, <1h suppressed."""
    t0 = datetime(2026, 11, 4, 13, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(test_settings)

    # 0. Task suitability for shorter preset
    assert task_suits_shorter_preset("admin") is True
    assert task_suits_shorter_preset("email") is True
    assert task_suits_shorter_preset("planning") is True
    assert task_suits_shorter_preset("deep_work") is False
    assert task_suits_shorter_preset("coding") is False
    assert task_suits_shorter_preset("deep_work", {"title": "Sprint #shortpomo"}) is True
    assert task_suits_shorter_preset("deep_work", {"description": "[25m] focus"}) is True
    assert task_suits_shorter_preset("deep_work", {"allow_shorter_preset": True}) is True

    # 1. Under one hour (< 60m): no automatic Pomodoro suggestion
    assert fit_pomodoro_preset("50_10", 15.0) == (None, False)
    assert fit_pomodoro_preset("50_10", 45.0) == (None, False)
    assert fit_pomodoro_preset("50_10", 50.0) == (None, False)
    assert fit_pomodoro_preset("50_10", 59.9) == (None, False)
    assert fit_pomodoro_preset("25_5", 15.0) == (None, False)
    assert fit_pomodoro_preset("25_5", 45.0) == (None, False)
    assert fit_pomodoro_preset("25_5", 59.9) == (None, False)

    # 2. Shorter preset (25/5) at >= 1 hour: stays 25/5 and NEVER upgrades
    assert fit_pomodoro_preset("25_5", 60.0) == ("25_5", False)
    assert fit_pomodoro_preset("25_5", 90.0) == ("25_5", False)
    assert fit_pomodoro_preset("25_5", 150.0) == ("25_5", False)

    # 3. 50/10 with 1h to 2h (60m to 119.9m):
    # - If task does NOT suit shorter preset: suggest nothing!
    assert fit_pomodoro_preset("50_10", 60.0, suits_shorter=False) == (None, False)
    assert fit_pomodoro_preset("50_10", 90.0, suits_shorter=False) == (None, False)
    assert fit_pomodoro_preset("50_10", 119.9, suits_shorter=False) == (None, False)
    # - If task DOES suit shorter preset: downshifts to 25_5
    assert fit_pomodoro_preset("50_10", 60.0, suits_shorter=True) == ("25_5", True)
    assert fit_pomodoro_preset("50_10", 90.0, suits_shorter=True) == ("25_5", True)
    assert fit_pomodoro_preset("50_10", 119.9, suits_shorter=True) == ("25_5", True)

    # 4. 50/10 at >= 2 hours (>= 120m): recommends 50_10
    assert fit_pomodoro_preset("50_10", 120.0) == ("50_10", False)
    assert fit_pomodoro_preset("50_10", 150.0) == ("50_10", False)

    # 5. All-day event unconstrained
    assert fit_pomodoro_preset("50_10", None) == ("50_10", False)
    assert fit_pomodoro_preset("25_5", None) == ("25_5", False)

    # 6. Integration via evaluate()
    policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    await policy.start()
    try:
        # A: 45-minute admin block (< 1h) -> suppressed
        await db.upsert_calendar_event(
            event_id="evt_45m_admin",
            title="Short Admin",
            start_time=now,
            end_time=now + timedelta(minutes=45),
            task_type="admin",
        )
        assert await policy.evaluate(now=now) is None

        # B: 55-minute deep work block (< 1h) -> suppressed
        now = now + timedelta(minutes=50)
        await db.upsert_calendar_event(
            event_id="evt_55m_deep",
            title="Short Deep Work",
            start_time=now,
            end_time=now + timedelta(minutes=55),
            task_type="deep_work",
        )
        assert await policy.evaluate(now=now) is None

        # C: Exactly 60-minute admin block (>= 1h) -> recommends 25_5
        now = now + timedelta(hours=1)
        await db.upsert_calendar_event(
            event_id="evt_60m_admin",
            title="One Hour Admin",
            start_time=now,
            end_time=now + timedelta(minutes=60),
            task_type="admin",
        )
        rec_60 = await policy.evaluate(now=now)
        assert rec_60 is not None
        assert rec_60["action_payload"]["kind"] == "25_5"
        assert rec_60["action_payload"]["downshifted"] is False
        await db.resolve_recommendation(rec_60["id"], "accepted", resolved_at=now)

        # D: 90-minute admin block -> recommends 25_5 (NEVER upgrades to 50_10!)
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_90m_admin",
            title="Ninety Min Admin",
            start_time=now,
            end_time=now + timedelta(minutes=90),
            task_type="admin",
        )
        rec_90 = await policy.evaluate(now=now)
        assert rec_90 is not None
        assert rec_90["action_payload"]["kind"] == "25_5"
        assert rec_90["action_payload"]["downshifted"] is False
        await db.resolve_recommendation(rec_90["id"], "accepted", resolved_at=now)

        # E: 90-minute deep work without short fallback -> None
        # (doesn't fit 50_10, doesn't suit 25_5)
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_90m_deep_nosuit",
            title="Deep Research",
            start_time=now,
            end_time=now + timedelta(minutes=90),
            task_type="deep_work",
        )
        assert await policy.evaluate(now=now) is None

        # F: 90-minute deep work WITH #shortpomo tag -> downshifts to 25_5
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_90m_deep_short",
            title="Deep Research #shortpomo",
            start_time=now,
            end_time=now + timedelta(minutes=90),
            task_type="deep_work",
        )
        rec_short = await policy.evaluate(now=now)
        assert rec_short is not None
        assert rec_short["action_payload"]["kind"] == "25_5"
        assert rec_short["action_payload"]["downshifted"] is True
        await db.resolve_recommendation(rec_short["id"], "accepted", resolved_at=now)

        # G: 110-minute deep work without short fallback -> None (< 120m)
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_110m_deep",
            title="Long Deep Work",
            start_time=now,
            end_time=now + timedelta(minutes=110),
            task_type="deep_work",
        )
        assert await policy.evaluate(now=now) is None

        # H: Exactly 120-minute deep work block (>= 2h) -> recommends 50_10
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_120m_deep",
            title="Two Hour Deep Work",
            start_time=now,
            end_time=now + timedelta(minutes=120),
            task_type="deep_work",
        )
        rec_120 = await policy.evaluate(now=now)
        assert rec_120 is not None
        assert rec_120["action_payload"]["kind"] == "50_10"
        assert rec_120["action_payload"]["downshifted"] is False
        await db.resolve_recommendation(rec_120["id"], "accepted", resolved_at=now)

        # I: 150-minute coding block (>= 2h) -> recommends 50_10
        now = now + timedelta(hours=3)
        await db.upsert_calendar_event(
            event_id="evt_150m_coding",
            title="Long Coding Session",
            start_time=now,
            end_time=now + timedelta(minutes=150),
            task_type="coding",
        )
        rec_150 = await policy.evaluate(now=now)
        assert rec_150 is not None
        assert rec_150["action_payload"]["kind"] == "50_10"
        assert rec_150["action_payload"]["downshifted"] is False
    finally:
        await policy.stop()


async def test_pomodoro_meeting_boundary_protection(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify that available time before meetings enforces feasibility thresholds."""
    t0 = datetime(2026, 11, 4, 15, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(test_settings)
    policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    await policy.start()
    try:
        # Active admin block with 75m available before meeting (>= 1h) -> recommends 25_5
        await db.upsert_calendar_event(
            event_id="evt_admin_work",
            title="Triage & Prep",
            start_time=now,
            end_time=now + timedelta(minutes=120),
            task_type="admin",
        )
        await db.upsert_calendar_event(
            event_id="evt_meeting_soon",
            title="Customer Sync Meeting",
            start_time=now + timedelta(minutes=75),
            end_time=now + timedelta(minutes=120),
            task_type="meeting",
        )

        rec = await policy.evaluate(now=now)
        assert rec is not None
        assert rec["action_payload"]["kind"] == "25_5"

        await db.resolve_recommendation(rec["id"], "accepted", resolved_at=now)

        # If meeting starts in 45 minutes (< 1h) -> under 1 hour, suppressed!
        now = now + timedelta(minutes=30)
        rec2 = await policy.evaluate(now=now)
        assert rec2 is None
    finally:
        await policy.stop()


async def test_pomodoro_stepped_dismissal_backoff_and_reset(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify stepped backoff on repeated dismissals and reset on acceptance."""
    t0 = datetime(2026, 11, 4, 9, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(test_settings, recommendation_cooldown_minutes=30)
    policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    await policy.start()
    try:
        await db.upsert_calendar_event(
            event_id="evt_persist_work",
            title="Big Project",
            start_time=now,
            end_time=now + timedelta(hours=4),
            task_type="deep_work",
        )

        # First recommendation
        rec1 = await policy.evaluate(now=now)
        assert rec1 is not None

        # 1st dismissal: 30m base cooldown
        await db.resolve_recommendation(rec1["id"], "dismissed", resolved_at=now)
        now = now + timedelta(minutes=25)
        assert await policy.evaluate(now=now) is None  # within 30m
        now = now + timedelta(minutes=6)  # at 31m: re-emits!
        rec2 = await policy.evaluate(now=now)
        assert rec2 is not None

        # 2nd consecutive dismissal: 2x base cooldown = 60m
        await db.resolve_recommendation(rec2["id"], "dismissed", resolved_at=now)
        now = now + timedelta(minutes=45)
        assert await policy.evaluate(now=now) is None  # within 60m
        now = now + timedelta(minutes=20)  # at 65m: re-emits!
        rec3 = await policy.evaluate(now=now)
        assert rec3 is not None

        # 3rd consecutive dismissal: suppressed for remainder of calendar block!
        await db.resolve_recommendation(rec3["id"], "dismissed", resolved_at=now)
        now = now + timedelta(minutes=90)
        assert await policy.evaluate(now=now) is None  # still suppressed within event block!

        # Acceptance resets the backoff counter
        now = now + timedelta(hours=3)
        await db.upsert_calendar_event(
            event_id="evt_persist_work_2",
            title="Another Project",
            start_time=now,
            end_time=now + timedelta(hours=4),
            task_type="deep_work",
        )
        rec4 = await policy.evaluate(now=now)
        assert rec4 is not None
        await db.resolve_recommendation(rec4["id"], "accepted", resolved_at=now)

        # Next cooldown is back to normal baseline (30m)
        now = now + timedelta(minutes=31)
        rec5 = await policy.evaluate(now=now)
        assert rec5 is not None
    finally:
        await policy.stop()


async def test_posture_independence_when_pomodoro_skipped(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify movement/posture recommendations fire independently even when Pomodoro is skipped."""
    t0 = datetime(2026, 11, 4, 8, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(test_settings)

    pomo_policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    posture_policy = PosturePolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)

    await pomo_policy.start()
    await posture_policy.start()
    try:
        # Calendar event is an un-timeboxed flow block
        await db.upsert_calendar_event(
            event_id="evt_flow_research",
            title="Deep Research #flow",
            start_time=now,
            end_time=now + timedelta(hours=3),
            task_type="deep_work",
        )

        # Pomodoro policy intentionally emits None
        pomo_rec = await pomo_policy.evaluate(now=now)
        assert pomo_rec is None

        # User sits down at desk
        await db.open_activity_span(
            activity="sitting",
            source="manual",
            confidence=1.0,
            reason="test_start",
            started_at=now,
        )

        # After 50 minutes of sitting, posture policy MUST still emit stand recommendation!
        now = now + timedelta(minutes=51)
        posture_rec = await posture_policy.evaluate(now=now)
        assert posture_rec is not None
        assert posture_rec["suggested_action"] in {"stand", "walk", "stand_or_walk"}
        assert posture_rec["action_payload"]["activity"] == "sitting"
        assert posture_rec["action_payload"]["elapsed_minutes"] >= 50
    finally:
        await pomo_policy.stop()
        await posture_policy.stop()


async def test_acceptance_resets_only_relevant_cooldown_history(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify accepting a recommendation resets ONLY its relevant history."""
    t0 = datetime(2026, 11, 4, 9, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(
        test_settings,
        recommendation_cooldown_minutes=30,
        max_sitting_minutes=50,
    )

    pomo_policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    posture_policy = PosturePolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)

    await pomo_policy.start()
    await posture_policy.start()
    try:
        # 1. Establish 2 consecutive dismissals on posture policy (stepping cooldown to 60m)
        await db.open_activity_span(activity="sitting", source="manual", started_at=t0)
        now = t0 + timedelta(minutes=55)
        posture_rec1 = await posture_policy.evaluate(now=now)
        assert posture_rec1 is not None
        await db.resolve_recommendation(posture_rec1["id"], "dismissed", resolved_at=now)

        # 31m later: fires again -> 2nd dismissal
        now = now + timedelta(minutes=31)
        posture_rec2 = await posture_policy.evaluate(now=now)
        assert posture_rec2 is not None
        await db.resolve_recommendation(posture_rec2["id"], "dismissed", resolved_at=now)

        # Confirm posture policy is in 60m cooldown (at 45m after rec2, returns None)
        now = now + timedelta(minutes=45)
        assert await posture_policy.evaluate(now=now) is None

        # 2. Meanwhile, a Pomodoro recommendation is generated and ACCEPTED
        await db.upsert_calendar_event(
            event_id="evt_pomo_isolated",
            title="Isolated Coding Sprint",
            start_time=now,
            end_time=now + timedelta(hours=4),
            task_type="coding",
        )
        pomo_rec = await pomo_policy.evaluate(now=now)
        assert pomo_rec is not None
        assert pomo_rec["dedupe_key"] == "pomodoro:event:evt_pomo_isolated"

        # ACCEPT the Pomodoro recommendation
        await db.resolve_recommendation(pomo_rec["id"], "accepted", resolved_at=now)

        # 3. VERIFY ISOLATION:
        # Accepting Pomodoro MUST NOT reset the 60m posture cooldown!
        # Posture cooldown was set at rec2. We are currently at rec2 + 45m.
        # Since 45m < 60m, posture policy MUST still return None!
        assert await posture_policy.evaluate(now=now) is None

        # Advancing past the 60m posture cooldown (e.g. at rec2 + 65m, which is now + 20m)
        now = now + timedelta(minutes=20)
        # Now posture alert fires because its own cooldown expired, not because of Pomodoro accept
        posture_rec3 = await posture_policy.evaluate(now=now)
        assert posture_rec3 is not None

        # And Pomodoro history for evt_pomo_isolated is reset to standard 30m baseline
        assert await pomo_policy.evaluate(now=now) is None  # within 30m of accept
        now = now + timedelta(minutes=15)  # at 35m after accept: re-evaluates
        assert await pomo_policy.evaluate(now=now) is not None
    finally:
        await pomo_policy.stop()
        await posture_policy.stop()


async def test_pomodoro_cognitive_intensity_and_restart_cost_preset_selection(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify cognitive intensity and restart cost guide preset selection without rigid maps."""
    t0 = datetime(2026, 11, 4, 10, 0, 0).astimezone()
    now = t0
    bus = EventBus()
    cfg = _pomodoro_config(test_settings)

    # 1. Direct unit checks on select_task_preset
    # Factor A: Mental effort / difficulty sustaining attention favors 25/5
    assert select_task_preset({"title": "Kernel Debugging #intense"}, "coding", cfg) == "25_5"
    assert (
        select_task_preset({"title": "Dense Proofs", "mental_effort": "high"}, "deep_work", cfg)
        == "25_5"
    )
    assert (
        select_task_preset({"title": "Algorithm Work", "difficulty": "demanding"}, "coding", cfg)
        == "25_5"
    )
    assert select_task_preset(
        {"title": "Refactoring", "difficulty_sustaining_attention": True}, "coding", cfg
    ) == "25_5"
    assert select_task_preset({"title": "Sprint Task (debugging)"}, "deep_work", cfg) == "25_5"

    # Factor B: High restart cost / uninterrupted concentration favors 50/10
    assert (
        select_task_preset({"title": "Core Architecture #architecture"}, "planning", cfg)
        == "50_10"
    )
    assert (
        select_task_preset({"title": "System Design", "restart_cost": "high"}, "planning", cfg)
        == "50_10"
    )
    assert select_task_preset({"title": "Strategy Doc #deep"}, "admin", cfg) == "50_10"

    # Explicit user preferences & event tags take precedence
    assert select_task_preset({"title": "Quick Sprint #25m"}, "deep_work", cfg) == "25_5"
    assert select_task_preset({"title": "Big Sprint #50m"}, "admin", cfg) == "50_10"
    assert select_task_preset({"preferred_preset": "25_5"}, "deep_work", cfg) == "25_5"
    assert select_task_preset({"preferred_preset": "50_10"}, "admin", cfg) == "50_10"
    assert select_task_preset({"preferred_preset": "none"}, "deep_work", cfg) is None

    # Limited context: fallback to conservative configurable defaults
    assert select_task_preset({"title": "Routine Coding"}, "coding", cfg) == "50_10"
    assert select_task_preset({"title": "Weekly Planning"}, "planning", cfg) == "25_5"

    # 2. Integration via policy.evaluate()
    policy = PomodoroPolicyModule(bus=bus, db=db, config=cfg, clock=lambda: now)
    await policy.start()
    try:
        # A. High mental effort coding block with 90m available:
        # Selects 25/5 as its primary preset; 90m >= 60m fits without downshifting!
        await db.upsert_calendar_event(
            event_id="evt_demanding_coding",
            title="Async Kernel Bug #intense",
            start_time=now,
            end_time=now + timedelta(minutes=90),
            task_type="coding",
        )
        rec_demand = await policy.evaluate(now=now)
        assert rec_demand is not None
        assert rec_demand["action_payload"]["kind"] == "25_5"
        assert rec_demand["action_payload"]["downshifted"] is False
        assert "Suggested Pomodoro timer: 25/5 block" in rec_demand["message"]
        await db.resolve_recommendation(rec_demand["id"], "accepted", resolved_at=now)

        # B. High restart cost planning block with 90m available:
        # Selects 50/10 as primary preset; 90m < 120m does not fit 50/10,
        # and high restart cost prevents downshifting to 25/5 -> returns None!
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_arch_planning_short",
            title="Platform Migration #architecture",
            start_time=now,
            end_time=now + timedelta(minutes=90),
            task_type="planning",
        )
        assert await policy.evaluate(now=now) is None

        # C. High restart cost planning block with 150m available (>= 2h):
        # 150m >= 120m fits 50/10 -> recommends 50/10!
        now = now + timedelta(hours=2)
        await db.upsert_calendar_event(
            event_id="evt_arch_planning_long",
            title="Platform Migration #architecture",
            start_time=now,
            end_time=now + timedelta(minutes=150),
            task_type="planning",
        )
        rec_arch = await policy.evaluate(now=now)
        assert rec_arch is not None
        assert rec_arch["action_payload"]["kind"] == "50_10"
        assert rec_arch["action_payload"]["downshifted"] is False
        assert "Suggested Pomodoro timer: focused 50/10 block" in rec_arch["message"]
    finally:
        await policy.stop()


def test_preset_validation_and_consistent_cognitive_factors(test_settings: Settings) -> None:
    """Verify alias normalization, config override validation, and cognitive factor consistency."""
    cfg = _pomodoro_config(test_settings)

    # 1. normalize_event_preset alias mapping
    assert normalize_event_preset("25_5") == "25_5"
    assert normalize_event_preset("25m") == "25_5"
    assert normalize_event_preset("25/5") == "25_5"
    assert normalize_event_preset("25") == "25_5"
    assert normalize_event_preset("50_10") == "50_10"
    assert normalize_event_preset("50m") == "50_10"
    assert normalize_event_preset("50/10") == "50_10"
    assert normalize_event_preset("50") == "50_10"
    assert normalize_event_preset("none") == "none"
    assert normalize_event_preset("off") == "none"
    assert normalize_event_preset("no_timer") == "none"
    assert normalize_event_preset("notimer") == "none"
    assert normalize_event_preset("custom_45") is None
    assert normalize_event_preset(123) is None
    assert normalize_event_preset(None) is None

    # 2. Config opt-out override validation
    opt_out_cfg = dict(cfg)
    opt_out_cfg["deep_work"] = None  # User disabled Pomodoro for deep_work by default

    # Valid preset override revives the timer
    valid_override_evt = {"title": "Special Session", "preferred_preset": "25_5"}
    assert is_pomodoro_appropriate(valid_override_evt, "deep_work", opt_out_cfg) is True
    assert select_task_preset(valid_override_evt, "deep_work", opt_out_cfg) == "25_5"

    # Invalid preset CANNOT override config opt-out (even if title has #intense)
    invalid_override_evt = {"title": "Special Session #intense", "preferred_preset": "invalid_99"}
    assert is_pomodoro_appropriate(invalid_override_evt, "deep_work", opt_out_cfg) is False

    # Explicit 'none' preset remains opted out
    none_override_evt = {"title": "Special Session", "preferred_preset": "none"}
    assert is_pomodoro_appropriate(none_override_evt, "deep_work", opt_out_cfg) is False

    # 3. parse_cognitive_factors consistency across selection and fallback
    # concentration_need consistently indicates high restart cost
    conc_event = {"title": "Deep Architecture", "concentration_need": "high"}
    eff, cost = parse_cognitive_factors(conc_event)
    assert eff is False
    assert cost is True
    # In select_task_preset: favors 50/10
    assert select_task_preset(conc_event, "planning", cfg) == "50_10"
    # In task_suits_shorter_preset: high restart cost suppresses downshift
    assert task_suits_shorter_preset("planning", conc_event, cfg) is False

    # #shortpomo tag allows shorter-preset fallback, but does NOT force 25/5 in select_task_preset
    short_fallback_evt = {"title": "Deep Research #shortpomo"}
    assert select_task_preset(short_fallback_evt, "deep_work", cfg) == "50_10"
    assert task_suits_shorter_preset("deep_work", short_fallback_evt, cfg) is True

    # #25m tag explicitly requests 25/5 in select_task_preset AND suits shorter preset
    short_exact_evt = {"title": "Deep Research #25m"}
    assert select_task_preset(short_exact_evt, "deep_work", cfg) == "25_5"
    assert task_suits_shorter_preset("deep_work", short_exact_evt, cfg) is True

    # 4. available_minutes=None represents an unconstrained schedule boundary
    assert fit_pomodoro_preset("50_10", None) == ("50_10", False)
    assert fit_pomodoro_preset("25_5", None) == ("25_5", False)


