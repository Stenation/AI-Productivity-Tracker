"""Tests for TimerManager, ManualControlModule, and core `/api/*` REST endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, SystemStateSnapshot
from jarvis.core.settings import Settings
from jarvis.core.timer import TimerManager
from jarvis.web.api import router as api_router


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


@pytest.fixture
async def api_env(
    test_settings: Settings,
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    """Create a FastAPI app with `web.api.router`, `TimerManager`, and `manual_control`."""
    test_settings.enabled_modules.inputs = ["manual_control"]
    app = create_app(settings=test_settings)
    app.include_router(api_router)

    async with app.router.lifespan_context(app):
        timer_manager = TimerManager(
            bus=app.state.bus,
            db=app.state.db,
            settings=app.state.settings,
            tick_interval_seconds=0.01,
        )
        await timer_manager.start()
        app.state.timer_manager = timer_manager

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            try:
                yield app, client
            finally:
                await timer_manager.stop()


async def test_manual_control_set_and_full_cycle_persistence(
    api_env: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = api_env
    db: Database = app.state.db
    app.state.settings.posture.min_session_duration_seconds = 0

    # Initial /api/state snapshot
    state_resp = await client.get("/api/state")
    assert state_resp.status_code == 200
    state_body = state_resp.json()
    assert state_body["activity"] == "unknown"
    assert state_body["override_expires_at"] is None
    assert state_body["timer"]["active"] is False
    assert state_body["privacy_paused"] is False
    assert state_body["dnd_mode"] is False
    assert state_body["pending_recommendation_count"] == 0
    # Must validate cleanly against SystemStateSnapshot schema
    snapshot = SystemStateSnapshot.model_validate(state_body)
    assert snapshot.activity == "unknown"

    # First cycle from initial "unknown" -> "sitting", then full cycle:
    # sitting -> standing -> walking -> away -> sitting
    expected_sequence = ["sitting", "standing", "walking", "away", "sitting"]
    for expected in expected_sequence:
        cycle_resp = await client.post("/api/activity/cycle")
        assert cycle_resp.status_code == 200
        cycle_data = cycle_resp.json()
        assert cycle_data["activity"] == expected
        assert cycle_data["source"] == "manual_control"
        assert cycle_data["confidence"] == 1.0
        assert cycle_data["reason"] == "manual_override"
        assert cycle_data["override_expires_at"] is not None

    # Explicit POST /api/activity with custom ttl_minutes
    set_resp = await client.post(
        "/api/activity",
        json={"activity": "walking", "ttl_minutes": 45},
    )
    assert set_resp.status_code == 200
    set_data = set_resp.json()
    assert set_data["activity"] == "walking"
    assert set_data["confidence"] == 1.0
    started_dt = datetime.fromisoformat(set_data["started_at"])
    expires_dt = datetime.fromisoformat(set_data["override_expires_at"])
    assert int(round((expires_dt - started_dt).total_seconds())) == 45 * 60

    # HTMX form-encoded submission also works cleanly
    form_resp = await client.post(
        "/api/activity",
        data={"activity": "away", "ttl_minutes": "15"},
    )
    assert form_resp.status_code == 200
    assert form_resp.json()["activity"] == "away"

    # Invalid manual state ("unknown") is rejected with 422
    bad_resp = await client.post("/api/activity", json={"activity": "unknown"})
    assert bad_resp.status_code == 422

    # History endpoint reflects all persisted manual spans
    hist_resp = await client.get("/api/history/activity")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["count"] == 7
    assert hist_data["spans"][0]["activity"] == "away"
    assert hist_data["spans"][0]["ended_at"] is None
    assert hist_data["spans"][1]["activity"] == "walking"
    assert hist_data["spans"][1]["ended_at"] is not None
    assert "away" in hist_data["totals_seconds"]

    # Verify SQLite rows directly
    raw_spans = await db.list_activity_spans()
    assert len(raw_spans) == 7
    assert all(s["source"] == "manual_control" for s in raw_spans)
    assert all(s["confidence"] == 1.0 for s in raw_spans)


async def test_api_accidental_cycle_under_30s_resumes_sitting_session(
    api_env: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = api_env
    db: Database = app.state.db

    # User sits down initially with 60m override
    post_resp = await client.post(
        "/api/activity",
        json={"activity": "sitting", "ttl_minutes": 60},
    )
    assert post_resp.status_code == 200
    current_span = await db.get_current_activity_span()
    assert current_span is not None
    span_id = current_span["id"]

    # Backdate started_at to 30 minutes ago
    t0 = datetime.now().astimezone() - timedelta(minutes=30)
    await db.connection.execute(
        "UPDATE activity_spans SET started_at = ? WHERE id = ?",
        (t0.isoformat(), span_id),
    )
    await db.connection.commit()
    await app.state.resolver.start()

    state0 = (await client.get("/api/state")).json()
    assert state0["activity"] == "sitting"
    assert state0["elapsed_seconds"] >= 1800

    # User accidentally cycles: stand -> walk -> away -> sit in under 30s
    c1 = await client.post("/api/activity/cycle")
    assert c1.json()["activity"] == "standing"

    c2 = await client.post("/api/activity/cycle")
    assert c2.json()["activity"] == "walking"

    c3 = await client.post("/api/activity/cycle")
    assert c3.json()["activity"] == "away"

    c4 = await client.post("/api/activity/cycle")
    assert c4.json()["activity"] == "sitting"

    # State should have resumed sitting at ~30m, NOT restarted at 0!
    state_after = (await client.get("/api/state")).json()
    assert state_after["activity"] == "sitting"
    assert state_after["elapsed_seconds"] >= 1800

    # History should only contain the 1 sitting span, NOT the accidental stand/walk/away spans
    hist = (await client.get("/api/history/activity")).json()
    assert hist["count"] == 1
    assert hist["spans"][0]["activity"] == "sitting"
    assert hist["spans"][0]["ended_at"] is None


async def test_timer_manager_work_break_completed_transitions(
    db: Database,
) -> None:
    current_time = datetime(2026, 11, 3, 14, 0, 0).astimezone()

    def fake_clock() -> datetime:
        return current_time

    bus = EventBus()
    timer_events: list[PlatformEvent] = []
    bus.subscribe("timer_state_changed", timer_events.append)

    tm = TimerManager(
        bus=bus,
        db=db,
        tick_interval_seconds=0.01,
        clock=fake_clock,
    )
    await tm.start()

    started = await tm.start_timer("25_5", calendar_event_id="evt_admin")
    assert started["active"] is True
    assert started["kind"] == "25_5"
    assert started["phase"] == "work"
    assert started["planned_duration_seconds"] == 1500
    assert started["remaining_seconds"] == 1500
    assert started["started_at"] == current_time.isoformat()
    assert len(timer_events) == 1
    assert timer_events[0].payload["transition"] == "started"

    # Advance 10 minutes into 25-minute work block
    current_time = current_time + timedelta(minutes=10)
    await asyncio.sleep(0.03)
    mid_status = tm.get_status()
    assert mid_status["active"] is True
    assert mid_status["phase"] == "work"
    assert mid_status["remaining_seconds"] == 900

    # Advance to end of 25-minute work block -> transitions automatically to 5-minute break
    current_time = current_time + timedelta(minutes=15)
    await _wait_until(lambda: tm.get_status()["phase"] == "break")

    break_status = tm.get_status()
    assert break_status["active"] is True
    assert break_status["kind"] == "25_5"
    assert break_status["phase"] == "break"
    assert break_status["planned_duration_seconds"] == 300
    assert break_status["remaining_seconds"] == 300
    assert any(ev.payload.get("transition") == "break_started" for ev in timer_events)

    # Advance 5 minutes through break -> transitions automatically to completed/idle
    current_time = current_time + timedelta(minutes=5)
    await _wait_until(lambda: tm.get_status()["active"] is False)

    done_status = tm.get_status()
    assert done_status["active"] is False
    assert done_status["phase"] == "idle"
    assert done_status["kind"] is None
    assert done_status["remaining_seconds"] == 0
    assert done_status["planned_duration_seconds"] is None
    assert done_status["started_at"] is None
    assert timer_events[-1].payload["transition"] == "completed"
    assert timer_events[-1].payload["status"] == "completed"

    spans = await db.list_timer_spans(limit=10)
    assert len(spans) == 2
    assert spans[0]["kind"] == "25_5_break"
    assert spans[0]["status"] == "completed"
    assert spans[0]["actual_duration_seconds"] == 300
    assert spans[1]["kind"] == "25_5"
    assert spans[1]["status"] == "completed"
    assert spans[1]["actual_duration_seconds"] == 1500
    await tm.stop()


async def test_api_timer_presets_custom_and_stop_persistence(
    api_env: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = api_env
    api_db: Database = app.state.db

    start_50_resp = await client.post("/api/timers/start", json={"kind": "50_10"})
    assert start_50_resp.status_code == 200
    body_50 = start_50_resp.json()
    assert body_50["active"] is True
    assert body_50["kind"] == "50_10"
    assert body_50["phase"] == "work"
    assert body_50["planned_duration_seconds"] == 3000

    state_with_timer = (await client.get("/api/state")).json()
    assert state_with_timer["timer"]["active"] is True
    assert state_with_timer["timer"]["kind"] == "50_10"
    assert state_with_timer["active_timer"] is not None

    # Starting a custom timer replaces the running 50_10 timer cleanly
    custom_resp = await client.post(
        "/api/timers/start",
        json={"kind": "custom", "work_minutes": 15, "break_minutes": 3},
    )
    assert custom_resp.status_code == 200
    custom_body = custom_resp.json()
    assert custom_body["kind"] == "custom"
    assert custom_body["planned_duration_seconds"] == 900

    stop_resp = await client.post("/api/timers/stop", json={"reason": "stopped"})
    assert stop_resp.status_code == 200
    stop_body = stop_resp.json()
    assert stop_body["stopped"] is True
    assert stop_body["active"] is False

    # Unknown preset without work_minutes returns 422
    invalid_timer_resp = await client.post(
        "/api/timers/start",
        json={"kind": "99_99"},
    )
    assert invalid_timer_resp.status_code == 422

    api_spans = await api_db.list_timer_spans(limit=10)
    assert len(api_spans) == 2
    assert all(row["status"] == "stopped" for row in api_spans)
    assert await api_db.get_active_timer() is None


async def test_timer_startup_marks_leftover_running_timers_interrupted_without_resuming(
    db: Database,
) -> None:
    t0 = datetime(2026, 11, 3, 9, 0, 0).astimezone()
    startup_time = t0 + timedelta(minutes=12, seconds=20)

    completed_row = await db.start_timer_span(
        kind="25_5",
        planned_duration_seconds=1500,
        started_at=t0 - timedelta(hours=1),
    )
    await db.end_timer_span(
        completed_row["id"],
        status="completed",
        ended_at=t0 - timedelta(minutes=35),
    )

    leftover_running = await db.start_timer_span(
        kind="50_10",
        planned_duration_seconds=3000,
        calendar_event_id="evt_crash",
        started_at=t0,
    )
    assert await db.get_active_timer() is not None

    bus = EventBus()
    tm = TimerManager(
        bus=bus,
        db=db,
        tick_interval_seconds=0.05,
        clock=lambda: startup_time,
    )
    interrupted_count = await tm.start()
    try:
        assert interrupted_count == 1
        # Mandatory rule: leftover running timer is NEVER silently resumed
        status = tm.get_status()
        assert status["active"] is False
        assert status["phase"] == "idle"
        assert status["kind"] is None
        assert status["remaining_seconds"] == 0
        assert await db.get_active_timer() is None

        spans_by_id = {row["id"]: row for row in await db.list_timer_spans(limit=10)}
        interrupted_row = spans_by_id[leftover_running["id"]]
        assert interrupted_row["status"] == "interrupted"
        assert interrupted_row["ended_at"] == startup_time.isoformat()
        assert interrupted_row["actual_duration_seconds"] == 12 * 60 + 20

        # Previously completed timer remains untouched
        assert spans_by_id[completed_row["id"]]["status"] == "completed"
        assert spans_by_id[completed_row["id"]]["actual_duration_seconds"] == 1500
    finally:
        await tm.stop()


async def test_recommendations_accept_snooze_dismiss_non_destructive_and_waybar_shortcut(
    api_env: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = api_env
    db: Database = app.state.db
    bus: EventBus = app.state.bus
    timer_manager: TimerManager = app.state.timer_manager

    resolved_events: list[PlatformEvent] = []
    timer_events: list[PlatformEvent] = []
    activity_events: list[PlatformEvent] = []
    bus.subscribe("recommendation_resolved", resolved_events.append)
    bus.subscribe("timer_state_changed", timer_events.append)
    bus.subscribe("activity_reported", activity_events.append)
    bus.subscribe("activity_state_changed", activity_events.append)

    t0 = datetime(2026, 11, 3, 15, 0, 0).astimezone()
    rec_accept = await db.create_recommendation(
        policy="pomodoro_policy",
        message="Start a 50/10 block and stand up.",
        suggested_action="start_timer_and_stand",
        action_payload={"kind": "50_10", "activity": "standing"},
        created_at=t0,
    )
    rec_dismiss = await db.create_recommendation(
        policy="posture_policy",
        message="You have been sitting for 55 minutes.",
        suggested_action="stand_or_walk",
        created_at=t0 + timedelta(seconds=10),
    )
    rec_snooze = await db.create_recommendation(
        policy="meeting_policy",
        message="Meeting starts in 10 minutes.",
        suggested_action="return_to_desk",
        created_at=t0 + timedelta(seconds=20),
    )

    list_resp = await client.get("/api/recommendations")
    assert list_resp.status_code == 200
    assert list_resp.json()["count"] == 3
    assert (await client.get("/api/state")).json()["pending_recommendation_count"] == 3

    # Milestone 1 non-destructive rule: accepting MUST NOT auto-start timers or change activity
    accept_resp = await client.post(
        f"/api/recommendations/{rec_accept['id']}/accept",
        json={"metadata": {"surface": "dashboard"}},
    )
    assert accept_resp.status_code == 200
    accepted_rec = accept_resp.json()["recommendation"]
    assert accepted_rec["status"] == "accepted"
    assert accepted_rec["resolved_at"] is not None
    assert accepted_rec["resolution_metadata"] == {"surface": "dashboard"}

    assert timer_manager.get_status()["active"] is False
    assert await db.get_active_timer() is None
    assert timer_events == []
    assert app.state.resolver.get_effective_state().activity == "unknown"
    assert await db.list_activity_spans() == []
    assert activity_events == []
    assert len(resolved_events) == 1
    assert resolved_events[0].payload["status"] == "accepted"
    assert resolved_events[0].payload["recommendation_id"] == rec_accept["id"]

    # Dismiss by ID
    dismiss_resp = await client.post(
        f"/api/recommendations/{rec_dismiss['id']}/dismiss",
        json={"reason": "busy"},
    )
    assert dismiss_resp.status_code == 200
    assert dismiss_resp.json()["recommendation"]["status"] == "dismissed"
    assert dismiss_resp.json()["recommendation"]["resolution_metadata"] == {
        "reason": "busy"
    }

    # Snooze by ID
    snooze_id_resp = await client.post(
        f"/api/recommendations/{rec_snooze['id']}/snooze",
        json={"snooze_minutes": 10},
    )
    assert snooze_id_resp.status_code == 200
    assert snooze_id_resp.json()["recommendation"]["status"] == "snoozed"
    assert snooze_id_resp.json()["recommendation"]["resolution_metadata"] == {
        "snooze_minutes": 10
    }
    assert (await client.get("/api/recommendations")).json()["count"] == 0

    # Waybar convenience endpoint `POST /api/recommendations/snooze` (no ID)
    older_pending = await db.create_recommendation(
        policy="posture_policy",
        message="Older pending nudge",
        created_at=t0 + timedelta(minutes=1),
    )
    newest_pending = await db.create_recommendation(
        policy="focus_policy",
        message="Most recent pending nudge",
        created_at=t0 + timedelta(minutes=2),
    )

    waybar_snooze_1 = await client.post("/api/recommendations/snooze")
    assert waybar_snooze_1.status_code == 200
    wb_body_1 = waybar_snooze_1.json()
    assert wb_body_1["snoozed"] is True
    assert wb_body_1["recommendation"]["id"] == newest_pending["id"]
    assert wb_body_1["recommendation"]["status"] == "snoozed"

    # Older pending is still pending until snoozed next
    remaining_pending = (await client.get("/api/recommendations")).json()["recommendations"]
    assert [r["id"] for r in remaining_pending] == [older_pending["id"]]

    waybar_snooze_2 = await client.post("/api/recommendations/snooze")
    assert waybar_snooze_2.status_code == 200
    assert waybar_snooze_2.json()["recommendation"]["id"] == older_pending["id"]

    # When no pending recommendations exist, Waybar snooze is a clean no-op
    waybar_noop = await client.post("/api/recommendations/snooze")
    assert waybar_noop.status_code == 200
    assert waybar_noop.json()["snoozed"] is False
    assert waybar_noop.json()["status"] == "noop"

    # Non-existent recommendation ID returns 404
    missing_resp = await client.post("/api/recommendations/99999/accept")
    assert missing_resp.status_code == 404
    assert len(resolved_events) == 5


async def test_privacy_controls_and_external_event_ingestion(
    api_env: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = api_env
    db: Database = app.state.db
    bus: EventBus = app.state.bus

    privacy_events: list[PlatformEvent] = []
    phone_events: list[PlatformEvent] = []
    window_events: list[PlatformEvent] = []
    bus.subscribe("privacy_mode_changed", privacy_events.append)
    bus.subscribe("phone_state_changed", phone_events.append)
    bus.subscribe("window_focus_changed", window_events.append)

    # 1. Normal external worker event ingestion via POST /api/events
    ev_resp = await client.post(
        "/api/events",
        json={
            "type": "window_focus_changed",
            "source": "hyprland_ipc",
            "payload": {"app_class": "kitty", "title": "nvim", "category": "focused"},
            "dedupe_key": "win-1",
        },
    )
    assert ev_resp.status_code == 200
    assert ev_resp.json()["dispatched"] is True
    assert len(window_events) == 1

    # Duplicate dedupe_key reports duplicate and is not re-dispatched
    dup_resp = await client.post(
        "/api/events",
        json={
            "type": "window_focus_changed",
            "source": "hyprland_ipc",
            "payload": {"app_class": "kitty", "title": "nvim", "category": "focused"},
            "dedupe_key": "win-1",
        },
    )
    assert dup_resp.status_code == 200
    assert dup_resp.json()["dispatched"] is False
    assert dup_resp.json()["status"] == "duplicate"
    assert len(window_events) == 1

    # 2. Phone webhook event blocked when allow_phone_webhook=False (default)
    phone_payload: dict[str, Any] = {
        "type": "phone_state_changed",
        "source": "s24_webhook",
        "payload": {"unlocked_seconds": 65},
    }
    blocked_phone = await client.post("/api/events", json=phone_payload)
    assert blocked_phone.status_code == 403
    assert len(phone_events) == 0

    # Enabling allow_phone_webhook allows phone event ingestion
    app.state.settings.privacy.allow_phone_webhook = True
    allowed_phone = await client.post("/api/events", json=phone_payload)
    assert allowed_phone.status_code == 200
    assert allowed_phone.json()["dispatched"] is True
    assert len(phone_events) == 1

    # 3. Privacy pause updates runtime state, persists to DB settings, emits event,
    #    and suppresses external /api/events ingestion
    pause_resp = await client.post("/api/privacy/pause")
    assert pause_resp.status_code == 200
    assert pause_resp.json()["global_pause"] is True
    assert app.state.settings.privacy.global_pause is True
    assert await db.get_setting("privacy.global_pause") is True
    assert (await db.get_setting("privacy"))["global_pause"] is True
    assert len(privacy_events) == 1
    assert privacy_events[-1].payload["global_pause"] is True

    paused_ingest = await client.post(
        "/api/events",
        json={
            "type": "window_focus_changed",
            "source": "hyprland_ipc",
            "payload": {"app_class": "firefox", "title": "Reddit"},
        },
    )
    assert paused_ingest.status_code == 200
    assert paused_ingest.json()["dispatched"] is False
    assert paused_ingest.json()["reason"] == "global_pause"
    assert len(window_events) == 1

    # 4. Privacy resume restores normal ingestion
    resume_resp = await client.post("/api/privacy/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["global_pause"] is False
    assert await db.get_setting("privacy.global_pause") is False
    assert len(privacy_events) == 2
    assert privacy_events[-1].payload["global_pause"] is False

    resumed_ingest = await client.post(
        "/api/events",
        json={
            "type": "window_focus_changed",
            "source": "hyprland_ipc",
            "payload": {"app_class": "firefox", "title": "Docs"},
        },
    )
    assert resumed_ingest.status_code == 200
    assert resumed_ingest.json()["dispatched"] is True
    assert len(window_events) == 2

    # 5. DND toggle & explicit set updates runtime, DB settings, /api/state, and EventBus
    dnd_toggle_on = await client.post("/api/privacy/dnd")
    assert dnd_toggle_on.status_code == 200
    assert dnd_toggle_on.json()["dnd_mode"] is True
    assert app.state.settings.privacy.dnd_mode is True
    assert await db.get_setting("privacy.dnd_mode") is True
    assert (await client.get("/api/state")).json()["dnd_mode"] is True
    assert len(privacy_events) == 3
    assert privacy_events[-1].payload["dnd_mode"] is True

    dnd_off = await client.post("/api/privacy/dnd", json={"dnd_mode": False})
    assert dnd_off.status_code == 200
    assert dnd_off.json()["dnd_mode"] is False
    assert await db.get_setting("privacy.dnd_mode") is False
    assert len(privacy_events) == 4
    assert privacy_events[-1].payload["dnd_mode"] is False


async def test_modules_endpoints_and_window_calendar_state(
    api_env: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = api_env
    db: Database = app.state.db

    now_utc = datetime.now(tz=UTC)
    await db.upsert_calendar_event(
        event_id="evt_now",
        title="Deep Work: Control Plane",
        start_time=now_utc - timedelta(minutes=10),
        end_time=now_utc + timedelta(minutes=50),
        task_type="coding",
        suitable_for_walking=False,
        recommended_pomodoro="50_10",
    )
    await db.upsert_calendar_event(
        event_id="evt_next",
        title="Architecture Sync",
        start_time=now_utc + timedelta(minutes=55),
        end_time=now_utc + timedelta(minutes=85),
        task_type="meeting",
        suitable_for_walking=False,
    )
    await db.open_window_span(
        app_class="code",
        window_title="api.py - Jarvis",
        category="focused",
        classifier="app_class",
        started_at=now_utc - timedelta(minutes=5),
    )

    state = (await client.get("/api/state")).json()
    assert state["current_event"] is not None
    assert state["current_event"]["event_id"] == "evt_now"
    assert state["next_event"] is not None
    assert state["next_event"]["event_id"] == "evt_next"
    assert state["focus"] == "focused"

    win_hist = (await client.get("/api/history/windows")).json()
    assert win_hist["count"] == 1
    assert win_hist["spans"][0]["app_class"] == "code"
    assert win_hist["totals_seconds"]["focused"] >= 299

    # Module listing, runtime disable, and re-enable
    mods_resp = await client.get("/api/modules")
    assert mods_resp.status_code == 200
    mods_by_name = {m["module"]: m for m in mods_resp.json()["modules"]}
    assert mods_by_name["manual_control"]["enabled"] is True
    assert mods_by_name["manual_control"]["status"] == "ok"

    dis_resp = await client.post("/api/modules/manual_control/disable")
    assert dis_resp.status_code == 200
    assert dis_resp.json()["enabled"] is False
    assert dis_resp.json()["status"] == "disabled"

    en_resp = await client.post("/api/modules/manual_control/enable")
    assert en_resp.status_code == 200
    assert en_resp.json()["enabled"] is True
    assert en_resp.json()["status"] == "ok"

    unknown_mod_resp = await client.post("/api/modules/does_not_exist_xyz/enable")
    assert unknown_mod_resp.status_code == 404
