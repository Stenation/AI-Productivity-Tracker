"""Unit and integration tests for FocusPolicyModule (`observe_only` guarantee)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import Settings
from jarvis.core.timer import TimerManager
from jarvis.modules.inputs.hyprland_ipc import HyprlandIpcModule
from jarvis.modules.outputs.desktop_notify import DesktopNotifyModule
from jarvis.modules.policies.focus_policy import FocusPolicyModule, classify_focus_context


@pytest.fixture(autouse=True)
def _isolate_hyprland_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent any ambient host Hyprland socket env vars from leaking into tests."""
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    xdg_dir = tmp_path / "xdg_runtime"
    xdg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))


def test_classify_focus_context_rules() -> None:
    """Verify deterministic focus classification rules and X/Firefox boundary safety."""
    assert classify_focus_context("code", "focus_policy.py - jarvis") == (
        "focused",
        "app_class",
    )
    assert classify_focus_context("Zotero", "Paper PDF") == ("focused", "app_class")
    assert classify_focus_context("firefox", "Twitch - Stream — Mozilla Firefox") == (
        "distracted",
        "title_pattern",
    )
    assert classify_focus_context("firefox", "FastAPI Lifespan Docs — Mozilla Firefox") == (
        "unknown",
        "default",
    )
    assert classify_focus_context(
        "firefox",
        "Zoom Meeting — Mozilla Firefox",
        calendar_task_type="meeting",
    ) == ("neutral", "calendar_task")
    assert classify_focus_context("", "") == ("neutral", "desktop")


async def test_focus_policy_subscriptions_and_context_classification(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify FocusPolicyModule subscribes to all 4 topics and classifies observed context."""
    t_now = datetime(2026, 11, 4, 9, 30, 0).astimezone()
    bus = EventBus()

    policy = FocusPolicyModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        clock=lambda: t_now,
    )
    await policy.start()
    try:
        health_initial = await policy.health()
        assert health_initial["module"] == "focus_policy"
        assert health_initial["status"] == "ok"
        assert health_initial["mode"] == "observe_only"
        assert health_initial["configured_mode"] == "observe_only"

        # 1. Calendar context changed -> active coding block
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=t_now,
                payload={
                    "current_event": {
                        "event_id": "evt_deep_1",
                        "title": "Deep Work: Architecture",
                        "task_type": "deep_work",
                        "start_time": (t_now - timedelta(minutes=10)).isoformat(),
                        "end_time": (t_now + timedelta(minutes=50)).isoformat(),
                    }
                },
            )
        )

        # 2. Window focus changed with `work` alias -> normalized to `focused`
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=t_now,
                payload={
                    "app_class": "code",
                    "title": "resolver.py - jarvis",
                    "category": "work",
                },
            )
        )
        snap_focused = await policy.get_context_snapshot(now=t_now + timedelta(seconds=45))
        assert snap_focused["focus_state"] == "focused"
        assert snap_focused["app_class"] == "code"
        assert snap_focused["window_title"] == "resolver.py - jarvis"
        assert snap_focused["window_elapsed_seconds"] == 45
        assert snap_focused["calendar_event_id"] == "evt_deep_1"
        assert snap_focused["is_productivity_block"] is True

        # 3. Window focus changed with `distraction` alias -> normalized to `distracted`
        t_dist = t_now + timedelta(minutes=2)
        t_now = t_dist
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=t_dist,
                payload={
                    "app_class": "firefox",
                    "title": "Reddit - /r/archlinux",
                    "category": "distraction",
                },
            )
        )
        snap_dist = await policy.get_context_snapshot(now=t_dist + timedelta(seconds=30))
        assert snap_dist["focus_state"] == "distracted"
        assert snap_dist["distraction_grace_exceeded"] is False

        # 4. Input idle changed -> tracked cleanly
        t_idle = t_dist + timedelta(minutes=1)
        t_now = t_idle
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t_idle,
                payload={"idle": True, "idle_seconds": 60},
            )
        )
        snap_idle = await policy.get_context_snapshot(now=t_idle)
        assert snap_idle["idle"] is True
        assert snap_idle["idle_since"] is not None

        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t_idle + timedelta(seconds=10),
                payload={"idle": False},
            )
        )
        assert (await policy.get_context_snapshot(now=t_idle + timedelta(seconds=10)))[
            "idle"
        ] is False
    finally:
        await policy.stop()

    # After stop(), events no longer mutate stopped module state
    count_after_stop = (await policy.health())["observed_window_events"]
    await bus.publish(
        PlatformEvent(
            type="window_focus_changed",
            source="hyprland_ipc",
            payload={"app_class": "kitty", "title": "htop"},
        )
    )
    assert (await policy.health())["observed_window_events"] == count_after_stop


async def test_focus_policy_strict_observe_only_zero_warnings_during_prolonged_distraction(
    db: Database,
    test_settings: Settings,
) -> None:
    """Guarantee ZERO recommendations, notifications, screenshots, or roasts in Milestone 2."""
    t0 = datetime(2026, 11, 4, 10, 0, 0).astimezone()
    t_cursor = t0

    # Even if a caller configures aggressive flags, Milestone 2 FocusPolicyModule MUST stay
    # strictly observe_only and emit zero nudges or screen-check requests.
    cfg = test_settings.model_dump()
    cfg["privacy"]["hyprland_ipc_mode"] = "notify_on_distraction"
    cfg["privacy"]["allow_screenshots"] = True
    cfg["privacy"]["allow_vlm_screenshots"] = True
    cfg["privacy"]["allow_roast_mode"] = True
    cfg["privacy"]["dnd_mode"] = False
    cfg["focus"]["distraction_grace_seconds"] = 60

    await db.upsert_calendar_event(
        event_id="evt_coding_focus",
        title="Core Coding Sprint",
        start_time=t0 - timedelta(minutes=5),
        end_time=t0 + timedelta(minutes=55),
        task_type="coding",
        recommended_pomodoro="50_10",
    )

    bus = EventBus()
    emitted_events: list[PlatformEvent] = []
    bus.subscribe("*", emitted_events.append)

    notify_calls: list[list[str]] = []

    async def recording_notify(argv: list[str]) -> int:
        notify_calls.append(list(argv))
        return 0

    timer_manager = TimerManager(bus=bus, db=db, settings=test_settings)
    await timer_manager.start()
    await timer_manager.start_timer(kind="50_10", calendar_event_id="evt_coding_focus")

    notifier = DesktopNotifyModule(
        bus=bus,
        db=db,
        config=cfg,
        runner=recording_notify,
        clock=lambda: t_cursor,
    )
    await notifier.start()

    policy = FocusPolicyModule(
        bus=bus,
        db=db,
        config=cfg,
        clock=lambda: t_cursor,
    )
    await policy.start()

    try:
        assert policy.configured_mode() == "notify_on_distraction"
        assert policy.effective_mode() == "observe_only"

        # User opens YouTube / Reddit / Twitch for 25 minutes straight during coding + 50/10 timer
        for offset_mins, title in (
            (0, "Funny Cat Compilation - YouTube — Mozilla Firefox"),
            (8, "Reddit - /r/gaming — Mozilla Firefox"),
            (16, "Twitch - Live Stream — Mozilla Firefox"),
        ):
            t_cursor = t0 + timedelta(minutes=offset_mins)
            await bus.publish(
                PlatformEvent(
                    type="window_focus_changed",
                    source="hyprland_ipc",
                    timestamp=t_cursor,
                    payload={
                        "app_class": "firefox",
                        "title": title,
                        "category": "distracted",
                    },
                )
            )
            t_eval = t_cursor + timedelta(minutes=7)
            assert await policy.evaluate(now=t_eval) is None
            snap = await policy.get_context_snapshot(now=t_eval)
            assert snap["focus_state"] == "distracted"
            assert snap["distraction_grace_exceeded"] is True
            assert snap["mode"] == "observe_only"

        # Strict assertions: ZERO recommendations, ZERO screen checks, ZERO notifications
        assert await db.list_recommendations() == []
        assert await db.get_pending_recommendations() == []
        forbidden_types = {
            "recommendation_created",
            "screen_check_requested",
            "screen_check_completed",
        }
        assert [e for e in emitted_events if e.type in forbidden_types] == []
        assert notify_calls == []
    finally:
        await policy.stop()
        await notifier.stop()
        await timer_manager.stop()


async def test_focus_policy_privacy_pause_and_registry_e2e_integration(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify privacy global_pause clearing and full FastAPI + ModuleRegistry wiring."""
    t0 = datetime(2026, 11, 4, 15, 0, 0).astimezone()
    bus = EventBus()

    policy = FocusPolicyModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        clock=lambda: t0,
    )
    await policy.start()
    try:
        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=t0,
                payload={
                    "app_class": "obsidian",
                    "title": "Research Notes",
                    "category": "focused",
                },
            )
        )
        assert (await policy.get_context_snapshot(now=t0))["focus_state"] == "focused"

        # Enable global privacy pause -> clears retained window title/class and ignores events
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t0 + timedelta(minutes=1),
                payload={"global_pause": True, "privacy_paused": True},
            )
        )
        paused_snap = await policy.get_context_snapshot(now=t0 + timedelta(minutes=1))
        assert paused_snap["privacy_paused"] is True
        assert paused_snap["focus_state"] == "unknown"
        assert paused_snap["app_class"] is None
        assert paused_snap["window_title"] is None

        await bus.publish(
            PlatformEvent(
                type="window_focus_changed",
                source="hyprland_ipc",
                timestamp=t0 + timedelta(minutes=2),
                payload={
                    "app_class": "firefox",
                    "title": "Secret Document",
                    "category": "unknown",
                },
            )
        )
        assert (await policy.get_context_snapshot(now=t0 + timedelta(minutes=2)))[
            "window_title"
        ] is None
    finally:
        await policy.stop()

    # ModuleRegistry enable/disable + `/api/state` focus integration with HyprlandIpcModule
    test_settings.enabled_modules.policies = ["focus_policy"]
    app = create_app(settings=test_settings)
    async with app.router.lifespan_context(app):
        registry: ModuleRegistry = app.state.registry
        assert registry.is_enabled("focus_policy") is True

        reader = asyncio.StreamReader()
        hypr = HyprlandIpcModule(
            bus=app.state.bus,
            db=app.state.db,
            config=test_settings.model_dump(),
            stream_reader=reader,
        )
        await hypr.start()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                await hypr.handle_raw_line("activewindow>>code,focus_policy.py - jarvis\n")
                state_focused = (await client.get("/api/state")).json()
                assert state_focused["focus"] == "focused"
                assert state_focused["pending_recommendation_count"] == 0

                await hypr.handle_raw_line(
                    "activewindow>>firefox,YouTube - Music — Mozilla Firefox\n"
                )
                state_distracted = (await client.get("/api/state")).json()
                assert state_distracted["focus"] == "distracted"
                assert state_distracted["pending_recommendation_count"] == 0

                win_hist = (await client.get("/api/history/windows")).json()
                assert win_hist["count"] == 2

                mods = (await client.get("/api/modules")).json()["modules"]
                by_name = {m["module"]: m for m in mods}
                assert by_name["focus_policy"]["enabled"] is True
                assert by_name["focus_policy"]["status"] == "ok"
        finally:
            reader.feed_eof()
            await hypr.stop()
