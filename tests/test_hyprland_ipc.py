"""Unit and integration tests for HyprlandIpcModule window and idle observation."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import PROJECT_ROOT, Settings
from jarvis.modules.inputs.hyprland_ipc import (
    HyprlandIpcModule,
    classify_window,
    parse_socket2_line,
    resolve_hyprland_socket2_path,
)
from jarvis.modules.inputs.manual_control import ManualControlModule


@pytest.fixture(autouse=True)
def _isolate_hyprland_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent any ambient host Hyprland socket env vars from leaking into tests."""
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    xdg_dir = tmp_path / "xdg_runtime"
    xdg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))


@pytest.fixture
def short_sock_dir() -> Iterator[Path]:
    """Provide a short `/tmp` directory so AF_UNIX socket paths stay well under 108 bytes."""
    with tempfile.TemporaryDirectory(prefix="jv-hypr-") as raw_dir:
        yield Path(raw_dir)


async def _wait_until(
    predicate: Any,
    *,
    timeout: float = 2.0,
    interval: float = 0.01,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        res = predicate()
        if asyncio.iscoroutine(res):
            res = await res
        if res:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for async condition")
        await asyncio.sleep(interval)


def test_socket_path_resolution_xdg_fallback_and_overrides(tmp_path: Path) -> None:
    """Verify `.socket2.sock` path resolution order and explicit overrides."""
    explicit_ctor = tmp_path / "ctor.sock"
    explicit_cfg = tmp_path / "cfg.sock"

    # 1. Explicit constructor socket_path wins over everything
    assert (
        resolve_hyprland_socket2_path(
            socket_path=explicit_ctor,
            config={"socket_path": str(explicit_cfg)},
            env={"HYPRLAND_INSTANCE_SIGNATURE": "sig1", "XDG_RUNTIME_DIR": str(tmp_path)},
        )
        == explicit_ctor
    )

    # 2. Explicit config socket_path wins over env vars
    assert (
        resolve_hyprland_socket2_path(
            config={"hyprland_ipc": {"socket_path": str(explicit_cfg)}},
            env={"HYPRLAND_INSTANCE_SIGNATURE": "sig1", "XDG_RUNTIME_DIR": str(tmp_path)},
        )
        == explicit_cfg
    )

    # 3. Missing HYPRLAND_INSTANCE_SIGNATURE returns None
    assert (
        resolve_hyprland_socket2_path(
            env={"XDG_RUNTIME_DIR": str(tmp_path)},
        )
        is None
    )

    # 4. XDG_RUNTIME_DIR/hypr/$HIS/.socket2.sock is preferred first
    xdg_root = tmp_path / "xdg"
    xdg_sock = xdg_root / "hypr" / "sig_xdg" / ".socket2.sock"
    xdg_sock.parent.mkdir(parents=True)
    xdg_sock.touch()

    assert (
        resolve_hyprland_socket2_path(
            env={
                "HYPRLAND_INSTANCE_SIGNATURE": "sig_xdg",
                "XDG_RUNTIME_DIR": str(xdg_root),
            },
        )
        == xdg_sock
    )

    # 5. Falls back to /tmp/hypr/$HIS/.socket2.sock when XDG socket is absent on disk
    unique_sig = f"jarvis_test_sig_{tmp_path.name}"
    tmp_sock = Path("/tmp") / "hypr" / unique_sig / ".socket2.sock"
    tmp_sock.parent.mkdir(parents=True, exist_ok=True)
    tmp_sock.touch()
    try:
        assert (
            resolve_hyprland_socket2_path(
                env={
                    "HYPRLAND_INSTANCE_SIGNATURE": unique_sig,
                    "XDG_RUNTIME_DIR": str(xdg_root),
                },
            )
            == tmp_sock
        )
        # Also resolves /tmp fallback when XDG_RUNTIME_DIR is unset
        assert (
            resolve_hyprland_socket2_path(
                env={"HYPRLAND_INSTANCE_SIGNATURE": unique_sig},
            )
            == tmp_sock
        )
    finally:
        tmp_sock.unlink(missing_ok=True)
        with pytest.raises(OSError):
            # Ignore if non-empty, otherwise remove cleanly
            if not any(tmp_sock.parent.iterdir()):
                tmp_sock.parent.rmdir()
                raise OSError("cleaned")


def test_parse_socket2_line_and_classify_window_precedence() -> None:
    """Verify `.socket2.sock` parsing (including commas/`>>` in titles) and classification."""
    assert parse_socket2_line("workspace>>2\n") is None
    assert parse_socket2_line("activewindowv2>>55d8f2a0\n") is None
    assert parse_socket2_line("malformed_line_without_separator\n") is None

    # Titles containing commas and `>>`
    assert parse_socket2_line(
        b"activewindow>>code,hyprland_ipc.py, line 42, col 8 >> Visual Studio Code\n"
    ) == ("code", "hyprland_ipc.py, line 42, col 8 >> Visual Studio Code")

    # Empty / desktop focus variants
    assert parse_socket2_line("activewindow>>,\n") == ("", "")
    assert parse_socket2_line("activewindow>>\n") == ("", "")

    # Classification precedence:
    # 1. work_app_classes beats distraction keywords in file names
    res1 = classify_window("code", "reddit_scraper.py - Visual Studio Code")
    assert res1.category == "focused"
    assert res1.classifier == "app_class"
    assert res1.is_call_window is False

    res2 = classify_window("Alacritty", "zsh")
    assert res2.category == "focused"
    assert res2.classifier == "app_class"

    # 2. distraction_title_patterns catches YouTube/Reddit/X without false-matching Firefox
    res3 = classify_window("firefox", "Arch Linux Wiki — Mozilla Firefox")
    assert res3.category == "unknown"
    assert res3.classifier == "default"

    res4 = classify_window("firefox", "Reddit - /r/hyprland — Mozilla Firefox")
    assert res4.category == "distracted"
    assert res4.classifier == "title_rule"

    res5 = classify_window("firefox", "Home / X — Mozilla Firefox")
    assert res5.category == "distracted"
    assert res5.classifier == "title_rule"

    res6 = classify_window("firefox", "Lo-Fi Beats - YouTube — Mozilla Firefox")
    assert res6.category == "distracted"
    assert res6.classifier == "title_rule"

    # 3. Calendar task correlation for unmatched apps
    res7 = classify_window(
        "firefox",
        "Google Meet — Mozilla Firefox",
        calendar_task_type="meeting",
    )
    assert res7.category == "focused"
    assert res7.classifier == "calendar_correlation"
    assert res7.is_call_window is True

    res8 = classify_window(
        "mpv",
        "Break stretch video",
        calendar_task_type="break",
    )
    assert res8.category == "neutral"
    assert res8.classifier == "calendar_correlation"

    # 4. Empty desktop
    res_desktop = classify_window("", "")
    assert res_desktop.category == "neutral"
    assert res_desktop.classifier == "desktop"
    assert res_desktop.confidence == 1.0


async def test_unix_socket_stream_dedupe_desktop_calendar_link_and_stop_close(
    db: Database,
    test_settings: Settings,
    short_sock_dir: Path,
) -> None:
    """Test live Unix domain socket streaming, comma titles, dedupe, desktop focus, and SQLite."""
    sock_path = short_sock_dir / "s2.sock"
    connected_writers: list[asyncio.StreamWriter] = []
    client_connected = asyncio.Event()

    async def _handle_client(
        _reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        connected_writers.append(writer)
        client_connected.set()

    server = await asyncio.start_unix_server(_handle_client, path=str(sock_path))

    t_cursor = datetime(2026, 11, 4, 10, 0, 0).astimezone()

    def clock() -> datetime:
        return t_cursor

    # Seed an active coding event in calendar_cache so window_spans link calendar_event_id
    await db.upsert_calendar_event(
        event_id="cal_coding_1",
        title="Backend Hyprland IPC",
        start_time=t_cursor - timedelta(minutes=15),
        end_time=t_cursor + timedelta(minutes=45),
        task_type="coding",
    )

    bus = EventBus()
    focus_events: list[PlatformEvent] = []
    bus.subscribe("window_focus_changed", focus_events.append)

    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        socket_path=sock_path,
        clock=clock,
    )

    try:
        await mod.start()
        await asyncio.wait_for(client_connected.wait(), timeout=2.0)
        assert (await mod.health())["status"] == "ok"

        writer = connected_writers[0]

        # 1. Window with commas in title + duplicate consecutive sub-events + non-window noise
        writer.write(
            b"workspace>>1\n"
            b"activewindow>>code,schema.py, settings.py - jarvis, Workspace\n"
            b"activewindowv2>>55aa01\n"
            b"activewindow>>code,schema.py, settings.py - jarvis, Workspace\n"
        )
        await writer.drain()
        await _wait_until(lambda: len(focus_events) == 1)

        assert focus_events[0].payload["app_class"] == "code"
        assert (
            focus_events[0].payload["title"]
            == "schema.py, settings.py - jarvis, Workspace"
        )
        assert focus_events[0].payload["category"] == "focused"
        assert focus_events[0].payload["calendar_event_id"] == "cal_coding_1"

        spans_after_first = await db.list_window_spans()
        assert len(spans_after_first) == 1
        assert spans_after_first[0]["app_class"] == "code"
        assert (
            spans_after_first[0]["window_title"]
            == "schema.py, settings.py - jarvis, Workspace"
        )
        assert spans_after_first[0]["category"] == "focused"
        assert spans_after_first[0]["calendar_event_id"] == "cal_coding_1"
        assert spans_after_first[0]["ended_at"] is None
        assert spans_after_first[0]["metadata"] == {
            "category": "focused",
            "classifier": "app_class",
            "confidence": 0.95,
            "is_call_window": False,
        }

        # 2. Switch to distracted browser tab -> closes previous span, opens distracted span
        t_cursor = t_cursor + timedelta(minutes=5)
        writer.write(b"activewindow>>firefox,Reddit, r/linux, top posts - Mozilla Firefox\n")
        await writer.drain()
        await _wait_until(lambda: len(focus_events) == 2)

        assert focus_events[1].payload["category"] == "distracted"
        cur_span = await db.get_current_window_span()
        assert cur_span is not None
        assert cur_span["app_class"] == "firefox"
        assert cur_span["category"] == "distracted"
        assert cur_span["classifier"] == "title_rule"
        assert cur_span["metadata"] == {
            "category": "distracted",
            "classifier": "title_rule",
            "confidence": 0.90,
            "is_call_window": False,
        }

        # 3. Switch to empty workspace (`activewindow>>,`) -> closes current span, no blank row
        t_cursor = t_cursor + timedelta(minutes=3)
        writer.write(b"activewindow>>,\n")
        await writer.drain()
        await _wait_until(lambda: len(focus_events) == 3)

        assert focus_events[2].payload["app_class"] == ""
        assert focus_events[2].payload["title"] == ""
        assert focus_events[2].payload["span_id"] is None
        assert await db.get_current_window_span() is None
        assert len(await db.list_window_spans()) == 2

        # 4. Focus an unclassified utility app -> opens `unknown` span
        t_cursor = t_cursor + timedelta(minutes=2)
        writer.write(b"activewindow>>pavucontrol,Volume Control\n")
        await writer.drain()
        await _wait_until(lambda: len(focus_events) == 4)

        assert focus_events[3].payload["category"] == "unknown"
        pavu_span = await db.get_current_window_span()
        assert pavu_span is not None
        assert pavu_span["classifier"] == "default"
        assert pavu_span["metadata"] == {
            "category": "unknown",
            "classifier": "default",
            "confidence": 0.0,
            "is_call_window": False,
        }

        # 5. Stopping module cleanly closes the currently open window span
        t_cursor = t_cursor + timedelta(minutes=4)
        await mod.stop()

        assert await db.get_current_window_span() is None
        all_spans = await db.list_window_spans()
        assert len(all_spans) == 3
        assert all(span["ended_at"] is not None for span in all_spans)
        assert all_spans[0]["ended_at"] == t_cursor.isoformat()
    finally:
        await mod.stop()
        for w in connected_writers:
            w.close()
        server.close()
        await server.wait_closed()


async def test_hyprctl_seed_missing_socket_degraded_isolation_and_reconnect(
    db: Database,
    test_settings: Settings,
    short_sock_dir: Path,
) -> None:
    """Verify hyprctl seeding, safe degraded isolation, and bounded-backoff reconnect."""
    t0 = datetime(2026, 11, 4, 11, 0, 0).astimezone()
    bus = EventBus()
    health_events: list[PlatformEvent] = []
    focus_events: list[PlatformEvent] = []
    bus.subscribe("module_health_changed", health_events.append)
    bus.subscribe("window_focus_changed", focus_events.append)

    # Part A: StreamReader + injectable hyprctl_runner seeds initial active window & dedupes
    reader = asyncio.StreamReader()

    async def fake_hyprctl(argv: list[str]) -> str:
        assert argv[-2:] == ["activewindow", "-j"]
        return json.dumps({"class": "kitty", "title": "nvim src/jarvis/core/app.py"})

    seeded_mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        hyprctl_runner=fake_hyprctl,
        clock=lambda: t0,
    )
    await seeded_mod.start()
    try:
        assert len(focus_events) == 1
        assert focus_events[0].payload["app_class"] == "kitty"
        assert focus_events[0].payload["category"] == "focused"
        seeded_span = await db.get_current_window_span()
        assert seeded_span is not None
        assert seeded_span["app_class"] == "kitty"

        # Feeding the identical initial window via socket2 does not duplicate the span
        reader.feed_data(b"activewindow>>kitty,nvim src/jarvis/core/app.py\n")
        reader.feed_eof()
        await asyncio.sleep(0.03)
        assert len(focus_events) == 1
        assert len(await db.list_window_spans()) == 1
    finally:
        await seeded_mod.stop()

    # Part B: Start when socket does NOT exist yet -> marks `degraded`, core stays healthy;
    #         then create socket server -> reconnects with bounded backoff and marks `ok`!
    delayed_sock = short_sock_dir / "delayed.sock"
    reconn_mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        socket_path=delayed_sock,
        reconnect_initial_seconds=0.02,
        reconnect_max_seconds=0.08,
        clock=lambda: t0 + timedelta(minutes=10),
    )
    await reconn_mod.start()

    server: asyncio.AbstractServer | None = None
    writers: list[asyncio.StreamWriter] = []
    try:
        # Immediately degraded without crashing
        initial_health = await reconn_mod.health()
        assert initial_health["status"] == "degraded"
        assert "not found" in str(initial_health["last_error"])

        db_health = {row["module"]: row for row in await db.list_module_health()}
        assert db_health["hyprland_ipc"]["status"] == "degraded"
        assert any(
            evt.payload.get("module") == "hyprland_ipc"
            and evt.payload.get("status") == "degraded"
            for evt in health_events
        )

        # Now start the Unix socket server so bounded backoff reconnect succeeds
        async def _on_reconnected(
            _r: asyncio.StreamReader,
            w: asyncio.StreamWriter,
        ) -> None:
            writers.append(w)
            w.write(b"activewindow>>obsidian,Daily Note - Obsidian\n")
            await w.drain()

        server = await asyncio.start_unix_server(_on_reconnected, path=str(delayed_sock))

        await _wait_until(lambda: len(focus_events) >= 2)
        await _wait_until(
            lambda: reconn_mod._last_error is None
        )

        recovered_health = await reconn_mod.health()
        assert recovered_health["status"] == "ok"
        assert recovered_health["last_error"] is None

        db_health_recovered = {row["module"]: row for row in await db.list_module_health()}
        assert db_health_recovered["hyprland_ipc"]["status"] == "ok"
        assert focus_events[-1].payload["app_class"] == "obsidian"
        assert focus_events[-1].payload["category"] == "focused"
    finally:
        await reconn_mod.stop()
        for w in writers:
            w.close()
        if server is not None:
            server.close()
            await server.wait_closed()

    # Part C: Enabling `hyprland_ipc` in full FastAPI app with NO Hyprland running
    #         marks `hyprland_ipc` degraded in SQLite without disrupting core/timers/Waybar
    test_settings.enabled_modules.inputs = ["manual_control", "hyprland_ipc"]
    test_settings.enabled_modules.outputs = ["waybar_status", "desktop_notify"]
    app = create_app(settings=test_settings)
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.02)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8000",
        ) as client:
            # Core API, manual posture cycling, timers, and Waybar remain 100% operational
            cycle_resp = await client.post("/api/activity/cycle")
            assert cycle_resp.status_code == 200
            assert cycle_resp.json()["activity"] == "sitting"

            timer_resp = await client.post("/api/timers/start", json={"kind": "25_5"})
            assert timer_resp.status_code == 200
            assert timer_resp.json()["active"] is True

            assert test_settings.waybar.status_file.exists()

            app_db: Database = app.state.db
            mod_health_rows = {
                row["module"]: row for row in await app_db.list_module_health()
            }
            assert mod_health_rows["hyprland_ipc"]["status"] == "degraded"
            assert "HYPRLAND_INSTANCE_SIGNATURE" in str(
                mod_health_rows["hyprland_ipc"]["last_error"]
            )


async def test_privacy_global_pause_closes_span_and_suppresses_recording(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify `privacy.global_pause` closes open window spans and blocks recording while paused."""
    t_cursor = datetime(2026, 11, 4, 13, 0, 0).astimezone()
    bus = EventBus()
    focus_events: list[PlatformEvent] = []
    bus.subscribe("window_focus_changed", focus_events.append)

    reader = asyncio.StreamReader()
    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        clock=lambda: t_cursor,
    )
    await mod.start()
    try:
        # 1. Normal window focus opens a span
        await mod.handle_raw_line("activewindow>>code,main.py - jarvis\n", now=t_cursor)
        open_span = await db.get_current_window_span()
        assert open_span is not None
        assert open_span["app_class"] == "code"
        assert len(focus_events) == 1

        # 2. Enable global privacy pause -> closes open span immediately
        t_pause = t_cursor + timedelta(minutes=5)
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t_pause,
                payload={"global_pause": True, "privacy_paused": True},
            )
        )
        assert await db.get_current_window_span() is None
        spans_at_pause = await db.list_window_spans()
        assert len(spans_at_pause) == 1
        assert spans_at_pause[0]["ended_at"] == t_pause.isoformat()

        # 3. Sensitive windows while paused are neither stored in SQLite nor published on bus
        t_during_pause = t_pause + timedelta(minutes=2)
        await mod.handle_raw_line(
            "activewindow>>keepassxc,Personal Vault - KeePassXC\n",
            now=t_during_pause,
        )
        await mod.handle_raw_line(
            "activewindow>>firefox,Private Banking — Mozilla Firefox\n",
            now=t_during_pause + timedelta(seconds=30),
        )
        assert await db.get_current_window_span() is None
        assert len(await db.list_window_spans()) == 1
        assert len(focus_events) == 1
        all_titles = [s["window_title"] for s in await db.list_window_spans()]
        assert all("Vault" not in t and "Banking" not in t for t in all_titles)

        # 4. Resume privacy -> subsequent window events record normally again
        t_resume = t_pause + timedelta(minutes=10)
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t_resume,
                payload={"global_pause": False, "privacy_paused": False},
            )
        )
        await mod.handle_raw_line(
            "activewindow>>Alacritty,pytest -v\n",
            now=t_resume + timedelta(seconds=5),
        )
        resumed_span = await db.get_current_window_span()
        assert resumed_span is not None
        assert resumed_span["app_class"] == "Alacritty"
        assert resumed_span["category"] == "focused"
        assert len(focus_events) == 2
    finally:
        reader.feed_eof()
        await mod.stop()


async def test_input_idle_changed_threshold_manual_override_ttl_and_hypridle_conf(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify idle tracking, `away` reporting, manual override TTL, and hypridle example conf."""
    # Configure 1-sample sensor consensus on resolver to test direct manual TTL vs idle sensor
    test_settings.posture.sensor_consensus_samples = 1
    test_settings.posture.sensor_consensus_window_samples = 3

    t0 = datetime(2026, 11, 4, 14, 0, 0).astimezone()
    bus = EventBus()
    resolver = StateResolver(bus=bus, db=db, settings=test_settings)
    await resolver.start()

    manual = ManualControlModule(bus=bus, db=db, config=test_settings.model_dump())
    await manual.start()

    activity_reports: list[PlatformEvent] = []
    bus.subscribe("activity_reported", activity_reports.append)

    reader = asyncio.StreamReader()
    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        idle_threshold_seconds=300.0,
    )
    await mod.start()

    try:
        # 1. Idle below 300s threshold tracks `is_idle` + `idle_since` without emitting `away`
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t0,
                payload={"idle": True, "idle_seconds": 120},
            )
        )
        assert mod.is_idle is True
        assert mod.idle_since == t0 - timedelta(seconds=120)
        assert [e for e in activity_reports if e.source == "hyprland_idle"] == []

        # Resume (`idle=False`) clears idle tracking before threshold
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t0 + timedelta(seconds=30),
                payload={"idle": False, "idle_seconds": 0},
            )
        )
        assert mod.is_idle is False
        assert mod.idle_since is None

        # 2. Manual override (`standing`, TTL=30m) takes precedence over idle `away` within TTL
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="manual_control",
                confidence=1.0,
                timestamp=t0 + timedelta(minutes=1),
                payload={
                    "activity": "standing",
                    "ttl_minutes": 30,
                    "reason": "manual_override",
                },
            )
        )
        assert resolver.get_effective_state(now=t0 + timedelta(minutes=2)).activity == "standing"

        # Idle exceeds 300s at t0 + 10m (inside 30m manual TTL):
        # HyprlandIpcModule emits `activity_reported(source="hyprland_idle", activity="away", 0.7)`
        # AND StateResolver honors manual override TTL (keeps `standing`!)
        t_inside_ttl = t0 + timedelta(minutes=10)
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t_inside_ttl,
                payload={"idle": True, "idle_seconds": 305},
            )
        )
        idle_reports = [e for e in activity_reports if e.source == "hyprland_idle"]
        assert len(idle_reports) == 1
        assert idle_reports[0].payload["activity"] == "away"
        assert idle_reports[0].confidence == pytest.approx(0.7)
        assert resolver.get_effective_state(now=t_inside_ttl).activity == "standing"

        # Reset idle and advance past the 30m manual override TTL (t0 + 35m)
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t0 + timedelta(minutes=12),
                payload={"idle": False},
            )
        )

        t_after_ttl = t0 + timedelta(minutes=35)
        await bus.publish(
            PlatformEvent(
                type="input_idle_changed",
                source="hypridle",
                timestamp=t_after_ttl,
                payload={"idle": True, "idle_seconds": 300},
            )
        )
        eff_after = resolver.get_effective_state(now=t_after_ttl + timedelta(seconds=5))
        assert eff_after.activity == "away"
        assert eff_after.source == "hyprland_idle"
        assert eff_after.confidence == pytest.approx(0.7)
    finally:
        reader.feed_eof()
        await mod.stop()
        await manual.stop()
        await resolver.stop()

    # 3. Injectable async `idle_monitor` stream also drives `away` detection cleanly
    async def _fake_idle_monitor() -> AsyncIterator[dict[str, Any]]:
        yield {"idle": True, "idle_seconds": 310}
        yield {"idle": False, "idle_seconds": 0}

    bus2 = EventBus()
    monitor_reports: list[PlatformEvent] = []
    bus2.subscribe("activity_reported", monitor_reports.append)

    reader2 = asyncio.StreamReader()
    mon_mod = HyprlandIpcModule(
        bus=bus2,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader2,
        idle_monitor=_fake_idle_monitor(),
        idle_threshold_seconds=300.0,
    )
    await mon_mod.start()
    try:
        await _wait_until(lambda: len(monitor_reports) == 1)
        assert monitor_reports[0].source == "hyprland_idle"
        assert monitor_reports[0].payload["activity"] == "away"
        assert monitor_reports[0].confidence == pytest.approx(0.7)
    finally:
        reader2.feed_eof()
        await mon_mod.stop()

    # 4. Validate `scripts/hypridle-posture.example.conf` & ingest its payloads via `/api/events`
    conf_path = PROJECT_ROOT / "scripts" / "hypridle-posture.example.conf"
    assert conf_path.is_file()
    conf_text = conf_path.read_text(encoding="utf-8")
    assert "listener {" in conf_text
    assert "on-timeout" in conf_text
    assert "on-resume" in conf_text
    assert "http://127.0.0.1:8000/api/events" in conf_text

    extracted_jsons = re.findall(r"-d '(\{.*?\})'", conf_text)
    assert len(extracted_jsons) == 2
    timeout_payload = json.loads(extracted_jsons[0])
    resume_payload = json.loads(extracted_jsons[1])
    assert timeout_payload["type"] == "input_idle_changed"
    assert timeout_payload["payload"]["idle"] is True
    assert resume_payload["type"] == "input_idle_changed"
    assert resume_payload["payload"]["idle"] is False

    app = create_app(settings=test_settings)
    async with app.router.lifespan_context(app):
        # Register a hyprland_ipc instance on app.state.bus with a fake stream reader
        api_reader = asyncio.StreamReader()
        api_hypr = HyprlandIpcModule(
            bus=app.state.bus,
            db=app.state.db,
            config=test_settings.model_dump(),
            stream_reader=api_reader,
        )
        await api_hypr.start()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                r1 = await client.post("/api/events", json=timeout_payload)
                assert r1.status_code == 200
                assert r1.json()["dispatched"] is True
                assert api_hypr.is_idle is True

                r2 = await client.post("/api/events", json=resume_payload)
                assert r2.status_code == 200
                assert r2.json()["dispatched"] is True
                assert api_hypr.is_idle is False
        finally:
            api_reader.feed_eof()
            await api_hypr.stop()

    # ModuleRegistry discovery sanity check
    reg_bus = EventBus()
    registry = ModuleRegistry(bus=reg_bus, db=db, settings=test_settings)
    assert registry._resolve_category("hyprland_ipc") == "inputs"


async def test_window_classification_work_window_metadata(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify that a work window produces category='focused' and stores classification metadata."""
    t0 = datetime(2026, 11, 4, 16, 0, 0).astimezone()
    bus = EventBus()
    focus_events: list[PlatformEvent] = []
    bus.subscribe("window_focus_changed", focus_events.append)

    reader = asyncio.StreamReader()
    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        clock=lambda: t0,
    )
    await mod.start()
    try:
        await mod.handle_raw_line("activewindow>>code,main.py - jarvis\n", now=t0)
        assert len(focus_events) == 1
        assert focus_events[0].payload["category"] == "focused"
        assert focus_events[0].payload["classifier"] == "app_class"

        span = await db.get_current_window_span()
        assert span is not None
        assert span["app_class"] == "code"
        assert span["category"] == "focused"
        assert span["classifier"] == "app_class"
        assert span["metadata"] == {
            "category": "focused",
            "classifier": "app_class",
            "confidence": 0.95,
            "is_call_window": False,
        }
    finally:
        reader.feed_eof()
        await mod.stop()


async def test_window_classification_distraction_window(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify that a distraction window produces category='distracted' and stores metadata."""
    t0 = datetime(2026, 11, 4, 16, 10, 0).astimezone()
    bus = EventBus()
    focus_events: list[PlatformEvent] = []
    bus.subscribe("window_focus_changed", focus_events.append)

    reader = asyncio.StreamReader()
    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        clock=lambda: t0,
    )
    await mod.start()
    try:
        await mod.handle_raw_line(
            "activewindow>>firefox,Twitch - Live Stream — Mozilla Firefox\n", now=t0
        )
        assert len(focus_events) == 1
        assert focus_events[0].payload["category"] == "distracted"
        assert focus_events[0].payload["classifier"] == "title_rule"

        span = await db.get_current_window_span()
        assert span is not None
        assert span["app_class"] == "firefox"
        assert span["category"] == "distracted"
        assert span["classifier"] == "title_rule"
        assert span["metadata"] == {
            "category": "distracted",
            "classifier": "title_rule",
            "confidence": 0.90,
            "is_call_window": False,
        }
    finally:
        reader.feed_eof()
        await mod.stop()


async def test_window_classification_call_window_sets_flag(
    db: Database,
    test_settings: Settings,
) -> None:
    """Verify that a call window sets is_call_window=True in span metadata."""
    t0 = datetime(2026, 11, 4, 16, 20, 0).astimezone()
    bus = EventBus()
    focus_events: list[PlatformEvent] = []
    bus.subscribe("window_focus_changed", focus_events.append)

    reader = asyncio.StreamReader()
    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        clock=lambda: t0,
    )
    await mod.start()
    try:
        # Test 1: Native call app (zoom)
        await mod.handle_raw_line("activewindow>>zoom,Zoom Meeting\n", now=t0)
        assert len(focus_events) == 1
        span1 = await db.get_current_window_span()
        assert span1 is not None
        assert span1["metadata"]["is_call_window"] is True

        # Test 2: In-browser call window (Google Meet)
        t1 = t0 + timedelta(minutes=5)
        await mod.handle_raw_line(
            "activewindow>>firefox,Meet - abc-defg-hij — Mozilla Firefox\n", now=t1
        )
        assert len(focus_events) == 2
        span2 = await db.get_current_window_span()
        assert span2 is not None
        assert span2["metadata"]["is_call_window"] is True

        # Test 3: Microsoft Teams
        t2 = t0 + timedelta(minutes=10)
        await mod.handle_raw_line(
            "activewindow>>teams,Microsoft Teams\n", now=t2
        )
        assert len(focus_events) == 3
        span3 = await db.get_current_window_span()
        assert span3 is not None
        assert span3["metadata"]["is_call_window"] is True
    finally:
        reader.feed_eof()
        await mod.stop()


async def test_window_classification_failure_and_unknown_preserves_persistence(
    db: Database,
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that classification exceptions or unknown results do not break span persistence."""
    t0 = datetime(2026, 11, 4, 16, 30, 0).astimezone()
    bus = EventBus()
    focus_events: list[PlatformEvent] = []
    bus.subscribe("window_focus_changed", focus_events.append)

    reader = asyncio.StreamReader()
    mod = HyprlandIpcModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        stream_reader=reader,
        clock=lambda: t0,
    )
    await mod.start()
    try:
        # Case A: Unknown window classification (neither work, distraction, nor call)
        await mod.handle_raw_line(
            "activewindow>>gucharmap,Character Map\n", now=t0
        )
        assert len(focus_events) == 1
        assert focus_events[0].payload["category"] == "unknown"
        assert focus_events[0].payload["classifier"] == "default"

        span_unknown = await db.get_current_window_span()
        assert span_unknown is not None
        assert span_unknown["app_class"] == "gucharmap"
        assert span_unknown["category"] == "unknown"
        assert span_unknown["classifier"] == "default"
        assert span_unknown["metadata"] == {
            "category": "unknown",
            "classifier": "default",
            "confidence": 0.0,
            "is_call_window": False,
        }

        # Case B: classify_window raises an unexpected exception
        def _failing_classify(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Simulated classifier catastrophic failure")

        monkeypatch.setattr(mod, "classify", _failing_classify)

        t1 = t0 + timedelta(minutes=2)
        # Should not raise exception; span should still be persisted with fallback
        span_row = await mod.handle_raw_line(
            "activewindow>>code,crash_test.py\n", now=t1
        )
        assert span_row is not None
        assert len(focus_events) == 2
        assert focus_events[1].payload["category"] == "unknown"
        assert focus_events[1].payload["classifier"] == "default"

        span_crash = await db.get_current_window_span()
        assert span_crash is not None
        assert span_crash["app_class"] == "code"
        assert span_crash["category"] == "unknown"
        assert span_crash["classifier"] == "default"
        assert span_crash["metadata"] == {
            "category": "unknown",
            "classifier": "default",
            "confidence": 0.0,
            "is_call_window": False,
        }
    finally:
        reader.feed_eof()
        await mod.stop()

