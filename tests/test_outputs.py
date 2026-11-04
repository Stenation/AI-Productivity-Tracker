"""Unit and integration tests for WaybarStatusModule, DesktopNotifyModule, scripts, and systemd."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import PROJECT_ROOT, Settings
from jarvis.modules.outputs.desktop_notify import DesktopNotifyModule
from jarvis.modules.outputs.waybar_status import WaybarStatusModule


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


def _read_status_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 1
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


async def test_waybar_atomic_json_generation_and_activity_states(
    db: Database,
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_file = tmp_path / "waybar" / "posture_waybar.json"
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replace_calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    t0 = datetime(2026, 11, 3, 10, 0, 0).astimezone()
    now = t0
    bus = EventBus()

    waybar = WaybarStatusModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        status_file=status_file,
        signal_waybar=False,
        refresh_interval_seconds=60.0,
        clock=lambda: now,
    )
    await waybar.start()
    try:
        # Initial startup renders unknown state atomically via temp file + os.replace
        assert status_file.exists()
        assert len(replace_calls) == 1
        src_tmp, dst_target = replace_calls[0]
        assert src_tmp.parent == status_file.parent
        assert dst_target == status_file
        assert list(status_file.parent.glob("*.tmp")) == []

        initial = _read_status_json(status_file)
        assert initial["class"] == "unknown"
        assert "UNKNOWN 0m" in initial["text"]
        assert "Unknown (0m)" in initial["tooltip"]
        assert "Manual override: none" in initial["tooltip"]

        # Cycle through sitting -> standing -> walking -> away with elapsed durations & override TTL
        override_exp = t0 + timedelta(minutes=30)
        await bus.publish(
            PlatformEvent(
                type="activity_state_changed",
                source="resolver",
                timestamp=t0,
                payload={
                    "previous_activity": "unknown",
                    "activity": "sitting",
                    "activity_source": "manual_control",
                    "reason": "manual_override",
                    "started_at": t0.isoformat(),
                    "override_expires_at": override_exp.isoformat(),
                    "span_id": 1,
                },
            )
        )
        now = t0 + timedelta(minutes=14, seconds=30)
        sit_payload = await waybar.refresh()
        assert sit_payload["class"] == "sitting"
        assert sit_payload["text"] == "󰔄 SIT 14m"
        assert "Sitting (14m)" in sit_payload["tooltip"]
        expected_hm = override_exp.astimezone().strftime("%H:%M")
        assert (
            f"Manual override: until {expected_hm} (16m, source: manual_control)"
            in sit_payload["tooltip"]
        )

        # Once clock advances past override_expires_at, manual override resets to 'none'
        now = t0 + timedelta(minutes=31)
        expired_payload = await waybar.refresh()
        assert expired_payload["text"] == "󰔄 SIT 31m"
        assert "Manual override: none" in expired_payload["tooltip"]

        for state, expected_label in (
            ("standing", "STAND"),
            ("walking", "WALK"),
            ("away", "AWAY"),
        ):
            t_state = now
            await bus.publish(
                PlatformEvent(
                    type="activity_state_changed",
                    source="resolver",
                    timestamp=t_state,
                    payload={
                        "previous_activity": "sitting",
                        "activity": state,
                        "activity_source": "manual_control",
                        "reason": "manual_override",
                        "started_at": t_state.isoformat(),
                        "override_expires_at": None,
                        "span_id": 2,
                    },
                )
            )
            now = t_state + timedelta(minutes=7)
            rendered = await waybar.refresh()
            assert rendered["class"] == state
            assert f"{expected_label} 7m" in rendered["text"]
            assert list(status_file.parent.glob("*.tmp")) == []
    finally:
        await waybar.stop()


async def test_waybar_timers_calendar_recommendations_and_privacy_dnd(
    db: Database,
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    status_file = tmp_path / "posture_waybar.json"
    t0 = datetime(2026, 11, 3, 14, 0, 0).astimezone()
    now = t0
    bus = EventBus()

    waybar = WaybarStatusModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        status_file=status_file,
        signal_waybar=False,
        refresh_interval_seconds=60.0,
        clock=lambda: now,
    )
    await waybar.start()
    try:
        # Walking for 12m
        walk_start = t0 - timedelta(minutes=12)
        await bus.publish(
            PlatformEvent(
                type="activity_state_changed",
                source="resolver",
                timestamp=t0,
                payload={
                    "previous_activity": "sitting",
                    "activity": "walking",
                    "activity_source": "swalk_ble",
                    "reason": "sensor_consensus",
                    "started_at": walk_start.isoformat(),
                    "override_expires_at": None,
                    "span_id": 10,
                },
            )
        )

        # Start 50_10 Pomodoro with 22m remaining + calendar context (ARCHITECTURE.MD §10)
        await bus.publish(
            PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=t0,
                payload={
                    "active": True,
                    "timer_id": 1,
                    "kind": "50_10",
                    "phase": "work",
                    "previous_phase": "idle",
                    "remaining_seconds": 22 * 60,
                    "planned_duration_seconds": 50 * 60,
                    "started_at": (t0 - timedelta(minutes=28)).isoformat(),
                    "status": "running",
                    "transition": "started",
                },
            )
        )
        await bus.publish(
            PlatformEvent(
                type="calendar_context_changed",
                source="google_calendar",
                timestamp=t0,
                payload={
                    "current_event": {
                        "event_id": "evt-1",
                        "title": "Backend refactor",
                        "task_type": "coding",
                        "start_time": (t0 - timedelta(minutes=15)).isoformat(),
                        "end_time": (t0 + timedelta(minutes=30)).isoformat(),
                    },
                    "next_event": {
                        "event_id": "evt-2",
                        "title": "Meeting",
                        "task_type": "meeting",
                        "start_time": (t0 + timedelta(minutes=38)).isoformat(),
                        "end_time": (t0 + timedelta(minutes=68)).isoformat(),
                    },
                },
            )
        )

        data = _read_status_json(status_file)
        assert data["text"] == "󰗉 WALK 12m · 50/10 22m"
        assert data["class"] == "walking"
        assert "Walking (12m)" in data["tooltip"]
        assert "Timer: 50/10 (work, 22m left)" in data["tooltip"]
        assert "Current: Backend refactor" in data["tooltip"]
        assert "Next: Meeting in 38m" in data["tooltip"]
        assert "Manual override: none" in data["tooltip"]
        assert "Pending recommendations: 0" in data["tooltip"]

        # Break transition updates timer badge to BREAK
        await bus.publish(
            PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=t0,
                payload={
                    "active": True,
                    "timer_id": 2,
                    "kind": "50_10",
                    "phase": "break",
                    "previous_phase": "work",
                    "remaining_seconds": 10 * 60,
                    "planned_duration_seconds": 10 * 60,
                    "started_at": t0.isoformat(),
                    "status": "running",
                    "transition": "break_started",
                },
            )
        )
        break_data = _read_status_json(status_file)
        assert break_data["text"] == "󰗉 WALK 12m · 50/10 BREAK 10m"
        assert "Timer: 50/10 (break, 10m left)" in break_data["tooltip"]

        # Pending recommendation created & resolved updates tooltip count
        rec = await db.create_recommendation(
            policy="meeting_policy",
            message="Meeting starts in 10 minutes. Stop walking.",
            suggested_action="return_to_desk",
            dedupe_key="meeting:evt-2",
            created_at=t0,
        )
        await bus.publish(
            PlatformEvent(
                type="recommendation_created",
                source="meeting_policy",
                timestamp=t0,
                payload={
                    "recommendation_id": rec["id"],
                    "id": rec["id"],
                    "policy": "meeting_policy",
                    "message": rec["message"],
                    "suggested_action": "return_to_desk",
                    "action_payload": None,
                    "dedupe_key": "meeting:evt-2",
                    "status": "pending",
                    "created_at": t0.isoformat(),
                },
            )
        )
        assert "Pending recommendations: 1" in _read_status_json(status_file)["tooltip"]

        await db.resolve_recommendation(int(rec["id"]), "accepted", resolved_at=t0)
        await bus.publish(
            PlatformEvent(
                type="recommendation_resolved",
                source="api",
                timestamp=t0,
                payload={
                    "recommendation_id": rec["id"],
                    "id": rec["id"],
                    "status": "accepted",
                },
            )
        )
        assert "Pending recommendations: 0" in _read_status_json(status_file)["tooltip"]

        # Timer completed removes timer countdown from text
        await bus.publish(
            PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=t0,
                payload={
                    "active": False,
                    "timer_id": 2,
                    "kind": "50_10",
                    "phase": "idle",
                    "previous_phase": "break",
                    "remaining_seconds": 0,
                    "planned_duration_seconds": 600,
                    "started_at": t0.isoformat(),
                    "status": "completed",
                    "transition": "completed",
                },
            )
        )
        assert _read_status_json(status_file)["text"] == "󰗉 WALK 12m"

        # Enable DND mode -> class becomes "dnd", DND indicator shown
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t0,
                payload={
                    "global_pause": False,
                    "privacy_paused": False,
                    "dnd_mode": True,
                },
            )
        )
        dnd_data = _read_status_json(status_file)
        assert dnd_data["class"] == "dnd"
        assert "DND" in dnd_data["text"]
        assert "Mode: DND enabled" in dnd_data["tooltip"]

        # Enable global privacy pause -> class becomes "paused", PAUSED indicator shown
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t0,
                payload={
                    "global_pause": True,
                    "privacy_paused": True,
                    "dnd_mode": True,
                },
            )
        )
        paused_data = _read_status_json(status_file)
        assert paused_data["class"] == "paused"
        assert "PAUSED" in paused_data["text"]
        assert "DND" in paused_data["text"]
        assert "Privacy: PAUSED · DND enabled" in paused_data["tooltip"]

        # Resume normal operation
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t0,
                payload={
                    "global_pause": False,
                    "privacy_paused": False,
                    "dnd_mode": False,
                },
            )
        )
        resumed = _read_status_json(status_file)
        assert resumed["class"] == "walking"
        assert resumed["text"] == "󰗉 WALK 12m"
    finally:
        await waybar.stop()


async def test_waybar_periodic_refresh_signaling_recovery_and_registry(
    db: Database,
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = datetime(2026, 11, 3, 15, 0, 0).astimezone()
    now = t0
    bus = EventBus()

    # Pre-seed SQLite open span + manual override to test restart recovery (UC-01)
    override_exp = t0 + timedelta(minutes=25)
    await db.open_activity_span(
        activity="standing",
        source="waybar",
        confidence=1.0,
        reason="manual_override",
        started_at=t0 - timedelta(minutes=5),
        metadata={"ttl_minutes": 30, "override_expires_at": override_exp.isoformat()},
    )

    # Create a fake pkill executable that exits 1 (simulating Waybar not running) while logging args
    pkill_log = tmp_path / "pkill_args.log"
    fake_pkill = tmp_path / "fake-pkill"
    fake_pkill.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{pkill_log}"\nexit 1\n',
        encoding="utf-8",
    )
    fake_pkill.chmod(0o755)

    status_file = tmp_path / "periodic_waybar.json"
    waybar = WaybarStatusModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        status_file=status_file,
        refresh_interval_seconds=0.02,
        signal_waybar=True,
        pkill_binary=str(fake_pkill),
        clock=lambda: now,
    )
    await waybar.start()
    try:
        # Recovered standing 5m + manual override on start, and signaled Waybar without raising
        initial = _read_status_json(status_file)
        assert initial["class"] == "standing"
        assert initial["text"] == "󰋑 STAND 5m"
        assert "source: waybar" in initial["tooltip"]
        assert "-RTMIN+8 waybar" in pkill_log.read_text(encoding="utf-8")

        # Advance clock by 8 minutes -> periodic loop updates elapsed duration automatically
        now = t0 + timedelta(minutes=8)
        await _wait_until(
            lambda: _read_status_json(status_file)["text"] == "󰋑 STAND 13m"
        )
    finally:
        await waybar.stop()

    # Missing pkill binary never raises
    missing_waybar = WaybarStatusModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        status_file=tmp_path / "missing_pkill_waybar.json",
        signal_waybar=True,
        pkill_binary=str(tmp_path / "does-not-exist-pkill"),
        clock=lambda: now,
    )
    await missing_waybar.start()
    assert (await missing_waybar.health())["status"] == "ok"
    await missing_waybar.stop()

    # ModuleRegistry lifecycle integration (isolated via POSTURE_WAYBAR_FILE + mocked pkill)
    reg_file = tmp_path / "registry_waybar.json"
    monkeypatch.setenv("POSTURE_WAYBAR_FILE", str(reg_file))

    async def _noop_subproc(*args: Any, **kwargs: Any) -> Any:
        class _DummyProc:
            returncode = 0

            async def wait(self) -> int:
                return 0

        return _DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _noop_subproc)

    registry = ModuleRegistry(bus=bus, db=db, settings=test_settings)
    assert await registry.enable_module("waybar_status") is True
    assert registry.is_enabled("waybar_status") is True
    assert reg_file.exists()
    assert await registry.disable_module("waybar_status") is True
    assert registry.is_enabled("waybar_status") is False


async def test_posture_status_script_online_offline_and_static_configs(
    tmp_path: Path,
) -> None:
    script_path = PROJECT_ROOT / "scripts" / "posture-status.py"
    example_waybar_cfg = PROJECT_ROOT / "scripts" / "waybar-config.example.jsonc"
    service_unit = PROJECT_ROOT / "systemd" / "posture-core.service"

    assert script_path.exists()
    assert os.access(script_path, os.X_OK)

    # Ensure stdlib-only imports in posture-status.py
    script_src = script_path.read_text(encoding="utf-8")
    assert "jarvis" not in script_src
    assert "pydantic" not in script_src

    async def run_status_script(env_file: Path) -> tuple[int, str]:
        env = {**os.environ, "POSTURE_WAYBAR_FILE": str(env_file)}
        proc = await asyncio.create_subprocess_exec(
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_b, stderr_b = await proc.communicate()
        assert stderr_b == b""
        return int(proc.returncode or 0), stdout_b.decode("utf-8")

    # 1. Missing file -> offline JSON
    missing_path = tmp_path / "missing.json"
    rc, out = await run_status_script(missing_path)
    assert rc == 0
    assert len(out.splitlines()) == 1
    assert json.loads(out) == {
        "text": "POSTURE OFFLINE",
        "tooltip": "posture-core.service is not running",
        "class": "offline",
    }

    # 2. Invalid/corrupt JSON or missing 'text' -> offline JSON
    corrupt_path = tmp_path / "corrupt.json"
    for invalid_content in (
        "{not valid json",
        "[]",
        '{"tooltip": "missing text"}',
        '{"text": "   "}',
    ):
        corrupt_path.write_text(invalid_content, encoding="utf-8")
        rc, out = await run_status_script(corrupt_path)
        assert rc == 0
        assert json.loads(out) == {
            "text": "POSTURE OFFLINE",
            "tooltip": "posture-core.service is not running",
            "class": "offline",
        }

    # 3. Valid online file -> prints exact single-line JSON
    online_path = tmp_path / "online.json"
    expected_online = {
        "text": "󰗉 WALK 12m · 50/10 22m",
        "tooltip": "Walking (12m)\nCurrent: Backend refactor\nNext: Meeting in 38m",
        "class": "walking",
    }
    online_path.write_text(
        json.dumps(expected_online, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rc, out = await run_status_script(online_path)
    assert rc == 0
    assert len(out.splitlines()) == 1
    assert json.loads(out) == expected_online

    # 4. Validate waybar-config.example.jsonc (strip // comments and parse JSON)
    raw_jsonc = example_waybar_cfg.read_text(encoding="utf-8")
    stripped = re.sub(r"^\s*//.*$", "", raw_jsonc, flags=re.MULTILINE)
    waybar_cfg = json.loads(stripped)
    custom_posture = waybar_cfg["custom/posture"]
    assert custom_posture["exec"] == "/home/steff/Projects/jarvis/scripts/posture-status.py"
    assert custom_posture["return-type"] == "json"
    assert custom_posture["signal"] == 8
    assert "http://127.0.0.1:8000/api/activity/cycle" in custom_posture["on-click"]
    assert "http://127.0.0.1:8000" in custom_posture["on-click-right"]
    assert (
        "http://127.0.0.1:8000/api/recommendations/snooze"
        in custom_posture["on-click-middle"]
    )

    # 5. Validate systemd/posture-core.service directives
    unit_text = service_unit.read_text(encoding="utf-8")
    assert ".venv/bin" not in unit_text
    assert "WorkingDirectory=%h/Projects/jarvis" in unit_text
    assert "Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin" in unit_text
    assert "Environment=PYTHONUNBUFFERED=1" in unit_text
    assert (
        "ExecStart=/usr/bin/env uv run --all-groups jarvis" in unit_text
        or "ExecStart=/usr/bin/env uv run jarvis" in unit_text
    )


async def test_desktop_notify_subprocess_dispatch_and_shell_safety(
    db: Database,
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    # Real executable notify-send stub writing JSON lines of received argv
    capture_file = tmp_path / "notifications.jsonl"
    sentinel_file = tmp_path / "pwned_sentinel"
    fake_notify = tmp_path / "fake-notify-send"
    fake_notify.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"with open({str(capture_file)!r}, 'a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    fake_notify.chmod(0o755)

    t0 = datetime(2026, 11, 3, 16, 0, 0).astimezone()
    bus = EventBus()
    notifier = DesktopNotifyModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        notify_binary=str(fake_notify),
        expire_time_ms=8000,
        clock=lambda: t0,
    )
    await notifier.start()
    try:
        # 1. Recommendation with shell metacharacters is passed safely without shell expansion
        dangerous_msg = f"You sat 50m; $(touch {sentinel_file}) `id` 'quote' \"dbl\""
        await bus.publish(
            PlatformEvent(
                type="recommendation_created",
                source="posture_policy",
                timestamp=t0,
                payload={
                    "recommendation_id": 1,
                    "id": 1,
                    "policy": "posture_policy",
                    "message": dangerous_msg,
                    "suggested_action": "stand_or_walk",
                    "action_payload": {"elapsed_minutes": 50},
                    "dedupe_key": "posture:sitting",
                    "status": "pending",
                    "created_at": t0.isoformat(),
                },
            )
        )
        assert not sentinel_file.exists()

        # 2. Timer 'started' and 'stopped' transitions do NOT emit notifications
        for ignored_transition in ("started", "stopped"):
            await bus.publish(
                PlatformEvent(
                    type="timer_state_changed",
                    source="timer_manager",
                    timestamp=t0,
                    payload={
                        "active": ignored_transition == "started",
                        "timer_id": 5,
                        "kind": "25_5",
                        "phase": "work" if ignored_transition == "started" else "idle",
                        "previous_phase": "idle",
                        "remaining_seconds": 1500 if ignored_transition == "started" else 0,
                        "planned_duration_seconds": 1500,
                        "started_at": t0.isoformat(),
                        "status": "running" if ignored_transition == "started" else "stopped",
                        "transition": ignored_transition,
                    },
                )
            )

        # 3. Timer 'break_started' and 'completed' transitions DO emit notifications
        await bus.publish(
            PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=t0,
                payload={
                    "active": True,
                    "timer_id": 6,
                    "kind": "25_5",
                    "phase": "break",
                    "previous_phase": "work",
                    "remaining_seconds": 300,
                    "planned_duration_seconds": 300,
                    "started_at": t0.isoformat(),
                    "status": "running",
                    "transition": "break_started",
                },
            )
        )
        await bus.publish(
            PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=t0,
                payload={
                    "active": False,
                    "timer_id": 6,
                    "kind": "25_5",
                    "phase": "idle",
                    "previous_phase": "break",
                    "remaining_seconds": 0,
                    "planned_duration_seconds": 300,
                    "started_at": t0.isoformat(),
                    "status": "completed",
                    "transition": "completed",
                },
            )
        )

        captured = [
            json.loads(line)
            for line in capture_file.read_text(encoding="utf-8").splitlines()
        ]
        assert len(captured) == 3

        assert captured[0] == [
            "--app-name",
            "Posture Platform",
            "--urgency",
            "normal",
            "--expire-time",
            "8000",
            "--",
            "Posture Recommendation",
            dangerous_msg,
        ]
        assert captured[1][-2:] == [
            "Timer — Break Started",
            "Work block complete (25/5). Take a 5m break.",
        ]
        assert captured[2][-2:] == [
            "Timer — Completed",
            "25/5 timer cycle complete. Break finished.",
        ]
        assert (await notifier.health())["status"] == "ok"
    finally:
        await notifier.stop()


async def test_desktop_notify_privacy_dnd_suppression_and_degraded_resilience(
    db: Database,
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    t0 = datetime(2026, 11, 3, 17, 0, 0).astimezone()
    bus = EventBus()
    calls: list[list[str]] = []

    async def recording_runner(argv: Any) -> int:
        calls.append(list(argv))
        return 0

    # Start with DND enabled in SQLite settings
    await db.set_setting("privacy.dnd_mode", True)

    notifier = DesktopNotifyModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        runner=recording_runner,
        clock=lambda: t0,
    )
    await notifier.start()
    try:
        # Suppressed while DND is active
        await bus.publish(
            PlatformEvent(
                type="recommendation_created",
                source="focus_policy",
                timestamp=t0,
                payload={
                    "recommendation_id": 10,
                    "id": 10,
                    "policy": "focus_policy",
                    "message": "Off-task window detected.",
                    "status": "pending",
                },
            )
        )
        assert calls == []

        # Switch DND off, turn global_pause on -> still suppressed
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t0,
                payload={
                    "global_pause": True,
                    "privacy_paused": True,
                    "dnd_mode": False,
                },
            )
        )
        await bus.publish(
            PlatformEvent(
                type="timer_state_changed",
                source="timer_manager",
                timestamp=t0,
                payload={
                    "active": False,
                    "timer_id": 11,
                    "kind": "50_10",
                    "phase": "idle",
                    "previous_phase": "work",
                    "remaining_seconds": 0,
                    "planned_duration_seconds": 3000,
                    "status": "completed",
                    "transition": "completed",
                },
            )
        )
        assert calls == []

        # Resume both global_pause=False and dnd_mode=False -> notifications flow again
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                timestamp=t0,
                payload={
                    "global_pause": False,
                    "privacy_paused": False,
                    "dnd_mode": False,
                },
            )
        )
        await bus.publish(
            PlatformEvent(
                type="recommendation_created",
                source="meeting_policy",
                timestamp=t0,
                payload={
                    "recommendation_id": 12,
                    "id": 12,
                    "policy": "meeting_policy",
                    "message": "Meeting starts in 10 minutes.",
                    "status": "pending",
                },
            )
        )
        assert len(calls) == 1
        assert calls[0][-2:] == ["Upcoming Meeting", "Meeting starts in 10 minutes."]
    finally:
        await notifier.stop()

    # Clear persisted DND setting before resilience checks
    await db.set_setting("privacy.dnd_mode", False)

    health_events: list[PlatformEvent] = []
    bus.subscribe("module_health_changed", health_events.append)

    # Missing binary resilience via real subprocess_exec -> marks degraded, never crashes
    missing_notifier = DesktopNotifyModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        notify_binary=str(tmp_path / "nonexistent-notify-send"),
        clock=lambda: t0,
    )
    await missing_notifier.start()
    try:
        dispatched = await missing_notifier.send_notification(
            "Posture Recommendation",
            "Consider standing up.",
        )
        assert dispatched is False
        health = await missing_notifier.health()
        assert health["status"] == "degraded"
        assert "FileNotFoundError" in str(health["last_error"])

        db_health = {row["module"]: row for row in await db.list_module_health()}
        assert db_health["desktop_notify"]["status"] == "degraded"
        assert any(
            evt.payload.get("module") == "desktop_notify"
            and evt.payload.get("status") == "degraded"
            for evt in health_events
        )
    finally:
        await missing_notifier.stop()

    # Non-zero exit code resilience + recovery to ok
    outcomes: list[tuple[int, str, str]] = [
        (1, "", "org.freedesktop.Notifications unavailable"),
        (0, "", ""),
    ]

    async def flaky_runner(_argv: Any) -> tuple[int, str, str]:
        return outcomes.pop(0)

    flaky_notifier = DesktopNotifyModule(
        bus=bus,
        db=db,
        config=test_settings.model_dump(),
        runner=flaky_runner,
        clock=lambda: t0,
    )
    await flaky_notifier.start()
    try:
        assert await flaky_notifier.send_notification("Test", "First fails") is False
        assert (await flaky_notifier.health())["status"] == "degraded"
        assert "unavailable" in str((await flaky_notifier.health())["last_error"])

        assert await flaky_notifier.send_notification("Test", "Second recovers") is True
        assert (await flaky_notifier.health())["status"] == "ok"
        assert (await flaky_notifier.health())["last_error"] is None
    finally:
        await flaky_notifier.stop()

    # ModuleRegistry lifecycle check
    registry = ModuleRegistry(bus=bus, db=db, settings=test_settings)
    assert await registry.enable_module("desktop_notify") is True
    assert registry.is_enabled("desktop_notify") is True
    assert await registry.disable_module("desktop_notify") is True
    assert registry.is_enabled("desktop_notify") is False
