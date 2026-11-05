"""Tests for core schema, EventBus, StateResolver, ModuleRegistry, and FastAPI app."""

from __future__ import annotations

import logging
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis import __version__
from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import (
    ActivitySpan,
    CalendarEvent,
    ModuleHealth,
    PlatformEvent,
    Recommendation,
    SystemStateSnapshot,
    TimerSpan,
)
from jarvis.core.settings import Settings
from jarvis.main import main
from jarvis.modules.base import BaseModule


def test_schema_models_enforce_timezone_aware_datetimes_and_validate_rows() -> None:
    event = PlatformEvent(
        type="activity_reported",
        source="manual_control",
        payload={"activity": "sitting"},
    )
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() is not None

    # Naive datetimes are promoted to local timezone-aware datetimes automatically
    naive_event = PlatformEvent(
        type="window_focus_changed",
        source="hyprland_ipc",
        timestamp=datetime(2026, 11, 3, 16, 0, 0),
    )
    assert naive_event.timestamp.tzinfo is not None
    # Subtraction between default aware timestamp and promoted timestamp never raises TypeError
    _ = event.timestamp - naive_event.timestamp

    utc_now = datetime(2026, 11, 3, 15, 0, 0, tzinfo=UTC)
    span = ActivitySpan(
        id=1,
        activity="walking",
        source="swalk_ble",
        confidence=0.95,
        reason="sensor_consensus",
        started_at=utc_now,
        metadata_json='{"speed_kmh": 3.2}',  # type: ignore[call-arg]
    )
    assert span.metadata == {"speed_kmh": 3.2}
    assert span.started_at.tzinfo is not None

    timer = TimerSpan(
        id=2,
        kind="50_10",
        planned_duration_seconds=3000,
        status="running",
    )
    rec = Recommendation(
        id=3,
        policy="posture_policy",
        message="Stand up for 10 minutes.",
        suggested_action="stand_or_walk",
        dedupe_key="posture:max_sitting",
        action_payload_json='{"target": "standing"}',  # type: ignore[call-arg]
    )
    assert rec.dedupe_key == "posture:max_sitting"
    assert rec.action_payload == {"target": "standing"}
    assert rec.status == "pending"

    cal = CalendarEvent(
        event_id="evt_1",
        title="Deep Work: Core Spine",
        task_type="coding",
        suitable_for_walking=False,
        recommended_pomodoro="50_10",
        start_time=utc_now,
        end_time=utc_now + timedelta(hours=1),
    )
    health = ModuleHealth(module="manual_control", status="ok", enabled=True)

    snapshot = SystemStateSnapshot(
        activity="walking",
        source="swalk_ble",
        confidence=0.95,
        reason="sensor_consensus",
        elapsed_seconds=120,
        focus="focused",
        active_timer=timer,
        current_event=cal,
        pending_recommendations=[rec],
    )
    assert snapshot.active_timer is not None
    assert snapshot.current_event is not None
    assert len(snapshot.pending_recommendations) == 1
    assert health.updated_at.tzinfo is not None


async def test_event_bus_typed_wildcard_dedupe_and_error_isolation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus(max_dedupe_keys=2)
    typed_received: list[PlatformEvent] = []
    wildcard_received: list[PlatformEvent] = []
    sync_received: list[str] = []

    async def failing_subscriber(event: PlatformEvent) -> None:
        raise RuntimeError(f"boom on {event.type}")

    async def typed_subscriber(event: PlatformEvent) -> None:
        typed_received.append(event)

    async def wildcard_subscriber(event: PlatformEvent) -> None:
        wildcard_received.append(event)

    def sync_subscriber(event: PlatformEvent) -> None:
        sync_received.append(event.type)

    bus.subscribe("activity_reported", failing_subscriber)
    bus.subscribe("activity_reported", typed_subscriber)
    bus.subscribe("activity_reported", sync_subscriber)
    bus.subscribe("*", wildcard_subscriber)

    caplog.set_level(logging.ERROR)
    ev1 = PlatformEvent(
        type="activity_reported",
        source="manual_control",
        payload={"activity": "sitting"},
        dedupe_key="key-1",
    )
    assert await bus.publish(ev1) is True
    assert len(typed_received) == 1
    assert len(wildcard_received) == 1
    assert sync_received == ["activity_reported"]
    assert "boom on activity_reported" in caplog.text

    # Duplicate dedupe_key within bounded cache is suppressed
    assert await bus.publish(ev1) is False
    assert len(typed_received) == 1
    assert len(wildcard_received) == 1

    # Other event type reaches wildcard subscriber but not activity_reported subscriber
    ev2 = PlatformEvent(
        type="timer_state_changed",
        source="pomodoro_policy",
        payload={"status": "running"},
        dedupe_key="key-2",
    )
    assert await bus.publish(ev2) is True
    assert len(typed_received) == 1
    assert len(wildcard_received) == 2

    # Publishing a 3rd distinct dedupe_key evicts "key-1" (max_dedupe_keys=2)
    ev3 = PlatformEvent(
        type="privacy_mode_changed",
        source="api",
        payload={"global_pause": False},
        dedupe_key="key-3",
    )
    assert await bus.publish(ev3) is True
    assert len(wildcard_received) == 3

    # "key-1" was evicted, so publishing it again succeeds; "key-2" is now evicted
    assert await bus.publish(ev1) is True
    assert len(typed_received) == 2
    assert len(wildcard_received) == 4

    # Unsubscribe removes callbacks cleanly
    assert bus.unsubscribe("activity_reported", typed_subscriber) is True
    assert bus.unsubscribe("activity_reported", typed_subscriber) is False
    assert bus.unsubscribe(wildcard_subscriber) is True

    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            payload={"activity": "standing"},
        )
    )
    assert len(typed_received) == 2
    assert len(wildcard_received) == 4
    assert len(bus.recent_events()) == 5


async def test_state_resolver_manual_override_ttl_precedence_and_expiry_fallback(
    db: Database,
    test_settings: Settings,
) -> None:
    bus = EventBus()
    resolver = StateResolver(bus=bus, db=db, settings=test_settings)
    await resolver.start()

    state_changes: list[PlatformEvent] = []
    bus.subscribe("activity_state_changed", state_changes.append)

    t0 = datetime(2026, 11, 3, 14, 0, 0).astimezone()

    # 1. Manual override to "walking" with 30m TTL
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            confidence=0.6,  # Resolver must record manual overrides with confidence=1.0
            timestamp=t0,
            payload={"activity": "walking", "ttl_minutes": 30},
        )
    )

    eff = resolver.get_effective_state(now=t0 + timedelta(minutes=5))
    assert eff.activity == "walking"
    assert eff["activity"] == "walking"
    assert eff.source == "manual_control"
    assert eff.confidence == 1.0
    assert eff.reason == "manual_override"
    assert eff.elapsed_seconds == 300
    assert eff.override_expires_at == t0 + timedelta(minutes=30)
    assert len(state_changes) == 1
    assert state_changes[0].payload["previous_activity"] == "unknown"
    assert state_changes[0].payload["activity"] == "walking"
    assert state_changes[0].payload["reason"] == "manual_override"

    # 2. Sensor sends 5 agreeing "sitting" samples inside TTL -> manual override wins
    for idx in range(5):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="webcam_pose",
                confidence=0.9,
                timestamp=t0 + timedelta(minutes=10, seconds=idx * 5),
                payload={"activity": "sitting"},
            )
        )

    eff_during_ttl = resolver.get_effective_state(now=t0 + timedelta(minutes=15))
    assert eff_during_ttl.activity == "walking"
    assert eff_during_ttl.override_expires_at == t0 + timedelta(minutes=30)
    assert len(state_changes) == 1

    # 3. After 30m TTL expires: override_expires_at clears, retaining state until consensus
    t_after_ttl = t0 + timedelta(minutes=31)
    eff_expired = resolver.get_effective_state(now=t_after_ttl)
    assert eff_expired.activity == "walking"
    assert eff_expired.override_expires_at is None
    assert eff_expired.elapsed_seconds == 31 * 60

    # 4. Now 4-of-5 webcam samples report "sitting" -> falls back to sensor consensus
    for idx in range(4):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="webcam_pose",
                confidence=0.85,
                timestamp=t_after_ttl + timedelta(seconds=idx * 5),
                payload={"activity": "sitting"},
            )
        )

    t_transition = t_after_ttl + timedelta(seconds=15)
    eff_sensor = resolver.get_effective_state(now=t_transition + timedelta(seconds=30))
    assert eff_sensor.activity == "sitting"
    assert eff_sensor.source == "webcam_pose"
    assert eff_sensor.confidence == pytest.approx(0.85)
    assert eff_sensor.reason == "sensor_consensus"
    assert eff_sensor.override_expires_at is None
    assert eff_sensor.elapsed_seconds == 30
    assert len(state_changes) == 2
    assert state_changes[1].payload["previous_activity"] == "walking"
    assert state_changes[1].payload["activity"] == "sitting"
    assert state_changes[1].payload["reason"] == "sensor_consensus"

    # Verify SQLite closed the manual "walking" span and opened the "sitting" span
    spans = await db.list_activity_spans()
    assert len(spans) == 2
    assert spans[0]["activity"] == "sitting"
    assert spans[0]["ended_at"] is None
    assert spans[1]["activity"] == "walking"
    assert spans[1]["ended_at"] == t_transition.isoformat()

    await resolver.stop()


async def test_state_resolver_consensus_weak_conflicting_privacy_and_startup_recovery(
    db: Database,
    test_settings: Settings,
) -> None:
    test_settings.posture.sensor_consensus_samples = 4
    test_settings.posture.sensor_consensus_window_samples = 5

    bus = EventBus()
    resolver = StateResolver(bus=bus, db=db, settings=test_settings)
    await resolver.start()

    t0 = datetime(2026, 11, 3, 10, 0, 0).astimezone()

    # Establish initial "sitting" via 4 agreeing sensor samples
    for idx in range(4):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="webcam_pose",
                confidence=0.8,
                timestamp=t0 + timedelta(seconds=idx * 5),
                payload={"activity": "sitting"},
            )
        )
    assert resolver.get_effective_state(now=t0 + timedelta(seconds=20)).activity == "sitting"

    # Weak evidence (low confidence < 0.5) never changes state even with 5 samples
    t_weak = t0 + timedelta(minutes=2)
    for idx in range(5):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="webcam_pose",
                confidence=0.25,
                timestamp=t_weak + timedelta(seconds=idx * 5),
                payload={"activity": "standing"},
            )
        )
    assert resolver.get_effective_state(now=t_weak + timedelta(seconds=30)).activity == "sitting"

    # Conflicting evidence (3 "standing" + 2 "away" in 5-sample window < 4 required) retains state
    t_conflict = t0 + timedelta(minutes=5)
    pattern = ["standing", "away", "standing", "away", "standing"]
    for idx, act in enumerate(pattern):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="webcam_pose",
                confidence=0.9,
                timestamp=t_conflict + timedelta(seconds=idx * 5),
                payload={"activity": act},
            )
        )
    assert (
        resolver.get_effective_state(now=t_conflict + timedelta(seconds=30)).activity
        == "sitting"
    )

    # One more "standing" pushes oldest "standing" out -> 3 "standing" + 2 "away";
    # a second consecutive "standing" evicts "away" -> 4 of 5 "standing" -> transitions!
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="webcam_pose",
            confidence=0.9,
            timestamp=t_conflict + timedelta(seconds=25),
            payload={"activity": "standing"},
        )
    )
    assert (
        resolver.get_effective_state(now=t_conflict + timedelta(seconds=26)).activity
        == "sitting"
    )

    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="webcam_pose",
            confidence=0.9,
            timestamp=t_conflict + timedelta(seconds=30),
            payload={"activity": "standing"},
        )
    )
    assert (
        resolver.get_effective_state(now=t_conflict + timedelta(seconds=31)).activity
        == "standing"
    )

    # Higher-priority BLE walking sensor overrides and blocks simultaneous webcam reports
    t_ble = t0 + timedelta(minutes=10)
    for idx in range(4):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="swalk_ble",
                confidence=0.95,
                timestamp=t_ble + timedelta(seconds=idx * 2),
                payload={"activity": "walking"},
            )
        )
    assert resolver.get_effective_state(now=t_ble + timedelta(seconds=7)).activity == "walking"

    for idx in range(5):
        await bus.publish(
            PlatformEvent(
                type="activity_reported",
                source="webcam_pose",
                confidence=0.9,
                timestamp=t_ble + timedelta(seconds=8 + idx),
                payload={"activity": "sitting"},
            )
        )
    assert resolver.get_effective_state(now=t_ble + timedelta(seconds=15)).activity == "walking"

    # Privacy pause suppresses activity reports until resumed
    await bus.publish(
        PlatformEvent(
            type="privacy_mode_changed",
            source="api",
            timestamp=t_ble + timedelta(minutes=1),
            payload={"global_pause": True},
        )
    )
    assert resolver.get_effective_state().privacy_paused is True

    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=t_ble + timedelta(minutes=2),
            payload={"activity": "away", "ttl_minutes": 15},
        )
    )
    assert resolver.get_effective_state().activity == "walking"

    await bus.publish(
        PlatformEvent(
            type="privacy_mode_changed",
            source="api",
            timestamp=t_ble + timedelta(minutes=3),
            payload={"global_pause": False},
        )
    )
    assert resolver.get_effective_state().privacy_paused is False

    # Set manual override near current wall clock, stop resolver, and recover on new instance
    live_start = datetime.now().astimezone() - timedelta(minutes=5)
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=live_start,
            payload={"activity": "away", "ttl_minutes": 30},
        )
    )
    await resolver.stop()

    bus2 = EventBus()
    recovered_resolver = StateResolver(bus=bus2, db=db, settings=test_settings)
    recovered_state = await recovered_resolver.start()
    assert recovered_state.activity == "away"
    assert recovered_state.source == "manual_control"
    assert recovered_state.reason == "manual_override"
    assert recovered_state.elapsed_seconds >= 299
    assert recovered_state.override_expires_at == live_start + timedelta(minutes=30)
    await recovered_resolver.stop()


async def test_module_registry_enable_disable_and_degraded_isolation(
    db: Database,
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_log: list[str] = []

    class HealthyInputModule(BaseModule):
        name = "fake_healthy_input"

        async def start(self) -> None:
            lifecycle_log.append("healthy:start")

        async def stop(self) -> None:
            lifecycle_log.append("healthy:stop")

    class FailingPolicyModule(BaseModule):
        name = "fake_failing_policy"

        async def start(self) -> None:
            lifecycle_log.append("failing:start")
            raise RuntimeError("GPU worker unreachable")

        async def stop(self) -> None:
            lifecycle_log.append("failing:stop")

    healthy_mod = types.ModuleType("jarvis.modules.inputs.fake_healthy_input")
    healthy_mod.HealthyInputModule = HealthyInputModule  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "jarvis.modules.inputs.fake_healthy_input", healthy_mod
    )

    failing_mod = types.ModuleType("jarvis.modules.policies.fake_failing_policy")
    failing_mod.FailingPolicyModule = FailingPolicyModule  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "jarvis.modules.policies.fake_failing_policy", failing_mod
    )

    empty_mod = types.ModuleType("jarvis.modules.outputs.fake_invalid_output")
    monkeypatch.setitem(
        sys.modules, "jarvis.modules.outputs.fake_invalid_output", empty_mod
    )

    test_settings.enabled_modules.inputs = ["fake_healthy_input", "missing_input_mod"]
    test_settings.enabled_modules.policies = ["fake_failing_policy"]
    test_settings.enabled_modules.outputs = ["fake_invalid_output"]

    bus = EventBus()
    health_events: list[PlatformEvent] = []
    bus.subscribe("module_health_changed", health_events.append)

    registry = ModuleRegistry(bus=bus, db=db, settings=test_settings)
    statuses = await registry.start_all()

    assert statuses == {
        "fake_invalid_output": "degraded",
        "fake_failing_policy": "degraded",
        "fake_healthy_input": "ok",
        "missing_input_mod": "degraded",
    }
    assert registry.is_enabled("fake_healthy_input") is True
    assert registry.is_enabled("fake_failing_policy") is False
    assert registry.is_enabled("missing_input_mod") is False
    assert registry.is_enabled("fake_invalid_output") is False

    healthy_instance = registry.get_module("fake_healthy_input")
    assert healthy_instance is not None
    assert "posture" in healthy_instance.config

    # Verify degraded modules were persisted to SQLite module_health without crashing
    db_health = {row["module"]: row for row in await db.list_module_health()}
    assert db_health["fake_healthy_input"]["status"] == "ok"
    assert db_health["fake_healthy_input"]["last_error"] is None
    assert db_health["fake_failing_policy"]["status"] == "degraded"
    assert "GPU worker unreachable" in db_health["fake_failing_policy"]["last_error"]
    assert db_health["missing_input_mod"]["status"] == "degraded"
    assert "ModuleNotFoundError" in db_health["missing_input_mod"]["last_error"]
    assert db_health["fake_invalid_output"]["status"] == "degraded"
    assert "BaseModule" in db_health["fake_invalid_output"]["last_error"]

    # Runtime disable and re-enable of healthy module
    assert await registry.disable_module("fake_healthy_input") is True
    assert registry.is_enabled("fake_healthy_input") is False
    assert registry.get_module("fake_healthy_input") is None
    assert "healthy:stop" in lifecycle_log

    db_health_after_disable = {
        row["module"]: row for row in await db.list_module_health()
    }
    assert db_health_after_disable["fake_healthy_input"]["status"] == "disabled"

    assert await registry.enable_module("fake_healthy_input") is True
    assert registry.is_enabled("fake_healthy_input") is True
    assert lifecycle_log.count("healthy:start") == 2

    # Attempting to enable a completely unknown module at runtime degrades safely
    assert await registry.enable_module("totally_unknown_module") is False
    unknown_status = registry.get_status("totally_unknown_module")
    assert unknown_status is not None
    assert unknown_status["status"] == "degraded"

    assert len(health_events) >= 6
    await registry.stop_all()
    assert registry.is_enabled("fake_healthy_input") is False


async def test_fastapi_lifespan_and_health_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped_flag = False

    class LifespanProbeModule(BaseModule):
        name = "lifespan_probe"

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            nonlocal stopped_flag
            stopped_flag = True

    probe_mod = types.ModuleType("jarvis.modules.inputs.lifespan_probe")
    probe_mod.LifespanProbeModule = LifespanProbeModule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis.modules.inputs.lifespan_probe", probe_mod)

    db_file = tmp_path / "app_lifespan.db"
    cfg_file = tmp_path / "app_config.yaml"
    cfg_file.write_text(
        f"""
server:
  host: "127.0.0.1"
  port: 8199
database:
  path: "{db_file}"
  wal_mode: true
enabled_modules:
  inputs:
    - lifespan_probe
    - broken_nonexistent_module
""",
        encoding="utf-8",
    )

    app = create_app(cfg_file)
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.settings, Settings)
        assert isinstance(app.state.db, Database)
        assert isinstance(app.state.bus, EventBus)
        assert isinstance(app.state.resolver, StateResolver)
        assert isinstance(app.state.registry, ModuleRegistry)
        assert "0001_initial.sql" in app.state.migrations

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["version"] == __version__
            assert body["activity"] == "unknown"
            assert body["database"] == "connected"
            assert "0001_initial.sql" in body["migrations"]
            assert body["degraded_modules"] == ["broken_nonexistent_module"]
            modules_by_name = {m["module"]: m for m in body["modules"]}
            assert modules_by_name["lifespan_probe"]["status"] == "ok"
            assert modules_by_name["broken_nonexistent_module"]["status"] == "degraded"

    # Lifespan shutdown must stop modules and close the SQLite connection cleanly
    assert stopped_flag is True
    assert app.state.db._conn is None


def test_main_entrypoint_invokes_uvicorn_with_configured_host_and_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_file = tmp_path / "cli_config.yaml"
    cfg_file.write_text(
        f"""
server:
  host: "127.0.0.1"
  port: 8765
database:
  path: "{tmp_path / 'cli.db'}"
""",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_uvicorn_run(app: Any, *, host: str, port: int, **kwargs: Any) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["kwargs"] = kwargs

    monkeypatch.setattr("jarvis.main.uvicorn.run", fake_uvicorn_run)
    main(["--config", str(cfg_file)])

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["app"].title == "Jarvis Posture Platform"


async def test_state_resolver_startup_detects_and_closes_stale_or_preboot_spans(
    db: Database,
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now().astimezone()
    yesterday = now - timedelta(days=1)
    fake_boot = now - timedelta(hours=2)

    await db.open_activity_span(
        activity="sitting",
        source="manual_control",
        reason="manual_override",
        started_at=yesterday,
        metadata={
            "ttl_minutes": 30.0,
            "override_expires_at": (yesterday + timedelta(minutes=30)).isoformat(),
        },
    )

    monkeypatch.setattr("jarvis.core.resolver.get_system_boot_time", lambda: fake_boot)

    bus = EventBus()
    resolver = StateResolver(bus=bus, db=db, settings=test_settings)
    state = await resolver.start()

    assert state.activity == "unknown"
    assert state.reason == "startup"

    spans = await db.list_activity_spans()
    assert len(spans) == 1
    assert spans[0]["ended_at"] is not None
    assert spans[0]["ended_at"] == (yesterday + timedelta(minutes=30)).isoformat()
    await resolver.stop()


async def test_resolver_accidental_cycle_under_30s_resumes_previous_duration(
    db: Database,
    test_settings: Settings,
) -> None:
    bus = EventBus()
    resolver = StateResolver(bus=bus, db=db, settings=test_settings)
    await resolver.start()

    events: list[PlatformEvent] = []
    bus.subscribe("activity_state_changed", events.append)

    t0 = datetime(2026, 11, 3, 10, 0, 0).astimezone()

    # 1. User starts sitting
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=t0,
            payload={"activity": "sitting", "ttl_minutes": 30},
        )
    )
    assert resolver.get_effective_state(now=t0).activity == "sitting"
    assert resolver.get_effective_state(now=t0 + timedelta(minutes=30)).elapsed_seconds == 1800

    # 2. At 30m, accidentally cycle to stand (2s)
    t1 = t0 + timedelta(minutes=30)
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=t1,
            payload={"activity": "standing", "ttl_minutes": 30},
        )
    )
    assert resolver.get_effective_state(now=t1).activity == "standing"

    # 3. Cycle to walking (2s)
    t2 = t1 + timedelta(seconds=2)
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=t2,
            payload={"activity": "walking", "ttl_minutes": 30},
        )
    )
    assert resolver.get_effective_state(now=t2).activity == "walking"

    # 4. Cycle to away (2s)
    t3 = t2 + timedelta(seconds=2)
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=t3,
            payload={"activity": "away", "ttl_minutes": 30},
        )
    )
    assert resolver.get_effective_state(now=t3).activity == "away"

    # 5. Cycle back to sitting (2s after away)
    t4 = t3 + timedelta(seconds=2)
    await bus.publish(
        PlatformEvent(
            type="activity_reported",
            source="manual_control",
            timestamp=t4,
            payload={"activity": "sitting", "ttl_minutes": 30},
        )
    )
    eff = resolver.get_effective_state(now=t4)
    assert eff.activity == "sitting"
    # Should NOT have reset to 0! Started_at should be original t0!
    assert eff.started_at == t0
    # Elapsed duration should be 30m + 6s = 1806s
    assert eff.elapsed_seconds == 1806

    # Verify activity_state_changed event carried original started_at
    last_event = events[-1]
    assert last_event.payload["activity"] == "sitting"
    assert last_event.payload["started_at"] == t0.isoformat()

    # Verify only the sitting span exists in DB (transient spans under 30s were not saved)
    spans = await db.list_activity_spans()
    assert len(spans) == 1
    assert spans[0]["activity"] == "sitting"
    assert spans[0]["started_at"] == t0.isoformat()
    assert spans[0]["ended_at"] is None

    await resolver.stop()


