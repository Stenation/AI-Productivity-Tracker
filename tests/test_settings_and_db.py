"""Tests for YAML + Pydantic settings, SQLite WAL migrations, and Database repository."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.core.database import Database
from jarvis.core.migrations import apply_migrations, get_applied_migrations, run_migrations
from jarvis.core.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIGRATIONS_DIR,
    PROJECT_ROOT,
    Settings,
    load_settings,
)


def test_default_config_yaml_parsing_and_privacy_defaults() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8000

    # ~ expansion on database.path and oauth_token_path
    assert "~" not in str(settings.database.path)
    assert settings.database.path.is_absolute()
    assert settings.database.path.name == "posture.db"
    assert settings.database.wal_mode is True

    assert "~" not in str(settings.google_calendar.oauth_token_path)
    assert settings.google_calendar.oauth_token_path.is_absolute()
    assert settings.google_calendar.read_only is True

    # Project-relative migrations/ resolution
    assert settings.migrations_dir == DEFAULT_MIGRATIONS_DIR
    assert settings.migrations_dir == PROJECT_ROOT / "migrations"
    assert settings.migrations_dir.is_dir()

    # Milestone 2 default modules enabled out of the box
    assert settings.enabled_modules.inputs == ["manual_control", "hyprland_ipc"]
    assert settings.enabled_modules.policies == [
        "posture_policy",
        "pomodoro_policy",
        "meeting_policy",
        "focus_policy",
    ]
    assert settings.enabled_modules.outputs == [
        "waybar_status",
        "desktop_notify",
    ]

    # Output module settings defaults
    assert settings.waybar.status_file == Path("/dev/shm/posture_waybar.json")
    assert settings.waybar.refresh_interval_seconds == 15.0
    assert settings.waybar.signal_waybar is True
    assert settings.waybar.waybar_signal == 8
    assert settings.desktop_notify.notify_binary == "notify-send"
    assert settings.desktop_notify.app_name == "Posture Platform"
    assert settings.desktop_notify.urgency == "normal"
    assert settings.desktop_notify.expire_time_ms is None

    # Privacy-safe defaults (SPEC.MD §7)
    assert settings.privacy.global_pause is False
    assert settings.privacy.dnd_mode is False
    assert settings.privacy.hyprland_ipc_mode == "observe_only"
    assert settings.privacy.allow_webcam is False
    assert settings.privacy.microphone_mode == "push_to_talk"
    assert settings.privacy.allow_screenshots is False
    assert settings.privacy.allow_vlm_screenshots is False
    assert settings.privacy.allow_tts is False
    assert settings.privacy.allow_calendar_writes is False
    assert settings.privacy.allow_phone_webhook is False
    assert settings.privacy.allow_external_notifications is False
    assert settings.privacy.dashboard_bind_local_only is True

    # Posture, Pomodoro, and Focus settings
    assert settings.posture.manual_override_ttl_minutes == 30
    assert settings.posture.max_sitting_minutes == 50
    assert settings.posture.max_walking_minutes == 45
    assert settings.posture.pre_meeting_warning_minutes == 10
    assert settings.posture.recommendation_cooldown_minutes == 30

    assert settings.pomodoro.get("deep_work") == "50_10"
    assert settings.pomodoro.get("coding") == "50_10"
    assert settings.pomodoro.get("admin") == "25_5"
    assert settings.pomodoro.get("meeting") is None
    assert settings.pomodoro.get("break") is None
    assert settings.pomodoro.get("unknown") is None
    assert settings.pomodoro.auto_start is False

    assert settings.focus.distraction_grace_seconds == 180
    assert "code" in settings.focus.work_app_classes
    assert "Reddit" in settings.focus.distraction_title_patterns


def test_custom_yaml_and_local_bind_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text(
        """
server:
  host: "127.0.0.1"
  port: 8123
migrations_dir: "migrations"
database:
  path: "~/custom-posture/test.db"
google_calendar:
  oauth_token_path: "~/custom-posture/token.json"
  credentials_path: "~/custom-posture/creds.json"
phone_bridge:
  token_path: "~/custom-posture/phone-token.txt"
posture:
  recommendation_cooldown_minutes: 45
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("JARVIS_CONFIG", str(yaml_file))
    settings = load_settings()
    assert settings.server.port == 8123
    assert "~" not in str(settings.database.path)
    assert "~" not in str(settings.google_calendar.oauth_token_path)
    assert settings.google_calendar.credentials_path is not None
    assert "~" not in str(settings.google_calendar.credentials_path)
    assert settings.phone_bridge.token_path is not None
    assert "~" not in str(settings.phone_bridge.token_path)
    assert settings.migrations_dir == PROJECT_ROOT / "migrations"
    assert settings.posture.recommendation_cooldown_minutes == 45

    bad_yaml = tmp_path / "non_local.yaml"
    bad_yaml.write_text(
        """
server:
  host: "0.0.0.0"
privacy:
  dashboard_bind_local_only: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        Settings.from_yaml(bad_yaml)


async def test_migrations_idempotency_lexical_order_and_wal_mode(tmp_path: Path) -> None:
    custom_migrations = tmp_path / "migrations"
    custom_migrations.mkdir()
    shutil.copy2(
        DEFAULT_MIGRATIONS_DIR / "0001_initial.sql",
        custom_migrations / "0001_initial.sql",
    )

    db_path = tmp_path / "wal_check.db"
    first_run = await run_migrations(db_path, custom_migrations)
    assert first_run == ["0001_initial.sql"]

    # Second run must be a no-op
    second_run = await run_migrations(db_path, custom_migrations)
    assert second_run == []

    # Add a second migration and confirm lexical application of only the new file
    (custom_migrations / "0002_seed_setting.sql").write_text(
        """
        INSERT INTO settings (key, value_json, updated_at)
        VALUES ('seeded', '"yes"', '2026-11-03T17:00:00+01:00');
        """,
        encoding="utf-8",
    )

    async with Database(db_path, migrations_dir=custom_migrations) as database:
        assert await database.get_journal_mode() == "wal"
        assert await database.get_busy_timeout() == 5000

        applied = await get_applied_migrations(database.connection)
        assert applied == ["0001_initial.sql", "0002_seed_setting.sql"]

        reapplied = await apply_migrations(database.connection, custom_migrations)
        assert reapplied == []
        assert await database.get_setting("seeded") == "yes"

        # Verify all 9 required tables and recommendation dedupe index exist
        async with database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ) as cursor:
            tables = {row[0] for row in await cursor.fetchall()}

        expected_tables = {
            "schema_meta",
            "activity_spans",
            "window_spans",
            "calendar_cache",
            "timer_spans",
            "recommendations",
            "calendar_drafts",
            "module_health",
            "settings",
        }
        assert expected_tables.issubset(tables)

        async with database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ) as cursor:
            indexes = {row[0] for row in await cursor.fetchall()}
        assert "idx_recommendations_dedupe_created" in indexes


async def test_activity_and_window_span_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "spans.db"
    t0 = datetime(2026, 11, 3, 15, 0, 0).astimezone()
    t1 = t0 + timedelta(minutes=25)
    t2 = t0 + timedelta(minutes=40)

    db1 = Database(db_file)
    await db1.connect()
    assert await db1.get_current_activity_span() is None

    s1 = await db1.open_activity_span(
        "sitting",
        "manual_control",
        1.0,
        "manual_override",
        t0,
        metadata={"ttl_minutes": 30},
    )
    assert s1["activity"] == "sitting"
    assert s1["ended_at"] is None
    assert s1["metadata"] == {"ttl_minutes": 30}
    assert datetime.fromisoformat(s1["started_at"]).tzinfo is not None

    # Opening a new span auto-closes the previous open span at started_at
    s2 = await db1.open_activity_span(
        "walking",
        "swalk_ble",
        0.92,
        "sensor_consensus",
        t1,
    )
    assert s2["activity"] == "walking"
    assert s2["ended_at"] is None

    current = await db1.get_current_activity_span()
    assert current is not None
    assert current["id"] == s2["id"]
    await db1.close()

    # Reconnect to verify durable persistence on disk
    async with Database(db_file) as db2:
        persisted_current = await db2.get_current_activity_span()
        assert persisted_current is not None
        assert persisted_current["activity"] == "walking"

        closed = await db2.close_current_activity_span(t2)
        assert closed is not None
        assert closed["id"] == s2["id"]
        assert closed["ended_at"] == t2.isoformat()
        assert await db2.get_current_activity_span() is None
        assert await db2.close_current_activity_span(t2) is None

        all_spans = await db2.list_activity_spans()
        assert [row["activity"] for row in all_spans] == ["walking", "sitting"]
        assert all_spans[1]["ended_at"] == t1.isoformat()

        recent_spans = await db2.list_activity_spans(since=t1)
        assert [row["activity"] for row in recent_spans] == ["walking"]

        # Window spans lifecycle
        w1 = await db2.open_window_span(
            app_class="code",
            window_title="database.py - Jarvis",
            category="focused",
            classifier="app_class",
            started_at=t0,
        )
        assert w1["ended_at"] is None
        w2 = await db2.open_window_span(
            app_class="firefox",
            window_title="Reddit - /r/archlinux",
            category="distracted",
            classifier="title_pattern",
            started_at=t1,
            metadata={"workspace": 2},
        )
        assert w2["metadata"] == {"workspace": 2}
        closed_w2 = await db2.close_current_window_span(t2)
        assert closed_w2 is not None
        assert closed_w2["ended_at"] == t2.isoformat()

        windows = await db2.list_window_spans(limit=10)
        assert [w["app_class"] for w in windows] == ["firefox", "code"]
        assert windows[1]["ended_at"] == t1.isoformat()

        # Update window span classification
        updated = await db2.update_window_span_classification(
            span_id=int(w1["id"]),
            category="distracted",
            classifier="llm",
            confidence=0.88,
            metadata_patch={"reason": "reclassified"},
        )
        assert updated is not None
        assert updated["category"] == "distracted"
        assert updated["classifier"] == "llm"
        assert updated["metadata"]["reason"] == "reclassified"
        assert updated["metadata"]["confidence"] == 0.88



async def test_timer_spans_and_interrupted_cleanup(db: Database) -> None:
    t0 = datetime(2026, 11, 3, 16, 0, 0).astimezone()

    # Completed timer should never be touched by interrupted cleanup
    completed_timer = await db.start_timer_span(
        "25_5", 1500, calendar_event_id="evt_done", started_at=t0 - timedelta(hours=1)
    )
    ended_done = await db.end_timer_span(
        completed_timer["id"],
        "completed",
        ended_at=t0 - timedelta(minutes=35),
    )
    assert ended_done is not None
    assert ended_done["status"] == "completed"
    assert ended_done["actual_duration_seconds"] == 1500

    # Start a running timer and simulate daemon restart 7m30s later
    running = await db.start_timer_span(
        "50_10", 3000, calendar_event_id="evt_focus", started_at=t0
    )
    active = await db.get_active_timer()
    assert active is not None
    assert active["id"] == running["id"]

    restart_at = t0 + timedelta(minutes=7, seconds=30)
    interrupted_count = await db.mark_running_timers_interrupted(restart_at)
    assert interrupted_count == 1
    assert await db.get_active_timer() is None

    # Calling again when none are running returns 0
    assert await db.mark_running_timers_interrupted(restart_at) == 0

    timers = await db.list_timer_spans(limit=10)
    by_id = {row["id"]: row for row in timers}
    assert by_id[running["id"]]["status"] == "interrupted"
    assert by_id[running["id"]]["ended_at"] == restart_at.isoformat()
    assert by_id[running["id"]]["actual_duration_seconds"] == 450
    assert by_id[completed_timer["id"]]["status"] == "completed"
    assert by_id[completed_timer["id"]]["actual_duration_seconds"] == 1500


async def test_recommendation_lifecycle_and_dedupe_cooldown_queries(db: Database) -> None:
    base = datetime(2026, 11, 3, 14, 0, 0).astimezone()
    dedupe = "posture:max_sitting"

    assert await db.find_recent_recommendation(dedupe, since=base - timedelta(minutes=30)) is None

    rec = await db.create_recommendation(
        policy="posture_policy",
        message="You have been sitting for 55 minutes.",
        suggested_action="stand_or_walk",
        action_payload={"recommended_activity": "standing"},
        dedupe_key=dedupe,
        created_at=base,
    )
    assert rec["status"] == "pending"
    assert rec["action_payload"] == {"recommended_activity": "standing"}

    fetched = await db.get_recommendation(rec["id"])
    assert fetched is not None
    assert fetched["dedupe_key"] == dedupe

    pending = await db.get_pending_recommendations()
    assert [r["id"] for r in pending] == [rec["id"]]

    # Pending recommendation suppresses duplicates even if created before `since`
    later_check = base + timedelta(hours=2)
    still_pending_hit = await db.find_recent_recommendation(
        dedupe, since=later_check - timedelta(minutes=30)
    )
    assert still_pending_hit is not None
    assert still_pending_hit["id"] == rec["id"]

    # Distinct dedupe_key is not suppressed
    assert (
        await db.find_recent_recommendation(
            "meeting:pre_warn:evt1", since=base - timedelta(minutes=30)
        )
        is None
    )

    # Resolve recommendation at base + 10m
    resolved_at = base + timedelta(minutes=10)
    resolved = await db.resolve_recommendation(
        rec["id"],
        "dismissed",
        metadata={"via": "waybar"},
        resolved_at=resolved_at,
    )
    assert resolved is not None
    assert resolved["status"] == "dismissed"
    assert resolved["resolution_metadata"] == {"via": "waybar"}
    assert await db.get_pending_recommendations() == []

    # Within 30m cooldown of resolution (at base + 25m -> since = base - 5m): suppressed
    within_cooldown = await db.find_recent_recommendation(
        dedupe, since=resolved_at - timedelta(minutes=15)
    )
    assert within_cooldown is not None
    assert within_cooldown["id"] == rec["id"]

    # After 30m cooldown expires (at base + 45m -> since = base + 15m > resolved_at): None
    after_cooldown = await db.find_recent_recommendation(
        dedupe, since=resolved_at + timedelta(minutes=5)
    )
    assert after_cooldown is None

    # Without `since`, returns most recent historical record for that dedupe_key
    latest = await db.find_recent_recommendation(dedupe)
    assert latest is not None
    assert latest["id"] == rec["id"]

    with pytest.raises(ValueError):
        await db.resolve_recommendation(rec["id"], "invalid_status")


async def test_calendar_cache_module_health_and_settings(db: Database) -> None:
    t_utc = datetime(2026, 11, 3, 15, 0, 0, tzinfo=UTC)

    await db.upsert_calendar_event(
        event_id="evt_coding",
        title="Backend refactor",
        start_time=t_utc,
        end_time=t_utc + timedelta(hours=1),
        task_type="coding",
        suitable_for_walking=False,
        recommended_pomodoro="50_10",
        raw={"summary": "Backend refactor"},
    )
    await db.upsert_calendar_event(
        event_id="evt_meeting",
        title="Team Sync",
        start_time=t_utc + timedelta(hours=1),
        end_time=t_utc + timedelta(hours=1, minutes=30),
        task_type="meeting",
        suitable_for_walking=False,
    )

    # Before first event
    cur_evt, next_evt = await db.get_current_and_next_events(
        now=t_utc - timedelta(minutes=15)
    )
    assert cur_evt is None
    assert next_evt is not None
    assert next_evt["event_id"] == "evt_coding"

    # During first event
    cur_evt, next_evt = await db.get_current_and_next_events(
        now=t_utc + timedelta(minutes=50)
    )
    assert cur_evt is not None
    assert cur_evt["event_id"] == "evt_coding"
    assert cur_evt["suitable_for_walking"] is False
    assert next_evt is not None
    assert next_evt["event_id"] == "evt_meeting"

    # Exact boundary at end of first / start of second
    cur_evt, next_evt = await db.get_current_and_next_events(
        now=t_utc + timedelta(hours=1)
    )
    assert cur_evt is not None
    assert cur_evt["event_id"] == "evt_meeting"
    assert next_evt is None

    # Module health upsert preserves last_event_at and clears last_error on recovery
    event_ts = t_utc.astimezone()
    await db.upsert_module_health("hyprland_ipc", "ok", last_event_at=event_ts)
    await db.upsert_module_health(
        "hyprland_ipc", "degraded", last_error="socket timeout"
    )
    health_rows = await db.list_module_health()
    assert len(health_rows) == 1
    assert health_rows[0]["status"] == "degraded"
    assert health_rows[0]["last_error"] == "socket timeout"
    assert health_rows[0]["last_event_at"] == event_ts.isoformat()

    await db.upsert_module_health("hyprland_ipc", "ok")
    recovered = (await db.list_module_health())[0]
    assert recovered["status"] == "ok"
    assert recovered["last_error"] is None
    assert recovered["last_event_at"] == event_ts.isoformat()

    # Settings key/value store
    assert await db.get_setting("missing_key", default="fallback") == "fallback"
    await db.set_setting("privacy_pause", True)
    assert await db.get_setting("privacy_pause") is True
    await db.set_setting("calibration", {"sit_y": 0.61, "stand_y": 0.42})
    assert await db.get_setting("calibration") == {"sit_y": 0.61, "stand_y": 0.42}


async def test_database_prune_retention(db: Database) -> None:
    now = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
    old_time = now - timedelta(days=100)
    recent_time = now - timedelta(days=30)

    # Old closed activity span vs recent closed activity span vs open activity span
    await db.open_activity_span(
        activity="sitting",
        source="manual",
        confidence=1.0,
        reason="manual_override",
        started_at=old_time,
    )
    await db.close_current_activity_span(ended_at=old_time + timedelta(hours=1))

    await db.open_activity_span(
        activity="standing",
        source="manual",
        confidence=1.0,
        reason="manual_override",
        started_at=recent_time,
    )
    await db.close_current_activity_span(ended_at=recent_time + timedelta(hours=1))

    # Currently open span (ended_at IS NULL)
    await db.open_activity_span(
        activity="walking",
        source="manual",
        confidence=1.0,
        reason="manual_override",
        started_at=old_time - timedelta(days=5),
    )

    # Window spans
    await db.open_window_span(
        app_class="code",
        window_title="old",
        category="coding",
        started_at=old_time,
    )
    await db.close_current_window_span(ended_at=old_time + timedelta(minutes=30))

    await db.open_window_span(
        app_class="code",
        window_title="recent",
        category="coding",
        started_at=recent_time,
    )
    await db.close_current_window_span(ended_at=recent_time + timedelta(minutes=30))

    # Recommendations
    old_rec = await db.create_recommendation(
        policy="posture_policy",
        suggested_action="suggest_stand",
        message="Stand up",
        created_at=old_time,
    )
    await db.resolve_recommendation(int(old_rec["id"]), "accepted")

    pending_rec = await db.create_recommendation(
        policy="posture_policy",
        suggested_action="suggest_sit",
        message="Sit down",
        created_at=old_time,
    )

    # Prune 90-day retention
    counts = await db.prune_retention(retention_days=90, now=now)
    assert counts.get("activity_spans") == 1
    assert counts.get("window_spans") == 1
    assert counts.get("recommendations") == 1

    # Remaining spans: recent span and open span preserved
    remaining_activity = await db.list_activity_spans(limit=10)
    assert len(remaining_activity) == 2
    activities = {r["activity"] for r in remaining_activity}
    assert activities == {"standing", "walking"}

    # Pending recommendation preserved even if created_at is old
    remaining_recs = await db.list_recommendations(limit=10)
    assert len(remaining_recs) == 1
    assert remaining_recs[0]["id"] == pending_rec["id"]


async def test_app_lifespan_pruning_and_persisted_settings(
    db: Database,
    test_settings: Settings,
) -> None:
    from jarvis.core.app import create_app

    # Seed persisted overrides
    await db.set_setting("privacy.global_pause", True)
    await db.set_setting("privacy.allow_tts", True)
    await db.set_setting("llm.text_model", "qwen3:14b")
    await db.set_setting("module.manual_control.enabled", True)

    app = create_app(settings=test_settings)
    async with app.router.lifespan_context(app):
        # Settings should have been updated from persisted DB values
        assert app.state.settings.privacy.global_pause is True
        assert app.state.settings.privacy.allow_tts is True
        assert app.state.settings.llm.text_model == "qwen3:14b"
        assert app.state.registry.is_enabled("manual_control") is True


async def test_activity_span_under_30s_discarded_and_resumes_previous(
    db: Database,
) -> None:
    t0 = datetime(2026, 11, 3, 10, 0, 0).astimezone()

    # 1. User sits for 30 minutes (t0 to t0 + 30m)
    s1 = await db.open_activity_span("sitting", "manual_control", started_at=t0)
    assert s1["activity"] == "sitting"
    assert s1["started_at"] == t0.isoformat()

    # 2. At 30m, accidentally cycles to standing
    t1 = t0 + timedelta(minutes=30)
    s2 = await db.open_activity_span("standing", "manual_control", started_at=t1)
    assert s2["activity"] == "standing"

    # Sitting span was >= 30s (30m), so it was saved with ended_at
    spans = await db.list_activity_spans()
    assert len(spans) == 2
    assert spans[0]["activity"] == "standing" and spans[0]["ended_at"] is None
    assert spans[1]["activity"] == "sitting" and spans[1]["ended_at"] == t1.isoformat()

    # 3. 5 seconds later, cycles to walking (standing lasted 5s < 30s)
    t2 = t1 + timedelta(seconds=5)
    s3 = await db.open_activity_span("walking", "manual_control", started_at=t2)
    # Standing was < 30s, so it was deleted and NOT saved
    spans = await db.list_activity_spans()
    assert len(spans) == 2
    assert spans[0]["activity"] == "walking" and spans[0]["ended_at"] is None
    assert spans[1]["activity"] == "sitting"

    # 4. 5 seconds later, cycles to away (walking lasted 5s < 30s)
    t3 = t2 + timedelta(seconds=5)
    s4 = await db.open_activity_span("away", "manual_control", started_at=t3)
    spans = await db.list_activity_spans()
    assert len(spans) == 2
    assert spans[0]["activity"] == "away" and spans[0]["ended_at"] is None
    assert spans[1]["activity"] == "sitting"

    # 5. 5 seconds later, cycles back to sitting (away lasted 5s < 30s)
    # Sitting was the previous valid span, ended 15 seconds ago -> RESUMES s1!
    t4 = t3 + timedelta(seconds=5)
    s5 = await db.open_activity_span("sitting", "manual_control", started_at=t4)
    assert s5["id"] == s1["id"]
    assert s5["started_at"] == t0.isoformat()
    assert s5["ended_at"] is None

    # Only 1 span in DB now: the resumed sitting span starting from t0
    spans = await db.list_activity_spans()
    assert len(spans) == 1
    assert spans[0]["id"] == s1["id"]
    assert spans[0]["activity"] == "sitting"
    assert spans[0]["started_at"] == t0.isoformat()
    assert spans[0]["ended_at"] is None

    # 6. If user now switches to standing and stays for 45s (>= 30s), it IS saved
    t5 = t4 + timedelta(minutes=10)
    s6 = await db.open_activity_span("standing", "manual_control", started_at=t5)
    t6 = t5 + timedelta(seconds=45)
    s7 = await db.open_activity_span("sitting", "manual_control", started_at=t6)
    # Standing was 45s >= 30s, so it was saved, and sitting starts a NEW span
    spans = await db.list_activity_spans()
    assert len(spans) == 3
    assert spans[0]["activity"] == "sitting" and spans[0]["started_at"] == t6.isoformat()
    assert spans[1]["activity"] == "standing" and spans[1]["ended_at"] == t6.isoformat()
    assert spans[2]["activity"] == "sitting" and spans[2]["ended_at"] == t5.isoformat()



