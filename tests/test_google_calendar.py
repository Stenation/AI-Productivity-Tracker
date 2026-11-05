"""Test suite for Google Calendar read-only sync module (FR-03, FR-14, FR-16, FR-17)."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.registry import ModuleRegistry
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import Settings
from jarvis.modules.inputs.google_calendar import (
    READONLY_SCOPES,
    GoogleCalendarModule,
    cli,
    run_auth_flow,
    save_token_file,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
async def calendar_db(test_settings: Settings) -> AsyncIterator[Database]:
    database = Database(
        test_settings.database.path,
        migrations_dir=test_settings.migrations_dir,
        wal_mode=test_settings.database.wal_mode,
        busy_timeout_ms=test_settings.database.busy_timeout_ms,
    )
    await database.connect(run_migrations=True)
    try:
        yield database
    finally:
        await database.close()


def test_readonly_scopes_strict() -> None:
    """Verify strictly read-only OAuth scope is configured with zero write permissions."""
    assert READONLY_SCOPES == ("https://www.googleapis.com/auth/calendar.events.readonly",)
    for scope in READONLY_SCOPES:
        assert "readonly" in scope
        assert "events.readonly" in scope
        assert not scope.endswith("/calendar")
        assert not scope.endswith("/calendar.events")


def test_save_token_file_strict_permissions(tmp_path: Path) -> None:
    """Verify save_token_file writes valid JSON and enforces strict 0600 permissions."""
    token_file = tmp_path / "posture-platform" / "google-token.json"
    sample_json = json.dumps({"token": "ya29.sample", "refresh_token": "1//sample_refresh"})

    saved = save_token_file(token_file, sample_json)
    assert saved == token_file
    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8") == sample_json

    st_mode = token_file.stat().st_mode & 0o777
    assert st_mode == 0o600

    # Overwrite an existing file with mode 0644 and verify it is tightened back to 0600
    token_file.chmod(0o644)
    assert token_file.stat().st_mode & 0o777 == 0o644

    save_token_file(token_file, sample_json)
    assert token_file.stat().st_mode & 0o777 == 0o600


def test_auth_cli_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify standalone CLI bootstrap entrypoint with mock OAuth flow."""
    cred_file = tmp_path / "google-credentials.json"
    cred_file.write_text(json.dumps({"installed": {"client_id": "test_id"}}), encoding="utf-8")
    token_file = tmp_path / "google-token.json"

    # Mock InstalledAppFlow
    class FakeFlow:
        def __init__(self, secrets_path: str, scopes: list[str]) -> None:
            self.secrets_path = secrets_path
            self.scopes = scopes

        def run_local_server(self, port: int = 0, open_browser: bool = True) -> Any:
            mock_creds = MagicMock()
            mock_creds.to_json.return_value = json.dumps(
                {
                    "token": "fake_access_token",
                    "refresh_token": "fake_refresh_token",
                    "scopes": self.scopes,
                }
            )
            return mock_creds

    # Run auth flow with flow_factory
    saved_path = run_auth_flow(
        credentials_path=cred_file,
        oauth_token_path=token_file,
        no_browser=True,
        port=0,
        flow_factory=FakeFlow,
    )
    assert saved_path == token_file
    assert token_file.exists()
    assert (token_file.stat().st_mode & 0o777) == 0o600
    saved_data = json.loads(token_file.read_text(encoding="utf-8"))
    assert saved_data["token"] == "fake_access_token"
    assert saved_data["scopes"] == list(READONLY_SCOPES)

    # Test CLI invocation without --auth (prints help and exits 0)
    assert cli([]) == 0

    # Test CLI with missing credentials file (exits 1)
    missing_cred = tmp_path / "non_existent_cred.json"
    ret = cli(["--auth", "--credentials", str(missing_cred), "--token", str(token_file)])
    assert ret == 1


async def test_lazy_import_and_missing_dependencies_degraded_health(
    bus: EventBus,
    calendar_db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify running without Google libraries installed cleanly marks health degraded."""
    import jarvis.modules.inputs.google_calendar as gc_mod

    # Force get_google_libraries to return None
    monkeypatch.setattr(gc_mod, "_GOOGLE_LIBS_OVERRIDE", None)
    monkeypatch.setattr(gc_mod, "get_google_libraries", lambda: None)

    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={
            "calendar_id": "primary",
            "oauth_token_path": "/tmp/missing-token.json",
        },
    )

    await module.start()
    assert module._running is True
    assert module._last_error is not None
    assert "uv sync --group calendar" in module._last_error

    health = await module.health()
    assert health["status"] == "degraded"
    assert "uv sync --group calendar" in health["last_error"]

    db_health = await calendar_db.list_module_health()
    by_mod = {row["module"]: row for row in db_health}
    assert by_mod["google_calendar"]["status"] == "degraded"
    assert "uv sync --group calendar" in by_mod["google_calendar"]["last_error"]

    await module.stop()
    assert module._running is False


async def test_sync_timed_and_allday_events_and_classification(
    bus: EventBus,
    calendar_db: Database,
) -> None:
    """Verify parsing timed and all-day events, timezone preservation, and classification."""
    fixed_now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=ZoneInfo("Europe/Berlin"))

    # Mock fetcher returning timed and all-day events
    fake_events = [
        {
            "id": "evt_deep_work",
            "summary": "Deep Work: Async Architecture",
            "description": "Refactoring distributed bus",
            "start": {"dateTime": "2026-11-04T10:30:00+01:00"},
            "end": {"dateTime": "2026-11-04T12:00:00+01:00"},
        },
        {
            "id": "evt_meeting",
            "summary": "1:1 Sync with Manager",
            "description": "Weekly status review",
            "start": {"dateTime": "2026-11-04T14:00:00+01:00"},
            "end": {"dateTime": "2026-11-04T14:30:00+01:00"},
        },
        {
            "id": "evt_allday",
            "summary": "Admin & Compliance Day",
            "description": "Quarterly timesheets and compliance forms",
            "start": {"date": "2026-11-04"},
            "end": {"date": "2026-11-05"},
        },
        {
            "id": "evt_color_coding",
            "summary": "Bugfix PR",
            "colorId": "8",  # Color 8 maps to coding
            "start": {"dateTime": "2026-11-04T15:00:00+01:00"},
            "end": {"dateTime": "2026-11-04T16:00:00+01:00"},
        },
        {
            "id": "evt_other_tz",
            "summary": "International Client Meeting",
            "start": {"dateTime": "2026-11-04T19:00:00+03:00"},  # UTC+3
            "end": {"dateTime": "2026-11-04T20:00:00+03:00"},
        },
    ]

    def fake_fetcher(params: dict[str, Any]) -> dict[str, Any]:
        assert params["calendarId"] == "primary"
        assert "timeMin" in params
        assert "timeMax" in params
        assert params["singleEvents"] is True
        return {
            "items": fake_events,
            "timeZone": "Europe/Berlin",
            "nextPageToken": None,
        }

    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={"calendar_id": "primary"},
        fetcher=fake_fetcher,
        clock=lambda: fixed_now,
    )

    synced = await module.sync_calendar(now=fixed_now)
    assert len(synced) == 5

    events_in_db = await calendar_db.list_calendar_events(calendar_id="primary")
    by_id = {row["event_id"]: row for row in events_in_db}

    # Verify deep work classification
    dw = by_id["evt_deep_work"]
    assert dw["title"] == "Deep Work: Async Architecture"
    assert dw["task_type"] == "deep_work"
    assert dw["suitable_for_walking"] is False
    assert dw["recommended_pomodoro"] == "50_10"

    # Verify meeting classification
    mtg = by_id["evt_meeting"]
    assert mtg["task_type"] == "meeting"
    assert mtg["suitable_for_walking"] is False
    assert mtg["recommended_pomodoro"] is None

    # Verify all-day event parsing and timezone
    allday = by_id["evt_allday"]
    assert allday["task_type"] == "admin"
    assert allday["suitable_for_walking"] is True
    assert allday["recommended_pomodoro"] == "25_5"
    start_dt = datetime.fromisoformat(allday["start_time"])
    assert start_dt.hour == 0 and start_dt.minute == 0

    # Verify color mapping
    col = by_id["evt_color_coding"]
    assert col["task_type"] == "coding"
    assert col["suitable_for_walking"] is False
    assert col["recommended_pomodoro"] == "50_10"

    # Verify other timezone preservation
    other_tz = by_id["evt_other_tz"]
    assert other_tz["task_type"] == "meeting"
    assert "19:00:00" in other_tz["start_time"] or "17:00:00" in other_tz["start_time"]


async def test_pagination_handling(
    bus: EventBus,
    calendar_db: Database,
) -> None:
    """Verify handling Google Calendar nextPageToken across multiple pages."""
    fixed_now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)
    calls: list[dict[str, Any]] = []

    def paginated_fetcher(params: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(params))
        page_token = params.get("pageToken")
        if page_token is None:
            return {
                "items": [
                    {
                        "id": "evt_page1",
                        "summary": "Page 1 Event",
                        "start": {"dateTime": "2026-11-04T10:00:00Z"},
                        "end": {"dateTime": "2026-11-04T11:00:00Z"},
                    }
                ],
                "nextPageToken": "token_for_page_2",
            }
        if page_token == "token_for_page_2":
            return {
                "items": [
                    {
                        "id": "evt_page2",
                        "summary": "Page 2 Event",
                        "start": {"dateTime": "2026-11-04T12:00:00Z"},
                        "end": {"dateTime": "2026-11-04T13:00:00Z"},
                    }
                ],
                "nextPageToken": None,
            }
        raise ValueError(f"Unexpected pageToken: {page_token}")

    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={"calendar_id": "primary"},
        fetcher=paginated_fetcher,
        clock=lambda: fixed_now,
    )

    synced = await module.sync_calendar(now=fixed_now)
    assert len(synced) == 2
    assert len(calls) == 2
    assert calls[0].get("pageToken") is None
    assert calls[1].get("pageToken") == "token_for_page_2"

    events = await calendar_db.list_calendar_events(calendar_id="primary")
    ids = {r["event_id"] for r in events}
    assert ids == {"evt_page1", "evt_page2"}


async def test_declined_and_cancelled_pruning(
    bus: EventBus,
    calendar_db: Database,
) -> None:
    """Verify skipping declined/cancelled events and pruning stale cache entries in sync window."""
    t0 = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    # Initial batch: 4 events in cache
    initial_items = [
        {
            "id": "evt_active",
            "summary": "Active Event",
            "start": {"dateTime": "2026-11-04T10:00:00Z"},
            "end": {"dateTime": "2026-11-04T11:00:00Z"},
        },
        {
            "id": "evt_to_cancel",
            "summary": "Event To Cancel Later",
            "start": {"dateTime": "2026-11-04T11:00:00Z"},
            "end": {"dateTime": "2026-11-04T12:00:00Z"},
        },
        {
            "id": "evt_to_decline",
            "summary": "Event To Decline Later",
            "start": {"dateTime": "2026-11-04T13:00:00Z"},
            "end": {"dateTime": "2026-11-04T14:00:00Z"},
        },
        {
            "id": "evt_to_delete",
            "summary": "Event Completely Deleted Later",
            "start": {"dateTime": "2026-11-04T15:00:00Z"},
            "end": {"dateTime": "2026-11-04T16:00:00Z"},
        },
    ]

    current_items = list(initial_items)

    def dynamic_fetcher(params: dict[str, Any]) -> dict[str, Any]:
        return {"items": current_items, "nextPageToken": None}

    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={"calendar_id": "primary"},
        fetcher=dynamic_fetcher,
        clock=lambda: t0,
    )

    # First sync
    res1 = await module.sync_calendar(now=t0)
    assert len(res1) == 4
    assert len(await calendar_db.list_calendar_events()) == 4

    # Second sync:
    # - evt_active remains active
    # - evt_to_cancel is returned with status="cancelled"
    # - evt_to_decline is returned with self responseStatus="declined"
    # - evt_to_delete is removed completely from Google API response
    current_items = [
        {
            "id": "evt_active",
            "summary": "Active Event",
            "start": {"dateTime": "2026-11-04T10:00:00Z"},
            "end": {"dateTime": "2026-11-04T11:00:00Z"},
        },
        {
            "id": "evt_to_cancel",
            "summary": "Cancelled Event",
            "status": "cancelled",
            "start": {"dateTime": "2026-11-04T11:00:00Z"},
            "end": {"dateTime": "2026-11-04T12:00:00Z"},
        },
        {
            "id": "evt_to_decline",
            "summary": "Declined Event",
            "attendees": [
                {"email": "organizer@example.com", "responseStatus": "accepted"},
                {"email": "steff@example.com", "self": True, "responseStatus": "declined"},
            ],
            "start": {"dateTime": "2026-11-04T13:00:00Z"},
            "end": {"dateTime": "2026-11-04T14:00:00Z"},
        },
    ]

    res2 = await module.sync_calendar(now=t0)
    assert len(res2) == 1
    assert res2[0]["event_id"] == "evt_active"

    # Only evt_active should remain cached; all 3 others must be pruned
    cached = await calendar_db.list_calendar_events()
    assert len(cached) == 1
    assert cached[0]["event_id"] == "evt_active"


async def test_calendar_context_changed_emission_and_deduplication(
    bus: EventBus,
    calendar_db: Database,
) -> None:
    """Verify deduplicated context emission on sync and punctual 60s boundary transitions."""
    t0 = datetime(2026, 11, 4, 9, 30, 0, tzinfo=UTC)
    current_time = t0

    events = [
        {
            "id": "evt_first",
            "summary": "Deep Work Block",
            "start": {"dateTime": "2026-11-04T10:00:00Z"},
            "end": {"dateTime": "2026-11-04T11:00:00Z"},
        },
        {
            "id": "evt_second",
            "summary": "Team Standup",
            "start": {"dateTime": "2026-11-04T11:15:00Z"},
            "end": {"dateTime": "2026-11-04T11:30:00Z"},
        },
    ]

    emitted_events: list[PlatformEvent] = []

    def _on_context_changed(event: PlatformEvent) -> None:
        emitted_events.append(event)

    bus.subscribe("calendar_context_changed", _on_context_changed)

    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={"calendar_id": "primary"},
        fetcher=lambda params: {"items": events, "nextPageToken": None},
        clock=lambda: current_time,
    )

    # 1. Initial sync at 09:30 (current=None, next=evt_first) -> Emits initial context
    await module.sync_calendar(now=current_time)
    assert len(emitted_events) == 1
    assert emitted_events[0].payload["current_event"] is None
    assert emitted_events[0].payload["next_event"]["event_id"] == "evt_first"
    assert emitted_events[0].payload["next_event_id"] == "evt_first"

    # 2. Second sync at 09:35 (no context change) -> Deduplicated, no new event
    current_time = t0 + timedelta(minutes=5)
    await module.sync_calendar(now=current_time)
    assert len(emitted_events) == 1

    # 3. Boundary check at 09:45 (still before 10:00) -> Deduplicated, no event
    current_time = t0 + timedelta(minutes=15)
    evt = await module._check_and_emit_context_changed(current_time)
    assert evt is None
    assert len(emitted_events) == 1

    # 4. Cross 10:00 start boundary (time = 10:02): current=evt_first, next=evt_second
    current_time = datetime(2026, 11, 4, 10, 2, 0, tzinfo=UTC)
    evt2 = await module._check_and_emit_context_changed(current_time)
    assert evt2 is not None
    assert len(emitted_events) == 2
    assert emitted_events[1].payload["current_event"]["event_id"] == "evt_first"
    assert emitted_events[1].payload["current_task_type"] == "deep_work"
    assert emitted_events[1].payload["next_event"]["event_id"] == "evt_second"

    # 5. Cross 11:00 end boundary (time = 11:05): between events (current=None, next=evt_second)
    current_time = datetime(2026, 11, 4, 11, 5, 0, tzinfo=UTC)
    evt3 = await module._check_and_emit_context_changed(current_time)
    assert evt3 is not None
    assert len(emitted_events) == 3
    assert emitted_events[2].payload["current_event"] is None
    assert emitted_events[2].payload["next_event"]["event_id"] == "evt_second"

    # 6. Cross 11:15 start boundary (time = 11:20): current=evt_second, next=None
    current_time = datetime(2026, 11, 4, 11, 20, 0, tzinfo=UTC)
    evt4 = await module._check_and_emit_context_changed(current_time)
    assert evt4 is not None
    assert len(emitted_events) == 4
    assert emitted_events[3].payload["current_event"]["event_id"] == "evt_second"
    assert emitted_events[3].payload["next_event"] is None

    # 7. Cross 11:30 end boundary (time = 11:35): both None
    current_time = datetime(2026, 11, 4, 11, 35, 0, tzinfo=UTC)
    evt5 = await module._check_and_emit_context_changed(current_time)
    assert evt5 is not None
    assert len(emitted_events) == 5
    assert emitted_events[4].payload["current_event"] is None
    assert emitted_events[4].payload["next_event"] is None


async def test_token_refresh_and_revoked_offline_resilience(
    bus: EventBus,
    calendar_db: Database,
    tmp_path: Path,
) -> None:
    """Verify headless daemon safe error isolation on missing/revoked token and network errors."""
    token_file = tmp_path / "google-token.json"

    # Seed an event in calendar_cache first to ensure offline cache retention
    t_now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)
    await calendar_db.upsert_calendar_event(
        event_id="evt_offline_preserved",
        title="Offline Preserved Block",
        start_time=t_now,
        end_time=t_now + timedelta(hours=1),
        task_type="coding",
    )

    # 1. Missing token file: sync fails cleanly without crashing daemon, marking degraded
    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={
            "calendar_id": "primary",
            "oauth_token_path": str(token_file),
        },
        clock=lambda: t_now,
    )

    res = await module.sync_calendar(now=t_now)
    assert res == []
    health = await module.health()
    assert health["status"] == "degraded"
    assert "OAuth token file not found" in health["last_error"]

    # Verify cached event remains preserved
    cached = await calendar_db.list_calendar_events()
    assert len(cached) == 1
    assert cached[0]["event_id"] == "evt_offline_preserved"

    # 2. Token refresh error (simulating revoked or expired refresh token)
    class FakeRefreshError(Exception):
        pass

    class FakeRefreshCreds:
        valid = False
        expired = True
        refresh_token = "mock_refresh"

        def refresh(self, req: Any) -> None:
            raise FakeRefreshError("invalid_grant: Token has been expired or revoked.")

    token_file.write_text(
        json.dumps({"token": "old", "refresh_token": "old_ref"}),
        encoding="utf-8",
    )

    module2 = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={
            "calendar_id": "primary",
            "oauth_token_path": str(token_file),
        },
        clock=lambda: t_now,
    )

    # Inject failure to simulate refresh error (revoked token)
    module2._get_service = lambda: (_ for _ in ()).throw(FakeRefreshError("Token revoked"))  # type: ignore[assignment]

    res2 = await module2.sync_calendar(now=t_now)
    assert res2 == []
    health2 = await module2.health()
    assert health2["status"] == "degraded"
    assert "Token revoked" in health2["last_error"]
    assert len(await calendar_db.list_calendar_events()) == 1

    # 3. Offline network / DNS failure (e.g. OSError / socket.gaierror)
    def offline_fetcher(params: dict[str, Any]) -> dict[str, Any]:
        raise OSError("Network is unreachable")

    module3 = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={"calendar_id": "primary"},
        fetcher=offline_fetcher,
        clock=lambda: t_now,
    )

    res3 = await module3.sync_calendar(now=t_now)
    assert res3 == []
    health3 = await module3.health()
    assert health3["status"] == "degraded"
    assert "Network is unreachable" in health3["last_error"]

    # Cached events still safely intact
    assert len(await calendar_db.list_calendar_events()) == 1


async def test_privacy_global_pause_respect(
    bus: EventBus,
    calendar_db: Database,
) -> None:
    """Verify sync and boundary loops strictly respect privacy.global_pause."""
    t_now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    emitted_events: list[PlatformEvent] = []
    bus.subscribe("calendar_context_changed", lambda ev: emitted_events.append(ev))

    events = [
        {
            "id": "evt_secret",
            "summary": "Private Meeting",
            "start": {"dateTime": "2026-11-04T10:00:00Z"},
            "end": {"dateTime": "2026-11-04T11:00:00Z"},
        }
    ]

    module = GoogleCalendarModule(
        bus=bus,
        db=calendar_db,
        config={
            "calendar_id": "primary",
            "privacy": {"global_pause": True},
        },
        fetcher=lambda params: {"items": events, "nextPageToken": None},
        clock=lambda: t_now,
    )

    await module.start()
    assert module._privacy_paused is True

    # With global_pause active, sync returns empty and emits nothing
    res = await module.sync_calendar(now=t_now)
    assert res == []
    assert len(emitted_events) == 0

    # Unpause via EventBus privacy_mode_changed
    await bus.publish(
        PlatformEvent(
            type="privacy_mode_changed",
            source="test",
            payload={"global_pause": False},
        )
    )
    # Yield to let event bus delivery task run
    await asyncio.sleep(0.01)
    assert module._privacy_paused is False

    # Now sync proceeds and emits context change
    res2 = await module.sync_calendar(now=t_now)
    assert len(res2) == 1
    assert len(emitted_events) == 1
    assert emitted_events[0].payload["current_event"]["event_id"] == "evt_secret"

    await module.stop()


async def test_database_calendar_cache_helpers(calendar_db: Database) -> None:
    """Test Database list, delete, and prune helpers for calendar_cache."""
    t1 = datetime(2026, 11, 4, 9, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 11, 4, 11, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 11, 4, 13, 0, 0, tzinfo=UTC)

    await calendar_db.upsert_calendar_event(
        event_id="e1",
        title="Event 1",
        start_time=t1,
        end_time=t1 + timedelta(hours=1),
        task_type="coding",
    )
    await calendar_db.upsert_calendar_event(
        event_id="e2",
        title="Event 2",
        start_time=t2,
        end_time=t2 + timedelta(hours=1),
        task_type="meeting",
    )
    await calendar_db.upsert_calendar_event(
        event_id="e3",
        title="Event 3",
        start_time=t3,
        end_time=t3 + timedelta(hours=1),
        task_type="admin",
    )

    # List all
    all_evts = await calendar_db.list_calendar_events()
    assert len(all_evts) == 3

    # List in window [t1, t2 + 30m] -> matches e1 and e2
    window_evts = await calendar_db.list_calendar_events(
        time_min=t1,
        time_max=t2 + timedelta(minutes=30),
    )
    assert [e["event_id"] for e in window_evts] == ["e1", "e2"]

    # Prune in window [t1 - 1h, t3 + 2h] keeping only e2
    pruned = await calendar_db.prune_stale_calendar_events(
        time_min=t1 - timedelta(hours=1),
        time_max=t3 + timedelta(hours=2),
        keep_event_ids=["e2"],
    )
    assert set(pruned) == {"e1", "e3"}

    remaining = await calendar_db.list_calendar_events()
    assert [e["event_id"] for e in remaining] == ["e2"]

    # Delete e2 directly
    assert await calendar_db.delete_calendar_event("e2") is True
    assert await calendar_db.delete_calendar_event("e2") is False
    assert len(await calendar_db.list_calendar_events()) == 0


async def test_registry_integration_lifecycle(
    calendar_db: Database,
    test_settings: Settings,
) -> None:
    """Verify ModuleRegistry cleanly loads, isolates, and stops google_calendar."""
    bus = EventBus()
    settings = test_settings.model_copy(deep=True)
    settings.enabled_modules.inputs.append("google_calendar")

    registry = ModuleRegistry(bus=bus, db=calendar_db, settings=settings)

    # Starting without calendar group installed cleanly marks degraded, never crashes
    statuses = await registry.start_all()
    assert "google_calendar" in statuses
    # If calendar libraries are not installed in .venv, status is degraded
    assert statuses["google_calendar"] in ("ok", "degraded")

    status_info = registry.get_status("google_calendar")
    assert status_info is not None
    assert status_info["category"] == "inputs"

    # Clean shutdown
    await registry.stop_all()
    assert registry.is_enabled("google_calendar") is False


@pytest.mark.parametrize(
    ("cmd", "expected_output"),
    [
        ([sys.executable, "-c", "import jarvis.modules.base"], ""),
        ([sys.executable, "-c", "import jarvis.core.bus"], ""),
        ([sys.executable, "-c", "import jarvis.core.app"], ""),
        (
            [sys.executable, "-m", "jarvis.modules.inputs.google_calendar", "--help"],
            "Posture Platform Google Calendar OAuth setup utility",
        ),
    ],
)
def test_import_cycle_and_cli_help_in_fresh_subprocesses(
    cmd: list[str],
    expected_output: str,
) -> None:
    """Ensure lightweight core/module imports and CLI execution without circular import errors."""
    import subprocess

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Command {cmd} failed with stderr:\n{result.stderr}"
    if expected_output:
        assert expected_output in result.stdout
