"""Shared pytest fixtures for Posture Platform tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jarvis.core.database import Database
from jarvis.core.settings import (
    DEFAULT_MIGRATIONS_DIR,
    DatabaseSettings,
    DesktopNotifySettings,
    Settings,
    WaybarSettings,
)


@pytest.fixture(autouse=True)
def _isolate_host_waybar_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any ambient host POSTURE_WAYBAR_FILE env var from leaking into tests."""
    monkeypatch.delenv("POSTURE_WAYBAR_FILE", raising=False)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return isolated Settings pointing to a temporary SQLite database and Waybar file."""
    db_file = tmp_path / "test_posture.db"
    noop_notify = tmp_path / "noop-notify-send"
    noop_notify.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    noop_notify.chmod(0o755)
    return Settings(
        database=DatabaseSettings(
            path=db_file,
            wal_mode=True,
            migrations_dir=DEFAULT_MIGRATIONS_DIR,
        ),
        waybar=WaybarSettings(
            status_file=tmp_path / "waybar.json",
            signal_waybar=False,
        ),
        desktop_notify=DesktopNotifySettings(
            notify_binary=str(noop_notify),
        ),
    )


@pytest.fixture
async def db(test_settings: Settings) -> AsyncIterator[Database]:
    """Provide an initialized async SQLite WAL Database backed by tmp_path."""
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

