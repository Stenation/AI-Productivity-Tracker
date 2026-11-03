"""Shared pytest fixtures for Posture Platform tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jarvis.core.database import Database
from jarvis.core.settings import DEFAULT_MIGRATIONS_DIR, DatabaseSettings, Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return isolated Settings pointing to a temporary SQLite database."""
    db_file = tmp_path / "test_posture.db"
    return Settings(
        database=DatabaseSettings(
            path=db_file,
            wal_mode=True,
            migrations_dir=DEFAULT_MIGRATIONS_DIR,
        )
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

