"""Versioned SQLite migration runner recording applied SQL files in schema_meta."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import aiosqlite


async def _ensure_schema_meta(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )
    await conn.commit()


async def get_applied_migrations(conn: aiosqlite.Connection) -> list[str]:
    """Return applied migration filenames in lexical order."""
    await _ensure_schema_meta(conn)
    async with conn.execute(
        "SELECT key FROM schema_meta WHERE key LIKE '%.sql' ORDER BY key ASC"
    ) as cursor:
        rows = await cursor.fetchall()
    return [str(row[0]) for row in rows]


async def apply_migrations(
    conn: aiosqlite.Connection,
    migrations_dir: Path | str,
) -> list[str]:
    """Apply unapplied migrations/*.sql in lexical order on an open connection."""
    migrations_path = Path(migrations_dir).expanduser()
    if not migrations_path.is_dir():
        raise FileNotFoundError(f"Migrations directory not found: {migrations_path}")

    await _ensure_schema_meta(conn)
    applied = set(await get_applied_migrations(conn))

    migration_files = sorted(migrations_path.glob("*.sql"), key=lambda p: p.name)
    newly_applied: list[str] = []

    for migration_file in migration_files:
        filename = migration_file.name
        if filename in applied:
            continue

        sql_script = migration_file.read_text(encoding="utf-8")
        await conn.executescript(sql_script)

        applied_at = datetime.now().astimezone().isoformat()
        await conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
            (filename, applied_at),
        )
        await conn.commit()
        newly_applied.append(filename)

    return newly_applied


async def run_migrations(
    db_path: Path | str,
    migrations_dir: Path | str,
) -> list[str]:
    """Connect to SQLite database at db_path, enable WAL mode, and apply migrations."""
    raw_path = str(db_path)
    if raw_path != ":memory:" and not raw_path.startswith("file:"):
        expanded = Path(raw_path).expanduser()
        expanded.parent.mkdir(parents=True, exist_ok=True)
        target: Path | str = expanded
    else:
        target = raw_path

    async with aiosqlite.connect(target) as conn:
        await conn.execute("PRAGMA busy_timeout = 5000;")
        await conn.execute("PRAGMA journal_mode = WAL;")
        return await apply_migrations(conn, migrations_dir)

