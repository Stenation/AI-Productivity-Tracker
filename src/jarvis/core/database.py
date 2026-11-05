"""Async SQLite WAL repository layer for the Posture Platform."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from jarvis.core.migrations import apply_migrations
from jarvis.core.schema import get_system_boot_time
from jarvis.core.settings import DEFAULT_MIGRATIONS_DIR

VALID_RECOMMENDATION_RESOLUTIONS = frozenset(
    {"accepted", "snoozed", "dismissed", "expired"}
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _to_dt(value: datetime | str | None = None) -> datetime:
    if value is None:
        return _now()
    if isinstance(value, str):
        return datetime.fromisoformat(value).astimezone()
    if isinstance(value, datetime):
        return value.astimezone()
    raise TypeError(f"Unsupported timestamp value: {value!r}")


def _to_iso(value: datetime | str | None = None) -> str:
    return _to_dt(value).isoformat()


def _loads_optional_json(raw: str | None) -> Any:
    if raw is None:
        return None
    return json.loads(raw)


def _dumps_optional_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _row_to_dict(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if "metadata_json" in data:
        data["metadata"] = _loads_optional_json(data["metadata_json"])
    if "action_payload_json" in data:
        data["action_payload"] = _loads_optional_json(data["action_payload_json"])
    if "resolution_metadata_json" in data:
        data["resolution_metadata"] = _loads_optional_json(
            data["resolution_metadata_json"]
        )
    if "suitable_for_walking" in data and data["suitable_for_walking"] is not None:
        data["suitable_for_walking"] = bool(data["suitable_for_walking"])
    return data


class Database:
    """Async SQLite repository backed by aiosqlite in WAL mode."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        migrations_dir: Path | str | None = None,
        wal_mode: bool = True,
        busy_timeout_ms: int = 5000,
    ) -> None:
        raw_path = str(db_path)
        if raw_path == ":memory:" or raw_path.startswith("file:"):
            self.db_path: Path | str = raw_path
        else:
            self.db_path = Path(raw_path).expanduser()
        self.migrations_dir = (
            Path(migrations_dir).expanduser()
            if migrations_dir is not None
            else DEFAULT_MIGRATIONS_DIR
        )
        self.wal_mode = wal_mode
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected; call connect() first.")
        return self._conn

    async def connect(self, *, run_migrations: bool = True) -> list[str]:
        """Open the SQLite connection, configure WAL + busy_timeout, and migrate."""
        async with self._lock:
            if self._conn is None:
                if isinstance(self.db_path, Path):
                    self.db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = await aiosqlite.connect(self.db_path)
                conn.row_factory = aiosqlite.Row
                await conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)};")
                if self.wal_mode:
                    await conn.execute("PRAGMA journal_mode = WAL;")
                await conn.execute("PRAGMA foreign_keys = ON;")
                await conn.commit()
                self._conn = conn

            if run_migrations:
                return await apply_migrations(self._conn, self.migrations_dir)
            return []

    async def initialize(self) -> list[str]:
        """Alias for connect(run_migrations=True) used in lifespan startup."""
        return await self.connect(run_migrations=True)

    async def close(self) -> None:
        """Close the open SQLite connection cleanly."""
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

    async def __aenter__(self) -> Database:
        await self.connect(run_migrations=True)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        await self.close()

    async def get_journal_mode(self) -> str:
        async with self._lock:
            async with self.connection.execute("PRAGMA journal_mode;") as cursor:
                row = await cursor.fetchone()
            return str(row[0]).lower() if row else ""

    async def get_busy_timeout(self) -> int:
        async with self._lock:
            async with self.connection.execute("PRAGMA busy_timeout;") as cursor:
                row = await cursor.fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Activity spans
    # ------------------------------------------------------------------

    async def get_current_activity_span(self) -> dict[str, Any] | None:
        async with self._lock:
            async with self.connection.execute(
                """
                SELECT * FROM activity_spans
                WHERE ended_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ) as cursor:
                row = await cursor.fetchone()
            return _row_to_dict(row)

    async def open_activity_span(
        self,
        activity: str,
        source: str,
        confidence: float = 1.0,
        reason: str = "state_change",
        started_at: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
        min_duration_seconds: float = 30.0,
        max_resume_gap_seconds: float = 120.0,
    ) -> dict[str, Any]:
        started_iso = _to_iso(started_at)
        metadata_json = _dumps_optional_json(metadata)
        current_dt = _to_dt(started_iso)

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                "SELECT id, started_at FROM activity_spans WHERE ended_at IS NULL"
            ) as cur:
                open_rows = await cur.fetchall()

            boot_dt = get_system_boot_time()
            for row in open_rows:
                row_id = row["id"]
                end_dt = current_dt
                row_start = _to_dt(row["started_at"])
                if boot_dt and row_start < boot_dt < end_dt:
                    end_dt = boot_dt
                duration_seconds = (end_dt - row_start).total_seconds()
                if min_duration_seconds > 0 and duration_seconds < min_duration_seconds:
                    await conn.execute(
                        "DELETE FROM activity_spans WHERE id = ?",
                        (row_id,),
                    )
                else:
                    await conn.execute(
                        "UPDATE activity_spans SET ended_at = ? WHERE id = ?",
                        (end_dt.isoformat(), row_id),
                    )

            if max_resume_gap_seconds > 0:
                async with conn.execute(
                    """
                    SELECT * FROM activity_spans
                    WHERE ended_at IS NOT NULL
                    ORDER BY started_at DESC, id DESC
                    LIMIT 1
                    """
                ) as last_cur:
                    last_row = await last_cur.fetchone()

                if last_row is not None and last_row["activity"] == activity:
                    last_end = _to_dt(last_row["ended_at"])
                    gap = (current_dt - last_end).total_seconds()
                    if -5.0 <= gap <= max_resume_gap_seconds:
                        span_id = last_row["id"]
                        await conn.execute(
                            """
                            UPDATE activity_spans
                            SET ended_at = NULL,
                                source = ?,
                                confidence = ?,
                                reason = ?,
                                metadata_json = COALESCE(?, metadata_json)
                            WHERE id = ?
                            """,
                            (source, float(confidence), reason, metadata_json, span_id),
                        )
                        await conn.commit()

                        async with conn.execute(
                            "SELECT * FROM activity_spans WHERE id = ?",
                            (span_id,),
                        ) as fetch_cursor:
                            row = await fetch_cursor.fetchone()
                        result = _row_to_dict(row)
                        assert result is not None
                        return result

            cursor = await conn.execute(
                """
                INSERT INTO activity_spans (
                  activity, source, confidence, reason, started_at, ended_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (activity, source, float(confidence), reason, started_iso, metadata_json),
            )
            span_id = cursor.lastrowid
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM activity_spans WHERE id = ?",
                (span_id,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def close_current_activity_span(
        self,
        ended_at: datetime | str | None = None,
        min_duration_seconds: float = 30.0,
    ) -> dict[str, Any] | None:
        ended_iso = _to_iso(ended_at)
        end_dt = _to_dt(ended_iso)

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                """
                SELECT id, started_at FROM activity_spans
                WHERE ended_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None

            span_id = int(row["id"])
            row_start = _to_dt(row["started_at"])
            boot_dt = get_system_boot_time()
            if boot_dt and row_start < boot_dt < end_dt:
                end_dt = boot_dt
            duration_seconds = (end_dt - row_start).total_seconds()
            if min_duration_seconds > 0 and duration_seconds < min_duration_seconds:
                await conn.execute("DELETE FROM activity_spans WHERE id = ?", (span_id,))
                await conn.commit()
                return None

            await conn.execute(
                "UPDATE activity_spans SET ended_at = ? WHERE id = ?",
                (end_dt.isoformat(), span_id),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM activity_spans WHERE id = ?",
                (span_id,),
            ) as fetch_cursor:
                closed_row = await fetch_cursor.fetchone()
            return _row_to_dict(closed_row)

    async def list_activity_spans(
        self,
        since: datetime | str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            conn = self.connection
            if since is None:
                async with conn.execute(
                    """
                    SELECT * FROM activity_spans
                    ORDER BY started_at DESC, id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ) as cursor:
                    rows = await cursor.fetchall()
                return [d for r in rows if (d := _row_to_dict(r)) is not None]

            since_dt = _to_dt(since)
            async with conn.execute(
                "SELECT * FROM activity_spans ORDER BY started_at DESC, id DESC"
            ) as cursor:
                rows = await cursor.fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                started_dt = _to_dt(row["started_at"])
                ended_raw = row["ended_at"]
                if (
                    started_dt >= since_dt
                    or ended_raw is None
                    or _to_dt(ended_raw) > since_dt
                ):
                    converted = _row_to_dict(row)
                    if converted is not None:
                        results.append(converted)
                        if len(results) >= int(limit):
                            break
            return results

    # ------------------------------------------------------------------
    # Timer spans
    # ------------------------------------------------------------------

    async def get_active_timer(self) -> dict[str, Any] | None:
        async with self._lock:
            async with self.connection.execute(
                """
                SELECT * FROM timer_spans
                WHERE status = 'running' AND ended_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ) as cursor:
                row = await cursor.fetchone()
            return _row_to_dict(row)

    async def mark_running_timers_interrupted(
        self,
        ended_at: datetime | str | None = None,
    ) -> int:
        end_dt = _to_dt(ended_at)
        end_iso = end_dt.isoformat()

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                """
                SELECT id, started_at, ended_at, actual_duration_seconds
                FROM timer_spans
                WHERE status = 'running' OR ended_at IS NULL
                """
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return 0

            for row in rows:
                row_id = int(row["id"])
                effective_end_dt = (
                    _to_dt(row["ended_at"]) if row["ended_at"] is not None else end_dt
                )
                effective_end_iso = (
                    str(row["ended_at"]) if row["ended_at"] is not None else end_iso
                )
                if row["actual_duration_seconds"] is not None:
                    actual_seconds = int(row["actual_duration_seconds"])
                else:
                    start_dt = _to_dt(row["started_at"])
                    actual_seconds = max(
                        0, int((effective_end_dt - start_dt).total_seconds())
                    )

                await conn.execute(
                    """
                    UPDATE timer_spans
                    SET status = 'interrupted',
                        ended_at = ?,
                        actual_duration_seconds = ?
                    WHERE id = ?
                    """,
                    (effective_end_iso, actual_seconds, row_id),
                )

            await conn.commit()
            return len(rows)

    async def start_timer_span(
        self,
        kind: str,
        planned_duration_seconds: int | None,
        calendar_event_id: str | None = None,
        started_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        started_iso = _to_iso(started_at)

        async with self._lock:
            conn = self.connection
            cursor = await conn.execute(
                """
                INSERT INTO timer_spans (
                  kind, calendar_event_id, started_at, ended_at,
                  planned_duration_seconds, actual_duration_seconds, status
                )
                VALUES (?, ?, ?, NULL, ?, NULL, 'running')
                """,
                (
                    kind,
                    calendar_event_id,
                    started_iso,
                    int(planned_duration_seconds)
                    if planned_duration_seconds is not None
                    else None,
                ),
            )
            timer_id = cursor.lastrowid
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM timer_spans WHERE id = ?",
                (timer_id,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def end_timer_span(
        self,
        timer_id: int,
        status: str = "completed",
        ended_at: datetime | str | None = None,
        actual_duration_seconds: int | None = None,
    ) -> dict[str, Any] | None:
        end_dt = _to_dt(ended_at)
        end_iso = end_dt.isoformat()

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                "SELECT * FROM timer_spans WHERE id = ?",
                (int(timer_id),),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is None:
                return None

            if actual_duration_seconds is None:
                start_dt = _to_dt(existing["started_at"])
                computed_seconds = max(0, int((end_dt - start_dt).total_seconds()))
            else:
                computed_seconds = max(0, int(actual_duration_seconds))

            await conn.execute(
                """
                UPDATE timer_spans
                SET status = ?, ended_at = ?, actual_duration_seconds = ?
                WHERE id = ?
                """,
                (status, end_iso, computed_seconds, int(timer_id)),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM timer_spans WHERE id = ?",
                (int(timer_id),),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            return _row_to_dict(row)

    async def list_timer_spans(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:
            async with self.connection.execute(
                """
                SELECT * FROM timer_spans
                ORDER BY started_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ) as cursor:
                rows = await cursor.fetchall()
            return [d for r in rows if (d := _row_to_dict(r)) is not None]

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    async def create_recommendation(
        self,
        policy: str,
        message: str,
        suggested_action: str | None = None,
        action_payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        created_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        created_iso = _to_iso(created_at)
        payload_json = _dumps_optional_json(action_payload)

        async with self._lock:
            conn = self.connection
            cursor = await conn.execute(
                """
                INSERT INTO recommendations (
                  policy, message, suggested_action, action_payload_json,
                  dedupe_key, status, created_at, resolved_at, resolution_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, 'pending', ?, NULL, NULL)
                """,
                (policy, message, suggested_action, payload_json, dedupe_key, created_iso),
            )
            rec_id = cursor.lastrowid
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM recommendations WHERE id = ?",
                (rec_id,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def get_recommendation(self, rec_id: int) -> dict[str, Any] | None:
        async with self._lock:
            async with self.connection.execute(
                "SELECT * FROM recommendations WHERE id = ?",
                (int(rec_id),),
            ) as cursor:
                row = await cursor.fetchone()
            return _row_to_dict(row)

    async def get_pending_recommendations(self) -> list[dict[str, Any]]:
        async with self._lock:
            async with self.connection.execute(
                """
                SELECT * FROM recommendations
                WHERE status = 'pending'
                ORDER BY created_at DESC, id DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
            return [d for r in rows if (d := _row_to_dict(r)) is not None]

    async def list_recommendations(
        self,
        limit: int = 50,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            if status is not None:
                async with self.connection.execute(
                    """
                    SELECT * FROM recommendations
                    WHERE status = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (status, int(limit)),
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with self.connection.execute(
                    """
                    SELECT * FROM recommendations
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ) as cursor:
                    rows = await cursor.fetchall()
            return [d for r in rows if (d := _row_to_dict(r)) is not None]

    async def find_recent_recommendation(
        self,
        dedupe_key: str,
        since: datetime | str | None = None,
    ) -> dict[str, Any] | None:
        async with self._lock:
            conn = self.connection
            if since is None:
                async with conn.execute(
                    """
                    SELECT * FROM recommendations
                    WHERE dedupe_key = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (dedupe_key,),
                ) as cursor:
                    row = await cursor.fetchone()
                return _row_to_dict(row)

            since_dt = _to_dt(since)
            async with conn.execute(
                """
                SELECT * FROM recommendations
                WHERE dedupe_key = ?
                ORDER BY created_at DESC, id DESC
                """,
                (dedupe_key,),
            ) as cursor:
                rows = await cursor.fetchall()

            for row in rows:
                if row["status"] == "pending":
                    return _row_to_dict(row)
                if _to_dt(row["created_at"]) >= since_dt:
                    return _row_to_dict(row)
                if (
                    row["resolved_at"] is not None
                    and _to_dt(row["resolved_at"]) >= since_dt
                ):
                    return _row_to_dict(row)
            return None

    async def resolve_recommendation(
        self,
        rec_id: int,
        status: str,
        metadata: dict[str, Any] | None = None,
        resolved_at: datetime | str | None = None,
    ) -> dict[str, Any] | None:
        if status not in VALID_RECOMMENDATION_RESOLUTIONS:
            raise ValueError(
                f"Invalid recommendation resolution status {status!r}; "
                f"expected one of {sorted(VALID_RECOMMENDATION_RESOLUTIONS)}"
            )

        resolved_iso = _to_iso(resolved_at)
        metadata_json = _dumps_optional_json(metadata)

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                "SELECT id FROM recommendations WHERE id = ?",
                (int(rec_id),),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is None:
                return None

            await conn.execute(
                """
                UPDATE recommendations
                SET status = ?, resolved_at = ?, resolution_metadata_json = ?
                WHERE id = ?
                """,
                (status, resolved_iso, metadata_json, int(rec_id)),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM recommendations WHERE id = ?",
                (int(rec_id),),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            return _row_to_dict(row)

    # ------------------------------------------------------------------
    # Window spans
    # ------------------------------------------------------------------

    async def get_current_window_span(self) -> dict[str, Any] | None:
        async with self._lock:
            async with self.connection.execute(
                """
                SELECT * FROM window_spans
                WHERE ended_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ) as cursor:
                row = await cursor.fetchone()
            return _row_to_dict(row)

    async def open_window_span(
        self,
        app_class: str,
        window_title: str,
        category: str = "unknown",
        classifier: str = "rule",
        calendar_event_id: str | None = None,
        started_at: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_iso = _to_iso(started_at)
        metadata_json = _dumps_optional_json(metadata)

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                "SELECT id, started_at FROM window_spans WHERE ended_at IS NULL"
            ) as cur:
                open_win_rows = await cur.fetchall()

            boot_dt = get_system_boot_time()
            for wrow in open_win_rows:
                wend_dt = _to_dt(started_iso)
                wstart_dt = _to_dt(wrow["started_at"])
                if boot_dt and wstart_dt < boot_dt < wend_dt:
                    wend_dt = boot_dt
                await conn.execute(
                    "UPDATE window_spans SET ended_at = ? WHERE id = ?",
                    (wend_dt.isoformat(), wrow["id"]),
                )
            cursor = await conn.execute(
                """
                INSERT INTO window_spans (
                  app_class, window_title, category, classifier,
                  calendar_event_id, started_at, ended_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    app_class,
                    window_title,
                    category,
                    classifier,
                    calendar_event_id,
                    started_iso,
                    metadata_json,
                ),
            )
            span_id = cursor.lastrowid
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM window_spans WHERE id = ?",
                (span_id,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def update_window_span_classification(
        self,
        span_id: int,
        *,
        category: str,
        classifier: str,
        confidence: float = 1.0,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Update the category, classifier, and metadata of an existing window span."""
        async with self._lock:
            conn = self.connection
            async with conn.execute(
                "SELECT * FROM window_spans WHERE id = ?",
                (span_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None

            row_dict = _row_to_dict(row) or {}
            existing_meta: dict[str, Any] = {}
            raw_meta = row_dict.get("metadata_json")
            if raw_meta:
                try:
                    import json
                    existing_meta = json.loads(raw_meta)
                except Exception:
                    pass

            if metadata_patch:
                existing_meta.update(metadata_patch)
            existing_meta["category"] = category
            existing_meta["classifier"] = classifier
            existing_meta["confidence"] = confidence

            import json
            meta_str = json.dumps(existing_meta)

            await conn.execute(
                """
                UPDATE window_spans
                SET category = ?, classifier = ?, metadata_json = ?
                WHERE id = ?
                """,
                (category, classifier, meta_str, span_id),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM window_spans WHERE id = ?",
                (span_id,),
            ) as fetch_cursor:
                updated_row = await fetch_cursor.fetchone()
            return _row_to_dict(updated_row)

    async def close_current_window_span(
        self,
        ended_at: datetime | str | None = None,
    ) -> dict[str, Any] | None:
        ended_iso = _to_iso(ended_at)

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                """
                SELECT id FROM window_spans
                WHERE ended_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None

            span_id = int(row["id"])
            await conn.execute(
                "UPDATE window_spans SET ended_at = ? WHERE ended_at IS NULL",
                (ended_iso,),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM window_spans WHERE id = ?",
                (span_id,),
            ) as fetch_cursor:
                closed_row = await fetch_cursor.fetchone()
            return _row_to_dict(closed_row)

    async def list_window_spans(
        self,
        limit: int = 100,
        since: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            conn = self.connection
            if since is None:
                async with conn.execute(
                    """
                    SELECT * FROM window_spans
                    ORDER BY started_at DESC, id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ) as cursor:
                    rows = await cursor.fetchall()
                return [d for r in rows if (d := _row_to_dict(r)) is not None]

            since_dt = _to_dt(since)
            async with conn.execute(
                "SELECT * FROM window_spans ORDER BY started_at DESC, id DESC"
            ) as cursor:
                rows = await cursor.fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                started_dt = _to_dt(row["started_at"])
                ended_raw = row["ended_at"]
                if (
                    started_dt >= since_dt
                    or ended_raw is None
                    or _to_dt(ended_raw) > since_dt
                ):
                    converted = _row_to_dict(row)
                    if converted is not None:
                        results.append(converted)
                        if len(results) >= int(limit):
                            break
            return results

    # ------------------------------------------------------------------
    # Calendar cache
    # ------------------------------------------------------------------

    async def upsert_calendar_event(
        self,
        event_id: str,
        title: str,
        start_time: datetime | str,
        end_time: datetime | str,
        calendar_id: str = "primary",
        description: str | None = None,
        color_id: str | None = None,
        task_type: str = "unknown",
        suitable_for_walking: bool | None = None,
        recommended_pomodoro: str | None = None,
        raw_json: dict[str, Any] | str | None = None,
        updated_at: datetime | str | None = None,
        *,
        raw: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_iso = _to_iso(start_time)
        end_iso = _to_iso(end_time)
        updated_iso = _to_iso(updated_at)

        walking_int: int | None
        if suitable_for_walking is None:
            walking_int = None
        else:
            walking_int = 1 if suitable_for_walking else 0

        serialized_raw: str | None
        if raw_json is not None:
            serialized_raw = (
                raw_json if isinstance(raw_json, str) else json.dumps(raw_json)
            )
        elif raw is not None:
            serialized_raw = json.dumps(raw)
        else:
            serialized_raw = None

        async with self._lock:
            conn = self.connection
            await conn.execute(
                """
                INSERT INTO calendar_cache (
                  event_id, calendar_id, title, description, color_id,
                  task_type, suitable_for_walking, recommended_pomodoro,
                  start_time, end_time, raw_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                  calendar_id = excluded.calendar_id,
                  title = excluded.title,
                  description = excluded.description,
                  color_id = excluded.color_id,
                  task_type = excluded.task_type,
                  suitable_for_walking = excluded.suitable_for_walking,
                  recommended_pomodoro = excluded.recommended_pomodoro,
                  start_time = excluded.start_time,
                  end_time = excluded.end_time,
                  raw_json = excluded.raw_json,
                  updated_at = excluded.updated_at
                """,
                (
                    event_id,
                    calendar_id,
                    title,
                    description,
                    color_id,
                    task_type,
                    walking_int,
                    recommended_pomodoro,
                    start_iso,
                    end_iso,
                    serialized_raw,
                    updated_iso,
                ),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM calendar_cache WHERE event_id = ?",
                (event_id,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def get_current_and_next_events(
        self,
        now: datetime | str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        now_dt = _to_dt(now)

        async with self._lock:
            async with self.connection.execute(
                "SELECT * FROM calendar_cache ORDER BY start_time ASC"
            ) as cursor:
                rows = await cursor.fetchall()

        active_candidates: list[tuple[float, float, dict[str, Any]]] = []
        upcoming_candidates: list[tuple[float, float, dict[str, Any]]] = []

        for row in rows:
            start_dt = _to_dt(row["start_time"])
            end_dt = _to_dt(row["end_time"])
            converted = _row_to_dict(row)
            if converted is None:
                continue

            if start_dt <= now_dt < end_dt:
                active_candidates.append(
                    (-start_dt.timestamp(), end_dt.timestamp(), converted)
                )
            elif start_dt > now_dt:
                upcoming_candidates.append(
                    (start_dt.timestamp(), end_dt.timestamp(), converted)
                )

        active_candidates.sort(key=lambda item: (item[0], item[1]))
        upcoming_candidates.sort(key=lambda item: (item[0], item[1]))

        current_event = active_candidates[0][2] if active_candidates else None
        next_event = upcoming_candidates[0][2] if upcoming_candidates else None
        return current_event, next_event

    async def delete_calendar_event(self, event_id: str) -> bool:
        """Delete a single cached calendar event by event_id."""
        async with self._lock:
            conn = self.connection
            cursor = await conn.execute(
                "DELETE FROM calendar_cache WHERE event_id = ?",
                (event_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def list_calendar_events(
        self,
        time_min: datetime | str | None = None,
        time_max: datetime | str | None = None,
        calendar_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List cached calendar events, optionally filtered by overlap window and calendar_id."""
        query = "SELECT * FROM calendar_cache WHERE 1=1"
        params: list[Any] = []
        if calendar_id is not None:
            query += " AND calendar_id = ?"
            params.append(calendar_id)
        if time_min is not None:
            query += " AND end_time > ?"
            params.append(_to_iso(time_min))
        if time_max is not None:
            query += " AND start_time < ?"
            params.append(_to_iso(time_max))
        query += " ORDER BY start_time ASC"

        async with self._lock:
            async with self.connection.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
            return [d for r in rows if (d := _row_to_dict(r)) is not None]

    async def prune_stale_calendar_events(
        self,
        time_min: datetime | str,
        time_max: datetime | str,
        keep_event_ids: Sequence[str] | set[str],
        calendar_id: str = "primary",
    ) -> list[str]:
        """Prune cached events in [time_min, time_max] not in keep_event_ids.

        Returns list of deleted event_ids.
        """
        time_min_iso = _to_iso(time_min)
        time_max_iso = _to_iso(time_max)
        keep_set = set(keep_event_ids)

        async with self._lock:
            conn = self.connection
            async with conn.execute(
                """
                SELECT event_id FROM calendar_cache
                WHERE calendar_id = ?
                  AND end_time > ?
                  AND start_time < ?
                """,
                (calendar_id, time_min_iso, time_max_iso),
            ) as cursor:
                rows = await cursor.fetchall()

            stale_ids = [
                str(row["event_id"])
                for row in rows
                if row["event_id"] not in keep_set
            ]

            if stale_ids:
                placeholders = ",".join("?" for _ in stale_ids)
                await conn.execute(
                    f"DELETE FROM calendar_cache WHERE event_id IN ({placeholders})",
                    tuple(stale_ids),
                )
                await conn.commit()

            return stale_ids

    # ------------------------------------------------------------------
    # Module health & settings
    # ------------------------------------------------------------------

    async def upsert_module_health(
        self,
        module: str,
        status: str,
        last_event_at: datetime | str | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        updated_iso = _to_iso()
        last_event_iso = _to_iso(last_event_at) if last_event_at is not None else None

        async with self._lock:
            conn = self.connection
            await conn.execute(
                """
                INSERT INTO module_health (
                  module, status, last_event_at, last_error, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(module) DO UPDATE SET
                  status = excluded.status,
                  last_event_at = COALESCE(excluded.last_event_at, module_health.last_event_at),
                  last_error = excluded.last_error,
                  updated_at = excluded.updated_at
                """,
                (module, status, last_event_iso, last_error, updated_iso),
            )
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM module_health WHERE module = ?",
                (module,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def get_module_health(self, module: str) -> dict[str, Any] | None:
        async with self._lock:
            async with self.connection.execute(
                "SELECT * FROM module_health WHERE module = ?",
                (module,),
            ) as cursor:
                row = await cursor.fetchone()
            return _row_to_dict(row)

    async def list_module_health(self) -> list[dict[str, Any]]:
        async with self._lock:
            async with self.connection.execute(
                "SELECT * FROM module_health ORDER BY module ASC"
            ) as cursor:
                rows = await cursor.fetchall()
            return [d for r in rows if (d := _row_to_dict(r)) is not None]

    async def get_setting(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            async with self.connection.execute(
                "SELECT value_json FROM settings WHERE key = ?",
                (key,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return default
            return json.loads(row["value_json"])

    async def set_setting(self, key: str, value: Any) -> dict[str, Any]:
        updated_iso = _to_iso()
        value_json = json.dumps(value)

        async with self._lock:
            conn = self.connection
            await conn.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value_json = excluded.value_json,
                  updated_at = excluded.updated_at
                """,
                (key, value_json, updated_iso),
            )
            await conn.commit()
        return {
            "key": key,
            "value": value,
            "value_json": value_json,
            "updated_at": updated_iso,
        }

    # ------------------------------------------------------------------
    # LLM interactions
    # ------------------------------------------------------------------

    async def record_llm_interaction(
        self,
        source: str,
        question: str,
        answer: str,
        suggested_action: str | None = None,
        confidence: float = 1.0,
        model: str = "",
        latency_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Insert a completed LLM interaction into llm_interactions."""
        created_iso = _to_iso(created_at)
        metadata_json = _dumps_optional_json(metadata)

        async with self._lock:
            conn = self.connection
            cursor = await conn.execute(
                """
                INSERT INTO llm_interactions (
                  source, question, answer, suggested_action, confidence,
                  model, latency_ms, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    question,
                    answer,
                    suggested_action,
                    float(confidence),
                    model,
                    latency_ms,
                    metadata_json,
                    created_iso,
                ),
            )
            interaction_id = cursor.lastrowid
            await conn.commit()

            async with conn.execute(
                "SELECT * FROM llm_interactions WHERE id = ?",
                (interaction_id,),
            ) as fetch_cursor:
                row = await fetch_cursor.fetchone()
            result = _row_to_dict(row)
            assert result is not None
            return result

    async def get_llm_interaction(self, interaction_id: int) -> dict[str, Any] | None:
        """Fetch an interaction by primary key."""
        async with self._lock:
            async with self.connection.execute(
                "SELECT * FROM llm_interactions WHERE id = ?",
                (int(interaction_id),),
            ) as cursor:
                row = await cursor.fetchone()
            return _row_to_dict(row)

    async def list_llm_interactions(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List recent LLM interactions ordered by creation time descending."""
        async with self._lock:
            async with self.connection.execute(
                """
                SELECT * FROM llm_interactions
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ) as cursor:
                rows = await cursor.fetchall()
            return [d for r in rows if (d := _row_to_dict(r)) is not None]

    # ------------------------------------------------------------------
    # Data retention cleanup
    # ------------------------------------------------------------------

    async def prune_retention(
        self,
        retention_days: int,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Prune historical spans, recommendations, and interactions older than retention_days.

        Returns a dictionary mapping table name to the number of deleted rows.
        """
        if retention_days <= 0:
            return {}

        ref_now = _to_dt(now)
        cutoff = ref_now - timedelta(days=int(retention_days))
        cutoff_iso = cutoff.isoformat()

        counts: dict[str, int] = {}
        async with self._lock:
            conn = self.connection

            # Prune completed activity spans older than cutoff
            cur = await conn.execute(
                "DELETE FROM activity_spans WHERE started_at < ? AND ended_at IS NOT NULL",
                (cutoff_iso,),
            )
            counts["activity_spans"] = cur.rowcount

            # Prune completed window spans older than cutoff
            cur = await conn.execute(
                "DELETE FROM window_spans WHERE started_at < ? AND ended_at IS NOT NULL",
                (cutoff_iso,),
            )
            counts["window_spans"] = cur.rowcount

            # Prune completed timer spans older than cutoff
            cur = await conn.execute(
                "DELETE FROM timer_spans WHERE started_at < ? AND ended_at IS NOT NULL",
                (cutoff_iso,),
            )
            counts["timer_spans"] = cur.rowcount

            # Prune resolved recommendations older than cutoff
            cur = await conn.execute(
                "DELETE FROM recommendations WHERE created_at < ? AND status != 'pending'",
                (cutoff_iso,),
            )
            counts["recommendations"] = cur.rowcount

            # Prune LLM interactions older than cutoff (if table exists)
            try:
                cur = await conn.execute(
                    "DELETE FROM llm_interactions WHERE created_at < ?",
                    (cutoff_iso,),
                )
                counts["llm_interactions"] = cur.rowcount
            except Exception:
                pass

            await conn.commit()

        return counts


