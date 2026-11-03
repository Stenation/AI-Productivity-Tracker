"""FastAPI application factory and lifespan wiring for the Posture Platform."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

from jarvis import __version__
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.migrations import get_applied_migrations
from jarvis.core.registry import ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import now_local
from jarvis.core.settings import Settings, load_settings


def create_app(
    config_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI application with full core lifespan management."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = (
            settings if settings is not None else load_settings(config_path)
        )
        db = Database(
            resolved_settings.database.path,
            migrations_dir=resolved_settings.migrations_dir,
            wal_mode=resolved_settings.database.wal_mode,
            busy_timeout_ms=resolved_settings.database.busy_timeout_ms,
        )
        await db.connect(run_migrations=True)
        await db.mark_running_timers_interrupted()
        migrations = await get_applied_migrations(db.connection)

        bus = EventBus()
        resolver = StateResolver(bus=bus, db=db, settings=resolved_settings)
        registry = ModuleRegistry(bus=bus, db=db, settings=resolved_settings)

        app.state.settings = resolved_settings
        app.state.db = db
        app.state.migrations = migrations
        app.state.bus = bus
        app.state.resolver = resolver
        app.state.registry = registry

        try:
            await resolver.start()
            await registry.start_all()
            yield
        finally:
            await registry.stop_all()
            await resolver.stop()
            await db.close()

    app = FastAPI(
        title="Jarvis Posture Platform",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/api/health")
    async def get_health(request: Request) -> dict[str, Any]:
        resolver: StateResolver = request.app.state.resolver
        registry: ModuleRegistry = request.app.state.registry
        effective = resolver.get_effective_state()
        modules = registry.list_modules()
        return {
            "status": "ok",
            "version": __version__,
            "activity": effective.activity,
            "privacy_paused": effective.privacy_paused,
            "database": "connected",
            "migrations": list(request.app.state.migrations),
            "modules": modules,
            "degraded_modules": [
                m["module"] for m in modules if m["status"] == "degraded"
            ],
            "timestamp": now_local().isoformat(),
        }

    return app
