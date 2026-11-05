"""FastAPI application factory and lifespan wiring for the Posture Platform."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from jarvis import __version__
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.migrations import get_applied_migrations
from jarvis.core.registry import ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import ensure_aware, now_local
from jarvis.core.settings import Settings, load_settings
from jarvis.core.timer import TimerManager
from jarvis.web.api import router as api_router
from jarvis.web.router import router as web_router
from jarvis.web.views import STATIC_DIR


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
        migrations = await get_applied_migrations(db.connection)

        # Prune historical data past retention window
        if resolved_settings.privacy.retention_days > 0:
            try:
                await db.prune_retention(resolved_settings.privacy.retention_days)
            except Exception as exc:
                logging.getLogger(__name__).warning("Failed to prune retention on startup: %s", exc)

        # Restore persisted runtime privacy overrides
        persisted_pause = await db.get_setting("privacy.global_pause")
        if isinstance(persisted_pause, bool):
            resolved_settings.privacy.global_pause = persisted_pause

        persisted_dnd = await db.get_setting("privacy.dnd_mode")
        if isinstance(persisted_dnd, bool):
            resolved_settings.privacy.dnd_mode = persisted_dnd

        persisted_tts = await db.get_setting("privacy.allow_tts")
        if isinstance(persisted_tts, bool):
            resolved_settings.privacy.allow_tts = persisted_tts

        persisted_mic = await db.get_setting("privacy.microphone_mode")
        if isinstance(persisted_mic, str) and persisted_mic in ("push_to_talk", "disabled"):
            resolved_settings.privacy.microphone_mode = persisted_mic

        # Restore persisted runtime LLM overrides
        persisted_model = await db.get_setting("llm.text_model")
        if isinstance(persisted_model, str) and persisted_model.strip():
            resolved_settings.llm.text_model = persisted_model.strip()

        persisted_keep_alive = await db.get_setting("llm.keep_alive")
        if isinstance(persisted_keep_alive, str) and persisted_keep_alive.strip():
            resolved_settings.llm.keep_alive = persisted_keep_alive.strip()

        bus = EventBus()
        timer_manager = TimerManager(bus=bus, db=db, settings=resolved_settings)
        resolver = StateResolver(bus=bus, db=db, settings=resolved_settings)
        registry = ModuleRegistry(bus=bus, db=db, settings=resolved_settings)

        app.state.settings = resolved_settings
        app.state.db = db
        app.state.migrations = migrations
        app.state.bus = bus
        app.state.timer_manager = timer_manager
        app.state.resolver = resolver
        app.state.registry = registry

        try:
            await timer_manager.start()
            await resolver.start()
            await registry.start_all()
            yield
        finally:
            await registry.stop_all()
            active_tm = getattr(app.state, "timer_manager", None)
            if isinstance(active_tm, TimerManager) and active_tm is not timer_manager:
                await active_tm.stop()
            await timer_manager.stop()
            await resolver.stop()
            try:
                curr_span = await db.get_current_activity_span()
                if curr_span:
                    s_reason = curr_span.get("reason")
                    s_meta = curr_span.get("metadata") or {}
                    s_exp = s_meta.get("override_expires_at")
                    is_active_override = (
                        s_reason == "manual_override"
                        and s_exp
                        and ensure_aware(datetime.fromisoformat(s_exp)) > now_local()
                    )
                    if not is_active_override:
                        await db.close_current_activity_span(ended_at=now_local())
            except Exception as exc:
                logging.getLogger(__name__).warning("Error closing activity span on shutdown: %s", exc)
            await db.close()

    app = FastAPI(
        title="Jarvis Posture Platform",
        version=__version__,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(api_router)
    app.include_router(web_router)

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
