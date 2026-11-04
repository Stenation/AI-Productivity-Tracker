"""Shared Jinja2 template renderer and view-context builders for HTMX pages and partials."""

from __future__ import annotations

import math
import re
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from jarvis import __version__
from jarvis.core.database import Database
from jarvis.core.registry import MODULE_CATEGORIES, ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import Recommendation, ensure_aware, now_local
from jarvis.core.settings import Settings
from jarvis.core.timer import TimerManager
from jarvis.modules.policies.pomodoro_policy import is_walking_suitable, select_pomodoro

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
MODULES_ROOT_DIR = WEB_DIR.parent / "modules"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_PRESET_RE = re.compile(r"^(\d+)_(\d+)$")
_CATEGORY_ORDER: dict[str, int] = {"inputs": 0, "policies": 1, "outputs": 2}


def is_htmx_request(request: Request) -> bool:
    """Return True when the incoming HTTP request carries `HX-Request: true`."""
    return request.headers.get("HX-Request", "").strip().lower() == "true"


def parse_dt(value: datetime | str | None) -> datetime | None:
    """Parse an ISO timestamp string or datetime into a timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value.strip():
        with suppress(ValueError):
            return ensure_aware(datetime.fromisoformat(value))
    return None


def format_duration(seconds: int | float | None) -> str:
    """Format elapsed seconds as compact human duration (e.g., `42s`, `12m 05s`, `1h 15m`)."""
    if seconds is None:
        return "—"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_countdown(seconds: int | float | None) -> str:
    """Format countdown seconds as `MM:SS` or `HH:MM:SS`."""
    if seconds is None:
        return "00:00"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp(value: datetime | str | None) -> str:
    """Format a timestamp in local `YYYY-MM-DD HH:MM:SS` form."""
    dt = parse_dt(value)
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def format_hm(value: datetime | str | None) -> str:
    """Format a timestamp in local `HH:MM` form."""
    dt = parse_dt(value)
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%H:%M")


def format_timer_kind(kind: str | None) -> str:
    """Format timer preset strings (`25_5` -> `25/5`, `25_5_break` -> `25/5 Break`)."""
    if not kind or not str(kind).strip():
        return "—"
    cleaned = str(kind).strip()
    is_break = cleaned.endswith("_break")
    base = cleaned.removesuffix("_break") if is_break else cleaned
    match = _PRESET_RE.match(base)
    label = f"{match.group(1)}/{match.group(2)}" if match else base.replace("_", " ").title()
    return f"{label} Break" if is_break else label


def render_template(
    request: Request,
    template_name: str,
    context: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> HTMLResponse:
    """Render a Jinja2 template with common metadata and optional HTMX headers."""
    merged_context: dict[str, Any] = {
        "version": __version__,
        **context,
    }
    return templates.TemplateResponse(
        request,
        template_name,
        merged_context,
        headers=headers,
    )


def _get_timer_manager(request: Request) -> TimerManager | None:
    for attr in ("timer_manager", "timer", "timers"):
        candidate = getattr(request.app.state, attr, None)
        if isinstance(candidate, TimerManager):
            return candidate
    return None


async def build_posture_context(request: Request) -> dict[str, Any]:
    """Build template context for the live posture & manual-override card."""
    settings: Settings = request.app.state.settings
    db: Database = request.app.state.db
    resolver: StateResolver = request.app.state.resolver

    now = now_local()
    effective = resolver.get_effective_state(now=now)
    db_span = await db.get_current_activity_span()

    activity = effective.activity
    started_at = effective.started_at
    source = effective.source
    confidence = effective.confidence
    reason = effective.reason
    override_expires_at = effective.override_expires_at

    if db_span is not None:
        db_act = str(db_span.get("activity", "unknown"))
        db_start = parse_dt(db_span.get("started_at"))
        if activity == "unknown" and db_act != "unknown" and db_start is not None:
            activity = db_act  # type: ignore[assignment]
            started_at = db_start
            source = str(db_span.get("source", source))
            confidence = float(db_span.get("confidence", confidence))
            reason = str(db_span.get("reason", reason))
        elif db_act == activity and db_start is not None and db_start < started_at:
            started_at = db_start

    elapsed_seconds = max(0, int((now - started_at).total_seconds()))
    override_active = override_expires_at is not None and override_expires_at > now
    override_mins_left = (
        max(1, math.ceil((override_expires_at - now).total_seconds() / 60))
        if override_active and override_expires_at is not None
        else 0
    )

    privacy_paused = bool(settings.privacy.global_pause or effective.privacy_paused)
    dnd_mode = bool(settings.privacy.dnd_mode)

    return {
        "posture": {
            "activity": activity,
            "activity_label": activity.capitalize(),
            "started_at": started_at.isoformat(),
            "started_hm": format_hm(started_at),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_display": format_duration(elapsed_seconds),
            "source": source,
            "confidence": round(confidence, 2),
            "reason": reason,
            "override_active": override_active,
            "override_mins_left": override_mins_left,
            "override_until_hm": format_hm(override_expires_at) if override_active else None,
            "privacy_paused": privacy_paused,
            "dnd_mode": dnd_mode,
            "default_ttl_minutes": settings.posture.manual_override_ttl_minutes,
        }
    }


async def build_timer_context(request: Request) -> dict[str, Any]:
    """Build template context for the Pomodoro timer card."""
    db: Database = request.app.state.db
    timer_manager = _get_timer_manager(request)
    now = now_local()

    if timer_manager is not None:
        raw = timer_manager.get_status()
    else:
        active_row = await db.get_active_timer()
        if active_row is not None:
            started_dt = parse_dt(active_row.get("started_at")) or now
            planned = active_row.get("planned_duration_seconds")
            elapsed = max(0, int((now - started_dt).total_seconds()))
            remaining = max(0, int(planned) - elapsed) if planned is not None else 0
            raw_kind = str(active_row.get("kind") or "custom")
            is_break = raw_kind.endswith("_break") or raw_kind == "break"
            base_kind = raw_kind.removesuffix("_break") if is_break else raw_kind
            raw = {
                "active": True,
                "timer_id": active_row.get("id"),
                "kind": base_kind,
                "phase": "break" if is_break else "work",
                "remaining_seconds": remaining,
                "planned_duration_seconds": planned,
                "elapsed_seconds": elapsed,
            }
        else:
            raw = {
                "active": False,
                "timer_id": None,
                "kind": None,
                "phase": "idle",
                "remaining_seconds": 0,
                "planned_duration_seconds": None,
                "elapsed_seconds": 0,
            }

    active = bool(raw.get("active"))
    planned_sec = raw.get("planned_duration_seconds")
    remaining_sec = int(raw.get("remaining_seconds") or 0)
    elapsed_sec = int(raw.get("elapsed_seconds") or 0)

    progress_pct = 0
    if active and planned_sec and int(planned_sec) > 0:
        progress_pct = min(100, max(0, int(round((elapsed_sec / int(planned_sec)) * 100))))

    phase = str(raw.get("phase") or "idle")
    kind = raw.get("kind")

    return {
        "timer": {
            "active": active,
            "timer_id": raw.get("timer_id"),
            "kind": kind,
            "kind_display": format_timer_kind(kind) if active else "Idle",
            "phase": phase,
            "phase_label": phase.capitalize(),
            "remaining_seconds": remaining_sec,
            "remaining_display": format_countdown(remaining_sec),
            "planned_seconds": planned_sec,
            "planned_display": format_duration(planned_sec) if planned_sec else "—",
            "progress_percent": progress_pct,
        }
    }


def _enrich_calendar_event(
    row: dict[str, Any] | None,
    *,
    settings: Settings,
    now: datetime,
) -> dict[str, Any] | None:
    if row is None:
        return None
    start_dt = parse_dt(row.get("start_time"))
    end_dt = parse_dt(row.get("end_time"))
    task_type = str(row.get("task_type") or "unknown")

    explicit_walk = row.get("suitable_for_walking")
    walk_ok = (
        bool(explicit_walk)
        if explicit_walk is not None
        else is_walking_suitable(task_type)
    )
    pomo = row.get("recommended_pomodoro") or select_pomodoro(task_type, settings)

    starts_in_mins: int | None = None
    ends_in_mins: int | None = None
    if start_dt is not None and start_dt > now:
        starts_in_mins = max(1, math.ceil((start_dt - now).total_seconds() / 60))
    if end_dt is not None and end_dt > now:
        ends_in_mins = max(1, math.ceil((end_dt - now).total_seconds() / 60))

    return {
        "event_id": row.get("event_id"),
        "title": row.get("title") or "Untitled event",
        "task_type": task_type,
        "task_label": task_type.replace("_", " ").title(),
        "time_window": f"{format_hm(start_dt)} – {format_hm(end_dt)}",
        "starts_in_minutes": starts_in_mins,
        "ends_in_minutes": ends_in_mins,
        "walking_suitable": walk_ok,
        "recommended_pomodoro": format_timer_kind(pomo) if pomo else None,
    }


async def build_calendar_context(request: Request) -> dict[str, Any]:
    """Build template context for current and upcoming calendar blocks."""
    settings: Settings = request.app.state.settings
    db: Database = request.app.state.db
    now = now_local()

    cur_row, next_row = await db.get_current_and_next_events(now=now)
    return {
        "calendar": {
            "current_event": _enrich_calendar_event(cur_row, settings=settings, now=now),
            "next_event": _enrich_calendar_event(next_row, settings=settings, now=now),
        }
    }


async def build_recommendations_context(
    request: Request,
    *,
    flash_message: str | None = None,
) -> dict[str, Any]:
    """Build template context for pending non-destructive policy recommendations."""
    db: Database = request.app.state.db
    rows = await db.get_pending_recommendations()

    items: list[dict[str, Any]] = []
    for row in rows:
        rec = Recommendation.model_validate(row)
        items.append(
            {
                "id": rec.id,
                "policy": rec.policy,
                "policy_label": rec.policy.replace("_", " ").title(),
                "message": rec.message,
                "suggested_action": rec.suggested_action,
                "action_label": (
                    rec.suggested_action.replace("_", " ").title()
                    if rec.suggested_action
                    else None
                ),
                "dedupe_key": rec.dedupe_key,
                "status": rec.status,
                "created_display": format_timestamp(rec.created_at),
                "created_hm": format_hm(rec.created_at),
            }
        )

    return {
        "recommendations": items,
        "pending_count": len(items),
        "flash_message": flash_message,
    }


async def build_privacy_context(request: Request) -> dict[str, Any]:
    """Build template context for live Privacy Pause and DND Mode controls."""
    settings: Settings = request.app.state.settings
    resolver: StateResolver = request.app.state.resolver
    effective = resolver.get_effective_state()

    privacy_paused = bool(settings.privacy.global_pause or effective.privacy_paused)
    dnd_mode = bool(settings.privacy.dnd_mode)

    return {
        "privacy": {
            "global_pause": privacy_paused,
            "privacy_paused": privacy_paused,
            "dnd_mode": dnd_mode,
            "hyprland_ipc_mode": settings.privacy.hyprland_ipc_mode,
            "allow_webcam": settings.privacy.allow_webcam,
            "microphone_mode": settings.privacy.microphone_mode,
            "allow_screenshots": settings.privacy.allow_screenshots,
            "allow_vlm_screenshots": settings.privacy.allow_vlm_screenshots,
            "allow_roast_mode": settings.privacy.allow_roast_mode,
            "allow_tts": settings.privacy.allow_tts,
            "allow_calendar_writes": settings.privacy.allow_calendar_writes,
            "allow_phone_webhook": settings.privacy.allow_phone_webhook,
            "allow_external_notifications": settings.privacy.allow_external_notifications,
            "dashboard_bind_local_only": settings.privacy.dashboard_bind_local_only,
            "retention_days": settings.privacy.retention_days,
        }
    }


def _discover_available_modules() -> dict[str, str]:
    discovered: dict[str, str] = {}
    for category in MODULE_CATEGORIES:
        cat_dir = MODULES_ROOT_DIR / category
        if not cat_dir.is_dir():
            continue
        for py_file in sorted(cat_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            discovered[py_file.stem] = category
    return discovered


async def build_modules_context(request: Request) -> dict[str, Any]:
    """Build template context for live module toggles and `module_health` inspection."""
    registry: ModuleRegistry = request.app.state.registry
    db: Database = request.app.state.db

    db_health_by_name = {
        str(row["module"]): row for row in await db.list_module_health()
    }
    discovered_on_disk = _discover_available_modules()

    by_name: dict[str, dict[str, Any]] = {}

    for rec in registry.list_modules():
        name = str(rec["module"])
        db_row = db_health_by_name.get(name, {})
        category = rec.get("category") or discovered_on_disk.get(name) or "other"
        last_error = rec.get("last_error") or db_row.get("last_error")
        by_name[name] = {
            "module": name,
            "category": category,
            "enabled": bool(rec.get("enabled")),
            "status": str(rec.get("status") or db_row.get("status") or "disabled"),
            "last_event_at": db_row.get("last_event_at"),
            "last_event_display": format_timestamp(db_row.get("last_event_at")),
            "last_error": last_error,
            "updated_display": format_timestamp(db_row.get("updated_at")),
        }

    for name, category in discovered_on_disk.items():
        if name in by_name:
            continue
        db_row = db_health_by_name.get(name, {})
        by_name[name] = {
            "module": name,
            "category": category,
            "enabled": False,
            "status": str(db_row.get("status") or "disabled"),
            "last_event_at": db_row.get("last_event_at"),
            "last_event_display": format_timestamp(db_row.get("last_event_at")),
            "last_error": db_row.get("last_error"),
            "updated_display": format_timestamp(db_row.get("updated_at")),
        }

    for name, db_row in db_health_by_name.items():
        if name in by_name:
            continue
        by_name[name] = {
            "module": name,
            "category": discovered_on_disk.get(name) or "other",
            "enabled": False,
            "status": str(db_row.get("status") or "disabled"),
            "last_event_at": db_row.get("last_event_at"),
            "last_event_display": format_timestamp(db_row.get("last_event_at")),
            "last_error": db_row.get("last_error"),
            "updated_display": format_timestamp(db_row.get("updated_at")),
        }

    sorted_modules = sorted(
        by_name.values(),
        key=lambda m: (_CATEGORY_ORDER.get(str(m["category"]), 99), str(m["module"])),
    )

    return {
        "modules": sorted_modules,
        "enabled_count": sum(1 for m in sorted_modules if m["enabled"]),
        "degraded_count": sum(1 for m in sorted_modules if m["status"] == "degraded"),
    }


async def build_dashboard_context(request: Request) -> dict[str, Any]:
    """Assemble full `/` dashboard template context."""
    posture_ctx = await build_posture_context(request)
    timer_ctx = await build_timer_context(request)
    calendar_ctx = await build_calendar_context(request)
    recs_ctx = await build_recommendations_context(request)
    privacy_ctx = await build_privacy_context(request)

    return {
        "active_page": "dashboard",
        **posture_ctx,
        **timer_ctx,
        **calendar_ctx,
        **recs_ctx,
        **privacy_ctx,
    }


async def build_history_context(request: Request) -> dict[str, Any]:
    """Assemble `/history` context: daily totals, activity spans, timer spans, and recs."""
    db: Database = request.app.state.db
    now = now_local()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    today_spans = await db.list_activity_spans(since=today_start, limit=1000)
    daily_seconds: dict[str, int] = {
        "sitting": 0,
        "standing": 0,
        "walking": 0,
        "away": 0,
        "unknown": 0,
    }
    for span in today_spans:
        act = str(span.get("activity") or "unknown")
        s_dt = parse_dt(span.get("started_at"))
        if s_dt is None:
            continue
        e_dt = parse_dt(span.get("ended_at")) or now
        clipped_start = max(s_dt, today_start)
        clipped_end = min(e_dt, now)
        dur = max(0, int((clipped_end - clipped_start).total_seconds()))
        daily_seconds[act] = daily_seconds.get(act, 0) + dur

    daily_totals = [
        {
            "activity": key,
            "label": key.capitalize(),
            "seconds": daily_seconds.get(key, 0),
            "display": format_duration(daily_seconds.get(key, 0)),
        }
        for key in ("sitting", "standing", "walking", "away")
    ]

    raw_activity_spans = await db.list_activity_spans(limit=50)
    activity_spans: list[dict[str, Any]] = []
    for row in raw_activity_spans:
        s_dt = parse_dt(row.get("started_at"))
        e_dt = parse_dt(row.get("ended_at"))
        end_ref = e_dt or now
        dur_sec = max(0, int((end_ref - s_dt).total_seconds())) if s_dt else 0
        act = str(row.get("activity") or "unknown")
        activity_spans.append(
            {
                "id": row.get("id"),
                "activity": act,
                "activity_label": act.capitalize(),
                "source": row.get("source") or "system",
                "reason": row.get("reason") or "—",
                "confidence": round(float(row.get("confidence", 1.0)), 2),
                "started_display": format_timestamp(s_dt),
                "ended_display": format_timestamp(e_dt) if e_dt else "Active now",
                "is_open": e_dt is None,
                "duration_display": format_duration(dur_sec),
            }
        )

    raw_timer_spans = await db.list_timer_spans(limit=50)
    timer_spans: list[dict[str, Any]] = []
    for row in raw_timer_spans:
        s_dt = parse_dt(row.get("started_at"))
        e_dt = parse_dt(row.get("ended_at"))
        status = str(row.get("status") or "unknown")
        actual_sec = row.get("actual_duration_seconds")
        if actual_sec is None and s_dt is not None:
            actual_sec = max(0, int(((e_dt or now) - s_dt).total_seconds()))
        planned_sec = row.get("planned_duration_seconds")
        timer_spans.append(
            {
                "id": row.get("id"),
                "kind": row.get("kind"),
                "kind_display": format_timer_kind(row.get("kind")),
                "status": status,
                "status_label": status.capitalize(),
                "started_display": format_timestamp(s_dt),
                "ended_display": format_timestamp(e_dt) if e_dt else "Running",
                "planned_display": format_duration(planned_sec),
                "actual_display": format_duration(actual_sec),
                "calendar_event_id": row.get("calendar_event_id"),
            }
        )

    raw_recs = await db.list_recommendations(limit=50)
    recommendation_history: list[dict[str, Any]] = []
    for row in raw_recs:
        rec = Recommendation.model_validate(row)
        recommendation_history.append(
            {
                "id": rec.id,
                "policy": rec.policy,
                "policy_label": rec.policy.replace("_", " ").title(),
                "message": rec.message,
                "suggested_action": rec.suggested_action or "—",
                "dedupe_key": rec.dedupe_key or "—",
                "status": rec.status,
                "status_label": rec.status.capitalize(),
                "created_display": format_timestamp(rec.created_at),
                "resolved_display": (
                    format_timestamp(rec.resolved_at) if rec.resolved_at else "—"
                ),
            }
        )

    return {
        "active_page": "history",
        "today_date": now.strftime("%Y-%m-%d"),
        "daily_totals": daily_totals,
        "activity_spans": activity_spans,
        "timer_spans": timer_spans,
        "recommendation_history": recommendation_history,
    }


async def build_settings_context(request: Request) -> dict[str, Any]:
    """Assemble `/settings` context: module toggles, health, privacy, and policy thresholds."""
    settings: Settings = request.app.state.settings
    modules_ctx = await build_modules_context(request)
    privacy_ctx = await build_privacy_context(request)

    pomodoro_mappings = [
        ("deep_work", settings.pomodoro.deep_work),
        ("coding", settings.pomodoro.coding),
        ("reading", settings.pomodoro.reading),
        ("admin", settings.pomodoro.admin),
        ("email", settings.pomodoro.email),
        ("planning", settings.pomodoro.planning),
        ("meeting", settings.pomodoro.meeting),
        ("break", settings.pomodoro.break_),
        ("unknown", settings.pomodoro.unknown),
    ]

    return {
        "active_page": "settings",
        **modules_ctx,
        **privacy_ctx,
        "server": {
            "host": settings.server.host,
            "port": settings.server.port,
        },
        "posture_cfg": settings.posture.model_dump(),
        "pomodoro_cfg": {
            "auto_start": settings.pomodoro.auto_start,
            "recommendation_cooldown_minutes": (
                settings.pomodoro.recommendation_cooldown_minutes
            ),
            "mappings": [
                {
                    "task_type": task,
                    "preset": format_timer_kind(preset) if preset else "None",
                }
                for task, preset in pomodoro_mappings
            ],
        },
        "focus_cfg": settings.focus.model_dump(),
        "waybar_cfg": {
            "status_file": str(settings.waybar.status_file),
            "refresh_interval_seconds": settings.waybar.refresh_interval_seconds,
            "signal_waybar": settings.waybar.signal_waybar,
            "waybar_signal": settings.waybar.waybar_signal,
        },
    }

