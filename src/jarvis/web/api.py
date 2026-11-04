"""Core REST API endpoints for manual control, timers, recommendations, history, and privacy."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.llm import (
    LlmConnectionError,
    LlmError,
    LlmTimeoutError,
    LlmValidationError,
)
from jarvis.core.registry import ModuleRegistry
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import (
    ActivitySpan,
    AskRequest,
    AskResponse,
    CalendarEvent,
    FocusState,
    ManualActivityState,
    PlatformEvent,
    Recommendation,
    TimerSpan,
    ensure_aware,
    now_local,
)
from jarvis.core.settings import Settings
from jarvis.core.timer import TimerManager
from jarvis.modules.inputs.manual_control import ManualControlModule
from jarvis.modules.outputs.llm_ollama import LlmOllamaModule
from jarvis.web.views import (
    build_modules_context,
    build_posture_context,
    build_privacy_context,
    build_recommendations_context,
    build_timer_context,
    is_htmx_request,
    render_template,
)

router = APIRouter(prefix="/api", tags=["api"])

_FOCUS_CATEGORY_MAP: dict[str, FocusState] = {
    "focused": "focused",
    "work": "focused",
    "productive": "focused",
    "neutral": "neutral",
    "distracted": "distracted",
    "distraction": "distracted",
    "unknown": "unknown",
}


class SetActivityRequest(BaseModel):
    """Payload for `POST /api/activity`."""

    model_config = ConfigDict(extra="ignore")

    activity: ManualActivityState
    ttl_minutes: int | float | None = Field(default=None, gt=0)


class CycleActivityRequest(BaseModel):
    """Optional payload for `POST /api/activity/cycle`."""

    model_config = ConfigDict(extra="ignore")

    ttl_minutes: int | float | None = Field(default=None, gt=0)


class RecommendationResolutionRequest(BaseModel):
    """Optional payload for recommendation accept/snooze/dismiss endpoints."""

    model_config = ConfigDict(extra="ignore")

    metadata: dict[str, Any] | None = None
    reason: str | None = None
    snooze_minutes: int | None = Field(default=None, ge=1)

    def to_metadata(self, *, include_snooze: bool = False) -> dict[str, Any] | None:
        merged: dict[str, Any] = dict(self.metadata) if self.metadata else {}
        if self.reason is not None:
            merged["reason"] = self.reason
        if include_snooze and self.snooze_minutes is not None:
            merged["snooze_minutes"] = self.snooze_minutes
        return merged or None


class TimerStartRequest(BaseModel):
    """Payload for `POST /api/timers/start`."""

    model_config = ConfigDict(extra="ignore")

    kind: str | None = None
    work_minutes: int | float | None = Field(default=None, gt=0)
    break_minutes: int | float | None = Field(default=None, ge=0)
    work_seconds: int | None = Field(default=None, gt=0)
    break_seconds: int | None = Field(default=None, ge=0)
    calendar_event_id: str | None = None


class TimerStopRequest(BaseModel):
    """Optional payload for `POST /api/timers/stop`."""

    model_config = ConfigDict(extra="ignore")

    reason: str = "stopped"


class PrivacyDndRequest(BaseModel):
    """Optional payload for `POST /api/privacy/dnd` (toggles DND when omitted)."""

    model_config = ConfigDict(extra="ignore")

    dnd_mode: bool | None = None
    enabled: bool | None = None


class PrivacyTtsRequest(BaseModel):
    """Optional payload for `POST /api/privacy/tts` (toggles TTS when omitted)."""

    model_config = ConfigDict(extra="ignore")

    allow_tts: bool | None = None
    enabled: bool | None = None


class PrivacyMicRequest(BaseModel):
    """Optional payload for `POST /api/privacy/microphone`."""

    model_config = ConfigDict(extra="ignore")

    microphone_mode: str | None = None
    enabled: bool | None = None


class SettingsLlmRequest(BaseModel):
    """Payload for updating LLM runtime settings."""

    model_config = ConfigDict(extra="ignore")

    text_model: str | None = None
    keep_alive: str | None = None


def _parse_json_or_form[ModelT: BaseModel](
    model_cls: type[ModelT],
    *,
    allow_empty: bool = False,
) -> Callable[[Request], Coroutine[Any, Any, ModelT]]:
    async def _dependency(request: Request) -> ModelT:
        content_type = request.headers.get("content-type", "").lower()
        if (
            "application/x-www-form-urlencoded" in content_type
            or "multipart/form-data" in content_type
        ):
            form = await request.form()
            raw_data: Any = {k: v for k, v in form.items() if v != ""}
        else:
            raw_bytes = await request.body()
            if not raw_bytes or not raw_bytes.strip():
                if allow_empty:
                    raw_data = {}
                else:
                    raw_data = {}
            else:
                try:
                    raw_data = json.loads(raw_bytes)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid JSON payload: {exc.msg}",
                    ) from exc

        try:
            return model_cls.model_validate(raw_data)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    return _dependency


_SetActivityDep = Annotated[
    SetActivityRequest,
    Depends(_parse_json_or_form(SetActivityRequest)),
]
_CycleActivityDep = Annotated[
    CycleActivityRequest,
    Depends(_parse_json_or_form(CycleActivityRequest, allow_empty=True)),
]
_EventBodyDep = Annotated[
    PlatformEvent,
    Depends(_parse_json_or_form(PlatformEvent)),
]
_RecResolutionDep = Annotated[
    RecommendationResolutionRequest,
    Depends(_parse_json_or_form(RecommendationResolutionRequest, allow_empty=True)),
]
_TimerStartDep = Annotated[
    TimerStartRequest,
    Depends(_parse_json_or_form(TimerStartRequest, allow_empty=True)),
]
_TimerStopDep = Annotated[
    TimerStopRequest,
    Depends(_parse_json_or_form(TimerStopRequest, allow_empty=True)),
]
_PrivacyDndDep = Annotated[
    PrivacyDndRequest,
    Depends(_parse_json_or_form(PrivacyDndRequest, allow_empty=True)),
]
_PrivacyTtsDep = Annotated[
    PrivacyTtsRequest,
    Depends(_parse_json_or_form(PrivacyTtsRequest, allow_empty=True)),
]
_PrivacyMicDep = Annotated[
    PrivacyMicRequest,
    Depends(_parse_json_or_form(PrivacyMicRequest, allow_empty=True)),
]
_SettingsLlmDep = Annotated[
    SettingsLlmRequest,
    Depends(_parse_json_or_form(SettingsLlmRequest, allow_empty=True)),
]
_AskDep = Annotated[
    AskRequest,
    Depends(_parse_json_or_form(AskRequest)),
]


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _get_db(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def _get_bus(request: Request) -> EventBus:
    return request.app.state.bus  # type: ignore[no-any-return]


def _get_resolver(request: Request) -> StateResolver:
    return request.app.state.resolver  # type: ignore[no-any-return]


def _get_registry(request: Request) -> ModuleRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


def _get_timer_manager(request: Request) -> TimerManager | None:
    for attr in ("timer_manager", "timer", "timers"):
        candidate = getattr(request.app.state, attr, None)
        if isinstance(candidate, TimerManager):
            return candidate
    return None


def _require_timer_manager(request: Request) -> TimerManager:
    manager = _get_timer_manager(request)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="TimerManager is not available on app.state",
        )
    return manager


def _get_manual_control(request: Request) -> ManualControlModule:
    registry = _get_registry(request)
    active_mod = registry.get_module("manual_control")
    if isinstance(active_mod, ManualControlModule):
        return active_mod
    return ManualControlModule(
        bus=_get_bus(request),
        db=_get_db(request),
        config=_get_settings(request).model_dump(),
    )


def _is_phone_webhook_event(event: PlatformEvent) -> bool:
    if event.type == "phone_state_changed":
        return True
    lowered_source = event.source.lower()
    return "phone" in lowered_source or "s24" in lowered_source


async def _resolve_and_emit_recommendation(
    *,
    db: Database,
    bus: EventBus,
    rec_id: int,
    status: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    now = now_local()
    resolved_row = await db.resolve_recommendation(
        rec_id=rec_id,
        status=status,
        metadata=metadata,
        resolved_at=now,
    )
    if resolved_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation {rec_id} not found",
        )

    serialized = Recommendation.model_validate(resolved_row).model_dump(mode="json")
    await bus.publish(
        PlatformEvent(
            type="recommendation_resolved",
            source="api",
            timestamp=now,
            payload={
                "recommendation_id": rec_id,
                "id": rec_id,
                "policy": serialized["policy"],
                "status": status,
                "suggested_action": serialized["suggested_action"],
                "resolved_at": serialized["resolved_at"],
                "resolution_metadata": serialized["resolution_metadata"],
            },
        )
    )
    return serialized


async def _persist_and_publish_privacy(
    *,
    settings: Settings,
    db: Database,
    bus: EventBus,
) -> dict[str, Any]:
    global_pause = bool(settings.privacy.global_pause)
    dnd_mode = bool(settings.privacy.dnd_mode)
    allow_tts = bool(settings.privacy.allow_tts)
    microphone_mode = settings.privacy.microphone_mode

    await db.set_setting("privacy.global_pause", global_pause)
    await db.set_setting("privacy.dnd_mode", dnd_mode)
    await db.set_setting("privacy.allow_tts", allow_tts)
    await db.set_setting("privacy.microphone_mode", microphone_mode)
    await db.set_setting(
        "privacy",
        {
            "global_pause": global_pause,
            "dnd_mode": dnd_mode,
            "allow_tts": allow_tts,
            "microphone_mode": microphone_mode,
        },
    )

    await bus.publish(
        PlatformEvent(
            type="privacy_mode_changed",
            source="api",
            timestamp=now_local(),
            payload={
                "global_pause": global_pause,
                "privacy_paused": global_pause,
                "dnd_mode": dnd_mode,
                "allow_tts": allow_tts,
                "microphone_mode": microphone_mode,
            },
        )
    )
    return {
        "status": "ok",
        "global_pause": global_pause,
        "privacy_paused": global_pause,
        "dnd_mode": dnd_mode,
        "allow_tts": allow_tts,
        "microphone_mode": microphone_mode,
    }


@router.get("/state")
async def get_state(request: Request) -> dict[str, Any]:
    """Return composite snapshot of effective activity, timer, calendar, privacy, and recs."""
    settings = _get_settings(request)
    db = _get_db(request)
    resolver = _get_resolver(request)
    timer_manager = _get_timer_manager(request)

    now = now_local()
    effective = resolver.get_effective_state()

    active_timer_row = await db.get_active_timer()
    active_timer_model = (
        TimerSpan.model_validate(active_timer_row).model_dump(mode="json")
        if active_timer_row is not None
        else None
    )

    if timer_manager is not None:
        timer_status = timer_manager.get_status()
    elif active_timer_row is not None:
        started_dt = ensure_aware(datetime.fromisoformat(str(active_timer_row["started_at"])))
        planned = active_timer_row.get("planned_duration_seconds")
        elapsed = max(0, int((now - started_dt).total_seconds()))
        remaining = max(0, int(planned) - elapsed) if planned is not None else 0
        raw_kind = str(active_timer_row["kind"])
        is_break = raw_kind.endswith("_break") or raw_kind == "break"
        base_kind = raw_kind.removesuffix("_break") if is_break else raw_kind
        timer_status = {
            "active": True,
            "timer_id": active_timer_row["id"],
            "kind": base_kind,
            "phase": "break" if is_break else "work",
            "remaining_seconds": remaining,
            "planned_duration_seconds": planned,
            "elapsed_seconds": elapsed,
            "calendar_event_id": active_timer_row.get("calendar_event_id"),
            "started_at": started_dt.isoformat(),
        }
    else:
        timer_status = {
            "active": False,
            "timer_id": None,
            "kind": None,
            "phase": "idle",
            "remaining_seconds": 0,
            "planned_duration_seconds": None,
            "elapsed_seconds": 0,
            "calendar_event_id": None,
            "started_at": None,
        }

    cur_evt_row, next_evt_row = await db.get_current_and_next_events(now=now)
    current_event = (
        CalendarEvent.model_validate(cur_evt_row).model_dump(mode="json")
        if cur_evt_row is not None
        else None
    )
    next_event = (
        CalendarEvent.model_validate(next_evt_row).model_dump(mode="json")
        if next_evt_row is not None
        else None
    )

    current_window = await db.get_current_window_span()
    focus_state: FocusState = "unknown"
    window_category: str = "unknown"
    app_class: str | None = None
    window_title: str | None = None
    is_call: bool = False

    registry = _get_registry(request)
    llm_enabled = registry.is_enabled("llm_ollama")

    privacy_paused = bool(settings.privacy.global_pause or effective.privacy_paused)
    dnd_mode = bool(settings.privacy.dnd_mode)

    focus_mod = registry.get_module("focus_policy")
    if (
        focus_mod is not None
        and hasattr(focus_mod, "get_focus_summary")
        and not privacy_paused
    ):
        focus_summary = await focus_mod.get_focus_summary(now=now)
        focus_state = focus_summary.get("focus_state", "unknown")
        app_class = focus_summary.get("app_class")
        window_title = focus_summary.get("window_title")
        is_call = bool(focus_summary.get("is_call_window", False))
        window_category = focus_state
    elif current_window is not None and not privacy_paused:
        raw_cat = str(current_window.get("category", "unknown")).lower()
        focus_state = _FOCUS_CATEGORY_MAP.get(raw_cat, "unknown")
        window_category = raw_cat
        app_class = current_window.get("app_class")
        window_title = current_window.get("window_title")
        meta = current_window.get("metadata")
        if isinstance(meta, dict) and meta.get("is_call_window") is not None:
            is_call = bool(meta["is_call_window"])
        else:
            from jarvis.core.classification import is_call_window
            is_call = is_call_window(app_class or "", window_title or "")

    current_window_payload = (
        {
            "id": current_window.get("id") if current_window else None,
            "app_class": app_class,
            "window_title": window_title,
            "title": window_title,
            "category": window_category,
            "focus": focus_state,
            "is_call_window": is_call,
        }
        if (app_class or window_title or current_window) and not privacy_paused
        else None
    )

    focus_context_payload = {
        "focus_state": focus_state,
        "focus": focus_state,
        "app_class": app_class if not privacy_paused else None,
        "window_title": window_title if not privacy_paused else None,
        "category": window_category if not privacy_paused else "unknown",
        "current_window_category": window_category if not privacy_paused else "unknown",
        "is_call_window": is_call if not privacy_paused else False,
        "call_window": is_call if not privacy_paused else False,
        "mode": settings.privacy.hyprland_ipc_mode,
    }

    pending_rows = await db.get_pending_recommendations()
    pending_recommendations = [
        Recommendation.model_validate(row).model_dump(mode="json")
        for row in pending_rows
    ]

    return {
        "activity": effective.activity,
        "source": effective.source,
        "confidence": effective.confidence,
        "reason": effective.reason,
        "started_at": effective.started_at.isoformat(),
        "elapsed_seconds": effective.elapsed_seconds,
        "override_expires_at": (
            effective.override_expires_at.isoformat()
            if effective.override_expires_at is not None
            else None
        ),
        "focus": focus_state,
        "focus_context": focus_context_payload,
        "current_window": current_window_payload,
        "window_context": current_window_payload,
        "current_window_category": window_category if not privacy_paused else "unknown",
        "window_category": window_category if not privacy_paused else "unknown",
        "call_window": is_call if not privacy_paused else False,
        "is_call_window": is_call if not privacy_paused else False,
        "llm_enabled": llm_enabled,
        "llm": {
            "enabled": llm_enabled,
            "provider": settings.llm.provider,
            "text_model": settings.llm.text_model,
            "keep_alive": settings.llm.keep_alive,
        },
        "voice_status": settings.privacy.microphone_mode,
        "voice": {
            "status": settings.privacy.microphone_mode,
            "microphone_mode": settings.privacy.microphone_mode,
            "allow_tts": bool(settings.privacy.allow_tts),
            "tts_enabled": registry.is_enabled("tts_piper"),
        },
        "timer": timer_status,
        "active_timer": active_timer_model,
        "current_event": current_event,
        "next_event": next_event,
        "calendar": {
            "current_event": current_event,
            "next_event": next_event,
        },
        "privacy_paused": privacy_paused,
        "global_pause": privacy_paused,
        "dnd_mode": dnd_mode,
        "privacy": {
            "global_pause": privacy_paused,
            "dnd_mode": dnd_mode,
            "allow_phone_webhook": bool(settings.privacy.allow_phone_webhook),
            "hyprland_ipc_mode": settings.privacy.hyprland_ipc_mode,
            "allow_tts": bool(settings.privacy.allow_tts),
            "microphone_mode": settings.privacy.microphone_mode,
        },
        "pending_recommendation_count": len(pending_recommendations),
        "pending_recommendations": pending_recommendations,
        "timestamp": now.isoformat(),
    }


@router.post("/activity", response_model=None)
async def set_activity(
    request: Request,
    payload: _SetActivityDep,
) -> HTMLResponse | dict[str, Any]:
    """Set manual activity state with optional TTL override."""
    manual_control = _get_manual_control(request)
    resolver = _get_resolver(request)

    await manual_control.set_activity(
        activity=payload.activity,
        ttl_minutes=payload.ttl_minutes,
    )
    if is_htmx_request(request):
        context = await build_posture_context(request)
        return render_template(
            request,
            "_partials/posture_card.html",
            context,
            headers={"HX-Trigger": "activityChanged"},
        )

    effective = resolver.get_effective_state()
    return {
        "status": "ok",
        "activity": effective.activity,
        "source": effective.source,
        "confidence": effective.confidence,
        "reason": effective.reason,
        "started_at": effective.started_at.isoformat(),
        "elapsed_seconds": effective.elapsed_seconds,
        "override_expires_at": (
            effective.override_expires_at.isoformat()
            if effective.override_expires_at is not None
            else None
        ),
        "privacy_paused": effective.privacy_paused,
    }


@router.post("/activity/cycle", response_model=None)
async def cycle_activity(
    request: Request,
    payload: _CycleActivityDep,
) -> HTMLResponse | dict[str, Any]:
    """Cycle manual activity state (`sitting -> standing -> walking -> away`)."""
    manual_control = _get_manual_control(request)
    resolver = _get_resolver(request)

    previous_activity = resolver.get_effective_state().activity
    await manual_control.cycle_activity(
        current_activity=previous_activity,
        ttl_minutes=payload.ttl_minutes,
    )
    if is_htmx_request(request):
        context = await build_posture_context(request)
        return render_template(
            request,
            "_partials/posture_card.html",
            context,
            headers={"HX-Trigger": "activityChanged"},
        )

    effective = resolver.get_effective_state()
    return {
        "status": "ok",
        "previous_activity": previous_activity,
        "activity": effective.activity,
        "source": effective.source,
        "confidence": effective.confidence,
        "reason": effective.reason,
        "started_at": effective.started_at.isoformat(),
        "elapsed_seconds": effective.elapsed_seconds,
        "override_expires_at": (
            effective.override_expires_at.isoformat()
            if effective.override_expires_at is not None
            else None
        ),
        "privacy_paused": effective.privacy_paused,
    }


@router.post("/events")
async def ingest_event(
    request: Request,
    event: _EventBodyDep,
) -> dict[str, Any]:
    """Ingest an external PlatformEvent from a local worker or webhook."""
    settings = _get_settings(request)
    resolver = _get_resolver(request)
    bus = _get_bus(request)

    if _is_phone_webhook_event(event) and not settings.privacy.allow_phone_webhook:
        raise HTTPException(
            status_code=403,
            detail="Phone webhook events are disabled by privacy settings",
        )

    if settings.privacy.global_pause or resolver.get_effective_state().privacy_paused:
        return {
            "status": "ignored",
            "dispatched": False,
            "reason": "global_pause",
            "event_type": event.type,
            "source": event.source,
        }

    dispatched = await bus.publish(event)
    return {
        "status": "accepted" if dispatched else "duplicate",
        "dispatched": dispatched,
        "event_type": event.type,
        "source": event.source,
        "timestamp": event.timestamp.isoformat(),
    }


@router.get("/recommendations")
async def list_recommendations(request: Request) -> dict[str, Any]:
    """List pending non-destructive policy recommendations."""
    db = _get_db(request)
    rows = await db.get_pending_recommendations()
    items = [
        Recommendation.model_validate(row).model_dump(mode="json")
        for row in rows
    ]
    return {
        "recommendations": items,
        "count": len(items),
    }


@router.post("/recommendations/snooze", response_model=None)
async def snooze_latest_recommendation(
    request: Request,
    payload: _RecResolutionDep,
) -> HTMLResponse | dict[str, Any]:
    """Snooze the most recent pending recommendation (Waybar middle-click shortcut)."""
    db = _get_db(request)
    bus = _get_bus(request)

    pending = await db.get_pending_recommendations()
    if not pending:
        if is_htmx_request(request):
            context = await build_recommendations_context(
                request,
                flash_message="No pending recommendations to snooze.",
            )
            return render_template(
                request,
                "_partials/recommendations_card.html",
                context,
                headers={"HX-Trigger": "recommendationChanged"},
            )
        return {
            "status": "noop",
            "snoozed": False,
            "recommendation": None,
        }

    latest_id = int(pending[0]["id"])
    serialized = await _resolve_and_emit_recommendation(
        db=db,
        bus=bus,
        rec_id=latest_id,
        status="snoozed",
        metadata=payload.to_metadata(include_snooze=True),
    )
    if is_htmx_request(request):
        context = await build_recommendations_context(
            request,
            flash_message=f"Recommendation #{latest_id} snoozed.",
        )
        return render_template(
            request,
            "_partials/recommendations_card.html",
            context,
            headers={"HX-Trigger": "recommendationChanged"},
        )
    return {
        "status": "ok",
        "snoozed": True,
        "recommendation": serialized,
    }


@router.post("/recommendations/{rec_id}/accept", response_model=None)
async def accept_recommendation(
    rec_id: int,
    request: Request,
    payload: _RecResolutionDep,
) -> HTMLResponse | dict[str, Any]:
    """Accept a recommendation non-destructively (records status + emits event only)."""
    serialized = await _resolve_and_emit_recommendation(
        db=_get_db(request),
        bus=_get_bus(request),
        rec_id=rec_id,
        status="accepted",
        metadata=payload.to_metadata(),
    )
    if is_htmx_request(request):
        context = await build_recommendations_context(
            request,
            flash_message=(
                f"Recommendation #{rec_id} accepted "
                "(recorded only — no automatic timer or posture change)."
            ),
        )
        return render_template(
            request,
            "_partials/recommendations_card.html",
            context,
            headers={"HX-Trigger": "recommendationChanged"},
        )
    return {
        "status": "ok",
        "recommendation": serialized,
    }


@router.post("/recommendations/{rec_id}/snooze", response_model=None)
async def snooze_recommendation(
    rec_id: int,
    request: Request,
    payload: _RecResolutionDep,
) -> HTMLResponse | dict[str, Any]:
    """Snooze a specific recommendation by ID."""
    serialized = await _resolve_and_emit_recommendation(
        db=_get_db(request),
        bus=_get_bus(request),
        rec_id=rec_id,
        status="snoozed",
        metadata=payload.to_metadata(include_snooze=True),
    )
    if is_htmx_request(request):
        context = await build_recommendations_context(
            request,
            flash_message=f"Recommendation #{rec_id} snoozed.",
        )
        return render_template(
            request,
            "_partials/recommendations_card.html",
            context,
            headers={"HX-Trigger": "recommendationChanged"},
        )
    return {
        "status": "ok",
        "recommendation": serialized,
    }


@router.post("/recommendations/{rec_id}/dismiss", response_model=None)
async def dismiss_recommendation(
    rec_id: int,
    request: Request,
    payload: _RecResolutionDep,
) -> HTMLResponse | dict[str, Any]:
    """Dismiss a specific recommendation by ID."""
    serialized = await _resolve_and_emit_recommendation(
        db=_get_db(request),
        bus=_get_bus(request),
        rec_id=rec_id,
        status="dismissed",
        metadata=payload.to_metadata(),
    )
    if is_htmx_request(request):
        context = await build_recommendations_context(
            request,
            flash_message=f"Recommendation #{rec_id} dismissed.",
        )
        return render_template(
            request,
            "_partials/recommendations_card.html",
            context,
            headers={"HX-Trigger": "recommendationChanged"},
        )
    return {
        "status": "ok",
        "recommendation": serialized,
    }


@router.post("/timers/start", response_model=None)
async def start_timer(
    request: Request,
    payload: _TimerStartDep,
) -> HTMLResponse | dict[str, Any]:
    """Start a local Pomodoro ('25_5', '50_10') or custom timer."""
    timer_manager = _require_timer_manager(request)
    try:
        timer_status = await timer_manager.start_timer(
            kind=payload.kind,
            work_minutes=payload.work_minutes,
            break_minutes=payload.break_minutes,
            work_seconds=payload.work_seconds,
            break_seconds=payload.break_seconds,
            calendar_event_id=payload.calendar_event_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if is_htmx_request(request):
        context = await build_timer_context(request)
        return render_template(
            request,
            "_partials/timer_card.html",
            context,
            headers={"HX-Trigger": "timerChanged"},
        )

    return {
        "status": "ok",
        **timer_status,
        "timer": timer_status,
    }


@router.post("/timers/stop", response_model=None)
async def stop_timer(
    request: Request,
    payload: _TimerStopDep,
) -> HTMLResponse | dict[str, Any]:
    """Stop the currently active local timer."""
    timer_manager = _require_timer_manager(request)
    stop_result = await timer_manager.stop_timer(reason=payload.reason)
    if is_htmx_request(request):
        context = await build_timer_context(request)
        return render_template(
            request,
            "_partials/timer_card.html",
            context,
            headers={"HX-Trigger": "timerChanged"},
        )

    current_status = timer_manager.get_status()
    return {
        "status": "ok",
        **stop_result,
        "timer": current_status,
    }


@router.get("/history/activity")
async def get_activity_history(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    since: Annotated[datetime | None, Query()] = None,
) -> dict[str, Any]:
    """Return persisted effective activity spans and per-activity duration aggregates."""
    db = _get_db(request)
    now = now_local()
    rows = await db.list_activity_spans(since=since, limit=limit)

    serialized: list[dict[str, Any]] = []
    totals_seconds: dict[str, int] = {}

    for row in rows:
        span_model = ActivitySpan.model_validate(row)
        end_dt = span_model.ended_at if span_model.ended_at is not None else now
        duration = max(0, int((end_dt - span_model.started_at).total_seconds()))
        totals_seconds[span_model.activity] = (
            totals_seconds.get(span_model.activity, 0) + duration
        )
        serialized.append(span_model.model_dump(mode="json"))

    return {
        "spans": serialized,
        "count": len(serialized),
        "totals_seconds": totals_seconds,
    }


@router.get("/history/windows")
async def get_window_history(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    since: Annotated[datetime | None, Query()] = None,
) -> dict[str, Any]:
    """Return persisted window/focus spans and per-category duration aggregates."""
    db = _get_db(request)
    now = now_local()
    rows = await db.list_window_spans(limit=limit, since=since)

    totals_seconds: dict[str, int] = {}
    for row in rows:
        started_dt = ensure_aware(datetime.fromisoformat(str(row["started_at"])))
        ended_raw = row.get("ended_at")
        end_dt = (
            ensure_aware(datetime.fromisoformat(str(ended_raw)))
            if ended_raw is not None
            else now
        )
        duration = max(0, int((end_dt - started_dt).total_seconds()))
        category = str(row.get("category", "unknown"))
        totals_seconds[category] = totals_seconds.get(category, 0) + duration

    return {
        "spans": rows,
        "count": len(rows),
        "totals_seconds": totals_seconds,
    }


@router.get("/modules")
async def list_modules(request: Request) -> dict[str, Any]:
    """Return runtime status and SQLite health details for all tracked modules."""
    registry = _get_registry(request)
    db = _get_db(request)

    db_health_by_name = {
        str(row["module"]): row for row in await db.list_module_health()
    }
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for rec in registry.list_modules():
        mod_name = str(rec["module"])
        seen.add(mod_name)
        db_row = db_health_by_name.get(mod_name, {})
        merged.append(
            {
                **rec,
                "last_event_at": db_row.get("last_event_at"),
                "updated_at": db_row.get("updated_at"),
            }
        )

    for mod_name, db_row in sorted(db_health_by_name.items()):
        if mod_name in seen:
            continue
        merged.append(
            {
                "module": mod_name,
                "category": None,
                "enabled": False,
                "status": db_row.get("status", "disabled"),
                "last_error": db_row.get("last_error"),
                "last_event_at": db_row.get("last_event_at"),
                "updated_at": db_row.get("updated_at"),
            }
        )

    return {
        "modules": merged,
        "count": len(merged),
    }


@router.post("/modules/{name}/enable", response_model=None)
async def enable_module(name: str, request: Request) -> HTMLResponse | dict[str, Any]:
    """Dynamically enable and start a module by name."""
    registry = _get_registry(request)
    if registry.get_status(name) is None and registry._resolve_category(name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown module '{name}'",
        )

    ok = await registry.enable_module(name)
    if is_htmx_request(request):
        context = await build_modules_context(request)
        return render_template(
            request,
            "_partials/modules_card.html",
            context,
            headers={"HX-Trigger": "modulesChanged"},
        )

    status_info = registry.get_status(name) or {
        "module": name,
        "category": None,
        "enabled": ok,
        "status": "ok" if ok else "degraded",
        "last_error": None,
    }
    return {
        "ok": ok,
        **status_info,
    }


@router.post("/modules/{name}/disable", response_model=None)
async def disable_module(name: str, request: Request) -> HTMLResponse | dict[str, Any]:
    """Dynamically stop and disable a module by name."""
    registry = _get_registry(request)
    if registry.get_status(name) is None and registry._resolve_category(name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown module '{name}'",
        )

    ok = await registry.disable_module(name)
    if is_htmx_request(request):
        context = await build_modules_context(request)
        return render_template(
            request,
            "_partials/modules_card.html",
            context,
            headers={"HX-Trigger": "modulesChanged"},
        )

    status_info = registry.get_status(name) or {
        "module": name,
        "category": None,
        "enabled": False,
        "status": "disabled",
        "last_error": None,
    }
    return {
        "ok": ok,
        **status_info,
    }


@router.post("/privacy/pause", response_model=None)
async def pause_privacy(request: Request) -> HTMLResponse | dict[str, Any]:
    """Enable global privacy pause, persist setting, and broadcast privacy_mode_changed."""
    settings = _get_settings(request)
    settings.privacy.global_pause = True
    result = await _persist_and_publish_privacy(
        settings=settings,
        db=_get_db(request),
        bus=_get_bus(request),
    )
    if is_htmx_request(request):
        context = await build_privacy_context(request)
        return render_template(
            request,
            "_partials/privacy_card.html",
            context,
            headers={"HX-Trigger": "privacyChanged"},
        )
    return result


@router.post("/privacy/resume", response_model=None)
async def resume_privacy(request: Request) -> HTMLResponse | dict[str, Any]:
    """Disable global privacy pause, persist setting, and broadcast privacy_mode_changed."""
    settings = _get_settings(request)
    settings.privacy.global_pause = False
    result = await _persist_and_publish_privacy(
        settings=settings,
        db=_get_db(request),
        bus=_get_bus(request),
    )
    if is_htmx_request(request):
        context = await build_privacy_context(request)
        return render_template(
            request,
            "_partials/privacy_card.html",
            context,
            headers={"HX-Trigger": "privacyChanged"},
        )
    return result


@router.post("/privacy/dnd", response_model=None)
async def set_or_toggle_dnd(
    request: Request,
    payload: _PrivacyDndDep,
) -> HTMLResponse | dict[str, Any]:
    """Set or toggle DND mode, persist setting, and broadcast privacy_mode_changed."""
    settings = _get_settings(request)
    if payload.dnd_mode is not None:
        settings.privacy.dnd_mode = bool(payload.dnd_mode)
    elif payload.enabled is not None:
        settings.privacy.dnd_mode = bool(payload.enabled)
    else:
        settings.privacy.dnd_mode = not bool(settings.privacy.dnd_mode)

    result = await _persist_and_publish_privacy(
        settings=settings,
        db=_get_db(request),
        bus=_get_bus(request),
    )
    if is_htmx_request(request):
        context = await build_privacy_context(request)
        return render_template(
            request,
            "_partials/privacy_card.html",
            context,
            headers={"HX-Trigger": "privacyChanged"},
        )
    return result


@router.post("/privacy/tts", response_model=None)
async def set_or_toggle_tts(
    request: Request,
    payload: _PrivacyTtsDep,
) -> HTMLResponse | dict[str, Any]:
    """Set or toggle TTS audio permission (FR-25)."""
    settings = _get_settings(request)
    if payload.allow_tts is not None:
        settings.privacy.allow_tts = bool(payload.allow_tts)
    elif payload.enabled is not None:
        settings.privacy.allow_tts = bool(payload.enabled)
    else:
        settings.privacy.allow_tts = not bool(settings.privacy.allow_tts)

    result = await _persist_and_publish_privacy(
        settings=settings,
        db=_get_db(request),
        bus=_get_bus(request),
    )
    if is_htmx_request(request):
        context = await build_privacy_context(request)
        return render_template(
            request,
            "_partials/privacy_card.html",
            context,
            headers={"HX-Trigger": "privacyChanged"},
        )
    return result


@router.post("/privacy/microphone", response_model=None)
async def set_or_toggle_microphone(
    request: Request,
    payload: _PrivacyMicDep,
) -> HTMLResponse | dict[str, Any]:
    """Set or toggle microphone privacy mode (push_to_talk | disabled)."""
    settings = _get_settings(request)
    if payload.microphone_mode is not None:
        norm = payload.microphone_mode.strip().lower()
        if norm in ("push_to_talk", "disabled"):
            settings.privacy.microphone_mode = norm  # type: ignore[assignment]
    elif payload.enabled is not None:
        settings.privacy.microphone_mode = "push_to_talk" if payload.enabled else "disabled"
    else:
        settings.privacy.microphone_mode = (
            "disabled" if settings.privacy.microphone_mode == "push_to_talk" else "push_to_talk"
        )

    result = await _persist_and_publish_privacy(
        settings=settings,
        db=_get_db(request),
        bus=_get_bus(request),
    )
    if is_htmx_request(request):
        context = await build_privacy_context(request)
        return render_template(
            request,
            "_partials/privacy_card.html",
            context,
            headers={"HX-Trigger": "privacyChanged"},
        )
    return result


@router.post("/settings/llm", response_model=None)
async def update_llm_settings(
    request: Request,
    payload: _SettingsLlmDep,
) -> HTMLResponse | dict[str, Any]:
    """Update runtime text model or keep-alive duration for local LLM (FR-23)."""
    settings = _get_settings(request)
    db = _get_db(request)
    registry = _get_registry(request)

    if payload.text_model is not None and payload.text_model.strip():
        settings.llm.text_model = payload.text_model.strip()
        await db.set_setting("llm.text_model", settings.llm.text_model)
    if payload.keep_alive is not None and payload.keep_alive.strip():
        settings.llm.keep_alive = payload.keep_alive.strip()
        await db.set_setting("llm.keep_alive", settings.llm.keep_alive)

    await db.set_setting(
        "llm",
        {
            "text_model": settings.llm.text_model,
            "keep_alive": settings.llm.keep_alive,
        },
    )

    llm_mod = registry.get_module("llm_ollama")
    if llm_mod is not None:
        from contextlib import suppress
        with suppress(Exception):
            await llm_mod.reload_config(settings.model_dump())

    response_payload = {
        "status": "ok",
        "text_model": settings.llm.text_model,
        "keep_alive": settings.llm.keep_alive,
    }
    if is_htmx_request(request):
        msg = (
            f"<div class='alert alert-success p-2'>Updated LLM model to "
            f"<code>{settings.llm.text_model}</code> (keep-alive: "
            f"<code>{settings.llm.keep_alive}</code>).</div>"
        )
        return HTMLResponse(
            msg,
            headers={"HX-Trigger": "settingsChanged"},
        )
    return response_payload


@router.post("/settings", response_model=None)
async def update_general_settings(
    request: Request,
    payload: _SettingsLlmDep,
) -> HTMLResponse | dict[str, Any]:
    """Alias for runtime settings update."""
    return await update_llm_settings(request, payload)


@router.post("/ask", response_model=None)
async def ask_assistant(
    request: Request,
    payload: _AskDep,
) -> HTMLResponse | JSONResponse | dict[str, Any]:
    """Ask the local LLM assistant a question in platform context (FR-23)."""
    settings = _get_settings(request)
    resolver = _get_resolver(request)
    registry = _get_registry(request)

    # 1. Privacy global pause check -> 409 Conflict
    if settings.privacy.global_pause or resolver.get_effective_state().privacy_paused:
        msg = "Privacy global pause is active. Assistant requests are suspended."
        if is_htmx_request(request):
            return HTMLResponse(
                f"<div class='alert alert-warning p-2'>{msg}</div>",
                status_code=409,
            )
        return JSONResponse(
            status_code=409,
            content={
                "detail": msg,
                "error": "privacy_paused",
                "message": msg,
            },
        )

    # 1b. Microphone mode disabled check for voice requests -> 409 Conflict
    if payload.source == "voice" and settings.privacy.microphone_mode == "disabled":
        msg = "Microphone is disabled in privacy settings."
        if is_htmx_request(request):
            return HTMLResponse(
                f"<div class='alert alert-warning p-2'>{msg}</div>",
                status_code=409,
            )
        return JSONResponse(
            status_code=409,
            content={
                "detail": msg,
                "error": "microphone_disabled",
                "message": msg,
            },
        )

    # 2. Disabled module or missing dependency check -> 503 Service Unavailable
    llm_mod = registry.get_module("llm_ollama")
    if not registry.is_enabled("llm_ollama") or llm_mod is None:
        if not LlmOllamaModule.is_available():
            msg = (
                "Module 'llm_ollama' dependency group is missing. "
                "Install with 'uv sync --group llm' and enable llm_ollama in Settings."
            )
        else:
            msg = "Module llm_ollama is disabled. Enable llm_ollama in Settings."
        if is_htmx_request(request):
            return HTMLResponse(
                f"<div class='alert alert-danger p-2'>{msg}</div>",
                status_code=503,
            )
        return JSONResponse(
            status_code=503,
            content={
                "detail": msg,
                "error": "module_unavailable",
                "message": msg,
            },
        )

    # 3. Call LlmOllamaModule.ask(...)
    try:
        result = await llm_mod.ask(
            question=payload.question,
            source=payload.source,
            speak=payload.speak,
            resolver=resolver,
            timer_manager=_get_timer_manager(request),
            settings=settings,
        )
    except LlmValidationError as exc:
        msg = f"Invalid model response: {exc}"
        if is_htmx_request(request):
            return HTMLResponse(
                f"<div class='alert alert-danger p-2'>{msg}</div>",
                status_code=502,
            )
        return JSONResponse(
            status_code=502,
            content={
                "detail": msg,
                "error": "validation_error",
                "message": str(exc),
            },
        )
    except (LlmConnectionError, LlmTimeoutError, LlmError) as exc:
        msg = f"Ollama service error: {exc}"
        if is_htmx_request(request):
            return HTMLResponse(
                f"<div class='alert alert-danger p-2'>{msg}</div>",
                status_code=503,
            )
        return JSONResponse(
            status_code=503,
            content={
                "detail": msg,
                "error": "ollama_error",
                "message": str(exc),
            },
        )
    except Exception as exc:
        msg = f"Unexpected error processing assistant question: {exc}"
        if is_htmx_request(request):
            return HTMLResponse(
                f"<div class='alert alert-danger p-2'>{msg}</div>",
                status_code=503,
            )
        return JSONResponse(
            status_code=503,
            content={
                "detail": msg,
                "error": "assistant_error",
                "message": str(exc),
            },
        )

    # 4. Handle HTMX partial response
    if is_htmx_request(request):
        suggested = result.get("suggested_action")
        action_html = (
            f"<div class='text-muted small mt-1'>Suggested action: <code>{suggested}</code></div>"
            if suggested
            else ""
        )
        conf_str = f"{result['confidence']:.2f}"
        latency_val = result.get("latency_ms")
        latency_str = f" | Latency: {latency_val}ms" if latency_val is not None else ""
        meta_str = f"Model: {result['model']}{latency_str} | Confidence: {conf_str}"
        html = (
            f"""<div class="assistant-response card p-3 mb-2">\n"""
            f"""  <div class="answer">{result['answer']}</div>\n"""
            f"""  {action_html}\n"""
            f"""  <div class="meta text-muted small mt-2">{meta_str}</div>\n"""
            f"""</div>"""
        )
        return HTMLResponse(
            content=html,
            status_code=200,
            headers={"HX-Trigger": "assistantAnswered"},
        )

    return AskResponse(
        answer=result["answer"],
        suggested_action=result.get("suggested_action"),
        confidence=float(result["confidence"]),
        model=result["model"],
        interaction_id=result.get("interaction_id", 0),
        latency_ms=result.get("latency_ms"),
    ).model_dump(mode="json")


@router.get("/ask/history")
async def get_ask_history(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    """Return recent LLM interactions."""
    db = _get_db(request)
    interactions = await db.list_llm_interactions(limit=limit)
    return {
        "interactions": interactions,
        "count": len(interactions),
    }
