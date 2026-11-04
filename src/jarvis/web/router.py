"""Server-rendered HTMX dashboard pages and live-polling HTML partials."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from jarvis.web.views import (
    build_calendar_context,
    build_dashboard_context,
    build_history_context,
    build_modules_context,
    build_posture_context,
    build_privacy_context,
    build_recommendations_context,
    build_settings_context,
    build_timer_context,
    render_template,
)

router = APIRouter(tags=["web"])


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Render the live HTMX control-plane dashboard (`/`)."""
    context = await build_dashboard_context(request)
    return render_template(request, "dashboard.html", context)


@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request) -> HTMLResponse:
    """Render the activity, timer, and recommendation history page (`/history`)."""
    context = await build_history_context(request)
    return render_template(request, "history.html", context)


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    """Render live module toggles, health inspection, and threshold settings (`/settings`)."""
    context = await build_settings_context(request)
    return render_template(request, "settings.html", context)


@router.get("/partials/posture", response_class=HTMLResponse)
async def posture_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for live effective posture and manual override controls."""
    context = await build_posture_context(request)
    return render_template(request, "_partials/posture_card.html", context)


@router.get("/partials/timer", response_class=HTMLResponse)
async def timer_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for local Pomodoro timer countdown and controls."""
    context = await build_timer_context(request)
    return render_template(request, "_partials/timer_card.html", context)


@router.get("/partials/calendar", response_class=HTMLResponse)
async def calendar_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for current and upcoming calendar blocks."""
    context = await build_calendar_context(request)
    return render_template(request, "_partials/calendar_card.html", context)


@router.get("/partials/recommendations", response_class=HTMLResponse)
async def recommendations_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for pending non-destructive policy recommendations."""
    context = await build_recommendations_context(request)
    return render_template(request, "_partials/recommendations_card.html", context)


@router.get("/partials/privacy", response_class=HTMLResponse)
async def privacy_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for Privacy Pause and DND Mode switches."""
    context = await build_privacy_context(request)
    return render_template(request, "_partials/privacy_card.html", context)


@router.get("/partials/history", response_class=HTMLResponse)
async def history_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for daily totals, activity spans, timers, and recommendations."""
    context = await build_history_context(request)
    return render_template(request, "_partials/history_content.html", context)


@router.get("/partials/modules", response_class=HTMLResponse)
async def modules_partial(request: Request) -> HTMLResponse:
    """Render HTMX partial for live module enable/disable controls and health status."""
    context = await build_modules_context(request)
    return render_template(request, "_partials/modules_card.html", context)
