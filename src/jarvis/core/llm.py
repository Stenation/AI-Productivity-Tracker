"""Compact context builder, Ollama HTTP client, and safety-boundary assistant logic."""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from typing import Any

from jarvis.core.database import Database
from jarvis.core.schema import ensure_aware, now_local
from jarvis.core.settings import Settings

logger = logging.getLogger(__name__)

# JSON schema enforced for structured assistant responses
RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Natural language response to the user's question.",
        },
        "suggested_action": {
            "type": ["string", "null"],
            "description": "Optional recommended action identifier.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0.0 and 1.0.",
        },
    },
    "required": ["answer", "confidence", "suggested_action"],
}

CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": [
                "deep_work",
                "coding",
                "reading",
                "admin",
                "email",
                "planning",
                "meeting",
                "break",
            ],
            "description": "Classified calendar task category.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0.0 and 1.0.",
        },
    },
    "required": ["task_type", "confidence"],
}

_THINK_TAG_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


class LlmError(Exception):
    """Base exception for LLM operations."""


class LlmConnectionError(LlmError):
    """Raised when the Ollama server is unreachable or connection is refused."""


class LlmTimeoutError(LlmError):
    """Raised when an Ollama request times out."""


class LlmValidationError(LlmError):
    """Raised when Ollama returns an unparseable or schema-violating response."""


def clean_think_tags(raw_text: str) -> str:
    """Defensively strip <think> tags (closed or unclosed) and code fences."""
    cleaned = _THINK_TAG_PATTERN.sub("", raw_text)
    cleaned = _UNCLOSED_THINK_PATTERN.sub("", cleaned).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def validate_llm_response(parsed: Any) -> dict[str, Any]:
    """Validate parsed JSON dictionary against the assistant response schema."""
    if not isinstance(parsed, dict):
        raise LlmValidationError(f"Expected JSON object, got {type(parsed).__name__}")

    if (
        "answer" not in parsed
        or not isinstance(parsed["answer"], str)
        or not parsed["answer"].strip()
    ):
        raise LlmValidationError("Response missing required non-empty string field 'answer'")

    if "confidence" not in parsed:
        raise LlmValidationError("Response missing required field 'confidence'")

    try:
        confidence = float(parsed["confidence"])
    except (TypeError, ValueError) as exc:
        raise LlmValidationError(
            f"Field 'confidence' is not a number: {parsed.get('confidence')}"
        ) from exc

    if not (0.0 <= confidence <= 1.0):
        raise LlmValidationError(
            f"Field 'confidence' must be between 0.0 and 1.0, got {confidence}"
        )

    suggested_action = parsed.get("suggested_action")
    if suggested_action is not None and not isinstance(suggested_action, str):
        type_name = type(suggested_action).__name__
        raise LlmValidationError(
            f"Field 'suggested_action' must be string or null, got {type_name}"
        )

    action_clean = (
        suggested_action.strip()
        if isinstance(suggested_action, str) and suggested_action.strip()
        else None
    )

    return {
        "answer": parsed["answer"].strip(),
        "suggested_action": action_clean,
        "confidence": confidence,
    }


_MONTH_MAP: dict[str, int] = {
    "jan": 1, "januar": 1, "january": 1,
    "feb": 2, "februar": 2, "february": 2,
    "mär": 3, "märz": 3, "mar": 3, "maerz": 3, "march": 3,
    "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "jun": 6, "juni": 6, "june": 6,
    "jul": 7, "juli": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oktober": 10, "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dez": 12, "dezember": 12, "dec": 12, "december": 12,
}

_WEEKDAY_MAP: dict[str, int] = {
    "montag": 0, "monday": 0,
    "dienstag": 1, "tuesday": 1,
    "mittwoch": 2, "wednesday": 2,
    "donnerstag": 3, "thursday": 3,
    "freitag": 4, "friday": 4,
    "samstag": 5, "sonnabend": 5, "saturday": 5,
    "sonntag": 6, "sunday": 6,
}


def parse_calendar_query_range(
    question: str, now: datetime
) -> tuple[datetime, datetime, bool, str]:
    """Parse calendar date range from a user question.

    Returns:
        (time_min, time_max, is_explicit_range, range_description)
    Where time_max includes the full final day in the user's timezone.
    """
    tz = now.tzinfo
    q = question.lower()
    today_start = datetime.combine(now.date(), dt_time.min, tzinfo=tz)
    today_end = datetime.combine(now.date() + timedelta(days=1), dt_time.min, tzinfo=tz)

    # 1. ISO format date: YYYY-MM-DD
    iso_m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q)
    if iso_m:
        y, m, d = int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3))
        target_d = date(y, m, d)
        is_range = (
            any(w in q for w in ["bis", "until", "through", "to"]) or target_d >= now.date()
        )
        start_dt = (
            today_start if is_range else datetime.combine(target_d, dt_time.min, tzinfo=tz)
        )
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} (full day)"

    # 2. D.M.YYYY or D.M.YY or D.M. (e.g. 8.11.2026, 8.11.26, 8.11., 08.11.2026)
    dm_m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.?(?:(\d{4}|\d{2}))?\b", q)
    if dm_m:
        d = int(dm_m.group(1))
        m = int(dm_m.group(2))
        raw_y = dm_m.group(3)
        if raw_y:
            y = int(raw_y)
            if y < 100:
                y += 2000
        else:
            y = now.year
            if date(y, m, d) < now.date() and not any(w in q for w in ["letzte", "last", "vergangene"]):
                y += 1
        target_d = date(y, m, d)
        is_range = (
            any(w in q for w in ["bis", "until", "through", "to"]) or target_d >= now.date()
        )
        start_dt = (
            today_start if is_range else datetime.combine(target_d, dt_time.min, tzinfo=tz)
        )
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} (full day)"

    # 3. Named month (e.g. "8. November 2026", "Nov 8", "8th of November")
    month_pattern = "|".join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))
    named_m = re.search(
        r"\b(\d{1,2})\.?\s+(?:von\s+|of\s+)?(" + month_pattern + r")(?:\s+(\d{4}|\d{2}))?\b",
        q,
    )
    if not named_m:
        named_m = re.search(
            r"\b(" + month_pattern + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}|\d{2}))?\b",
            q,
        )
        if named_m:
            m = _MONTH_MAP[named_m.group(1)]
            d = int(named_m.group(2))
            raw_y = named_m.group(3)
        else:
            m = None
    else:
        d = int(named_m.group(1))
        m = _MONTH_MAP[named_m.group(2)]
        raw_y = named_m.group(3)

    if m is not None:
        if raw_y:
            y = int(raw_y)
            if y < 100:
                y += 2000
        else:
            y = now.year
            if date(y, m, d) < now.date() and not any(w in q for w in ["letzte", "last", "vergangene"]):
                y += 1
        target_d = date(y, m, d)
        is_range = (
            any(w in q for w in ["bis", "until", "through", "to"]) or target_d >= now.date()
        )
        start_dt = (
            today_start if is_range else datetime.combine(target_d, dt_time.min, tzinfo=tz)
        )
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} (full day)"

    # 4. Relative expressions:
    if "übermorgen" in q or "day after tomorrow" in q:
        target_d = now.date() + timedelta(days=2)
        start_dt = datetime.combine(target_d, dt_time.min, tzinfo=tz)
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{target_d} (full day)"

    if "morgen" in q or "tomorrow" in q:
        target_d = now.date() + timedelta(days=1)
        if any(w in q for w in ["bis", "until", "through", "und", "and"]):
            start_dt = today_start
        else:
            start_dt = datetime.combine(target_d, dt_time.min, tzinfo=tz)
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} (full day)"

    if "diese woche" in q or "this week" in q:
        days_ahead = 6 - now.weekday()
        target_d = now.date() + timedelta(days=days_ahead)
        start_dt = today_start
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} (end of week full day)"

    if "nächste woche" in q or "next week" in q:
        days_to_next_mon = (7 - now.weekday()) % 7
        if days_to_next_mon == 0:
            days_to_next_mon = 7
        next_mon = now.date() + timedelta(days=days_to_next_mon)
        next_sun = next_mon + timedelta(days=6)
        start_dt = datetime.combine(next_mon, dt_time.min, tzinfo=tz)
        end_dt = datetime.combine(next_sun + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{next_mon} to {next_sun} (next week full days)"

    for w_name, w_idx in _WEEKDAY_MAP.items():
        if f"bis {w_name}" in q or f"until {w_name}" in q or f"durch {w_name}" in q:
            days_ahead = (w_idx - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_d = now.date() + timedelta(days=days_ahead)
            start_dt = today_start
            end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
            return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} ({w_name} full day)"

    next_n_m = re.search(r"(?:nächsten?|kommenden?|next)\s+(\d+)\s+(?:tage|tagen|days)", q)
    if next_n_m:
        n_days = int(next_n_m.group(1))
        target_d = now.date() + timedelta(days=n_days)
        start_dt = today_start
        end_dt = datetime.combine(target_d + timedelta(days=1), dt_time.min, tzinfo=tz)
        return start_dt, end_dt, True, f"{start_dt.date()} to {target_d} (full day)"

    return today_start, today_end, False, f"{now.date()} (today)"


async def build_assistant_context(
    db: Database,
    resolver: Any | None = None,
    timer_manager: Any | None = None,
    settings: Settings | None = None,
    registry: Any | None = None,
    question: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a compact, privacy-safe JSON snapshot of platform state.

    Contains:
    - current activity, source, and elapsed duration
    - active desktop countdown timer status: kind, phase, remaining
    - calendar state: status, availability, date range, appointments, and current/next events
    - recent window categories and duration totals (never sensitive raw window titles)
    - pending recommendations
    """
    ref_now = ensure_aware(now) if now is not None else now_local()

    # 1. Effective activity state
    if resolver is not None and hasattr(resolver, "get_effective_state"):
        effective = resolver.get_effective_state()
        activity_data = {
            "activity": getattr(effective, "activity", "unknown"),
            "source": getattr(effective, "source", "system"),
            "elapsed_seconds": int(getattr(effective, "elapsed_seconds", 0)),
        }
    else:
        current_span = await db.get_current_activity_span()
        if current_span:
            start_dt = ensure_aware(datetime.fromisoformat(str(current_span["started_at"])))
            elapsed = max(0, int((ref_now - start_dt).total_seconds()))
            activity_data = {
                "activity": current_span.get("activity", "unknown"),
                "source": current_span.get("source", "system"),
                "elapsed_seconds": elapsed,
            }
        else:
            activity_data = {
                "activity": "unknown",
                "source": "system",
                "elapsed_seconds": 0,
            }

    # 2. Active desktop countdown timer status
    if timer_manager is not None and hasattr(timer_manager, "get_status"):
        timer_status = timer_manager.get_status()
        timer_data = {
            "kind": timer_status.get("kind"),
            "phase": timer_status.get("phase", "idle"),
            "remaining_seconds": timer_status.get("remaining_seconds", 0),
        }
    else:
        active_timer_row = await db.get_active_timer()
        if active_timer_row is not None:
            started_dt = ensure_aware(datetime.fromisoformat(str(active_timer_row["started_at"])))
            planned = active_timer_row.get("planned_duration_seconds")
            elapsed = max(0, int((ref_now - started_dt).total_seconds()))
            remaining = max(0, int(planned) - elapsed) if planned is not None else 0
            raw_kind = str(active_timer_row.get("kind", ""))
            is_break = raw_kind.endswith("_break") or raw_kind == "break"
            base_kind = raw_kind.removesuffix("_break") if is_break else raw_kind
            timer_data = {
                "kind": base_kind,
                "phase": "break" if is_break else "work",
                "remaining_seconds": remaining,
            }
        else:
            timer_data = {
                "kind": None,
                "phase": "idle",
                "remaining_seconds": 0,
            }

    # 3. Calendar: status, appointments within requested range, today's appointments, and current/next events
    start_of_today = datetime.combine(ref_now.date(), dt_time.min, tzinfo=ref_now.tzinfo)
    end_of_today = datetime.combine(
        ref_now.date() + timedelta(days=1), dt_time.min, tzinfo=ref_now.tzinfo
    )

    query_start = start_of_today
    query_end = end_of_today
    is_explicit_range = False
    range_desc = f"{ref_now.date()} (today)"

    if question:
        query_start, query_end, is_explicit_range, range_desc = parse_calendar_query_range(
            question, ref_now
        )

    cal_enabled = False
    if registry is not None:
        cal_enabled = bool(registry.is_enabled("google_calendar"))
    else:
        persisted = await db.get_setting("module.google_calendar.enabled")
        if persisted is not None:
            cal_enabled = bool(persisted)
        elif settings is not None:
            inputs_enabled = getattr(settings.enabled_modules, "inputs", [])
            cal_enabled = "google_calendar" in inputs_enabled
        else:
            health_check = await db.get_module_health("google_calendar")
            if health_check is not None:
                cal_enabled = health_check.get("status") != "disabled"
            else:
                cached_any = await db.list_calendar_events()
                cal_enabled = bool(cached_any)

    def _format_event(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        st_raw = str(row.get("start_time", ""))
        et_raw = str(row.get("end_time", ""))
        time_window = ""
        date_str = ""
        status = "upcoming"
        try:
            s_dt = ensure_aware(datetime.fromisoformat(st_raw))
            e_dt = ensure_aware(datetime.fromisoformat(et_raw))
            date_str = s_dt.strftime("%Y-%m-%d")
            time_window = f"{s_dt.strftime('%H:%M')} – {e_dt.strftime('%H:%M')}"
            if ref_now > e_dt:
                status = "past"
            elif ref_now >= s_dt:
                status = "active"
            else:
                status = "upcoming"
        except Exception:
            time_window = f"{st_raw} – {et_raw}"
        walking_val = row.get("suitable_for_walking")
        walking_bool = bool(walking_val) if walking_val is not None else None
        return {
            "title": row.get("title", ""),
            "date": date_str,
            "status": status,
            "task_type": row.get("task_type", "unknown"),
            "start_time": st_raw,
            "end_time": et_raw,
            "time_window": time_window,
            "suitable_for_walking": walking_bool,
        }

    cur_row = None
    next_row = None
    events_today: list[dict[str, Any]] = []
    events_range: list[dict[str, Any]] = []
    data_complete = True
    incomplete_reason: str | None = None

    if not cal_enabled:
        calendar_status = "disabled"
        calendar_available = False
        data_complete = False
        calendar_message = (
            "Calendar integration is disabled in settings. Calendar data is unavailable."
        )
        incomplete_reason = calendar_message
    else:
        health_row = await db.get_module_health("google_calendar")
        cal_instance = registry.get_module("google_calendar") if registry is not None else None
        registry_status = registry.get_status("google_calendar") if registry is not None else None
        effective_status = registry_status or (health_row.get("status") if health_row else None)
        last_error = (health_row.get("last_error") if health_row else None) or (
            getattr(cal_instance, "_last_error", None) if cal_instance is not None else None
        )

        if effective_status == "degraded" or last_error:
            data_complete = False
            err_lower = (last_error or "").lower()
            if any(
                kw in err_lower
                for kw in ("oauth", "auth", "token", "credential", "permission", "filenotfound")
            ):
                calendar_status = "not_authenticated"
                calendar_available = False
                calendar_message = (
                    "Google Calendar authentication failed or token is missing/expired. "
                    "Calendar data is unavailable."
                )
            else:
                calendar_status = "fetch_failed"
                calendar_available = False
                calendar_message = (
                    "Google Calendar event fetch failed. Calendar data is unavailable."
                )
            incomplete_reason = calendar_message
        else:
            has_synced = False
            if (
                cal_instance is not None
                and getattr(cal_instance, "_last_sync_at", None) is not None
            ):
                has_synced = True
            elif (
                health_row is not None
                and health_row.get("last_event_at") is not None
                and health_row.get("status") == "ok"
            ):
                has_synced = True
            else:
                cached_all = await db.list_calendar_events()
                if cached_all:
                    has_synced = True

            if not has_synced:
                calendar_status = "not_fetched_yet"
                calendar_available = False
                data_complete = False
                calendar_message = (
                    "Calendar events have not been fetched yet or are temporarily unavailable."
                )
                incomplete_reason = calendar_message
            else:
                cal_id = (
                    settings.google_calendar.calendar_id
                    if settings is not None and hasattr(settings, "google_calendar")
                    else None
                )
                today_rows = await db.list_calendar_events(
                    time_min=start_of_today,
                    time_max=end_of_today,
                    calendar_id=cal_id,
                )
                if is_explicit_range and (
                    query_start != start_of_today or query_end != end_of_today
                ):
                    range_rows = await db.list_calendar_events(
                        time_min=query_start,
                        time_max=query_end,
                        calendar_id=cal_id,
                    )
                else:
                    range_rows = today_rows

                cur_row, next_row = await db.get_current_and_next_events(now=ref_now)
                events_today = [_format_event(r) for r in today_rows if r is not None]
                events_range = [_format_event(r) for r in range_rows if r is not None]

                # Check data completeness for requested range
                synced_until_val = await db.get_setting("calendar.synced_until")
                synced_until_dt: datetime | None = None
                if synced_until_val is not None:
                    try:
                        synced_until_dt = ensure_aware(
                            datetime.fromisoformat(str(synced_until_val))
                        )
                    except Exception:
                        pass
                if synced_until_dt is None and registry is not None:
                    cal_mod = registry.get_module("google_calendar")
                    if cal_mod is not None:
                        synced_until_dt = getattr(cal_mod, "_synced_until", None)

                if synced_until_dt is not None:
                    if query_end > synced_until_dt:
                        data_complete = False
                        incomplete_reason = (
                            f"Calendar sync is only confirmed through {synced_until_dt.strftime('%Y-%m-%d %H:%M')}, "
                            f"but requested range extends through {query_end.strftime('%Y-%m-%d')}. "
                            "Events beyond this sync point may be missing."
                        )
                else:
                    # synced_until is not explicitly stored in settings
                    tomorrow_end = datetime.combine(
                        ref_now.date() + timedelta(days=2), dt_time.min, tzinfo=ref_now.tzinfo
                    )
                    if query_end > tomorrow_end:
                        all_cached = await db.list_calendar_events(calendar_id=cal_id)
                        max_cached_end: datetime | None = None
                        for ev in all_cached:
                            try:
                                ev_end = ensure_aware(
                                    datetime.fromisoformat(
                                        str(ev.get("end_time") or ev.get("start_time"))
                                    )
                                )
                                if max_cached_end is None or ev_end > max_cached_end:
                                    max_cached_end = ev_end
                            except Exception:
                                pass
                        if max_cached_end is None or query_end > max_cached_end:
                            data_complete = False
                            max_str = (
                                max_cached_end.strftime("%Y-%m-%d")
                                if max_cached_end
                                else "today"
                            )
                            incomplete_reason = (
                                f"Calendar cache only contains events through {max_str}; requested range "
                                f"extends through {query_end.strftime('%Y-%m-%d')}. Calendar data for later dates is incomplete."
                            )

                if is_explicit_range:
                    if not events_range:
                        calendar_status = "no_events"
                        calendar_available = True
                        calendar_message = f"No calendar appointments scheduled for {range_desc}."
                    else:
                        calendar_status = "events_found"
                        calendar_available = True
                        calendar_message = f"{len(events_range)} appointment(s) scheduled for {range_desc}."
                else:
                    if not events_today:
                        calendar_status = "no_events"
                        calendar_available = True
                        calendar_message = "No calendar appointments scheduled for today."
                    else:
                        calendar_status = "events_found"
                        calendar_available = True
                        calendar_message = f"{len(events_today)} appointment(s) scheduled for today."

    calendar_data = {
        "status": calendar_status,
        "available": calendar_available,
        "data_complete": data_complete,
        "incomplete_reason": incomplete_reason,
        "message": calendar_message,
        "date": ref_now.date().isoformat(),
        "current_time": ref_now.isoformat(),
        "query_range": {
            "start": query_start.isoformat(),
            "end": query_end.isoformat(),
            "description": range_desc,
            "full_final_day_included": True,
        },
        "events": events_range,
        "events_today": events_today,
        "current_event": _format_event(cur_row),
        "next_event": _format_event(next_row),
    }

    # 4. Recent window categories and duration totals (NO raw window titles)
    recent_spans = await db.list_window_spans(limit=25)
    category_totals: dict[str, int] = {}
    recent_categories: list[str] = []
    for span in recent_spans:
        cat = str(span.get("category", "unknown"))
        if cat not in recent_categories:
            recent_categories.append(cat)
        try:
            st = ensure_aware(datetime.fromisoformat(str(span["started_at"])))
            et = (
                ensure_aware(datetime.fromisoformat(str(span["ended_at"])))
                if span.get("ended_at")
                else now
            )
            duration = max(0, int((et - st).total_seconds()))
        except (ValueError, TypeError):
            duration = 0
        category_totals[cat] = category_totals.get(cat, 0) + duration

    window_focus_data = {
        "category_totals_seconds": category_totals,
        "recent_categories": recent_categories,
    }

    # 5. Pending recommendations
    pending_recs = await db.get_pending_recommendations()
    recs_data = [
        {
            "id": r["id"],
            "policy": r["policy"],
            "message": r["message"],
            "suggested_action": r.get("suggested_action"),
        }
        for r in pending_recs
    ]

    return {
        "activity": activity_data,
        "timer": timer_data,
        "calendar": calendar_data,
        "window_focus": window_focus_data,
        "pending_recommendations": recs_data,
    }


class OllamaClient:
    """Async client communicating with local Ollama daemon at /api/chat."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:14b",
        keep_alive: str = "5m",
        timeout_seconds: float = 30.0,
        *,
        http_client: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds
        self._external_client = http_client is not None
        self._transport = transport
        self._client = http_client

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if internally owned."""
        if not self._external_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        keep_alive: str | None = None,
        think: bool = False,
        format_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send chat messages to POST /api/chat and return validated response dictionary."""
        effective_model = model or self.model
        effective_keep_alive = keep_alive or self.keep_alive
        effective_format = format_schema or RESPONSE_JSON_SCHEMA

        payload = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
            "keep_alive": effective_keep_alive,
            "think": think,
            "format": effective_format,
        }

        client = await self._get_client()
        url = f"{self.base_url}/api/chat"
        start_t = time.perf_counter()

        try:
            resp = await client.post(url, json=payload)
        except Exception as exc:
            import httpx

            if isinstance(exc, (httpx.ConnectError, ConnectionRefusedError, OSError)):
                raise LlmConnectionError(f"Ollama connection refused at {url}: {exc}") from exc
            if isinstance(exc, httpx.TimeoutException):
                raise LlmTimeoutError(
                    f"Ollama request timed out after {self.timeout_seconds}s"
                ) from exc
            raise LlmError(f"Ollama HTTP request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start_t) * 1000)

        if resp.status_code != 200:
            raise LlmError(f"Ollama returned HTTP status {resp.status_code}: {resp.text[:200]}")

        try:
            resp_data = resp.json()
        except Exception as exc:
            raise LlmValidationError(f"Ollama response was not valid JSON: {exc}") from exc

        content = resp_data.get("message", {}).get("content", "")
        cleaned_content = clean_think_tags(content)

        try:
            parsed = json.loads(cleaned_content)
        except Exception as exc:
            snippet = cleaned_content[:100]
            raise LlmValidationError(
                f"Ollama message content was not valid JSON: {exc}. Content snippet: {snippet!r}"
            ) from exc

        if format_schema is RESPONSE_JSON_SCHEMA or format_schema is None:
            validated = validate_llm_response(parsed)
            return {
                **validated,
                "model": resp_data.get("model", effective_model),
                "latency_ms": latency_ms,
                "raw": resp_data,
            }

        return {
            "parsed": parsed,
            "model": resp_data.get("model", effective_model),
            "latency_ms": latency_ms,
            "raw": resp_data,
        }

    async def ask(
        self,
        question: str,
        context: dict[str, Any] | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Ask a question given a platform context snapshot."""
        prompt_ctx = copy.deepcopy(context or {})
        q_lower = question.lower()
        asking_about_movement = any(
            w in q_lower for w in ("walk", "geh", "spazier", "step", "posture", "haltung", "bewegung")
        )
        if not asking_about_movement:
            if "calendar" in prompt_ctx and isinstance(prompt_ctx["calendar"], dict):
                for ev in prompt_ctx["calendar"].get("events", []):
                    if isinstance(ev, dict):
                        ev.pop("suitable_for_walking", None)
                for ev in prompt_ctx["calendar"].get("events_today", []):
                    if isinstance(ev, dict):
                        ev.pop("suitable_for_walking", None)

        # When asking what not to miss or what is upcoming/most important, filter out already elapsed events
        is_upcoming_or_unmissable = any(
            w in q_lower
            for w in (
                "verpass", "miss", "wichtigst", "important", "ansteh", "upcoming",
                "next", "nächste", "morgen", "tomorrow"
            )
        )
        is_explicitly_asking_past = any(
            w in q_lower
            for w in ("hatte", "hatten", "vergangen", "past", "war", "waren", "did i have")
        )
        if is_upcoming_or_unmissable and not is_explicitly_asking_past:
            if "calendar" in prompt_ctx and isinstance(prompt_ctx["calendar"], dict):
                if "events" in prompt_ctx["calendar"]:
                    upcoming = [
                        ev for ev in prompt_ctx["calendar"]["events"]
                        if isinstance(ev, dict) and ev.get("status") != "past"
                    ]
                    if upcoming:
                        prompt_ctx["calendar"]["events"] = upcoming
                if "events_today" in prompt_ctx["calendar"]:
                    upcoming_today = [
                        ev for ev in prompt_ctx["calendar"]["events_today"]
                        if isinstance(ev, dict) and ev.get("status") != "past"
                    ]
                    prompt_ctx["calendar"]["events_today"] = upcoming_today
        ctx_str = json.dumps(prompt_ctx, indent=2)
        lang_directive = (
            "- Language consistency: Always respond in the same language as the user's question "
            "(e.g. formulate the answer in natural German if the question is in German, "
            "or in English if the question is in English). Preserve calendar event titles, "
            "proper nouns, and app names in their original language.\n"
        )
        if language == "de":
            lang_directive += (
                "- Explicit user language: German ('de'). Answer in fluent, natural German.\n"
            )
        elif language == "en":
            lang_directive += (
                "- Explicit user language: English ('en'). Answer in fluent English.\n"
            )

        sys_prompt = system_prompt or (
            "You are Jarvis, a local posture, focus, and calendar assistant for Arch Linux.\n"
            "Answer the user's question accurately and concisely based on the context.\n"
            f"{lang_directive}"
            "Safety boundary: You may only answer questions and optionally suggest an action via "
            "'suggested_action' (e.g., 'start_timer', 'stand_up', 'take_break', or null). "
            "You must NEVER attempt to execute shell commands, directly change posture state, "
            "or modify settings.\n"
            "Conversational voice assistant style:\n"
            "- You speak directly to the user out loud as a personal voice assistant. Present answers in fluent,\n"
            "  natural conversational sentences instead of robotic numbered lists, raw bullet points, or parenthetical\n"
            "  database dumps (e.g. NEVER output '1. Aufstehen (07:00 – 07:15) - Coding').\n"
            "- Group appointments smoothly by time of day (morning, afternoon, evening) and transition naturally between them.\n"
            "Natural spoken time formatting (CRITICAL for text-to-speech):\n"
            "- NEVER use zero-padded digit times (e.g. NEVER write '07:00', '08:00', or '07:15').\n"
            "- In German:\n"
            "  * For full hours write 'um 7 Uhr', 'um 8 Uhr', '14 Uhr' (NEVER '07:00' or '7:00 Uhr').\n"
            "  * For time ranges write 'von 7 bis 8 Uhr' or 'von 7 Uhr bis 7:15 Uhr'.\n"
            "  * For minutes write 'um 7:15 Uhr' (or '7 Uhr 15') and 'um 8:30 Uhr' (or 'halb neun').\n"
            "- In English:\n"
            "  * For full hours write 'at 7 o'clock', 'at 8 o'clock', 'at 2 PM' (NEVER '07:00' or '7:00').\n"
            "  * For time ranges write 'from 7 to 8 o'clock' or 'from 7:00 to 7:15 AM'.\n"
            "  * For minutes write 'at 7:15 AM' and 'at 8:30 AM'.\n"
            "Calendar guidelines:\n"
            "- Current time & upcoming appointments:\n"
            "  * The current local time is given in 'current_time'. Events with status 'past' have\n"
            "    already occurred and cannot be missed ('verpassen'). When asked about what is coming up\n"
            "    or what not to miss, focus on upcoming events.\n"
            "- When calendar 'available' is false (e.g. status is disabled, not_authenticated,\n"
            "  fetch_failed, or not_fetched_yet), clearly state that calendar information is\n"
            "  unavailable or disabled and reference the reason given in the context.\n"
            "  NEVER claim or imply that the user has no appointments when calendar data\n"
            "  is unavailable.\n"
            "- When calendar 'status' is 'no_events' (available is true) and 'data_complete' is true,\n"
            "  inform the user that they have no appointments scheduled for the requested period.\n"
            "- Date range & multi-day queries:\n"
            "  * Use 'events' (or 'events_today' if query is only for today) which contains\n"
            "    all scheduled appointments within the requested date range (including the\n"
            "    full final day in the user's timezone).\n"
            "  * Check 'data_complete' and 'incomplete_reason' in the calendar context.\n"
            "  * Incomplete data requirement: If 'data_complete' is false or data is incomplete for the requested range,\n"
            "    you MUST explicitly state that the calendar data is incomplete or only\n"
            "    synced through a certain date/time, and that subsequent appointments in the\n"
            "    requested timeframe may be missing. NEVER make a definitive claim about an entire\n"
            "    period if the data is incomplete.\n"
            "- Movement suitability vs Appointment importance:\n"
            "  * 'suitable_for_walking' is purely an ergonomic/posture attribute indicating whether\n"
            "    physical movement/walking is feasible during that event. It has NO relationship to\n"
            "    appointment importance, priority, or urgency.\n"
            "  * NEVER confuse or conflate movement suitability with appointment importance.\n"
            "    Do NOT volunteer walking or movement advice when the user asks about appointment\n"
            "    importance or scheduling, unless the user explicitly asks about posture or physical activity.\n"
            "- Importance assessments:\n"
            "  * When asked to assess which appointment is the most important or unmissable ('nicht verpassen'),\n"
            "    evaluate UPCOMING appointments that have not yet ended (status != 'past'). A past event already took place\n"
            "    and cannot be missed. NEVER choose a past event as an appointment not to miss when upcoming appointments are scheduled.\n"
            "  * Medical and healthcare appointments (such as doctor visits, 'Augenarzt', clinic, dentist) must be prioritized\n"
            "    as crucial health commitments that cannot easily be rescheduled.\n"
            "  * You MUST explain the basis for your assessment (e.g. health necessity, academic deadlines)\n"
            "    and acknowledge uncertainty (importance is subjective and personal to the user).\n"
            f"Current platform context:\n{ctx_str}"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]

        return await self.chat(messages, model=model)


async def classify_event_with_llm(
    title: str = "",
    description: str | None = None,
    color_id: str | None = None,
    event: Any = None,
    *,
    client: OllamaClient | None = None,
    settings: Settings | None = None,
    db: Database | None = None,
    module_registry: Any = None,
) -> dict[str, Any] | None:
    """Implement optional LLM calendar event classification fallback (FR-16, FR-17)."""
    # 1. Check if classification fallback is enabled in settings
    if settings is not None:
        if not getattr(settings.llm, "enable_classification_fallback", False):
            return None

    # 2. Check if llm_ollama module is enabled if registry is provided
    if module_registry is not None and hasattr(module_registry, "is_enabled"):
        if not module_registry.is_enabled("llm_ollama"):
            return None

    effective_client = client or OllamaClient()
    own_client = client is None

    prompt = (
        "Classify the following calendar event into exactly one task type from: "
        "[deep_work, coding, reading, admin, email, planning, meeting, break].\n"
        f"Event title: {title}\n"
        f"Description: {description or 'None'}\n"
        f"Color ID: {color_id or 'None'}\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant categorizing calendar events into one of these task types: "
                "deep_work, coding, reading, admin, email, planning, meeting, break. "
                "Respond strictly with a JSON object containing 'task_type' and 'confidence'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        res = await effective_client.chat(messages, format_schema=CLASSIFICATION_JSON_SCHEMA)
        parsed = res.get("parsed") if isinstance(res.get("parsed"), dict) else res
        task_type = parsed.get("task_type")
        confidence = float(parsed.get("confidence", 0.8))
        if task_type in {
            "deep_work",
            "coding",
            "reading",
            "admin",
            "email",
            "planning",
            "meeting",
            "break",
        }:
            return {"task_type": task_type, "confidence": max(0.0, min(1.0, confidence))}
        return None
    except Exception as exc:
        logger.warning("classify_event_with_llm failed: %s", exc)
        return None
    finally:
        if own_client:
            await effective_client.close()


WINDOW_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["focused", "neutral", "distracted"],
            "description": "Window focus state classification.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0.0 and 1.0.",
        },
        "reason": {
            "type": "string",
            "description": "Brief reason for the classification.",
        },
    },
    "required": ["category", "confidence"],
}


async def classify_window_with_llm(
    app_class: str = "",
    window_title: str = "",
    *,
    calendar_task_type: str | None = None,
    client: OllamaClient | None = None,
    settings: Settings | None = None,
    db: Database | None = None,
    module_registry: Any = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Classify an ambiguous window into 'focused', 'neutral', or 'distracted' using a small local LLM."""
    clean_app = app_class.strip()
    clean_title = window_title.strip()
    if not clean_app and not clean_title:
        return {"category": "neutral", "confidence": 1.0}

    # 1. Check if classification fallback is enabled in settings
    if settings is not None:
        llm_enabled = getattr(settings.llm, "enable_window_classification_fallback", False)
        focus_enabled = (
            getattr(settings.focus, "enable_llm_fallback", False)
            if hasattr(settings, "focus")
            else False
        )
        if not (llm_enabled or focus_enabled):
            return None

    # 2. Check if llm_ollama module is enabled if registry is provided
    if module_registry is not None and hasattr(module_registry, "is_enabled"):
        if not module_registry.is_enabled("llm_ollama"):
            return None

    effective_model = model
    if effective_model is None and settings is not None:
        effective_model = getattr(settings.focus, "llm_model", None)
    if effective_model is None:
        effective_model = "qwen2.5:0.5b"

    effective_client = client or OllamaClient(model=effective_model)
    own_client = client is None

    cal_line = (
        f"Active Calendar Task: {calendar_task_type.strip()}\n"
        if calendar_task_type and calendar_task_type.strip() and calendar_task_type.strip().lower() != "none"
        else ""
    )

    prompt = (
        "Classify this window into exactly one category: [focused, neutral, distracted].\n"
        f"App: {clean_app or 'Unknown'}\n"
        f"Title: {clean_title or 'None'}\n"
        f"{cal_line}\n"
        "Rules:\n"
        "- focused: Work, software development, academic study/research, technical documentation, professional tools.\n"
        "- distracted: Gaming, browser games, app stores, shopping, social media, entertainment streaming, memes.\n"
        "- neutral: System utilities, audio/display controls, settings, file managers, desktop tools."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant classifying desktop window activity into exactly one of: "
                "focused, neutral, distracted. "
                "Respond strictly with a JSON object containing 'category', 'confidence', and optional 'reason'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        res = await effective_client.chat(
            messages,
            model=effective_model,
            format_schema=WINDOW_CLASSIFICATION_JSON_SCHEMA,
        )
        parsed = res.get("parsed") if isinstance(res.get("parsed"), dict) else res
        cat = parsed.get("category")
        confidence = float(parsed.get("confidence", 0.8))
        if cat in {"focused", "neutral", "distracted"}:
            return {
                "category": cat,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": parsed.get("reason", ""),
            }
        return None
    except Exception as exc:
        err_msg = str(exc)
        if "not found" in err_msg.lower() or "404" in err_msg:
            logger.warning(
                "Ollama window classification model '%s' is not installed. Run 'ollama pull %s' to enable window LLM classification.",
                effective_model,
                effective_model,
            )
        else:
            logger.warning("classify_window_with_llm failed: %s", exc)
        return None
    finally:
        if own_client:
            await effective_client.close()

