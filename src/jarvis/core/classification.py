"""Deterministic calendar-event classification module for Posture Platform (FR-16, FR-17).

Provides pure, deterministic classification of calendar events into task types,
determines Pomodoro durations and walking suitability, and supports an optional
async LLM fallback callable for ambiguous events.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta
from re import Pattern
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.schema import CalendarEvent, TaskType, now_local

logger = logging.getLogger(__name__)

VALID_TASK_TYPES: frozenset[TaskType] = frozenset({
    "deep_work",
    "coding",
    "reading",
    "admin",
    "email",
    "planning",
    "meeting",
    "break",
    "unknown",
})

DEFAULT_POMODORO_MAPPING: dict[str, str | None] = {
    "deep_work": "50_10",
    "coding": "50_10",
    "reading": "50_10",
    "admin": "25_5",
    "email": "25_5",
    "planning": "25_5",
    "meeting": None,
    "break": None,
    "unknown": None,
}

DEFAULT_WALKING_SUITABILITY: dict[str, bool | None] = {
    "admin": True,
    "email": True,
    "planning": True,
    "break": True,
    "deep_work": False,
    "coding": False,
    "meeting": False,
    "reading": False,
    "unknown": None,
}

DEFAULT_COLOR_MAPPINGS: dict[str, str] = {
    "1": "reading",
    "2": "deep_work",
    "3": "planning",
    "4": "break",
    "5": "email",
    "6": "admin",
    "7": "meeting",
    "8": "coding",
    "9": "reading",
    "10": "coding",
    "11": "deep_work",
}

DEFAULT_REGEX_RULES: list[tuple[Pattern[str], str]] = [
    (
        re.compile(
            r"(?i)\b(1:1|one-on-one|sync|standup|meeting|interview|retro|retrospective|huddle|call|debrief|presentation|demo)\b"
        ),
        "meeting",
    ),
    (
        re.compile(
            r"(?i)\b(break|lunch|dinner|breakfast|coffee|snack|stretch|rest|pause)\b"
        ),
        "break",
    ),
    (
        re.compile(
            r"(?i)\b(deep\s*work|focus\s*block|focus|study|studying|research|thesis|writing)\b"
        ),
        "deep_work",
    ),
    (
        re.compile(
            r"(?i)\b(code|coding|programming|refactor|refactoring|pr\s*review"
            r"|pull\s*request|debug|debugging|dev|frontend|backend|bug|bugs|bugfix)\b"
        ),
        "coding",
    ),
    (
        re.compile(r"(?i)\b(email|emails|inbox\s*zero|inbox|triage\s*inbox|mail)\b"),
        "email",
    ),
    (
        re.compile(
            r"(?i)\b(admin|administration|paperwork|expense|expenses|invoices?|timesheet|compliance|hr)\b"
        ),
        "admin",
    ),
    (
        re.compile(
            r"(?i)\b(planning|plan|sprint\s*plan|sprint\s*planning|roadmap|backlog|prioritization|weekly\s*review|okr)\b"
        ),
        "planning",
    ),
    (
        re.compile(
            r"(?i)\b(read|reading|book|audiobook|article|docs|documentation|paper|spec|specs)\b"
        ),
        "reading",
    ),
]

WALKING_OR_AUDIOBOOK_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(walk|walking|walker|walkingpad|walking-pad|treadmill|audiobook|audiobooks|audio-book|audio\s+book|audible|listen|listening)\b"
)


class TaskClassificationResult(BaseModel):
    """Result of calendar event task classification."""

    model_config = ConfigDict(extra="ignore")

    task_type: TaskType = "unknown"
    classifier: str = "default"
    suitable_for_walking: bool | None = None
    recommended_pomodoro: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=now_local)

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in type(self).model_fields


def is_walking_or_audiobook(text: str) -> bool:
    """Return True if text explicitly indicates walking or audiobook."""
    if not text:
        return False
    return bool(WALKING_OR_AUDIOBOOK_REGEX.search(text))


def get_recommended_pomodoro(
    task_type: str,
    pomodoro_settings: Any | None = None,
) -> str | None:
    """Return the recommended Pomodoro duration preset for a task type.

    Pomodoro mapping:
    - deep_work, coding, reading -> "50_10"
    - admin, email, planning -> "25_5"
    - meeting, break, unknown -> None
    """
    norm = str(task_type).strip().lower()
    if pomodoro_settings is not None and hasattr(pomodoro_settings, "get"):
        return pomodoro_settings.get(norm)
    return DEFAULT_POMODORO_MAPPING.get(norm)


def is_suitable_for_walking(
    task_type: str,
    *,
    title: str = "",
    description: str = "",
) -> bool | None:
    """Determine whether a task type is suitable for walking.

    Rules:
    - admin, email, planning, break -> True
    - deep_work, coding, meeting -> False
    - reading -> False unless title/description explicitly indicates walking or audiobook
    - unknown -> None
    """
    norm = str(task_type).strip().lower()
    if norm in {"admin", "email", "planning", "break"}:
        return True
    if norm in {"deep_work", "coding", "meeting"}:
        return False
    if norm == "reading":
        combined = f"{title}\n{description}".strip()
        return is_walking_or_audiobook(combined)
    return None


def _compile_regex_rules(
    rules: Sequence[tuple[str | Pattern[str], str]]
    | Sequence[dict[str, str]]
    | Mapping[str, str]
    | None,
) -> list[tuple[Pattern[str], str]]:
    """Compile rules into (Pattern, task_type) tuples."""
    if rules is None:
        return list(DEFAULT_REGEX_RULES)

    compiled: list[tuple[Pattern[str], str]] = []
    if isinstance(rules, Mapping):
        for pattern_str, task_type in rules.items():
            compiled.append((
                re.compile(pattern_str, re.IGNORECASE),
                str(task_type).strip().lower(),
            ))
        return compiled

    for item in rules:
        if isinstance(item, dict):
            pat = item.get("pattern", "")
            t_type = item.get("task_type", "unknown")
            if pat:
                compiled.append((
                    re.compile(str(pat), re.IGNORECASE),
                    str(t_type).strip().lower(),
                ))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            pat, t_type = item
            if isinstance(pat, Pattern):
                compiled.append((pat, str(t_type).strip().lower()))
            else:
                compiled.append((re.compile(str(pat), re.IGNORECASE), str(t_type).strip().lower()))

    return compiled if compiled else list(DEFAULT_REGEX_RULES)


def _match_explicit_mapping(
    title: str,
    explicit_mappings: Mapping[str, str],
) -> str | None:
    """Match explicit user mappings against event title with exact match first, then keyword."""
    clean_title = title.strip()
    if not clean_title or not explicit_mappings:
        return None

    lowered_title = clean_title.casefold()

    # Pass 1: Exact title match across all configured keys
    for key, task_type in explicit_mappings.items():
        if key.strip().casefold() == lowered_title:
            norm = str(task_type).strip().lower()
            if norm in VALID_TASK_TYPES:
                return norm

    # Pass 2: Keyword match in title (longer keys prioritized for specificity)
    sorted_mappings = sorted(
        explicit_mappings.items(),
        key=lambda item: len(str(item[0]).strip()),
        reverse=True,
    )
    for key, task_type in sorted_mappings:
        k_clean = str(key).strip()
        if not k_clean:
            continue
        pattern = rf"(?<!\w){re.escape(k_clean)}(?!\w)"
        if re.search(pattern, clean_title, re.IGNORECASE):
            norm = str(task_type).strip().lower()
            if norm in VALID_TASK_TYPES:
                return norm

    return None


def _extract_event_fields(
    event: CalendarEvent | dict[str, Any] | str | None,
    title: str | None,
    description: str | None,
    color_id: str | None,
) -> tuple[str, str | None, str | None, CalendarEvent | dict[str, Any] | None]:
    ev_title = ""
    ev_desc = None
    ev_color = None
    event_obj: CalendarEvent | dict[str, Any] | None = None

    if isinstance(event, CalendarEvent):
        ev_title = event.title
        ev_desc = event.description
        ev_color = event.color_id
        event_obj = event
    elif isinstance(event, dict):
        ev_title = str(event.get("title", ""))
        ev_desc = event.get("description")
        ev_color = event.get("color_id")
        event_obj = event
    elif isinstance(event, str):
        ev_title = event

    if title is not None:
        ev_title = title
    if description is not None:
        ev_desc = description
    if color_id is not None:
        ev_color = color_id

    if event_obj is None and (ev_title or ev_desc or ev_color):
        now = now_local()
        event_obj = CalendarEvent(
            event_id="transient",
            title=ev_title,
            description=ev_desc,
            color_id=str(ev_color) if ev_color is not None else None,
            start_time=now,
            end_time=now + timedelta(hours=1),
        )

    return ev_title, ev_desc, str(ev_color) if ev_color is not None else None, event_obj


def classify_calendar_event_deterministic(
    event: CalendarEvent | dict[str, Any] | str | None = None,
    *,
    title: str | None = None,
    description: str | None = None,
    color_id: str | None = None,
    explicit_mappings: Mapping[str, str] | None = None,
    color_mappings: Mapping[str, str] | None = None,
    regex_rules: Sequence[tuple[str | Pattern[str], str]]
    | Sequence[dict[str, str]]
    | Mapping[str, str]
    | None = None,
    settings: Any | None = None,
    pomodoro_settings: Any | None = None,
) -> TaskClassificationResult:
    """Deterministically classify a calendar event using precedence steps 1, 2, 3, and 5.

    Precedence:
    1. Explicit user title/keyword mapping.
    2. Calendar color ID mapping.
    3. Title/description regex rules.
    5. 'unknown' default.
    """
    clean_title, clean_desc, clean_color, _ = _extract_event_fields(
        event, title, description, color_id
    )

    # Resolve settings if provided
    cal_settings = None
    if settings is not None:
        if hasattr(settings, "calendar_classification"):
            cal_settings = settings.calendar_classification
        elif (
            hasattr(settings, "google_calendar")
            and hasattr(settings.google_calendar, "classification")
        ):
            cal_settings = settings.google_calendar.classification
        elif hasattr(settings, "explicit_mappings"):
            cal_settings = settings

        if pomodoro_settings is None and hasattr(settings, "pomodoro"):
            pomodoro_settings = settings.pomodoro

    active_explicit = (
        explicit_mappings
        if explicit_mappings is not None
        else (getattr(cal_settings, "explicit_mappings", {}) if cal_settings else {})
    )
    active_color_map = (
        color_mappings
        if color_mappings is not None
        else (
            getattr(cal_settings, "color_mappings", DEFAULT_COLOR_MAPPINGS)
            if cal_settings
            else DEFAULT_COLOR_MAPPINGS
        )
    )
    active_regex_rules = (
        regex_rules
        if regex_rules is not None
        else (getattr(cal_settings, "regex_rules", None) if cal_settings else None)
    )
    compiled_regexes = _compile_regex_rules(active_regex_rules)

    # Step 1: Explicit user title/keyword mapping
    if active_explicit:
        explicit_task = _match_explicit_mapping(clean_title, active_explicit)
        if explicit_task is not None and explicit_task != "unknown":
            return TaskClassificationResult(
                task_type=explicit_task,  # type: ignore[arg-type]
                classifier="explicit",
                suitable_for_walking=is_suitable_for_walking(
                    explicit_task, title=clean_title, description=clean_desc or ""
                ),
                recommended_pomodoro=get_recommended_pomodoro(
                    explicit_task, pomodoro_settings=pomodoro_settings
                ),
                confidence=1.0,
            )

    # Step 2: Calendar color ID mapping
    if clean_color and active_color_map:
        color_key = str(clean_color).strip()
        mapped_task = active_color_map.get(color_key)
        if mapped_task is not None:
            norm_task = str(mapped_task).strip().lower()
            if norm_task in VALID_TASK_TYPES and norm_task != "unknown":
                return TaskClassificationResult(
                    task_type=norm_task,  # type: ignore[arg-type]
                    classifier="color",
                    suitable_for_walking=is_suitable_for_walking(
                        norm_task, title=clean_title, description=clean_desc or ""
                    ),
                    recommended_pomodoro=get_recommended_pomodoro(
                        norm_task, pomodoro_settings=pomodoro_settings
                    ),
                    confidence=0.95,
                )

    # Step 3: Title/description regex rules
    search_texts = [clean_title]
    if clean_desc:
        search_texts.append(clean_desc)

    for pattern, rule_task in compiled_regexes:
        for text in search_texts:
            if text and pattern.search(text):
                norm_task = rule_task.strip().lower()
                if norm_task in VALID_TASK_TYPES and norm_task != "unknown":
                    return TaskClassificationResult(
                        task_type=norm_task,  # type: ignore[arg-type]
                        classifier="regex",
                        suitable_for_walking=is_suitable_for_walking(
                            norm_task, title=clean_title, description=clean_desc or ""
                        ),
                        recommended_pomodoro=get_recommended_pomodoro(
                            norm_task, pomodoro_settings=pomodoro_settings
                        ),
                        confidence=0.85,
                    )

    # Step 5: unknown default
    return TaskClassificationResult(
        task_type="unknown",
        classifier="default",
        suitable_for_walking=None,
        recommended_pomodoro=None,
        confidence=0.0,
    )


async def _call_llm_fallback(
    llm_callable: Callable[..., Any],
    *,
    event: Any,
    title: str,
    description: str | None,
    color_id: str | None,
) -> Any:
    sig = None
    try:
        sig = inspect.signature(llm_callable)
    except (ValueError, TypeError):
        pass

    res: Any = None
    if sig is not None:
        params = sig.parameters
        kwargs_to_pass: dict[str, Any] = {}
        if "title" in params:
            kwargs_to_pass["title"] = title
        if "description" in params:
            kwargs_to_pass["description"] = description
        if "color_id" in params:
            kwargs_to_pass["color_id"] = color_id
        if "event" in params:
            kwargs_to_pass["event"] = event

        if kwargs_to_pass:
            res = llm_callable(**kwargs_to_pass)
        else:
            pos_params = [
                p
                for p in params.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_var_positional = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
            )
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )

            if len(pos_params) == 0:
                if has_var_positional or has_var_keyword:
                    res = llm_callable(event if event is not None else title)
                else:
                    res = llm_callable()
            elif len(pos_params) == 1:
                first_name = pos_params[0].name.lower()
                if "title" in first_name or "text" in first_name:
                    res = llm_callable(title)
                else:
                    res = llm_callable(event if event is not None else title)
            elif len(pos_params) == 2:
                res = llm_callable(title, description)
            else:
                try:
                    res = llm_callable(event if event is not None else title)
                except TypeError:
                    res = llm_callable(title, description)
    else:
        try:
            res = llm_callable(event if event is not None else title)
        except TypeError:
            res = llm_callable(title=title, description=description)

    if inspect.isawaitable(res):
        return await res
    return res


def _normalize_llm_result(
    llm_res: Any,
    *,
    title: str,
    description: str,
    pomodoro_settings: Any | None = None,
) -> TaskClassificationResult | None:
    if llm_res is None:
        return None
    if isinstance(llm_res, TaskClassificationResult):
        return llm_res
    if isinstance(llm_res, dict):
        raw_task = str(llm_res.get("task_type", "")).strip().lower()
        if raw_task in VALID_TASK_TYPES and raw_task != "unknown":
            conf = float(llm_res.get("confidence", 0.8))
            conf = max(0.0, min(1.0, conf))
            return TaskClassificationResult(
                task_type=raw_task,  # type: ignore[arg-type]
                classifier="llm",
                suitable_for_walking=is_suitable_for_walking(
                    raw_task, title=title, description=description
                ),
                recommended_pomodoro=get_recommended_pomodoro(raw_task, pomodoro_settings),
                confidence=conf,
            )
        return None
    if isinstance(llm_res, str):
        clean = llm_res.strip().lower()
        if clean in VALID_TASK_TYPES and clean != "unknown":
            return TaskClassificationResult(
                task_type=clean,  # type: ignore[arg-type]
                classifier="llm",
                suitable_for_walking=is_suitable_for_walking(
                    clean, title=title, description=description
                ),
                recommended_pomodoro=get_recommended_pomodoro(clean, pomodoro_settings),
                confidence=0.8,
            )
    return None


async def classify_calendar_event(
    event: CalendarEvent | dict[str, Any] | str | None = None,
    *,
    title: str | None = None,
    description: str | None = None,
    color_id: str | None = None,
    explicit_mappings: Mapping[str, str] | None = None,
    color_mappings: Mapping[str, str] | None = None,
    regex_rules: Sequence[tuple[str | Pattern[str], str]]
    | Sequence[dict[str, str]]
    | Mapping[str, str]
    | None = None,
    llm_fallback: Callable[..., Awaitable[Any]] | None = None,
    settings: Any | None = None,
    pomodoro_settings: Any | None = None,
) -> TaskClassificationResult:
    """Classify a calendar event with strict precedence:

    1. Explicit user title/keyword mapping.
    2. Calendar color ID mapping.
    3. Title/description regex rules.
    4. Optional async LLM fallback callable.
    5. 'unknown' default.
    """
    result = classify_calendar_event_deterministic(
        event=event,
        title=title,
        description=description,
        color_id=color_id,
        explicit_mappings=explicit_mappings,
        color_mappings=color_mappings,
        regex_rules=regex_rules,
        settings=settings,
        pomodoro_settings=pomodoro_settings,
    )

    if result.task_type != "unknown":
        return result

    # Step 4: Optional async LLM fallback callable (only runs if deterministic result is unknown)
    if llm_fallback is not None:
        raw_title, raw_desc, raw_color, event_obj = _extract_event_fields(
            event, title, description, color_id
        )
        try:
            llm_res = await _call_llm_fallback(
                llm_fallback,
                event=event_obj,
                title=raw_title,
                description=raw_desc,
                color_id=raw_color,
            )
            parsed_result = _normalize_llm_result(
                llm_res,
                title=raw_title,
                description=raw_desc or "",
                pomodoro_settings=pomodoro_settings
                or (getattr(settings, "pomodoro", None) if settings is not None else None),
            )
            if parsed_result is not None and parsed_result.task_type != "unknown":
                return parsed_result
        except Exception as exc:
            logger.warning("Calendar classification LLM fallback failed: %s", exc, exc_info=True)

    # Step 5: unknown default
    return result


def apply_classification_to_event(
    event: CalendarEvent,
    result: TaskClassificationResult,
) -> CalendarEvent:
    """Enrich a CalendarEvent in-place with a classification result."""
    event.task_type = result.task_type
    event.suitable_for_walking = result.suitable_for_walking
    event.recommended_pomodoro = result.recommended_pomodoro
    return event
