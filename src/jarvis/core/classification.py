"""Deterministic calendar-event classification module for Posture Platform (FR-16, FR-17).

Provides pure, deterministic classification of calendar events into task types,
determines Pomodoro durations and walking suitability, and supports an optional
async LLM fallback callable for ambiguous events.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from re import Pattern
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.schema import CalendarEvent, FocusState, TaskType, now_local

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


ClassifierType = Literal["explicit", "color", "regex", "llm", "default"]


class TaskClassificationResult(BaseModel):
    """Result of calendar event task classification."""

    model_config = ConfigDict(extra="ignore")

    task_type: TaskType = "unknown"
    classifier: ClassifierType = "default"
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

    def __await__(self):
        async def _identity():
            return self

        return _identity().__await__()


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
    title_or_event: CalendarEvent | dict[str, Any] | str | None = None,
    description: str | None = None,
    color_id: str | None = None,
    *,
    event: CalendarEvent | dict[str, Any] | str | None = None,
    title: str | None = None,
) -> tuple[str, str | None, str | None, CalendarEvent | dict[str, Any] | None]:
    actual_event = event if event is not None else (
        title_or_event if not isinstance(title_or_event, str) else None
    )

    ev_title = ""
    ev_desc = None
    ev_color = None
    event_obj: CalendarEvent | dict[str, Any] | None = None

    if isinstance(actual_event, CalendarEvent):
        ev_title = actual_event.title
        ev_desc = actual_event.description
        ev_color = actual_event.color_id
        event_obj = actual_event
    elif isinstance(actual_event, dict):
        ev_title = str(actual_event.get("title", ""))
        ev_desc = actual_event.get("description")
        ev_color = actual_event.get("color_id")
        event_obj = actual_event

    if isinstance(title_or_event, str):
        ev_title = title_or_event
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
    title: CalendarEvent | dict[str, Any] | str | None = None,
    description: str | None = None,
    color_id: str | None = None,
    settings: Any | None = None,
    *,
    explicit_mappings: Mapping[str, str] | None = None,
    color_mappings: Mapping[str, str] | None = None,
    regex_rules: Sequence[tuple[str | Pattern[str], str]]
    | Sequence[dict[str, str]]
    | Mapping[str, str]
    | None = None,
    pomodoro_settings: Any | None = None,
    event: CalendarEvent | dict[str, Any] | str | None = None,
) -> TaskClassificationResult:
    """Deterministically classify a calendar event using precedence steps 1, 2, 3, and 5.

    Precedence:
    1. Explicit user title/keyword mapping.
    2. Calendar color ID mapping.
    3. Title/description regex rules.
    5. 'unknown' default.
    """
    clean_title, clean_desc, clean_color, _ = _extract_event_fields(
        title_or_event=title,
        description=description,
        color_id=color_id,
        event=event,
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
    title: CalendarEvent | dict[str, Any] | str | None = None,
    description: str | None = None,
    color_id: str | None = None,
    settings: Any | None = None,
    *,
    llm_classifier: Callable[..., Any] | None = None,
    llm_fallback: Callable[..., Any] | None = None,
    explicit_mappings: Mapping[str, str] | None = None,
    color_mappings: Mapping[str, str] | None = None,
    regex_rules: Sequence[tuple[str | Pattern[str], str]]
    | Sequence[dict[str, str]]
    | Mapping[str, str]
    | None = None,
    pomodoro_settings: Any | None = None,
    event: CalendarEvent | dict[str, Any] | str | None = None,
) -> TaskClassificationResult:
    """Classify a calendar event with strict precedence:

    1. Explicit user title/keyword mapping.
    2. Calendar color ID mapping.
    3. Title/description regex rules.
    4. Optional async LLM fallback callable.
    5. 'unknown' default.
    """
    effective_llm = llm_classifier if llm_classifier is not None else llm_fallback

    result = classify_calendar_event_deterministic(
        title=title,
        description=description,
        color_id=color_id,
        settings=settings,
        explicit_mappings=explicit_mappings,
        color_mappings=color_mappings,
        regex_rules=regex_rules,
        pomodoro_settings=pomodoro_settings,
        event=event,
    )

    if result.task_type != "unknown":
        return result

    # Step 4: Optional async LLM fallback hook (only runs if deterministic result is unknown)
    if effective_llm is not None:
        raw_title, raw_desc, raw_color, event_obj = _extract_event_fields(
            title_or_event=title,
            description=description,
            color_id=color_id,
            event=event,
        )
        try:
            llm_res = await _call_llm_fallback(
                effective_llm,
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


classify_calendar_event_sync = classify_calendar_event_deterministic
classify_calendar_event.sync = classify_calendar_event_sync  # type: ignore[attr-defined]


def apply_classification_to_event(
    event: CalendarEvent,
    result: TaskClassificationResult,
) -> CalendarEvent:
    """Enrich a CalendarEvent in-place with a classification result."""
    event.task_type = result.task_type
    event.suitable_for_walking = result.suitable_for_walking
    event.recommended_pomodoro = result.recommended_pomodoro
    return event


# ---------------------------------------------------------------------------
# Window focus classification (FR-08, FR-09)
# ---------------------------------------------------------------------------

VALID_FOCUS_CATEGORIES: frozenset[FocusState] = frozenset({
    "focused",
    "neutral",
    "distracted",
    "unknown",
})

DEFAULT_WORK_APP_CLASSES: tuple[str, ...] = (
    "code",
    "Alacritty",
    "kitty",
    "obsidian",
    "Zotero",
)

DEFAULT_DISTRACTION_APP_CLASSES: tuple[str, ...] = ()

DEFAULT_NEUTRAL_APP_CLASSES: tuple[str, ...] = ()

DEFAULT_DISTRACTION_TITLE_PATTERNS: tuple[str, ...] = (
    "YouTube",
    "Reddit",
    "Twitch",
    "Twitter",
    "X",
)

DEFAULT_CALL_APP_CLASSES: tuple[str, ...] = (
    "zoom",
    "zoom.real",
    "zoommtg",
    "teams",
    "microsoft teams",
    "ms-teams",
    "teams-for-linux",
    "com.microsoft.teams",
    "webex",
    "cisco webex",
    "google-meet",
)

DEFAULT_CALL_TITLE_PATTERNS: tuple[str, ...] = (
    "Zoom Meeting",
    "Zoom Cloud Meetings",
    "Google Meet",
    "Meet -",
    "meet.google.com",
    "Microsoft Teams",
    "Teams Meeting",
    "Discord Voice",
    "Voice Connected",
    "Huddle",
)

_READING_CORRELATION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(docs|documentation|wiki|paper|papers|spec|specs|specification|specifications|"
    r"book|books|article|articles|rfc|guide|guides|manual|manuals|tutorial|tutorials|"
    r"arxiv|read|reading|chapter|reference)\b"
)

_CODING_CORRELATION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(docs|documentation|wiki|github|gitlab|pull\s*request|pr\s*#|commit|diff|"
    r"stackoverflow|stack\s*overflow|crates\.io|pypi|npmjs|devdocs|api(\s+reference)?|"
    r"debugger|godoc|arch\s*linux\s*wiki|sourcegraph|code\s*review)\b"
)

_DEEP_WORK_DISTRACTION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(youtube|twitch|reddit|twitter|x|netflix|spotify|tiktok|instagram|facebook|"
    r"steam|game|games|gaming|video|videos|stream|streams|music|podcast|podcasts)\b"
)

_DEEP_WORK_FOCUSED_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(paper|papers|thesis|research|study|studying|deep\s*work|focus|docs|documentation|spec|specs)\b"
)

_MEETING_CORRELATION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(meeting|agenda|minutes|notes|sync|standup|1:1|one-on-one|interview|retro|retrospective|huddle|call)\b"
)

_EMAIL_CORRELATION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(email|emails|gmail|outlook|thunderbird|mail|inbox|fastmail|protonmail)\b"
)

_ADMIN_CORRELATION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(admin|administration|expense|expenses|invoice|invoices|timesheet|timesheets|"
    r"jira|linear|confluence|notion|hr|portal|paperwork|compliance)\b"
)

_PLANNING_CORRELATION_REGEX: Pattern[str] = re.compile(
    r"(?i)\b(planning|plan|roadmap|backlog|sprint|jira|linear|trello|asana|board|prioritization|okr|review)\b"
)


class WindowClassificationResult(BaseModel):
    """Result of window focus classification (FR-08, FR-09)."""

    model_config = ConfigDict(extra="ignore")

    category: FocusState = "unknown"
    classifier: str = "default"
    is_call_window: bool = False
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


def _title_matches_pattern(window_title: str, pattern: str) -> bool:
    """Check if window_title matches pattern with word boundaries for short tokens like 'X'."""
    cleaned = pattern.strip()
    if not cleaned or not window_title:
        return False
    if len(cleaned) <= 2:
        token_re = rf"(?<![A-Za-z0-9]){re.escape(cleaned)}(?![A-Za-z0-9])"
        return bool(re.search(token_re, window_title, re.IGNORECASE))
    return cleaned.lower() in window_title.lower()


def is_call_window(
    app_class: str,
    window_title: str,
    *,
    call_app_classes: Sequence[str] | None = None,
    call_title_patterns: Sequence[str] | None = None,
) -> bool:
    """Return True if the window belongs to a known call or meeting interface.

    Detects common call windows such as Zoom, Google Meet, Microsoft Teams,
    Discord voice, and Slack huddles.
    """
    clean_class = str(app_class).strip().lower() if app_class is not None else ""
    clean_title = str(window_title).strip() if window_title is not None else ""
    lowered_title = clean_title.lower()

    if not clean_class and not clean_title:
        return False

    # 1. Zoom
    if clean_class in {"zoom", "zoom.real", "zoommtg"}:
        return True
    if re.search(r"(?i)\bzoom(\s+(meeting|cloud\s+meetings))?\b", clean_title):
        return True

    # 2. Google Meet
    if clean_class in {"google-meet", "meet"}:
        return True
    if "google meet" in lowered_title or "meet.google.com" in lowered_title:
        return True
    if re.search(r"(?i)\bmeet\s*-\s*[a-z0-9\-]+", clean_title):
        return True

    # 3. Microsoft Teams
    if clean_class in {
        "teams",
        "microsoft teams",
        "ms-teams",
        "teams-for-linux",
        "com.microsoft.teams",
    }:
        return True
    if "microsoft teams" in lowered_title or "teams meeting" in lowered_title:
        return True

    # 4. Discord voice (explicitly voice/call indicator, not general chat)
    is_discord_app = (
        clean_class in {"discord", "vesktop", "discord-canary", "discord-ptb"}
        or "discord" in lowered_title
    )
    if is_discord_app:
        if (
            re.search(r"(?i)\b(voice(\s+connected|\s+channel)?|call)\b", clean_title)
            or "• voice" in lowered_title
            or "[voice]" in lowered_title
            or "(voice)" in lowered_title
        ):
            return True

    # 5. Slack huddles (explicitly huddles, not general chat)
    is_slack_app = clean_class == "slack" or "slack" in lowered_title
    if is_slack_app:
        if re.search(r"(?i)\bhuddle\b", clean_title):
            return True

    # 6. Webex / Jitsi
    if (
        clean_class in {"webex", "cisco webex"}
        or "webex meeting" in lowered_title
        or "jitsi meet" in lowered_title
        or "meet.jit.si" in lowered_title
    ):
        return True

    # 7. Configured patterns
    if call_app_classes:
        for c in call_app_classes:
            c_str = str(c).strip().lower()
            if c_str and clean_class == c_str:
                return True
    if call_title_patterns:
        for p in call_title_patterns:
            if _title_matches_pattern(clean_title, str(p)):
                return True

    return False


def _correlate_calendar_task(
    clean_class: str,
    clean_title: str,
    active_task_type: str,
    *,
    is_call: bool,
) -> tuple[FocusState, float] | None:
    """Conservatively correlate an active calendar task type with the window."""
    norm_task = active_task_type.strip().lower()

    if norm_task == "reading":
        if _READING_CORRELATION_REGEX.search(clean_title) or clean_class.lower() in {
            "evince",
            "okular",
            "zathura",
            "calibre",
        }:
            return ("focused", 0.80)
        return None

    if norm_task == "coding":
        if _CODING_CORRELATION_REGEX.search(clean_title):
            return ("focused", 0.80)
        return None

    if norm_task == "deep_work":
        if _DEEP_WORK_DISTRACTION_REGEX.search(clean_title):
            return ("distracted", 0.85)
        if _DEEP_WORK_FOCUSED_REGEX.search(clean_title):
            return ("focused", 0.80)
        return None

    if norm_task == "meeting":
        if is_call or _MEETING_CORRELATION_REGEX.search(clean_title):
            return ("focused", 0.85)
        return ("neutral", 0.70)

    if norm_task == "break":
        return ("neutral", 0.80)

    if norm_task == "email":
        if _EMAIL_CORRELATION_REGEX.search(clean_title):
            return ("focused", 0.80)
        return None

    if norm_task == "admin":
        if _ADMIN_CORRELATION_REGEX.search(clean_title):
            return ("focused", 0.80)
        return None

    if norm_task == "planning":
        if _PLANNING_CORRELATION_REGEX.search(clean_title):
            return ("focused", 0.80)
        return None

    return None


def classify_window(
    app_class: str,
    window_title: str,
    *,
    active_task_type: str | None = None,
    settings: Any | None = None,
    calendar_task_type: str | None = None,
    work_app_classes: Sequence[str] | None = None,
    distraction_app_classes: Sequence[str] | None = None,
    neutral_app_classes: Sequence[str] | None = None,
    distraction_title_patterns: Sequence[str] | None = None,
    work_title_patterns: Sequence[str] | None = None,
    neutral_title_patterns: Sequence[str] | None = None,
    call_app_classes: Sequence[str] | None = None,
    call_title_patterns: Sequence[str] | None = None,
    app_class_rules: Mapping[str, str] | None = None,
) -> WindowClassificationResult:
    """Deterministically classify a window focus state (FR-08, FR-09.1).

    Classification precedence:
    1. App-class rules.
    2. Window-title rules.
    3. Active calendar task-type correlation.
    4. 'unknown' default.

    Focus categories:
    - focused
    - neutral
    - distracted
    - unknown
    """
    clean_class = str(app_class).strip() if app_class is not None else ""
    clean_title = str(window_title).strip() if window_title is not None else ""

    effective_task_type = (
        active_task_type or calendar_task_type or ""
    ).strip().lower() or None

    # Resolve settings configuration
    focus_cfgs: list[Any] = []
    if settings is not None:
        if hasattr(settings, "classification"):
            focus_cfgs.append(settings.classification)
        if hasattr(settings, "focus"):
            focus_cfgs.append(settings.focus)
        if not focus_cfgs:
            focus_cfgs.append(settings)

    def _get_cfg(field_name: str, default: Any) -> Any:
        for cfg in focus_cfgs:
            if hasattr(cfg, field_name):
                val = getattr(cfg, field_name)
                if val is not None and val != [] and val != {}:
                    return val
            elif isinstance(cfg, Mapping) and field_name in cfg:
                val = cfg[field_name]
                if val is not None and val != [] and val != {}:
                    return val
        for cfg in focus_cfgs:
            if hasattr(cfg, field_name):
                val = getattr(cfg, field_name)
                if val is not None:
                    return val
            elif isinstance(cfg, Mapping) and field_name in cfg:
                val = cfg[field_name]
                if val is not None:
                    return val
        return default

    resolved_work_apps = (
        work_app_classes
        if work_app_classes is not None
        else _get_cfg("work_app_classes", DEFAULT_WORK_APP_CLASSES)
    )
    resolved_distraction_apps = (
        distraction_app_classes
        if distraction_app_classes is not None
        else _get_cfg("distraction_app_classes", DEFAULT_DISTRACTION_APP_CLASSES)
    )
    resolved_neutral_apps = (
        neutral_app_classes
        if neutral_app_classes is not None
        else _get_cfg("neutral_app_classes", DEFAULT_NEUTRAL_APP_CLASSES)
    )
    resolved_distraction_titles = (
        distraction_title_patterns
        if distraction_title_patterns is not None
        else _get_cfg("distraction_title_patterns", DEFAULT_DISTRACTION_TITLE_PATTERNS)
    )
    resolved_work_titles = (
        work_title_patterns
        if work_title_patterns is not None
        else _get_cfg("work_title_patterns", ())
    )
    resolved_neutral_titles = (
        neutral_title_patterns
        if neutral_title_patterns is not None
        else _get_cfg("neutral_title_patterns", ())
    )
    resolved_call_apps = (
        call_app_classes
        if call_app_classes is not None
        else _get_cfg("call_app_classes", DEFAULT_CALL_APP_CLASSES)
    )
    resolved_call_titles = (
        call_title_patterns
        if call_title_patterns is not None
        else _get_cfg("call_title_patterns", DEFAULT_CALL_TITLE_PATTERNS)
    )
    resolved_app_rules = (
        app_class_rules
        if app_class_rules is not None
        else _get_cfg("app_class_rules", {})
    )

    # Call-window detection
    call_detected = is_call_window(
        clean_class,
        clean_title,
        call_app_classes=resolved_call_apps,
        call_title_patterns=resolved_call_titles,
    )

    # Empty window / desktop
    if not clean_class and not clean_title:
        return WindowClassificationResult(
            category="neutral",
            classifier="desktop",
            is_call_window=False,
            confidence=1.0,
        )

    # Precedence 1: App-class rules
    if clean_class:
        lowered_class = clean_class.lower()

        # 1a. Explicit app-class mappings
        if resolved_app_rules and lowered_class in resolved_app_rules:
            mapped_cat = str(resolved_app_rules[lowered_class]).strip().lower()
            if mapped_cat in VALID_FOCUS_CATEGORIES and mapped_cat != "unknown":
                return WindowClassificationResult(
                    category=mapped_cat,  # type: ignore[arg-type]
                    classifier="app_class",
                    is_call_window=call_detected,
                    confidence=1.0,
                )

        # 1b. Productive / work app classes
        if resolved_work_apps and any(
            lowered_class == str(c).strip().lower()
            for c in resolved_work_apps
            if str(c).strip()
        ):
            return WindowClassificationResult(
                category="focused",
                classifier="app_class",
                is_call_window=call_detected,
                confidence=0.95,
            )

        # 1c. Distraction app classes
        if resolved_distraction_apps and any(
            lowered_class == str(c).strip().lower()
            for c in resolved_distraction_apps
            if str(c).strip()
        ):
            return WindowClassificationResult(
                category="distracted",
                classifier="app_class",
                is_call_window=call_detected,
                confidence=0.95,
            )

        # 1d. Neutral app classes
        if resolved_neutral_apps and any(
            lowered_class == str(c).strip().lower()
            for c in resolved_neutral_apps
            if str(c).strip()
        ):
            return WindowClassificationResult(
                category="neutral",
                classifier="app_class",
                is_call_window=call_detected,
                confidence=0.90,
            )

    # Precedence 2: Window-title rules
    if clean_title:
        if resolved_distraction_titles and any(
            _title_matches_pattern(clean_title, str(p))
            for p in resolved_distraction_titles
            if str(p).strip()
        ):
            return WindowClassificationResult(
                category="distracted",
                classifier="title_rule",
                is_call_window=call_detected,
                confidence=0.90,
            )

        if resolved_work_titles and any(
            _title_matches_pattern(clean_title, str(p))
            for p in resolved_work_titles
            if str(p).strip()
        ):
            return WindowClassificationResult(
                category="focused",
                classifier="title_rule",
                is_call_window=call_detected,
                confidence=0.90,
            )

        if resolved_neutral_titles and any(
            _title_matches_pattern(clean_title, str(p))
            for p in resolved_neutral_titles
            if str(p).strip()
        ):
            return WindowClassificationResult(
                category="neutral",
                classifier="title_rule",
                is_call_window=call_detected,
                confidence=0.85,
            )

    # Precedence 3: Active calendar task-type correlation
    if effective_task_type and effective_task_type != "unknown":
        correlation = _correlate_calendar_task(
            clean_class,
            clean_title,
            effective_task_type,
            is_call=call_detected,
        )
        if correlation is not None:
            cor_cat, cor_conf = correlation
            return WindowClassificationResult(
                category=cor_cat,
                classifier="calendar_correlation",
                is_call_window=call_detected,
                confidence=cor_conf,
            )

    # Precedence 4: unknown default
    return WindowClassificationResult(
        category="unknown",
        classifier="default",
        is_call_window=call_detected,
        confidence=0.0,
    )
