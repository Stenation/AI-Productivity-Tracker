"""Typed event contracts and Pydantic v2 domain models for the Posture Platform."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def now_local() -> datetime:
    """Return the current local timezone-aware datetime."""
    return datetime.now().astimezone()


def ensure_aware(value: datetime) -> datetime:
    """Ensure a datetime carries timezone offset information."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.astimezone()
    return value


def get_system_boot_time() -> datetime | None:
    """Return the system boot time as a timezone-aware datetime, if detectable."""
    try:
        uptime_sec = time.clock_gettime(time.CLOCK_BOOTTIME)
        return datetime.now(timezone.utc).astimezone() - timedelta(seconds=uptime_sec)
    except (AttributeError, OSError, ValueError):
        return None


AwareDatetime = Annotated[datetime, AfterValidator(ensure_aware)]

EventType = Literal[
    "activity_reported",
    "activity_state_changed",
    "window_focus_changed",
    "input_idle_changed",
    "phone_state_changed",
    "calendar_context_changed",
    "calendar_draft_created",
    "calendar_write_approved",
    "timer_state_changed",
    "recommendation_created",
    "recommendation_resolved",
    "screen_check_requested",
    "screen_check_completed",
    "module_health_changed",
    "privacy_mode_changed",
    "assistant_answered",
    "tts_requested",
]

ActivityState = Literal[
    "sitting",
    "standing",
    "walking",
    "away",
    "unknown",
]

ManualActivityState = Literal[
    "sitting",
    "standing",
    "walking",
    "away",
]

MANUAL_ACTIVITY_CYCLE: tuple[ActivityState, ...] = (
    "sitting",
    "standing",
    "walking",
    "away",
)

FocusState = Literal[
    "focused",
    "neutral",
    "distracted",
    "unknown",
]

TaskType = Literal[
    "deep_work",
    "coding",
    "reading",
    "admin",
    "email",
    "planning",
    "meeting",
    "break",
    "unknown",
]

RecommendationStatus = Literal[
    "pending",
    "accepted",
    "snoozed",
    "dismissed",
    "expired",
]


def _maybe_parse_json_dict(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if parsed is None or isinstance(parsed, dict):
            return parsed
    return None


class PlatformEvent(BaseModel):
    """Typed pub/sub event exchanged across inputs, resolver, policies, and outputs."""

    model_config = ConfigDict(extra="ignore")

    type: EventType
    source: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: AwareDatetime = Field(default_factory=now_local)
    dedupe_key: str | None = None


class ActivitySpan(BaseModel):
    """Persisted effective activity state span."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: int | None = None
    activity: ActivityState
    source: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = "state_change"
    started_at: AwareDatetime = Field(default_factory=now_local)
    ended_at: AwareDatetime | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _unpack_metadata_json(cls, data: Any) -> Any:
        if isinstance(data, dict) and "metadata" not in data and "metadata_json" in data:
            copied = dict(data)
            copied["metadata"] = _maybe_parse_json_dict(copied.get("metadata_json"))
            return copied
        return data


class TimerSpan(BaseModel):
    """Persisted Pomodoro or custom timer span."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: int | None = None
    kind: str
    calendar_event_id: str | None = None
    started_at: AwareDatetime = Field(default_factory=now_local)
    ended_at: AwareDatetime | None = None
    planned_duration_seconds: int | None = None
    actual_duration_seconds: int | None = None
    status: str = "running"


class Recommendation(BaseModel):
    """Non-destructive policy recommendation with lifecycle status and dedupe key."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: int | None = None
    policy: str
    message: str
    suggested_action: str | None = None
    action_payload: dict[str, Any] | None = None
    dedupe_key: str | None = None
    status: RecommendationStatus = "pending"
    created_at: AwareDatetime = Field(default_factory=now_local)
    resolved_at: AwareDatetime | None = None
    resolution_metadata: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _unpack_json_columns(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied: dict[str, Any] | None = None
        if "action_payload" not in data and "action_payload_json" in data:
            copied = dict(data)
            copied["action_payload"] = _maybe_parse_json_dict(
                copied.get("action_payload_json")
            )
        if "resolution_metadata" not in data and "resolution_metadata_json" in data:
            if copied is None:
                copied = dict(data)
            copied["resolution_metadata"] = _maybe_parse_json_dict(
                copied.get("resolution_metadata_json")
            )
        return copied if copied is not None else data


class CalendarEvent(BaseModel):
    """Cached Google Calendar event enriched with task classification."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    event_id: str
    calendar_id: str = "primary"
    title: str
    description: str | None = None
    color_id: str | None = None
    task_type: TaskType = "unknown"
    suitable_for_walking: bool | None = None
    recommended_pomodoro: str | None = None
    start_time: AwareDatetime
    end_time: AwareDatetime
    raw_json: str | None = None
    updated_at: AwareDatetime = Field(default_factory=now_local)


class ModuleHealth(BaseModel):
    """Health and runtime status record for an input, policy, or output module."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    module: str
    status: str = "ok"
    enabled: bool | None = None
    category: str | None = None
    last_event_at: AwareDatetime | None = None
    last_error: str | None = None
    updated_at: AwareDatetime = Field(default_factory=now_local)


class EffectiveActivityState(BaseModel):
    """Current resolved activity state exposed by StateResolver."""

    model_config = ConfigDict(extra="ignore")

    activity: ActivityState = "unknown"
    started_at: AwareDatetime = Field(default_factory=now_local)
    elapsed_seconds: int = Field(default=0, ge=0)
    source: str = "system"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = "startup"
    override_expires_at: AwareDatetime | None = None
    privacy_paused: bool = False

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in type(self).model_fields


class SystemStateSnapshot(BaseModel):
    """Composite snapshot of current platform state for API and dashboard views."""

    model_config = ConfigDict(extra="ignore")

    activity: ActivityState = "unknown"
    source: str = "system"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = "startup"
    started_at: AwareDatetime = Field(default_factory=now_local)
    elapsed_seconds: int = Field(default=0, ge=0)
    override_expires_at: AwareDatetime | None = None
    focus: FocusState = "unknown"
    active_timer: TimerSpan | None = None
    current_event: CalendarEvent | None = None
    next_event: CalendarEvent | None = None
    pending_recommendations: list[Recommendation] = Field(default_factory=list)
    privacy_paused: bool = False
    dnd_mode: bool = False
    timestamp: AwareDatetime = Field(default_factory=now_local)


class AskRequest(BaseModel):
    """Payload for POST /api/ask."""

    model_config = ConfigDict(extra="ignore")

    question: str
    source: Literal["dashboard", "voice"] | str = "dashboard"
    speak: bool = False
    language: str | None = None
    session_id: str | None = None
    reset_session: bool = False


class AskResponse(BaseModel):
    """Response payload for POST /api/ask."""

    model_config = ConfigDict(extra="ignore")

    answer: str
    suggested_action: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    model: str
    interaction_id: int | None = None
    latency_ms: int | None = None
    session_id: str | None = None


class LlmInteraction(BaseModel):
    """Persisted record of an LLM question/answer interaction."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: int | None = None
    source: str
    question: str
    answer: str
    suggested_action: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model: str
    latency_ms: int | None = None
    metadata: dict[str, Any] | None = None
    created_at: AwareDatetime = Field(default_factory=now_local)

    @model_validator(mode="before")
    @classmethod
    def _unpack_metadata_json(cls, data: Any) -> Any:
        if isinstance(data, dict) and "metadata" not in data and "metadata_json" in data:
            copied = dict(data)
            copied["metadata"] = _maybe_parse_json_dict(copied.get("metadata_json"))
            return copied
        return data
