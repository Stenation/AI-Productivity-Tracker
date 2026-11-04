"""Unit tests for deterministic calendar task classification (FR-16, FR-17)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from jarvis.core.classification import (
    DEFAULT_COLOR_MAPPINGS,
    DEFAULT_POMODORO_MAPPING,
    TaskClassificationResult,
    apply_classification_to_event,
    classify_calendar_event,
    classify_calendar_event_deterministic,
    get_recommended_pomodoro,
    is_suitable_for_walking,
    is_walking_or_audiobook,
)
from jarvis.core.schema import CalendarEvent, TaskType
from jarvis.core.settings import CalendarClassificationSettings, PomodoroSettings, Settings


def _create_event(
    title: str,
    description: str | None = None,
    color_id: str | None = None,
) -> CalendarEvent:
    now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)
    return CalendarEvent(
        event_id="test_event_1",
        title=title,
        description=description,
        color_id=color_id,
        start_time=now,
        end_time=now + timedelta(hours=1),
    )


# 1. Explicit mapping precedence tests


@pytest.mark.asyncio
async def test_explicit_mapping_precedence_over_color() -> None:
    # Explicit mapping takes precedence over color ID
    event = _create_event("Daily Standup", color_id="1")  # color "1" -> reading
    explicit_map = {"Daily Standup": "meeting"}

    result = await classify_calendar_event(
        event,
        explicit_mappings=explicit_map,
    )
    assert result.task_type == "meeting"
    assert result.classifier == "explicit"
    assert result.confidence == 1.0
    assert result.recommended_pomodoro is None
    assert result.suitable_for_walking is False


@pytest.mark.asyncio
async def test_explicit_mapping_precedence_over_regex() -> None:
    # Title "Code Review" matches regex "coding", but explicit mapping maps to "planning"
    event = _create_event("Code Review")
    explicit_map = {"Code Review": "planning"}

    result = await classify_calendar_event(
        event,
        explicit_mappings=explicit_map,
    )
    assert result.task_type == "planning"
    assert result.classifier == "explicit"
    assert result.confidence == 1.0
    assert result.recommended_pomodoro == "25_5"
    assert result.suitable_for_walking is True


@pytest.mark.asyncio
async def test_explicit_keyword_mapping_in_title() -> None:
    # Keyword match in title takes precedence
    event = _create_event("1:1 with Engineering Lead")
    explicit_map = {"1:1": "meeting"}

    result = await classify_calendar_event(
        event,
        explicit_mappings=explicit_map,
    )
    assert result.task_type == "meeting"
    assert result.classifier == "explicit"
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_explicit_mapping_precedence_over_llm_fallback() -> None:
    mock_llm = AsyncMock(return_value="coding")
    event = _create_event("Weekly Standup")
    explicit_map = {"Weekly Standup": "meeting"}

    result = await classify_calendar_event(
        event,
        explicit_mappings=explicit_map,
        llm_fallback=mock_llm,
    )
    assert result.task_type == "meeting"
    assert result.classifier == "explicit"
    mock_llm.assert_not_called()
    mock_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_mapping_via_settings() -> None:
    settings = Settings(
        calendar_classification=CalendarClassificationSettings(
            explicit_mappings={"Triage Session": "email"}
        )
    )
    event = _create_event("Triage Session")

    result = await classify_calendar_event(event, settings=settings)
    assert result.task_type == "email"
    assert result.classifier == "explicit"
    assert result.recommended_pomodoro == "25_5"
    assert result.suitable_for_walking is True


# 2. Color mapping precedence tests


@pytest.mark.asyncio
async def test_color_mapping_precedence_over_regex() -> None:
    # Title matches regex for coding, but color_id="1" maps to reading
    event = _create_event("Refactor database schema", color_id="1")

    result = await classify_calendar_event(event)
    assert result.task_type == "reading"
    assert result.classifier == "color"
    assert result.confidence == 0.95
    assert result.recommended_pomodoro == "50_10"
    assert result.suitable_for_walking is False


@pytest.mark.asyncio
async def test_color_mapping_precedence_over_llm_fallback() -> None:
    mock_llm = AsyncMock(return_value="deep_work")
    # Title has no regex match, color_id="7" maps to meeting
    event = _create_event("General Sync Session", color_id="7")

    result = await classify_calendar_event(event, llm_fallback=mock_llm)
    assert result.task_type == "meeting"
    assert result.classifier == "color"
    assert result.confidence == 0.95
    mock_llm.assert_not_called()
    mock_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_color_mapping_fallthrough_on_unmapped_color() -> None:
    # color_id="999" is unmapped, so regex should handle it
    event = _create_event("Daily Standup", color_id="999")

    result = await classify_calendar_event(event)
    assert result.task_type == "meeting"
    assert result.classifier == "regex"
    assert result.confidence == 0.85


@pytest.mark.asyncio
async def test_standard_color_mappings() -> None:
    for color_id, expected_task in DEFAULT_COLOR_MAPPINGS.items():
        event = _create_event("Unspecified Block", color_id=color_id)
        result = await classify_calendar_event(event)
        assert result.task_type == expected_task
        assert result.classifier == "color"


# 3. Regex classification tests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("title", "expected_task", "expected_pomodoro", "expected_walking"),
    [
        ("Deep Work: Distributed Systems", "deep_work", "50_10", False),
        ("Focus block for research", "deep_work", "50_10", False),
        ("Study paper on Raft consensus", "deep_work", "50_10", False),
        ("Writing dissertation chapter", "deep_work", "50_10", False),
        ("Refactor user auth backend", "coding", "50_10", False),
        ("Fix bug in API gateway", "coding", "50_10", False),
        ("PR review #402", "coding", "50_10", False),
        ("Frontend feature implementation", "coding", "50_10", False),
        ("Reading: Clean Architecture", "reading", "50_10", False),
        ("Read API specifications", "reading", "50_10", False),
        ("Submit monthly expense report", "admin", "25_5", True),
        ("HR timesheet submission", "admin", "25_5", True),
        ("Email triage and replies", "email", "25_5", True),
        ("Inbox Zero session", "email", "25_5", True),
        ("Sprint Planning Q1", "planning", "25_5", True),
        ("Roadmap and backlog prioritization", "planning", "25_5", True),
        ("Weekly review and OKR check", "planning", "25_5", True),
        ("Team Sync Call", "meeting", None, False),
        ("Daily standup meeting", "meeting", None, False),
        ("Client interview huddle", "meeting", None, False),
        ("Lunch break", "break", None, True),
        ("Coffee and stretch break", "break", None, True),
    ],
)
async def test_regex_rules_for_task_types(
    title: str,
    expected_task: TaskType,
    expected_pomodoro: str | None,
    expected_walking: bool | None,
) -> None:
    event = _create_event(title)
    result = await classify_calendar_event(event)
    assert result.task_type == expected_task
    assert result.classifier == "regex"
    assert result.confidence == 0.85
    assert result.recommended_pomodoro == expected_pomodoro
    assert result.suitable_for_walking == expected_walking


@pytest.mark.asyncio
async def test_regex_classification_on_description() -> None:
    # Generic title, but description clearly matches planning regex
    event = _create_event(
        title="Engineering Block",
        description="Sprint planning and backlog grooming with tech lead",
    )
    result = await classify_calendar_event(event)
    assert result.task_type == "planning"
    assert result.classifier == "regex"


@pytest.mark.asyncio
async def test_regex_precedence_over_llm_fallback() -> None:
    mock_llm = AsyncMock(return_value="meeting")
    event = _create_event("Frontend refactoring session")

    result = await classify_calendar_event(event, llm_fallback=mock_llm)
    assert result.task_type == "coding"
    assert result.classifier == "regex"
    mock_llm.assert_not_called()
    mock_llm.assert_not_awaited()


# 4. LLM fallback success and failure tests


@pytest.mark.asyncio
async def test_llm_fallback_success_string() -> None:
    mock_llm = AsyncMock(return_value="coding")
    event = _create_event("Quantum Flux Calibration")  # ambiguous title

    result = await classify_calendar_event(event, llm_fallback=mock_llm)
    assert result.task_type == "coding"
    assert result.classifier == "llm"
    assert result.confidence == 0.8
    assert result.recommended_pomodoro == "50_10"
    assert result.suitable_for_walking is False
    mock_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_fallback_success_dict() -> None:
    mock_llm = AsyncMock(return_value={"task_type": "deep_work", "confidence": 0.92})
    event = _create_event("Quantum Flux Calibration")

    result = await classify_calendar_event(event, llm_fallback=mock_llm)
    assert result.task_type == "deep_work"
    assert result.classifier == "llm"
    assert result.confidence == 0.92
    assert result.recommended_pomodoro == "50_10"
    assert result.suitable_for_walking is False


@pytest.mark.asyncio
async def test_llm_fallback_success_result_object() -> None:
    custom_res = TaskClassificationResult(
        task_type="admin",
        classifier="llm",
        suitable_for_walking=True,
        recommended_pomodoro="25_5",
        confidence=0.88,
    )
    mock_llm = AsyncMock(return_value=custom_res)
    event = _create_event("Quantum Flux Calibration")

    result = await classify_calendar_event(event, llm_fallback=mock_llm)
    assert result.task_type == "admin"
    assert result.classifier == "llm"
    assert result.confidence == 0.88


@pytest.mark.asyncio
async def test_llm_fallback_exception_handling(caplog: pytest.LogCaptureFixture) -> None:
    async def faulty_llm(*args: object, **kwargs: object) -> str:
        raise RuntimeError("Ollama daemon unreachable at 127.0.0.1:11434")

    event = _create_event("Quantum Flux Calibration")

    with caplog.at_level(logging.WARNING):
        result = await classify_calendar_event(event, llm_fallback=faulty_llm)

    assert result.task_type == "unknown"
    assert result.classifier == "default"
    assert result.confidence == 0.0
    assert result.recommended_pomodoro is None
    assert result.suitable_for_walking is None
    assert "LLM fallback failed" in caplog.text


@pytest.mark.asyncio
async def test_llm_fallback_returns_unknown_or_invalid() -> None:
    mock_unknown = AsyncMock(return_value="unknown")
    event = _create_event("Quantum Flux Calibration")

    res1 = await classify_calendar_event(event, llm_fallback=mock_unknown)
    assert res1.task_type == "unknown"
    assert res1.classifier == "default"

    mock_invalid = AsyncMock(return_value="non_existent_type")
    res2 = await classify_calendar_event(event, llm_fallback=mock_invalid)
    assert res2.task_type == "unknown"
    assert res2.classifier == "default"


@pytest.mark.asyncio
async def test_llm_fallback_signature_adaptation() -> None:
    async def llm_with_title_and_desc(title: str, description: str | None = None) -> str:
        assert title == "Custom Event Title"
        assert description == "Custom Event Description"
        return "planning"

    result = await classify_calendar_event(
        title="Custom Event Title",
        description="Custom Event Description",
        llm_fallback=llm_with_title_and_desc,
    )
    assert result.task_type == "planning"
    assert result.classifier == "llm"


# 5. Pomodoro mapping tests for every task type


@pytest.mark.parametrize(
    ("task_type", "expected_pomodoro"),
    [
        ("deep_work", "50_10"),
        ("coding", "50_10"),
        ("reading", "50_10"),
        ("admin", "25_5"),
        ("email", "25_5"),
        ("planning", "25_5"),
        ("meeting", None),
        ("break", None),
        ("unknown", None),
    ],
)
def test_pomodoro_mapping_for_every_task_type(
    task_type: TaskType,
    expected_pomodoro: str | None,
) -> None:
    assert get_recommended_pomodoro(task_type) == expected_pomodoro
    assert DEFAULT_POMODORO_MAPPING[task_type] == expected_pomodoro


def test_pomodoro_mapping_with_custom_settings() -> None:
    pomo_settings = PomodoroSettings(deep_work="90_15", admin="15_3")
    assert get_recommended_pomodoro("deep_work", pomo_settings) == "90_15"
    assert get_recommended_pomodoro("admin", pomo_settings) == "15_3"
    assert get_recommended_pomodoro("coding", pomo_settings) == "50_10"
    assert get_recommended_pomodoro("meeting", pomo_settings) is None


# 6. Walking suitability tests for every task type


@pytest.mark.parametrize(
    ("task_type", "expected_suitability"),
    [
        ("admin", True),
        ("email", True),
        ("planning", True),
        ("break", True),
        ("deep_work", False),
        ("coding", False),
        ("meeting", False),
        ("unknown", None),
    ],
)
def test_walking_suitability_for_every_task_type(
    task_type: TaskType,
    expected_suitability: bool | None,
) -> None:
    assert is_suitable_for_walking(task_type) == expected_suitability


def test_reading_walking_suitability_conditions() -> None:
    # Default reading without walking/audiobook keywords -> False
    assert is_suitable_for_walking("reading", title="Reading Design Patterns") is False
    assert (
        is_suitable_for_walking(
            "reading",
            title="Read book chapter",
            description="Reading at my desk with coffee",
        )
        is False
    )

    # Reading with walking in title -> True
    assert (
        is_suitable_for_walking("reading", title="Reading papers while walking")
        is True
    )

    # Reading with audiobook in title -> True
    assert (
        is_suitable_for_walking(
            "reading", title="Audiobook: Designing Data-Intensive Applications"
        )
        is True
    )
    assert is_suitable_for_walking("reading", title="Read audio book chapter 3") is True

    # Reading with walking pad / treadmill in title or description -> True
    assert (
        is_suitable_for_walking(
            "reading",
            title="Read documentation",
            description="Reading on the walking pad",
        )
        is True
    )
    assert (
        is_suitable_for_walking(
            "reading",
            title="Read arXiv papers",
            description="Treadmill reading block",
        )
        is True
    )

    # Reading with audible or listen in title or description -> True
    assert is_suitable_for_walking("reading", title="Audible: Pragmatic Programmer") is True
    assert (
        is_suitable_for_walking(
            "reading",
            title="Tech book",
            description="Listening to chapter 4",
        )
        is True
    )

    # Non-reading tasks ignore walking keywords per SPEC rules
    assert (
        is_suitable_for_walking("coding", title="Coding while walking on treadmill")
        is False
    )
    assert (
        is_suitable_for_walking("deep_work", title="Deep work on walking pad")
        is False
    )
    assert (
        is_suitable_for_walking("meeting", title="Walking 1:1 meeting")
        is False
    )


# 7. Unknown fallback behavior


@pytest.mark.asyncio
async def test_unknown_fallback_behavior() -> None:
    # Event with meaningless title and no color or fallback
    event = _create_event("XYZ-998811-Q")

    result = await classify_calendar_event(event)
    assert result.task_type == "unknown"
    assert result.classifier == "default"
    assert result.confidence == 0.0
    assert result.recommended_pomodoro is None
    assert result.suitable_for_walking is None


@pytest.mark.asyncio
async def test_empty_event_fallback() -> None:
    result = await classify_calendar_event("")
    assert result.task_type == "unknown"
    assert result.classifier == "default"
    assert result.confidence == 0.0
    assert result.recommended_pomodoro is None
    assert result.suitable_for_walking is None


# 8. Pure module, helper, and model tests


def test_task_classification_result_dict_access() -> None:
    res = TaskClassificationResult(
        task_type="deep_work",
        classifier="regex",
        suitable_for_walking=False,
        recommended_pomodoro="50_10",
        confidence=0.85,
    )
    assert res["task_type"] == "deep_work"
    assert res.get("classifier") == "regex"
    assert res.get("non_existent", "default_val") == "default_val"
    assert "task_type" in res

    with pytest.raises(KeyError):
        _ = res["invalid_key"]


def test_apply_classification_to_event() -> None:
    event = _create_event("Sprint Planning")
    assert event.task_type == "unknown"

    result = classify_calendar_event_deterministic(event)
    assert result.task_type == "planning"

    updated_event = apply_classification_to_event(event, result)
    assert updated_event.task_type == "planning"
    assert updated_event.suitable_for_walking is True
    assert updated_event.recommended_pomodoro == "25_5"


def test_is_walking_or_audiobook_helper() -> None:
    assert is_walking_or_audiobook("walk on the treadmill") is True
    assert is_walking_or_audiobook("listen to an audiobook") is True
    assert is_walking_or_audiobook("audible book") is True
    assert is_walking_or_audiobook("walking pad session") is True
    assert is_walking_or_audiobook("sitting at desk reading paper") is False
    assert is_walking_or_audiobook("") is False
