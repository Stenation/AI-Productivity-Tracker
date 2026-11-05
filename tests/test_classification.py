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
    WindowClassificationResult,
    apply_classification_to_event,
    classify_calendar_event,
    classify_calendar_event_deterministic,
    classify_calendar_event_sync,
    classify_window,
    get_recommended_pomodoro,
    is_call_window,
    is_suitable_for_walking,
    is_walking_or_audiobook,
)
from jarvis.core.schema import CalendarEvent, TaskType
from jarvis.core.settings import (
    CalendarClassificationSettings,
    ClassificationSettings,
    FocusSettings,
    PomodoroSettings,
    Settings,
)


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


# ===========================================================================
# Window focus classification tests (FR-08, FR-09)
# ===========================================================================


# 9. App-class precedence tests


def test_app_class_precedence_over_distraction_title_patterns() -> None:
    # Productive app class beats title containing distraction keywords
    res1 = classify_window("code", "Reddit - /r/python - Visual Studio Code")
    assert res1.category == "focused"
    assert res1.classifier == "app_class"
    assert res1.confidence == 0.95
    assert res1.is_call_window is False

    res2 = classify_window("Alacritty", "Lo-Fi Beats - YouTube")
    assert res2.category == "focused"
    assert res2.classifier == "app_class"

    res3 = classify_window("kitty", "Twitch chat bot dev")
    assert res3.category == "focused"
    assert res3.classifier == "app_class"

    res4 = classify_window("obsidian", "Notes on Twitter / X algorithm")
    assert res4.category == "focused"
    assert res4.classifier == "app_class"

    res5 = classify_window("Zotero", "YouTube video bibliography")
    assert res5.category == "focused"
    assert res5.classifier == "app_class"


def test_app_class_precedence_over_calendar_correlation() -> None:
    # Even during reading task, productive app-class rules take precedence
    res = classify_window(
        "code",
        "FastAPI Documentation",
        active_task_type="reading",
    )
    assert res.category == "focused"
    assert res.classifier == "app_class"
    assert res.confidence == 0.95


def test_app_class_case_insensitivity() -> None:
    assert classify_window("CODE", "main.py").category == "focused"
    assert classify_window("alacritty", "zsh").category == "focused"
    assert classify_window("Kitty", "bash").category == "focused"
    assert classify_window("Obsidian", "Daily Notes").category == "focused"
    assert classify_window("ZOTERO", "Library").category == "focused"


def test_explicit_app_class_rules_top_precedence() -> None:
    # Custom app_class_rules takes precedence over built-in work/distraction lists
    rules = {"firefox": "focused", "gimp": "neutral", "custom_game": "distracted"}
    res_ff = classify_window("firefox", "Reddit - /r/rust", app_class_rules=rules)
    assert res_ff.category == "focused"
    assert res_ff.classifier == "app_class"
    assert res_ff.confidence == 1.0

    res_gimp = classify_window("gimp", "design.png", app_class_rules=rules)
    assert res_gimp.category == "neutral"
    assert res_gimp.classifier == "app_class"

    res_game = classify_window("custom_game", "Game window", app_class_rules=rules)
    assert res_game.category == "distracted"
    assert res_game.classifier == "app_class"


def test_distraction_and_neutral_app_classes() -> None:
    # Custom distraction and neutral app classes
    res_dist = classify_window(
        "steam",
        "Documentation",
        distraction_app_classes=["steam"],
    )
    assert res_dist.category == "distracted"
    assert res_dist.classifier == "app_class"

    res_neutral = classify_window(
        "pavucontrol",
        "YouTube Volume",
        neutral_app_classes=["pavucontrol"],
    )
    assert res_neutral.category == "neutral"
    assert res_neutral.classifier == "app_class"


# 10. Title-rule precedence tests


def test_title_rule_precedence_for_distractions() -> None:
    # Unclassified browser app with distraction title patterns
    browser_apps = ["firefox", "google-chrome", "brave", "chromium"]
    for app in browser_apps:
        res_yt = classify_window(app, "Lo-Fi Beats - YouTube — Browser")
        assert res_yt.category == "distracted"
        assert res_yt.classifier == "title_rule"
        assert res_yt.confidence == 0.90

        res_reddit = classify_window(app, "Reddit: /r/hyprland — Browser")
        assert res_reddit.category == "distracted"
        assert res_reddit.classifier == "title_rule"

        res_twitch = classify_window(app, "Twitch - Live Stream")
        assert res_twitch.category == "distracted"
        assert res_twitch.classifier == "title_rule"

        res_twitter = classify_window(app, "Home / Twitter")
        assert res_twitter.category == "distracted"
        assert res_twitter.classifier == "title_rule"

        res_x = classify_window(app, "Trending / X — Browser")
        assert res_x.category == "distracted"
        assert res_x.classifier == "title_rule"


def test_title_rule_boundary_safety_for_short_patterns() -> None:
    # 'X' pattern must NOT match words containing x like Firefox, Linux, Next.js, Matrix
    res_firefox = classify_window("firefox", "Arch Linux Wiki — Mozilla Firefox")
    assert res_firefox.category == "unknown"
    assert res_firefox.classifier == "default"

    res_matrix = classify_window("firefox", "Matrix Protocol Documentation")
    assert res_matrix.category == "unknown"
    assert res_matrix.classifier == "default"

    res_next = classify_window("firefox", "Next.js App Router Guide")
    assert res_next.category == "unknown"
    assert res_next.classifier == "default"

    # But 'X' surrounded by non-alphanumeric boundaries MUST match
    res_x1 = classify_window("firefox", "Home / X")
    assert res_x1.category == "distracted"

    res_x2 = classify_window("firefox", "Steffen on X: 'Hello World'")
    assert res_x2.category == "distracted"


def test_title_rule_precedence_over_calendar_correlation() -> None:
    # Title rule beats calendar correlation (e.g. YouTube during reading or coding)
    res_read = classify_window(
        "firefox",
        "YouTube - Tech Conference Talk",
        active_task_type="reading",
    )
    assert res_read.category == "distracted"
    assert res_read.classifier == "title_rule"

    res_code = classify_window(
        "firefox",
        "Reddit - /r/programming Discussion",
        active_task_type="coding",
    )
    assert res_code.category == "distracted"
    assert res_code.classifier == "title_rule"


def test_work_and_neutral_title_patterns() -> None:
    res_work = classify_window(
        "firefox",
        "FastAPI Documentation — Mozilla Firefox",
        work_title_patterns=["FastAPI Documentation"],
    )
    assert res_work.category == "focused"
    assert res_work.classifier == "title_rule"

    res_neutral = classify_window(
        "custom_gui",
        "System Volume Settings",
        neutral_title_patterns=["System Volume Settings"],
    )
    assert res_neutral.category == "neutral"
    assert res_neutral.classifier == "title_rule"


# 11. Call-window detection tests


def test_call_window_detection_zoom() -> None:
    assert is_call_window("zoom", "") is True
    assert is_call_window("Zoom", "Team Sync") is True
    assert is_call_window("zoom.real", "Zoom Cloud Meetings") is True
    assert is_call_window("firefox", "Zoom Meeting - Sprint Review") is True

    res = classify_window("zoom", "Zoom Meeting")
    assert res.is_call_window is True


def test_call_window_detection_google_meet() -> None:
    assert is_call_window("firefox", "Google Meet — Mozilla Firefox") is True
    assert is_call_window("chrome", "Meet - abc-defg-hij") is True
    assert is_call_window("firefox", "https://meet.google.com/xyz-uvw-rst") is True
    assert is_call_window("google-meet", "Daily Standup") is True

    res = classify_window("firefox", "Meet - abc-defg-hij")
    assert res.is_call_window is True


def test_call_window_detection_microsoft_teams() -> None:
    assert is_call_window("teams", "Microsoft Teams") is True
    assert is_call_window("microsoft teams", "General") is True
    assert is_call_window("ms-teams", "Weekly Sync") is True
    assert is_call_window("teams-for-linux", "Daily Standup") is True
    assert is_call_window("firefox", "Teams Meeting | Microsoft Teams") is True

    res = classify_window("teams", "Microsoft Teams")
    assert res.is_call_window is True


def test_call_window_detection_discord_voice() -> None:
    # Discord voice or active call indicators
    assert is_call_window("discord", "Discord | Voice Connected") is True
    assert is_call_window("discord", "Discord Voice - Gaming") is True
    assert is_call_window("discord", "Incoming Call - Discord") is True
    assert is_call_window("vesktop", "Voice Channel - Friends") is True
    assert is_call_window("discord", "• Voice") is True
    assert is_call_window("discord", "Team Room (Voice)") is True

    # General Discord text chat must NOT be flagged as call window
    assert is_call_window("discord", "#general - Discord") is False
    assert is_call_window("discord", "#announcements") is False
    assert is_call_window("vesktop", "#dev-chat") is False

    res_voice = classify_window("discord", "Discord | Voice Connected")
    assert res_voice.is_call_window is True

    res_text = classify_window("discord", "#general - Discord")
    assert res_text.is_call_window is False


def test_call_window_detection_slack_huddles() -> None:
    # Slack huddle indicators
    assert is_call_window("slack", "Slack | Huddle") is True
    assert is_call_window("slack", "Dev Huddle") is True
    assert is_call_window("slack", "Slack Huddle with Alice") is True
    assert is_call_window("slack", "Team Call") is False
    assert is_call_window("slack", "Huddle") is True

    # General Slack text chat must NOT be flagged as call window
    assert is_call_window("slack", "#general - Slack") is False
    assert is_call_window("slack", "project-roadmap") is False

    res_huddle = classify_window("slack", "Slack | Huddle")
    assert res_huddle.is_call_window is True

    res_text = classify_window("slack", "#general - Slack")
    assert res_text.is_call_window is False


def test_non_call_windows() -> None:
    assert is_call_window("code", "main.py - Visual Studio Code") is False
    assert is_call_window("Alacritty", "zsh") is False
    assert is_call_window("firefox", "Arch Linux Wiki — Mozilla Firefox") is False
    assert is_call_window("firefox", "Wikipedia: Alan Turing") is False
    assert is_call_window("", "") is False


# 12. Calendar correlation tests


def test_calendar_correlation_reading_docs() -> None:
    # Browser documentation during reading task correlates to focused
    res_docs = classify_window(
        "firefox",
        "Python 3.12 Documentation",
        active_task_type="reading",
    )
    assert res_docs.category == "focused"
    assert res_docs.classifier == "calendar_correlation"
    assert res_docs.confidence == 0.80

    res_wiki = classify_window(
        "firefox",
        "Arch Linux Wiki — Mozilla Firefox",
        active_task_type="reading",
    )
    assert res_wiki.category == "focused"
    assert res_wiki.classifier == "calendar_correlation"

    res_book = classify_window(
        "firefox",
        "Designing Data-Intensive Applications - Book Chapter 3",
        active_task_type="reading",
    )
    assert res_book.category == "focused"
    assert res_book.classifier == "calendar_correlation"

    res_pdf = classify_window(
        "evince",
        "Raft Consensus Algorithm Paper.pdf",
        active_task_type="reading",
    )
    assert res_pdf.category == "focused"
    assert res_pdf.classifier == "calendar_correlation"


def test_calendar_correlation_coding_docs_and_repos() -> None:
    # Browser technical docs / repositories during coding task
    res_github = classify_window(
        "firefox",
        "GitHub - owner/repo: Pull Request #404",
        active_task_type="coding",
    )
    assert res_github.category == "focused"
    assert res_github.classifier == "calendar_correlation"
    assert res_github.confidence == 0.80

    res_so = classify_window(
        "firefox",
        "Stack Overflow: How to use asyncio subprocess in Python",
        active_task_type="coding",
    )
    assert res_so.category == "focused"
    assert res_so.classifier == "calendar_correlation"

    res_crates = classify_window(
        "firefox",
        "crates.io: tokio - Rust async runtime",
        active_task_type="coding",
    )
    assert res_crates.category == "focused"
    assert res_crates.classifier == "calendar_correlation"


def test_calendar_correlation_deep_work() -> None:
    # Deep work: media / entertainment correlates to distracted
    res_music = classify_window(
        "firefox",
        "Podcast: Software Engineering Daily",
        active_task_type="deep_work",
    )
    assert res_music.category == "distracted"
    assert res_music.classifier == "calendar_correlation"
    assert res_music.confidence == 0.85

    res_video = classify_window(
        "custom_player",
        "Funny Stream Video",
        active_task_type="deep_work",
    )
    assert res_video.category == "distracted"
    assert res_video.classifier == "calendar_correlation"

    # Deep work: research papers correlate to focused
    res_research = classify_window(
        "firefox",
        "Raft Consensus Research Paper",
        active_task_type="deep_work",
    )
    assert res_research.category == "focused"
    assert res_research.classifier == "calendar_correlation"

    # Deep work: ambiguous non-match falls back to unknown
    res_ambiguous = classify_window(
        "firefox",
        "Online Shopping Store",
        active_task_type="deep_work",
    )
    assert res_ambiguous.category == "unknown"
    assert res_ambiguous.classifier == "default"


def test_calendar_correlation_meeting() -> None:
    # Meeting task: call windows correlate to focused
    res_zoom = classify_window(
        "zoom",
        "Zoom Meeting - Weekly Standup",
        active_task_type="meeting",
    )
    assert res_zoom.category == "focused"
    assert res_zoom.classifier == "calendar_correlation"
    assert res_zoom.is_call_window is True

    res_meet = classify_window(
        "firefox",
        "Google Meet — Mozilla Firefox",
        active_task_type="meeting",
    )
    assert res_meet.category == "focused"
    assert res_meet.classifier == "calendar_correlation"
    assert res_meet.is_call_window is True

    # Meeting notes correlate to focused
    res_notes = classify_window(
        "firefox",
        "Sprint 42 Retrospective Notes",
        active_task_type="meeting",
    )
    assert res_notes.category == "focused"
    assert res_notes.classifier == "calendar_correlation"

    # General background browser during meeting correlates to neutral
    res_general = classify_window(
        "firefox",
        "Company Intranet Portal",
        active_task_type="meeting",
    )
    assert res_general.category == "neutral"
    assert res_general.classifier == "calendar_correlation"


def test_calendar_correlation_break() -> None:
    # Break task correlates general activity to neutral
    res_break1 = classify_window(
        "mpv",
        "Stretch and Relax Video",
        active_task_type="break",
    )
    assert res_break1.category == "neutral"
    assert res_break1.classifier == "calendar_correlation"

    res_break2 = classify_window(
        "firefox",
        "Cooking Recipes",
        active_task_type="break",
    )
    assert res_break2.category == "neutral"
    assert res_break2.classifier == "calendar_correlation"


def test_calendar_correlation_email_admin_planning() -> None:
    # Email task
    res_mail = classify_window(
        "firefox",
        "Gmail - Inbox (12)",
        active_task_type="email",
    )
    assert res_mail.category == "focused"
    assert res_mail.classifier == "calendar_correlation"

    # Admin task
    res_admin = classify_window(
        "firefox",
        "HR Timesheet and Expense Report - Portal",
        active_task_type="admin",
    )
    assert res_admin.category == "focused"
    assert res_admin.classifier == "calendar_correlation"

    # Planning task
    res_plan = classify_window(
        "firefox",
        "Q2 Roadmap and Backlog Prioritization - Jira",
        active_task_type="planning",
    )
    assert res_plan.category == "focused"
    assert res_plan.classifier == "calendar_correlation"


def test_calendar_correlation_conservative_fallback_to_unknown() -> None:
    # During reading, an unrelated store page must NOT be guessed as focused
    res_read = classify_window(
        "firefox",
        "Amazon.com: Electronics Store",
        active_task_type="reading",
    )
    assert res_read.category == "unknown"
    assert res_read.classifier == "default"

    # During coding, an unrelated real-estate page must NOT be guessed as focused
    res_code = classify_window(
        "firefox",
        "Apartments for Rent",
        active_task_type="coding",
    )
    assert res_code.category == "unknown"
    assert res_code.classifier == "default"


# 13. Unknown fallback and desktop tests


def test_unknown_fallback_for_unmatched_inputs() -> None:
    res = classify_window("firefox", "Random Blog Article")
    assert res.category == "unknown"
    assert res.classifier == "default"
    assert res.confidence == 0.0
    assert res.is_call_window is False

    res_app = classify_window("calculator", "Calculator")
    assert res_app.category == "unknown"
    assert res_app.classifier == "default"
    assert res_app.confidence == 0.0


def test_empty_window_and_desktop_fallback() -> None:
    res_empty = classify_window("", "")
    assert res_empty.category == "neutral"
    assert res_empty.classifier == "desktop"
    assert res_empty.confidence == 1.0
    assert res_empty.is_call_window is False

    res_spaces = classify_window("   ", "   ")
    assert res_spaces.category == "neutral"
    assert res_spaces.classifier == "desktop"


# 14. Deterministic behavior and result model tests


def test_deterministic_behavior_for_identical_inputs() -> None:
    # Repeated calls must return identical results
    test_cases = [
        ("code", "main.py", "coding"),
        ("firefox", "Reddit - /r/rust", "reading"),
        ("firefox", "FastAPI Docs", "reading"),
        ("zoom", "Zoom Meeting", "meeting"),
        ("firefox", "Meet - abc-defg-hij", "meeting"),
        ("slack", "Dev Huddle", None),
        ("discord", "#general", None),
        ("firefox", "Random Page", None),
        ("", "", None),
    ]

    for app_class, title, task_type in test_cases:
        r1 = classify_window(app_class, title, active_task_type=task_type)
        r2 = classify_window(app_class, title, active_task_type=task_type)
        assert r1.category == r2.category
        assert r1.classifier == r2.classifier
        assert r1.is_call_window == r2.is_call_window
        assert r1.confidence == r2.confidence


def test_classify_window_via_settings_object() -> None:
    custom_settings = Settings(
        focus=FocusSettings(
            work_app_classes=["custom_ide"],
            distraction_title_patterns=["DistractingSite"],
        )
    )
    res_ide = classify_window("custom_ide", "main.rs", settings=custom_settings)
    assert res_ide.category == "focused"
    assert res_ide.classifier == "app_class"

    res_dist = classify_window("firefox", "DistractingSite - Home", settings=custom_settings)
    assert res_dist.category == "distracted"
    assert res_dist.classifier == "title_rule"


def test_window_classification_result_dict_access() -> None:
    res = WindowClassificationResult(
        category="focused",
        classifier="app_class",
        is_call_window=False,
        confidence=0.95,
    )
    assert res["category"] == "focused"
    assert res["classifier"] == "app_class"
    assert res["is_call_window"] is False
    assert res["confidence"] == 0.95
    assert res.get("category") == "focused"
    assert res.get("non_existent", "fallback") == "fallback"
    assert "category" in res
    assert "is_call_window" in res

    with pytest.raises(KeyError):
        _ = res["non_existent_key"]


# 15. Comprehensive 5-stage cascade and ClassificationSettings tests


@pytest.mark.asyncio
async def test_strict_five_tier_precedence_cascade() -> None:
    """Test full strict precedence cascade: explicit > color > regex > llm > unknown."""
    # Stage 1: Explicit mapping present -> wins over color, regex, and LLM
    mock_llm = AsyncMock(return_value="deep_work")
    # color 1 -> reading, regex -> coding
    event_1 = _create_event("Refactor auth backend", color_id="1")
    res_explicit = await classify_calendar_event(
        event_1,
        explicit_mappings={"Refactor auth backend": "admin"},
        llm_classifier=mock_llm,
    )
    assert res_explicit.task_type == "admin"
    assert res_explicit.classifier == "explicit"
    assert res_explicit.confidence == 1.0
    mock_llm.assert_not_called()

    # Stage 2: No explicit mapping -> Color ID wins over regex and LLM
    res_color = await classify_calendar_event(
        event_1,
        llm_classifier=mock_llm,
    )
    assert res_color.task_type == "reading"
    assert res_color.classifier == "color"
    assert res_color.confidence == 0.95
    mock_llm.assert_not_called()

    # Stage 3: No explicit, unmapped color -> Regex wins over LLM
    event_2 = _create_event("Refactor auth backend")
    res_regex = await classify_calendar_event(
        event_2,
        llm_classifier=mock_llm,
    )
    assert res_regex.task_type == "coding"
    assert res_regex.classifier == "regex"
    assert res_regex.confidence == 0.85
    mock_llm.assert_not_called()

    # Stage 4: Ambiguous title (no regex match) -> LLM fallback wins over unknown
    event_3 = _create_event("Ambiguous Quantum Optimization")
    res_llm = await classify_calendar_event(
        event_3,
        llm_classifier=mock_llm,
    )
    assert res_llm.task_type == "deep_work"
    assert res_llm.classifier == "llm"
    assert res_llm.confidence == 0.8
    mock_llm.assert_awaited_once()

    # Stage 5: No LLM (or LLM returns None) -> default unknown
    mock_none_llm = AsyncMock(return_value=None)
    res_unknown = await classify_calendar_event(
        event_3,
        llm_classifier=mock_none_llm,
    )
    assert res_unknown.task_type == "unknown"
    assert res_unknown.classifier == "default"
    assert res_unknown.confidence == 0.0
    assert res_unknown.suitable_for_walking is None
    assert res_unknown.recommended_pomodoro is None


@pytest.mark.asyncio
async def test_classify_calendar_event_positional_arguments() -> None:
    """Verify classify_calendar_event accepts title, desc, color_id, settings positionally."""
    settings = Settings(
        classification=ClassificationSettings(
            explicit_mappings={"Special Session": "planning"}
        )
    )
    mock_llm = AsyncMock(return_value="coding")

    # Positional: title, description, color_id, settings
    res = await classify_calendar_event(
        "Special Session",
        "Quarterly roadmap review",
        "1",
        settings,
        llm_classifier=mock_llm,
    )
    assert res.task_type == "planning"
    assert res.classifier == "explicit"
    assert res.suitable_for_walking is True
    assert res.recommended_pomodoro == "25_5"
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_llm_classifier_success_with_task_classification_result() -> None:
    """Verify llm_classifier returning TaskClassificationResult object directly."""
    expected_result = TaskClassificationResult(
        task_type="coding",
        classifier="llm",
        suitable_for_walking=False,
        recommended_pomodoro="50_10",
        confidence=0.91,
    )
    mock_llm = AsyncMock(return_value=expected_result)

    res = await classify_calendar_event(
        "Ambiguous Tech Task",
        llm_classifier=mock_llm,
    )
    assert res.task_type == "coding"
    assert res.classifier == "llm"
    assert res.confidence == 0.91
    assert res.suitable_for_walking is False
    assert res.recommended_pomodoro == "50_10"
    mock_llm.assert_awaited_once()


def test_classify_calendar_event_sync_api() -> None:
    """Verify synchronous classification API functions and aliases."""
    res1 = classify_calendar_event_deterministic("Refactor auth backend")
    assert res1.task_type == "coding"
    assert res1.classifier == "regex"

    res2 = classify_calendar_event_sync("Refactor auth backend")
    assert res2.task_type == "coding"
    assert res2.classifier == "regex"

    res3 = classify_calendar_event.sync("Refactor auth backend")
    assert res3.task_type == "coding"
    assert res3.classifier == "regex"


def test_classification_settings_defaults_and_custom_overrides() -> None:
    """Verify ClassificationSettings holds both calendar and window overrides."""
    default_settings = ClassificationSettings()
    assert "1" in default_settings.color_mappings
    assert any(c == "code" for c in default_settings.work_app_classes)
    assert any("YouTube" in p for p in default_settings.distraction_title_patterns)
    assert any("zoom" in c for c in default_settings.call_app_classes)

    custom_settings = Settings(
        classification=ClassificationSettings(
            explicit_mappings={"All Hands": "meeting"},
            app_class_rules={"special_term": "focused"},
            work_app_classes=["custom_editor"],
        )
    )
    # Window classification uses ClassificationSettings
    res_win = classify_window("special_term", "Terminal", settings=custom_settings)
    assert res_win.category == "focused"
    assert res_win.classifier == "app_class"

    res_win2 = classify_window("custom_editor", "doc.txt", settings=custom_settings)
    assert res_win2.category == "focused"
    assert res_win2.classifier == "app_class"

    # Calendar classification uses ClassificationSettings
    res_cal = classify_calendar_event_deterministic(
        "All Hands", settings=custom_settings
    )
    assert res_cal.task_type == "meeting"
    assert res_cal.classifier == "explicit"


def test_tier1_bachelor_thesis_and_writing_tools() -> None:
    """Verify built-in detection of bachelor thesis writing and research tools (sioyek, zotero, etc.)."""
    assert classify_window("sioyek", "Attention is All You Need.pdf").category == "focused"
    assert classify_window("antigravity-ide", "jarvis - hyprland.lua").category == "focused"
    assert classify_window("zotero", "Zotero - My Library").category == "focused"
    assert classify_window("obsidian", "Thesis Outline.md").category == "focused"
    assert classify_window("texstudio", "thesis.tex").category == "focused"
    assert classify_window("cursor", "agent.py").category == "focused"


def test_tier1_firefox_websites_direct_classification() -> None:
    """Verify direct classification of Firefox sites using focus_rules.yaml."""
    rules_settings = Settings(focus=FocusSettings(rules_file="focus_rules.yaml"))

    # Academic & thesis research sites
    res_arxiv = classify_window(
        "firefox",
        "arXiv:2401.12345 [cs.AI] Attention Mechanisms — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_arxiv.category == "focused"

    res_scholar = classify_window(
        "firefox",
        "posture tracking - Google Scholar — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_scholar.category == "focused"

    res_overleaf = classify_window(
        "firefox",
        "Bachelor Thesis 2026 - Overleaf, Online LaTeX Editor — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_overleaf.category == "focused"

    # AI & developer tools
    res_gemini = classify_window(
        "firefox",
        "Fixing Icon Display - Google Gemini — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_gemini.category == "focused"

    res_cloud = classify_window(
        "firefox",
        "Zielgruppe – Google Auth Platform – Jarvis AI – Google Cloud Console — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_cloud.category == "focused"

    res_jarvis = classify_window(
        "firefox",
        "History · Jarvis Posture Platform — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_jarvis.category == "focused"

    res_icons = classify_window(
        "firefox",
        "Material Design Icons - Icon Library - Pictogrammers — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_icons.category == "focused"

    # Neutral and distracted
    res_cal = classify_window(
        "firefox",
        "Google Kalender - 4 Tage, ab Donnerstag, 5. November 2026 — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_cal.category == "neutral"

    res_yt = classify_window(
        "firefox",
        "YouTube — Mozilla Firefox",
        settings=rules_settings,
    )
    assert res_yt.category == "distracted"


def test_custom_rules_file_is_easily_replaceable(tmp_path: Any) -> None:
    """Verify users can provide a custom rules file without modifying source code."""
    custom_rules = tmp_path / "my_custom_rules.yaml"
    custom_rules.write_text(
        """
work_app_classes:
  - "custom_research_tool"
work_title_patterns:
  - "Niche Research Lab"
distraction_app_classes:
  - "custom_timewaster"
neutral_title_patterns:
  - "Custom Utility Dashboard"
""",
        encoding="utf-8",
    )

    settings = Settings(focus=FocusSettings(rules_file=custom_rules))

    assert classify_window("custom_research_tool", "Window 1", settings=settings).category == "focused"
    assert classify_window("firefox", "Niche Research Lab Portal", settings=settings).category == "focused"
    assert classify_window("custom_timewaster", "Game", settings=settings).category == "distracted"
    assert classify_window("firefox", "Custom Utility Dashboard", settings=settings).category == "neutral"


@pytest.mark.asyncio
async def test_classify_window_async_llm_fallback_and_cache() -> None:
    """Verify classify_window_async delegates unknown windows to LLM and caches result."""
    from jarvis.core.classification import (
        clear_window_llm_cache,
        classify_window_async,
        get_window_llm_cache,
    )

    clear_window_llm_cache()
    calls = []

    async def mock_llm(app_class: str, window_title: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((app_class, window_title))
        return {"category": "focused", "confidence": 0.85, "reason": "Academic paper"}

    # 1. Deterministic known app does NOT call LLM
    res_known = await classify_window_async(
        "antigravity-ide", "main.py", llm_fallback=mock_llm
    )
    assert res_known.category == "focused"
    assert res_known.classifier == "app_class"
    assert len(calls) == 0

    # 2. Unknown window calls LLM
    res_unknown = await classify_window_async(
        "firefox", "Unusual Science Blog Article", llm_fallback=mock_llm
    )
    assert res_unknown.category == "focused"
    assert res_unknown.classifier == "llm"
    assert res_unknown.confidence == 0.85
    assert len(calls) == 1

    # 3. Subsequent call is served from cache without invoking LLM again
    res_cached = await classify_window_async(
        "firefox", "Unusual Science Blog Article", llm_fallback=mock_llm
    )
    assert res_cached.category == "focused"
    assert res_cached.classifier == "llm"
    assert len(calls) == 1  # No second call!

    clear_window_llm_cache()

