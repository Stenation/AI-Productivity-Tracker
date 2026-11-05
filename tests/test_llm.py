"""Tests for Session M2-5: Local Ollama LLM Specialist (FR-23, FR-16, FR-17, SPEC §7)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi import FastAPI

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.classification import classify_calendar_event
from jarvis.core.database import Database
from jarvis.core.llm import (
    RESPONSE_JSON_SCHEMA,
    LlmValidationError,
    OllamaClient,
    build_assistant_context,
    classify_event_with_llm,
    clean_think_tags,
    parse_calendar_query_range,
    validate_llm_response,
)
from jarvis.core.schema import CalendarEvent, PlatformEvent
from jarvis.core.settings import Settings
from jarvis.core.timer import TimerManager
from jarvis.modules.outputs.llm_ollama import LlmOllamaModule
from jarvis.web.api import router as api_router


def make_fake_ollama_transport(
    response_content: str | dict[str, Any] | None = None,
    *,
    status_code: int = 200,
    model: str = "qwen3:14b",
    side_effect: Exception | None = None,
    capture_requests: list[dict[str, Any]] | None = None,
) -> httpx.MockTransport:
    """Create an injectable httpx.MockTransport simulating Ollama's /api/chat."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture_requests is not None:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            capture_requests.append(
                {
                    "url": str(request.url),
                    "method": request.method,
                    "body": body,
                }
            )

        if side_effect is not None:
            raise side_effect

        if isinstance(response_content, dict):
            content_str = json.dumps(response_content)
        elif response_content is not None:
            content_str = response_content
        else:
            content_str = json.dumps(
                {
                    "answer": "You have been sitting for 45 minutes. Time to stand up.",
                    "suggested_action": "stand_up",
                    "confidence": 0.95,
                }
            )

        resp_payload = {
            "model": model,
            "created_at": "2026-11-04T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": content_str,
            },
            "done": True,
            "total_duration": 120000000,
        }
        return httpx.Response(status_code, json=resp_payload)

    return httpx.MockTransport(handler)


@pytest.fixture
async def llm_test_env(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]]]:
    """Fixture providing FastAPI test client with llm_ollama enabled and mocked transport."""
    monkeypatch.setattr(LlmOllamaModule, "_dependency_override", True)
    test_settings.enabled_modules.inputs = ["manual_control"]
    test_settings.enabled_modules.outputs = ["waybar_status", "desktop_notify", "llm_ollama"]

    captured_requests: list[dict[str, Any]] = []
    fake_transport = make_fake_ollama_transport(capture_requests=captured_requests)
    fake_client = OllamaClient(
        base_url=test_settings.llm.base_url,
        model=test_settings.llm.text_model,
        keep_alive=test_settings.llm.keep_alive,
        transport=fake_transport,
    )

    app = create_app(settings=test_settings)
    app.include_router(api_router)

    async with app.router.lifespan_context(app):
        timer_manager = TimerManager(
            bus=app.state.bus,
            db=app.state.db,
            settings=app.state.settings,
            tick_interval_seconds=0.01,
        )
        await timer_manager.start()
        app.state.timer_manager = timer_manager

        # Inject fake client into the registered module instance
        llm_mod = app.state.registry.get_module("llm_ollama")
        if isinstance(llm_mod, LlmOllamaModule):
            llm_mod.client = fake_client

        asgi_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=asgi_transport, base_url="http://testserver"
        ) as client:
            try:
                yield app, client, captured_requests
            finally:
                await timer_manager.stop()


# ---------------------------------------------------------------------------
# 1. Migration 0002 & Database Repository Methods
# ---------------------------------------------------------------------------


async def test_migration_0002_schema_and_db_methods(db: Database) -> None:
    """Verify migration 0002 table, index, and database helper methods."""
    # Check that table llm_interactions exists
    async with db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_interactions'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None

    # Check index idx_llm_interactions_created_at exists
    async with db.connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND name='idx_llm_interactions_created_at'"
    ) as cursor:
        idx_row = await cursor.fetchone()
    assert idx_row is not None

    # Record interaction
    rec = await db.record_llm_interaction(
        source="dashboard",
        question="Should I take a break?",
        answer="Yes, standing break recommended.",
        suggested_action="take_break",
        confidence=0.92,
        model="qwen3:14b",
        latency_ms=150,
        metadata={"user_state": "sitting"},
    )
    assert rec["id"] is not None
    assert rec["source"] == "dashboard"
    assert rec["question"] == "Should I take a break?"
    assert rec["answer"] == "Yes, standing break recommended."
    assert rec["suggested_action"] == "take_break"
    assert rec["confidence"] == 0.92
    assert rec["model"] == "qwen3:14b"
    assert rec["latency_ms"] == 150
    assert rec["metadata"] == {"user_state": "sitting"}

    # List interactions
    interactions = await db.list_llm_interactions(limit=10)
    assert len(interactions) == 1
    assert interactions[0]["id"] == rec["id"]


# ---------------------------------------------------------------------------
# 2. Compact Context Builder
# ---------------------------------------------------------------------------


async def test_compact_context_builder(db: Database) -> None:
    """Verify build_assistant_context generates compact, privacy-safe JSON with NO window titles."""
    now = datetime.now().astimezone()

    # Seed an activity span
    await db.open_activity_span(
        activity="sitting",
        source="manual_control",
        started_at=now - timedelta(minutes=40),
    )

    # Seed an active timer
    await db.start_timer_span(
        kind="coding",
        planned_duration_seconds=3000,
        started_at=now - timedelta(minutes=10),
    )

    # Seed calendar events
    await db.upsert_calendar_event(
        event_id="cur-1",
        title="Deep Architecture Review",
        start_time=now - timedelta(minutes=15),
        end_time=now + timedelta(minutes=45),
        task_type="deep_work",
        suitable_for_walking=False,
    )
    await db.upsert_calendar_event(
        event_id="next-1",
        title="Walk and 1:1 Sync",
        start_time=now + timedelta(minutes=45),
        end_time=now + timedelta(minutes=75),
        task_type="meeting",
        suitable_for_walking=True,
    )

    # Seed window spans with sensitive titles
    await db.open_window_span(
        app_class="firefox",
        window_title="SECRET_PATIENT_RECORDS_CONFIDENTIAL - Browser",
        category="focused",
        started_at=now - timedelta(minutes=30),
    )
    await db.open_window_span(
        app_class="alacritty",
        window_title="steff@arch: ~/secrets/passwords.txt",
        category="distracted",
        started_at=now - timedelta(minutes=10),
    )

    # Seed a pending recommendation
    await db.create_recommendation(
        policy="posture_policy",
        message="You have been sitting for 40 minutes.",
        suggested_action="stand_up",
    )

    context = await build_assistant_context(db=db)

    # 1. Activity data present
    assert context["activity"]["activity"] == "sitting"
    assert context["activity"]["source"] == "manual_control"
    assert context["activity"]["elapsed_seconds"] >= 2400

    # 2. Timer data present
    assert context["timer"]["kind"] == "coding"
    assert context["timer"]["phase"] == "work"
    assert context["timer"]["remaining_seconds"] > 0

    # 3. Calendar data present
    assert context["calendar"]["current_event"]["title"] == "Deep Architecture Review"
    assert context["calendar"]["current_event"]["task_type"] == "deep_work"
    assert context["calendar"]["current_event"]["suitable_for_walking"] is False
    assert context["calendar"]["next_event"]["title"] == "Walk and 1:1 Sync"

    # 4. Window focus categories and durations present
    assert "focused" in context["window_focus"]["category_totals_seconds"]
    assert "distracted" in context["window_focus"]["category_totals_seconds"]

    # 5. PRIVACY CHECK: sensitive window titles must NEVER leak into assistant context
    context_json = json.dumps(context)
    assert "SECRET_PATIENT_RECORDS_CONFIDENTIAL" not in context_json
    assert "passwords.txt" not in context_json

    # 6. Pending recommendations present
    assert len(context["pending_recommendations"]) >= 1
    assert context["pending_recommendations"][0]["suggested_action"] == "stand_up"


# ---------------------------------------------------------------------------
# 3. Structured JSON Response, Persistence, and Bus Events
# ---------------------------------------------------------------------------


async def test_ask_endpoint_structured_json_and_events(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify POST /api/ask returns structured JSON, persists row, and emits assistant_answered."""
    app, client, captured = llm_test_env
    bus: EventBus = app.state.bus
    db: Database = app.state.db

    answered_events: list[PlatformEvent] = []
    bus.subscribe("assistant_answered", lambda ev: answered_events.append(ev))

    payload = {
        "question": "How long have I been sitting?",
        "source": "dashboard",
        "speak": False,
    }
    resp = await client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"] == "You have been sitting for 45 minutes. Time to stand up."
    assert data["suggested_action"] == "stand_up"
    assert data["confidence"] == 0.95
    assert data["model"] == "qwen3:14b"
    assert data["interaction_id"] > 0

    # Interaction persisted in SQLite
    interactions = await db.list_llm_interactions(limit=5)
    assert len(interactions) == 1
    assert interactions[0]["id"] == data["interaction_id"]
    assert interactions[0]["question"] == payload["question"]

    # Event emitted on EventBus
    assert len(answered_events) == 1
    assert answered_events[0].payload["answer"] == data["answer"]
    assert answered_events[0].payload["interaction_id"] == data["interaction_id"]


async def test_ask_endpoint_tts_conditions(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify tts_requested is emitted ONLY when speak=True, allow_tts=True, and dnd_mode=False."""
    app, client, _ = llm_test_env
    bus: EventBus = app.state.bus
    settings: Settings = app.state.settings

    tts_events: list[PlatformEvent] = []
    bus.subscribe("tts_requested", lambda ev: tts_events.append(ev))

    # Case A: speak=False -> no TTS
    settings.privacy.allow_tts = True
    settings.privacy.dnd_mode = False
    await client.post("/api/ask", json={"question": "q1", "speak": False})
    assert len(tts_events) == 0

    # Case B: speak=True, allow_tts=True, dnd_mode=False -> TTS emitted
    await client.post("/api/ask", json={"question": "q2", "speak": True})
    assert len(tts_events) == 1
    assert "sitting" in tts_events[0].payload["text"]

    # Case C: speak=True, allow_tts=True, dnd_mode=True -> suppressed by DND
    settings.privacy.dnd_mode = True
    await client.post("/api/ask", json={"question": "q3", "speak": True})
    assert len(tts_events) == 1  # count did not increase

    # Case D: speak=True, allow_tts=False, dnd_mode=False -> suppressed by allow_tts
    settings.privacy.dnd_mode = False
    settings.privacy.allow_tts = False
    await client.post("/api/ask", json={"question": "q4", "speak": True})
    assert len(tts_events) == 1  # count did not increase


async def test_ask_endpoint_htmx_partial(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify POST /api/ask renders HTML partial when HX-Request is true."""
    _, client, _ = llm_test_env
    resp = await client.post(
        "/api/ask",
        json={"question": "What's next?", "source": "dashboard"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "assistant-response" in resp.text
    assert "stand_up" in resp.text
    assert "HX-Trigger" in resp.headers
    assert resp.headers["HX-Trigger"] == "assistantAnswered"


# ---------------------------------------------------------------------------
# 4. Model Configurations (qwen3:14b and qwen2.5:3b)
# ---------------------------------------------------------------------------


async def test_qwen3_14b_and_qwen2_5_3b_configurations() -> None:
    """Verify OllamaClient correctly dispatches requests for both qwen3:14b and qwen2.5:3b."""
    # Test qwen3:14b
    reqs_14b: list[dict[str, Any]] = []
    transport_14b = make_fake_ollama_transport(model="qwen3:14b", capture_requests=reqs_14b)
    client_14b = OllamaClient(model="qwen3:14b", transport=transport_14b)
    res_14b = await client_14b.ask("Hello 14b")
    assert res_14b["model"] == "qwen3:14b"
    assert reqs_14b[0]["body"]["model"] == "qwen3:14b"
    await client_14b.close()

    # Test qwen2.5:3b
    reqs_3b: list[dict[str, Any]] = []
    transport_3b = make_fake_ollama_transport(model="qwen2.5:3b", capture_requests=reqs_3b)
    client_3b = OllamaClient(model="qwen2.5:3b", transport=transport_3b)
    res_3b = await client_3b.ask("Hello 3b")
    assert res_3b["model"] == "qwen2.5:3b"
    assert reqs_3b[0]["body"]["model"] == "qwen2.5:3b"
    await client_3b.close()


# ---------------------------------------------------------------------------
# 5. keep_alive="5m" and think=False Request Payload
# ---------------------------------------------------------------------------


async def test_keep_alive_and_think_false_payload() -> None:
    """Verify request settings: keep_alive='5m', think=False, stream=False, JSON format schema."""
    captured: list[dict[str, Any]] = []
    transport = make_fake_ollama_transport(capture_requests=captured)
    client = OllamaClient(keep_alive="5m", transport=transport)

    await client.ask("Test question")
    assert len(captured) == 1
    req_body = captured[0]["body"]

    assert req_body["keep_alive"] == "5m"
    assert req_body["think"] is False
    assert req_body["stream"] is False
    assert req_body["format"] == RESPONSE_JSON_SCHEMA
    await client.close()


# ---------------------------------------------------------------------------
# 6. Defensive <think> Content Stripping
# ---------------------------------------------------------------------------


def test_clean_think_tags_variations() -> None:
    """Verify clean_think_tags removes thinking tokens, markdown fences, and unclosed tags."""
    # Standard closed <think> tag
    raw1 = '<think>\nConsidering user activity...\n</think>\n{"answer": "Hello", "confidence": 1.0}'
    assert clean_think_tags(raw1) == '{"answer": "Hello", "confidence": 1.0}'

    # Case-insensitive tag with internal line breaks
    raw2 = '<THINK>pondering</THINK>{"answer": "Done"}'
    assert clean_think_tags(raw2) == '{"answer": "Done"}'

    # Markdown json fence
    raw3 = '```json\n{"answer": "Inside code fence"}\n```'
    assert clean_think_tags(raw3) == '{"answer": "Inside code fence"}'

    # Unclosed think tag
    raw4 = "<think>unclosed trailing monologue"
    assert clean_think_tags(raw4) == ""


async def test_ollama_client_strips_think_tags_defensively() -> None:
    """Verify OllamaClient parses responses cleanly even when <think> tags are returned."""
    raw_response = (
        "<think>\nThe user has been sitting for 45 minutes.\n"
        "Let's advise standing up.\n</think>\n"
        '{"answer": "Stand up and stretch.", "suggested_action": "stand_up", "confidence": 0.99}'
    )
    transport = make_fake_ollama_transport(response_content=raw_response)
    client = OllamaClient(transport=transport)
    res = await client.ask("What should I do?")
    assert res["answer"] == "Stand up and stretch."
    assert res["suggested_action"] == "stand_up"
    assert res["confidence"] == 0.99
    await client.close()


# ---------------------------------------------------------------------------
# 7. Malformed / Invalid Model Response Handling
# ---------------------------------------------------------------------------


def test_validate_llm_response_schema_rules() -> None:
    """Verify strict validation of LLM response dictionaries."""
    # Valid
    val = validate_llm_response({"answer": "Good", "suggested_action": "walk", "confidence": 0.8})
    assert val["answer"] == "Good"
    assert val["confidence"] == 0.8

    # Null suggested action is valid
    val_null = validate_llm_response(
        {"answer": "Good", "suggested_action": None, "confidence": 0.5}
    )
    assert val_null["suggested_action"] is None

    # Missing answer
    with pytest.raises(
        LlmValidationError, match="missing required non-empty string field 'answer'"
    ):
        validate_llm_response({"suggested_action": None, "confidence": 0.8})

    # Empty answer
    with pytest.raises(
        LlmValidationError, match="missing required non-empty string field 'answer'"
    ):
        validate_llm_response({"answer": "   ", "confidence": 0.8})

    # Missing confidence
    with pytest.raises(LlmValidationError, match="missing required field 'confidence'"):
        validate_llm_response({"answer": "Valid"})

    # Out of range confidence
    with pytest.raises(LlmValidationError, match="must be between 0.0 and 1.0"):
        validate_llm_response({"answer": "Valid", "confidence": 1.5})
    with pytest.raises(LlmValidationError, match="must be between 0.0 and 1.0"):
        validate_llm_response({"answer": "Valid", "confidence": -0.1})


async def test_malformed_response_marks_degraded_and_does_not_persist(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify malformed JSON returns 502, marks degraded, and does NOT persist."""
    app, client, _ = llm_test_env
    db: Database = app.state.db

    # Inject a transport that returns invalid JSON text
    bad_transport = make_fake_ollama_transport(
        response_content="This is plain text, not JSON at all!"
    )
    llm_mod = app.state.registry.get_module("llm_ollama")
    assert isinstance(llm_mod, LlmOllamaModule)
    llm_mod.client = OllamaClient(transport=bad_transport)

    resp = await client.post("/api/ask", json={"question": "hello?"})
    assert resp.status_code == 502
    err_body = resp.json()
    assert err_body["error"] == "validation_error"

    # Verify NOT persisted in SQLite
    interactions = await db.list_llm_interactions()
    assert len(interactions) == 0

    # Verify module is marked degraded
    health_row = await db.get_module_health("llm_ollama")
    assert health_row is not None
    assert health_row["status"] == "degraded"
    assert "LlmValidationError" in str(health_row["last_error"])


# ---------------------------------------------------------------------------
# 8. Disabled Module Handling
# ---------------------------------------------------------------------------


async def test_disabled_module_returns_503(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /api/ask returns 503 when llm_ollama is disabled."""
    monkeypatch.setattr(LlmOllamaModule, "_dependency_override", True)
    # llm_ollama omitted from enabled outputs
    test_settings.enabled_modules.outputs = ["waybar_status", "desktop_notify"]

    app = create_app(settings=test_settings)
    app.include_router(api_router)

    async with app.router.lifespan_context(app):
        asgi_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=asgi_transport, base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/ask", json={"question": "Are you there?"})
            assert resp.status_code == 503
            data = resp.json()
            assert "Enable llm_ollama in Settings" in data["detail"]


# ---------------------------------------------------------------------------
# 9. Dependency Group Missing Handling
# ---------------------------------------------------------------------------


async def test_missing_dependency_returns_503(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /api/ask returns 503 when ollama dependency group is missing."""
    monkeypatch.setattr(LlmOllamaModule, "_dependency_override", False)
    test_settings.enabled_modules.outputs = ["waybar_status", "desktop_notify", "llm_ollama"]

    app = create_app(settings=test_settings)
    app.include_router(api_router)

    async with app.router.lifespan_context(app):
        asgi_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=asgi_transport, base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/ask", json={"question": "Test missing dep"})
            assert resp.status_code == 503
            data = resp.json()
            assert "dependency group is missing" in data["detail"]
            assert "uv sync --group llm" in data["detail"]


# ---------------------------------------------------------------------------
# 10. Privacy Global Pause Handling
# ---------------------------------------------------------------------------


async def test_privacy_global_pause_returns_409(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify POST /api/ask returns 409 Conflict when privacy.global_pause is active."""
    app, client, _ = llm_test_env
    app.state.settings.privacy.global_pause = True

    # JSON caller
    resp = await client.post("/api/ask", json={"question": "How are my timers?"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "privacy_paused"
    assert "Privacy global pause is active" in body["detail"]

    # HTMX caller
    htmx_resp = await client.post(
        "/api/ask",
        json={"question": "How are my timers?"},
        headers={"HX-Request": "true"},
    )
    assert htmx_resp.status_code == 409
    assert "alert-warning" in htmx_resp.text


# ---------------------------------------------------------------------------
# 11. Ollama Outage / Timeout / Connection-Refused Isolation
# ---------------------------------------------------------------------------


async def test_ollama_connection_refused_isolation(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify connection error returns 503 cleanly and marks degraded."""
    app, client, _ = llm_test_env
    db: Database = app.state.db

    conn_error = httpx.ConnectError("Connection refused to 127.0.0.1:11434")
    failing_transport = make_fake_ollama_transport(side_effect=conn_error)

    llm_mod = app.state.registry.get_module("llm_ollama")
    assert isinstance(llm_mod, LlmOllamaModule)
    llm_mod.client = OllamaClient(transport=failing_transport)

    resp = await client.post("/api/ask", json={"question": "Is Ollama up?"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "ollama_error"

    # Module marked degraded
    health = await db.get_module_health("llm_ollama")
    assert health is not None
    assert health["status"] == "degraded"
    assert "LlmConnectionError" in str(health["last_error"])

    # Other endpoints continue working normally (core resilience)
    state_resp = await client.get("/api/state")
    assert state_resp.status_code == 200
    assert "activity" in state_resp.json()


async def test_ollama_timeout_isolation(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify timeout returns 503 cleanly and marks module degraded."""
    app, client, _ = llm_test_env
    db: Database = app.state.db

    timeout_error = httpx.ReadTimeout("Request timed out after 30s")
    failing_transport = make_fake_ollama_transport(side_effect=timeout_error)

    llm_mod = app.state.registry.get_module("llm_ollama")
    assert isinstance(llm_mod, LlmOllamaModule)
    llm_mod.client = OllamaClient(transport=failing_transport)

    resp = await client.post("/api/ask", json={"question": "Heavy prompt?"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "ollama_error"

    health = await db.get_module_health("llm_ollama")
    assert health is not None
    assert health["status"] == "degraded"
    assert "LlmTimeoutError" in str(health["last_error"])


# ---------------------------------------------------------------------------
# 12. History Endpoint (GET /api/ask/history)
# ---------------------------------------------------------------------------


async def test_get_ask_history_endpoint(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify GET /api/ask/history returns recent LLM interactions."""
    app, client, _ = llm_test_env
    db: Database = app.state.db

    for i in range(3):
        await db.record_llm_interaction(
            source="dashboard",
            question=f"Question {i}",
            answer=f"Answer {i}",
            suggested_action=None,
            confidence=0.9,
            model="qwen3:14b",
        )

    resp = await client.get("/api/ask/history?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["interactions"]) == 2
    assert data["interactions"][0]["question"] == "Question 2"
    assert data["interactions"][1]["question"] == "Question 1"


# ---------------------------------------------------------------------------
# 13. Optional Calendar Classification Fallback
# ---------------------------------------------------------------------------


async def test_optional_calendar_classification_fallback() -> None:
    """Verify classify_event_with_llm integrates with classify_calendar_event when enabled."""
    now = datetime.now(tz=UTC)
    ambiguous_event = CalendarEvent(
        event_id="ambig-1",
        title="Quarterly Strategy Deep Dive",
        start_time=now,
        end_time=now + timedelta(hours=2),
    )

    fake_transport = make_fake_ollama_transport(
        response_content={"task_type": "planning", "confidence": 0.88}
    )
    fake_client = OllamaClient(transport=fake_transport)

    # 1. Fallback disabled in settings -> returns None
    settings_disabled = Settings()
    settings_disabled.llm.enable_classification_fallback = False
    res_disabled = await classify_event_with_llm(
        title=ambiguous_event.title,
        client=fake_client,
        settings=settings_disabled,
    )
    assert res_disabled is None

    # 2. Fallback enabled in settings -> classifies into planning
    settings_enabled = Settings()
    settings_enabled.llm.enable_classification_fallback = True
    res_enabled = await classify_event_with_llm(
        title=ambiguous_event.title,
        client=fake_client,
        settings=settings_enabled,
    )
    assert res_enabled is not None
    assert res_enabled["task_type"] == "planning"
    assert res_enabled["confidence"] == 0.88

    # 3. Integration with classify_calendar_event
    async def fallback_callable(title: str, description: str | None = None, **kwargs: Any) -> Any:
        return await classify_event_with_llm(
            title=title,
            description=description,
            client=fake_client,
            settings=settings_enabled,
        )

    classified = await classify_calendar_event(
        ambiguous_event,
        llm_fallback=fallback_callable,
    )
    assert classified.task_type == "planning"
    assert classified.classifier == "llm"
    assert classified.confidence == 0.88
    await fake_client.close()


async def test_classify_window_with_llm_integration() -> None:
    """Verify classify_window_with_llm correctly parses Ollama responses and respects settings."""
    from jarvis.core.llm import classify_window_with_llm

    fake_transport = make_fake_ollama_transport(
        response_content={"category": "focused", "confidence": 0.92, "reason": "Academic literature search"}
    )
    fake_client = OllamaClient(transport=fake_transport)

    # 1. Disabled in settings -> returns None
    settings_off = Settings()
    res_off = await classify_window_with_llm(
        app_class="firefox",
        window_title="Some Academic Blog",
        client=fake_client,
        settings=settings_off,
    )
    assert res_off is None

    # 2. Enabled in settings -> returns classified result
    settings_on = Settings()
    settings_on.focus.enable_llm_fallback = True
    res_on = await classify_window_with_llm(
        app_class="firefox",
        window_title="Some Academic Blog",
        client=fake_client,
        settings=settings_on,
    )
    assert res_on is not None
    assert res_on["category"] == "focused"
    assert res_on["confidence"] == 0.92
    assert res_on["reason"] == "Academic literature search"

    await fake_client.close()



# ---------------------------------------------------------------------------
# 14. LLM Safety Boundary: Inability to Mutate State
# ---------------------------------------------------------------------------


async def test_llm_safety_boundary_cannot_mutate_state(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify LLM only suggests actions and cannot mutate state, timers, or settings."""
    app, client, _ = llm_test_env
    db: Database = app.state.db

    # Establish baseline activity: sitting
    manual_mod = app.state.registry.get_module("manual_control")
    assert manual_mod is not None
    await manual_mod.set_activity("sitting")

    # Establish baseline timer: 50m coding Pomodoro
    timer_mgr = app.state.timer_manager
    assert timer_mgr is not None
    await timer_mgr.start_timer(kind="coding", work_minutes=50)

    # Verify baseline state before LLM call
    state_before = await client.get("/api/state")
    assert state_before.json()["activity"] == "sitting"
    assert state_before.json()["timer"]["active"] is True
    assert state_before.json()["timer"]["kind"] == "coding"

    # Configure mock LLM to propose drastic action: stand_up and stop_timer
    dangerous_content = {
        "answer": "I have executed commands to switch posture to standing and stop your timer.",
        "suggested_action": "stop_timer_and_stand",
        "confidence": 0.99,
    }
    dangerous_transport = make_fake_ollama_transport(response_content=dangerous_content)
    llm_mod = app.state.registry.get_module("llm_ollama")
    assert isinstance(llm_mod, LlmOllamaModule)
    llm_mod.client = OllamaClient(transport=dangerous_transport)

    # Invoke LLM
    resp = await client.post(
        "/api/ask",
        json={"question": "Force standing mode and kill all timers immediately!"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["suggested_action"] == "stop_timer_and_stand"

    # CRITICAL SAFETY CHECKS:
    # 1. State did NOT change to standing; remained sitting
    state_after = await client.get("/api/state")
    assert state_after.json()["activity"] == "sitting"

    # 2. Timer did NOT stop; still active coding Pomodoro
    assert state_after.json()["timer"]["active"] is True
    assert state_after.json()["timer"]["kind"] == "coding"

    # 3. No unauthorized settings mutation occurred
    assert app.state.settings.privacy.global_pause is False

    # 4. Activity span in DB is still sitting
    curr_span = await db.get_current_activity_span()
    assert curr_span is not None
    assert curr_span["activity"] == "sitting"


# ---------------------------------------------------------------------------
# 14. Calendar -> Assistant Chat End-to-End Regression Tests
# ---------------------------------------------------------------------------


async def test_calendar_to_chat_disabled_state(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Verify disabled calendar produces unavailable status and never claims 'no appointments'."""
    app, client, captured = llm_test_env
    captured.clear()

    # Ensure google_calendar is NOT enabled
    assert app.state.registry.is_enabled("google_calendar") is False

    resp = await client.post("/api/ask", json={"question": "What appointments do I have today?"})
    assert resp.status_code == 200
    assert len(captured) == 1

    messages = captured[0]["body"]["messages"]
    sys_prompt = messages[0]["content"]
    assert "Calendar guidelines" in sys_prompt
    assert '"status": "disabled"' in sys_prompt
    assert '"available": false' in sys_prompt
    assert "Calendar integration is disabled in settings." in sys_prompt
    # Never leaks credentials or tokens
    assert "token" not in sys_prompt.lower() or "google-token.json" not in sys_prompt


async def test_calendar_to_chat_auth_and_fetch_failure_states(
    db: Database,
) -> None:
    """Verify auth failures and network fetch failures are cleanly distinguished and safe."""
    # 1. Not authenticated: token missing / OAuth failure
    await db.upsert_module_health(
        module="google_calendar",
        status="degraded",
        last_error="FileNotFoundError: OAuth token file not found at: /path/secret-token.json",
    )
    ctx_auth = await build_assistant_context(db=db)
    cal_auth = ctx_auth["calendar"]
    assert cal_auth["status"] == "not_authenticated"
    assert cal_auth["available"] is False
    assert "secret-token.json" not in json.dumps(cal_auth)  # credentials sanitized
    assert "authentication failed" in cal_auth["message"].lower()

    # 2. Fetch failed: connection error
    await db.upsert_module_health(
        module="google_calendar",
        status="degraded",
        last_error="ConnectTimeout: Google API timed out",
    )
    ctx_fetch = await build_assistant_context(db=db)
    cal_fetch = ctx_fetch["calendar"]
    assert cal_fetch["status"] == "fetch_failed"
    assert cal_fetch["available"] is False
    assert "fetch failed" in cal_fetch["message"].lower()


async def test_calendar_to_chat_not_fetched_yet_state(
    db: Database,
) -> None:
    """Verify enabled calendar with no initial sync is marked not_fetched_yet."""
    # Healthy record but last_event_at is None and cache is empty
    await db.upsert_module_health(
        module="google_calendar",
        status="ok",
        last_event_at=None,
        last_error=None,
    )
    ctx = await build_assistant_context(db=db)
    cal = ctx["calendar"]
    assert cal["status"] == "not_fetched_yet"
    assert cal["available"] is False
    assert "not been fetched yet" in cal["message"].lower()


async def test_calendar_to_chat_empty_and_populated_events(
    db: Database,
) -> None:
    """Verify empty day vs populated appointments in user's local timezone."""
    now = datetime.now().astimezone()
    today = now.date()

    # 1. Mark module synced with no events
    await db.upsert_module_health(
        module="google_calendar",
        status="ok",
        last_event_at=now,
        last_error=None,
    )
    ctx_empty = await build_assistant_context(db=db)
    cal_empty = ctx_empty["calendar"]
    assert cal_empty["status"] == "no_events"
    assert cal_empty["available"] is True
    assert cal_empty["events_today"] == []
    assert "No calendar appointments scheduled for today" in cal_empty["message"]

    # 2. Add multiple events: earlier today, ongoing, later today, and tomorrow
    t_earlier_start = datetime.combine(today, dt_time(8, 30), tzinfo=now.tzinfo)
    t_earlier_end = datetime.combine(today, dt_time(9, 30), tzinfo=now.tzinfo)

    t_noon_start = datetime.combine(today, dt_time(12, 0), tzinfo=now.tzinfo)
    t_noon_end = datetime.combine(today, dt_time(13, 0), tzinfo=now.tzinfo)

    t_evening_start = datetime.combine(today, dt_time(19, 0), tzinfo=now.tzinfo)
    t_evening_end = datetime.combine(today, dt_time(20, 30), tzinfo=now.tzinfo)

    tomorrow = today + timedelta(days=1)
    t_tomorrow_start = datetime.combine(tomorrow, dt_time(9, 0), tzinfo=now.tzinfo)
    t_tomorrow_end = datetime.combine(tomorrow, dt_time(10, 0), tzinfo=now.tzinfo)

    await db.upsert_calendar_event(
        event_id="evt_early",
        title="Morning Sync",
        start_time=t_earlier_start,
        end_time=t_earlier_end,
        task_type="meeting",
    )
    await db.upsert_calendar_event(
        event_id="evt_noon",
        title="Architecture Discussion",
        start_time=t_noon_start,
        end_time=t_noon_end,
        task_type="deep_work",
    )
    await db.upsert_calendar_event(
        event_id="evt_eve",
        title="Evening Workout",
        start_time=t_evening_start,
        end_time=t_evening_end,
        task_type="break",
    )
    await db.upsert_calendar_event(
        event_id="evt_tomorrow",
        title="Tomorrow Standup",
        start_time=t_tomorrow_start,
        end_time=t_tomorrow_end,
        task_type="meeting",
    )

    ctx_populated = await build_assistant_context(db=db)
    cal_pop = ctx_populated["calendar"]
    assert cal_pop["status"] == "events_found"
    assert cal_pop["available"] is True
    assert len(cal_pop["events_today"]) == 3

    titles = [e["title"] for e in cal_pop["events_today"]]
    assert titles == ["Morning Sync", "Architecture Discussion", "Evening Workout"]
    assert "Tomorrow Standup" not in titles

    # Time windows formatted in HH:MM
    assert cal_pop["events_today"][0]["time_window"] == "08:30 – 09:30"
    assert cal_pop["events_today"][1]["time_window"] == "12:00 – 13:00"
    assert cal_pop["events_today"][2]["time_window"] == "19:00 – 20:30"


# ---------------------------------------------------------------------------
# 15. Calendar Multi-Day Range, Doctor's Appointment & Completeness Tests
# ---------------------------------------------------------------------------


def test_parse_calendar_query_range() -> None:
    """Verify natural-language calendar query date range extraction and full final day inclusion."""
    berlin_tz = ZoneInfo("Europe/Berlin")
    ref_now = datetime(2026, 11, 5, 12, 40, 0, tzinfo=berlin_tz)

    # 1. German multi-day date: "Was ist bis zum 8.11.2026 der wichtigste Termin?"
    q1 = "Was ist bis zum 8.11.2026 der wichtigste Termin?"
    s1, e1, exp1, desc1 = parse_calendar_query_range(q1, ref_now)
    assert exp1 is True
    assert s1 == datetime(2026, 11, 5, 0, 0, 0, tzinfo=berlin_tz)
    # Full final day (Nov 8) in Europe/Berlin must be included: end_dt must be 2026-11-09T00:00:00+01:00
    assert e1 == datetime(2026, 11, 9, 0, 0, 0, tzinfo=berlin_tz)
    assert "2026-11-05 to 2026-11-08" in desc1

    # 2. Typos & 2-digit year: "was denkst du ist der wichtigste dtermin diese woche bis zum 8.11.26?"
    q2 = "was denkst du ist der wichtigste dtermin diese woche bis zum 8.11.26? also den den man nicht verpassen sollte"
    s2, e2, exp2, desc2 = parse_calendar_query_range(q2, ref_now)
    assert exp2 is True
    assert s2 == datetime(2026, 11, 5, 0, 0, 0, tzinfo=berlin_tz)
    assert e2 == datetime(2026, 11, 9, 0, 0, 0, tzinfo=berlin_tz)

    # 3. Tomorrow: "Welche Termine habe ich morgen?"
    q3 = "Welche Termine habe ich morgen?"
    s3, e3, exp3, desc3 = parse_calendar_query_range(q3, ref_now)
    assert exp3 is True
    assert s3 == datetime(2026, 11, 6, 0, 0, 0, tzinfo=berlin_tz)
    assert e3 == datetime(2026, 11, 7, 0, 0, 0, tzinfo=berlin_tz)
    assert "2026-11-06" in desc3

    # 4. Named month: "Termine bis 8. November"
    q4 = "Termine bis 8. November"
    s4, e4, exp4, desc4 = parse_calendar_query_range(q4, ref_now)
    assert exp4 is True
    assert s4 == datetime(2026, 11, 5, 0, 0, 0, tzinfo=berlin_tz)
    assert e4 == datetime(2026, 11, 9, 0, 0, 0, tzinfo=berlin_tz)

    # 5. Weekday: "Termine bis Sonntag" (2026-11-05 is Thursday, Sunday is 2026-11-08)
    q5 = "Termine bis Sonntag"
    s5, e5, exp5, desc5 = parse_calendar_query_range(q5, ref_now)
    assert exp5 is True
    assert s5 == datetime(2026, 11, 5, 0, 0, 0, tzinfo=berlin_tz)
    assert e5 == datetime(2026, 11, 9, 0, 0, 0, tzinfo=berlin_tz)

    # 6. Default / today: "Was steht heute noch an?"
    q6 = "Was steht heute noch an?"
    s6, e6, exp6, desc6 = parse_calendar_query_range(q6, ref_now)
    assert exp6 is False
    assert s6 == datetime(2026, 11, 5, 0, 0, 0, tzinfo=berlin_tz)
    assert e6 == datetime(2026, 11, 6, 0, 0, 0, tzinfo=berlin_tz)


async def test_multi_day_calendar_query_includes_doctor_appointment_tomorrow(
    db: Database,
) -> None:
    """Verify multi-day question up to 8.11.2026 includes doctor's appointment tomorrow and full final day."""
    berlin_tz = ZoneInfo("Europe/Berlin")
    ref_now = datetime(2026, 11, 5, 12, 40, 0, tzinfo=berlin_tz)

    # Seed events matching user's scenario
    # 1. Today (2026-11-05): Bachelorarbeit
    await db.upsert_calendar_event(
        event_id="evt_today_thesis",
        title="Bachelorarbeit",
        start_time=datetime(2026, 11, 5, 11, 30, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 5, 12, 30, tzinfo=berlin_tz),
        task_type="deep_work",
        suitable_for_walking=True,
    )
    # 2. Tomorrow (2026-11-06): Morning routine
    await db.upsert_calendar_event(
        event_id="evt_tomorrow_wake",
        title="Aufstehen, Zaehneputzen, Kacken",
        start_time=datetime(2026, 11, 6, 7, 0, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 6, 7, 15, tzinfo=berlin_tz),
        task_type="admin",
        suitable_for_walking=False,
    )
    # 3. Tomorrow (2026-11-06): Doctor's appointment (Augenarzt)
    await db.upsert_calendar_event(
        event_id="evt_tomorrow_doctor",
        title="Augenarzt",
        start_time=datetime(2026, 11, 6, 8, 30, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 6, 9, 30, tzinfo=berlin_tz),
        task_type="meeting",
        suitable_for_walking=False,
    )
    # 4. Tomorrow (2026-11-06): Thesis block
    await db.upsert_calendar_event(
        event_id="evt_tomorrow_thesis",
        title="Bachelorarbeit",
        start_time=datetime(2026, 11, 6, 10, 0, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 6, 13, 0, tzinfo=berlin_tz),
        task_type="deep_work",
        suitable_for_walking=True,
    )
    # 5. Day 3 (2026-11-07): Study session
    await db.upsert_calendar_event(
        event_id="evt_nov7_study",
        title="Spanisch Grammatik",
        start_time=datetime(2026, 11, 7, 15, 0, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 7, 16, 0, tzinfo=berlin_tz),
        task_type="reading",
        suitable_for_walking=None,
    )
    # 6. Final day (2026-11-08): Late evening review on the final day
    await db.upsert_calendar_event(
        event_id="evt_nov8_evening",
        title="Wochenabschluss & Review",
        start_time=datetime(2026, 11, 8, 20, 0, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 8, 21, 30, tzinfo=berlin_tz),
        task_type="planning",
        suitable_for_walking=False,
    )
    # 7. Beyond range (2026-11-09): Next week standup
    await db.upsert_calendar_event(
        event_id="evt_nov9_standup",
        title="Next Week Standup",
        start_time=datetime(2026, 11, 9, 9, 0, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 9, 9, 30, tzinfo=berlin_tz),
        task_type="meeting",
        suitable_for_walking=False,
    )

    # Confirm sync coverage through Nov 14
    await db.set_setting("calendar.synced_until", "2026-11-14T23:59:59+01:00")

    question = "Was ist bis zum 8.11.2026 der wichtigste Termin?"
    ctx = await build_assistant_context(db=db, question=question, now=ref_now)
    cal = ctx["calendar"]

    # Verify query range and full final day inclusion in Europe/Berlin
    assert cal["query_range"]["start"].startswith("2026-11-05T00:00:00")
    assert cal["query_range"]["end"].startswith("2026-11-09T00:00:00")
    assert cal["query_range"]["full_final_day_included"] is True

    # Data is complete because synced_until covers through Nov 14
    assert cal["data_complete"] is True
    assert cal["incomplete_reason"] is None

    # Events in range must contain all events through Nov 8 (including tomorrow's doctor appointment)
    event_titles = [e["title"] for e in cal["events"]]
    assert "Augenarzt" in event_titles  # Doctor's appointment tomorrow included!
    assert "Wochenabschluss & Review" in event_titles  # Full final day event included!
    assert "Next Week Standup" not in event_titles  # Nov 9 excluded!

    # Verify doctor's appointment structure
    doctor_evt = next(e for e in cal["events"] if e["title"] == "Augenarzt")
    assert doctor_evt["date"] == "2026-11-06"
    assert doctor_evt["time_window"] == "08:30 – 09:30"
    assert doctor_evt["suitable_for_walking"] is False

    # events_today must remain today only
    today_titles = [e["title"] for e in cal["events_today"]]
    assert today_titles == ["Bachelorarbeit"]
    assert "Augenarzt" not in today_titles


async def test_multi_day_calendar_incomplete_data_detection(
    db: Database,
) -> None:
    """Verify that if calendar sync only covers up to tomorrow, incomplete data is flagged."""
    berlin_tz = ZoneInfo("Europe/Berlin")
    ref_now = datetime(2026, 11, 5, 12, 40, 0, tzinfo=berlin_tz)

    await db.upsert_calendar_event(
        event_id="evt_today",
        title="Bachelorarbeit",
        start_time=datetime(2026, 11, 5, 11, 30, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 5, 12, 30, tzinfo=berlin_tz),
        task_type="deep_work",
    )
    await db.upsert_calendar_event(
        event_id="evt_doc",
        title="Augenarzt",
        start_time=datetime(2026, 11, 6, 8, 30, tzinfo=berlin_tz),
        end_time=datetime(2026, 11, 6, 9, 30, tzinfo=berlin_tz),
        task_type="meeting",
    )

    # Sync was only 24 hours ahead (until Nov 6 12:40)
    await db.set_setting("calendar.synced_until", "2026-11-06T12:40:00+01:00")

    question = "Was ist bis zum 8.11.2026 der wichtigste Termin?"
    ctx = await build_assistant_context(db=db, question=question, now=ref_now)
    cal = ctx["calendar"]

    assert cal["data_complete"] is False
    assert cal["incomplete_reason"] is not None
    assert "2026-11-06" in cal["incomplete_reason"]
    assert "2026-11-08" in cal["incomplete_reason"] or "2026-11-09" in cal["incomplete_reason"]


async def test_calendar_chat_prompt_guidelines_and_doctor_appointment(
    llm_test_env: tuple[FastAPI, httpx.AsyncClient, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify chat API receives requested date range, doctor's appointment, and guidelines separating walking suitability."""
    app, client, captured = llm_test_env
    captured.clear()
    db = app.state.db
    berlin_tz = ZoneInfo("Europe/Berlin")

    # Mark calendar healthy and synced, and mock registry.is_enabled
    orig_is_enabled = app.state.registry.is_enabled
    monkeypatch.setattr(
        app.state.registry,
        "is_enabled",
        lambda name: True if name == "google_calendar" else orig_is_enabled(name),
    )
    now_dt = datetime.now().astimezone()
    await db.upsert_module_health(
        module="google_calendar",
        status="ok",
        last_event_at=now_dt,
        last_error=None,
    )
    # Sync through 14 days
    await db.set_setting(
        "calendar.synced_until",
        (now_dt + timedelta(days=14)).isoformat(),
    )

    # Seed doctor's appointment tomorrow and thesis today
    tomorrow = now_dt.date() + timedelta(days=1)
    await db.upsert_calendar_event(
        event_id="evt_thesis_today",
        title="Bachelorarbeit",
        start_time=datetime.combine(now_dt.date(), dt_time(11, 30), tzinfo=now_dt.tzinfo),
        end_time=datetime.combine(now_dt.date(), dt_time(12, 30), tzinfo=now_dt.tzinfo),
        task_type="deep_work",
        suitable_for_walking=True,
    )
    await db.upsert_calendar_event(
        event_id="evt_doctor_tomorrow",
        title="Augenarzt",
        start_time=datetime.combine(tomorrow, dt_time(8, 30), tzinfo=now_dt.tzinfo),
        end_time=datetime.combine(tomorrow, dt_time(9, 30), tzinfo=now_dt.tzinfo),
        task_type="meeting",
        suitable_for_walking=False,
    )

    # Ask multi-day question
    question = "Was ist bis zum 8.11.2026 der wichtigste Termin?"
    resp = await client.post("/api/ask", json={"question": question})
    assert resp.status_code == 200
    assert len(captured) == 1

    sys_prompt = captured[0]["body"]["messages"][0]["content"]

    # Verify guidelines:
    # 1. Incomplete data warning requirement
    assert "Incomplete data requirement" in sys_prompt
    assert "NEVER make a definitive claim about an entire" in sys_prompt

    # 2. Movement suitability separated from appointment importance
    assert "Movement suitability vs Appointment importance" in sys_prompt
    assert "NEVER confuse or conflate movement suitability with appointment importance" in sys_prompt
    assert "Do NOT volunteer walking or movement advice" in sys_prompt

    # 3. Importance assessment must explain basis and acknowledge uncertainty
    assert "Importance assessments" in sys_prompt
    assert "explain the basis for your assessment" in sys_prompt
    assert "acknowledge uncertainty" in sys_prompt

    # 4. Context delivered to model contains doctor's appointment
    assert "Augenarzt" in sys_prompt
