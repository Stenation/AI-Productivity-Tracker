"""Full vertical-slice Milestone 1 & 2 E2E, HTMX control-plane, and failure isolation tests."""

from __future__ import annotations

import asyncio
import json
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.llm import OllamaClient
from jarvis.core.schema import PlatformEvent, now_local
from jarvis.core.settings import Settings, load_settings
from jarvis.modules.outputs.llm_ollama import LlmOllamaModule
from jarvis.modules.outputs.tts_piper import AudioPlaybackError, TtsPiperModule
from jarvis.modules.policies.posture_policy import PosturePolicyModule


async def _wait_until(
    predicate: Any,
    *,
    timeout: float = 3.0,
    interval: float = 0.02,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        res = predicate()
        if asyncio.iscoroutine(res):
            res = await res
        if res:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"Predicate did not become true within {timeout}s")
        await asyncio.sleep(interval)


def _create_recording_notify_binary(tmp_path: Path) -> tuple[Path, Path]:
    """Create a local executable fake `notify-send` recording argv to JSONL."""
    log_path = tmp_path / "notifications.jsonl"
    script_path = tmp_path / "fake_notify_send.py"
    script_path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"log = pathlib.Path({str(log_path)!r})\n"
        "with log.open('a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return script_path, log_path


def _read_notify_log(log_path: Path) -> list[list[str]]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@asynccontextmanager
async def _fake_hyprland_server(sock_path: Path) -> AsyncIterator[list[asyncio.StreamWriter]]:
    """Provide a simulated Hyprland Unix domain socket server and yield connected writers."""
    writers: list[asyncio.StreamWriter] = []

    async def _handle_client(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.append(writer)

    server = await asyncio.start_unix_server(_handle_client, path=str(sock_path))
    try:
        yield writers
    finally:
        for w in writers:
            with suppress(Exception):
                w.close()
        server.close()
        await server.wait_closed()


def _build_e2e_settings(
    tmp_path: Path,
    *,
    notify_binary: Path,
    hyprland_socket: Path | None = None,
) -> tuple[Settings, Path]:
    """Load real `config.yaml` defaults and isolate DB, Waybar JSON, and notify binary."""
    settings = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    waybar_path = tmp_path / "posture_waybar.json"

    settings.database.path = tmp_path / "e2e_posture.db"
    settings.waybar.status_file = waybar_path
    settings.waybar.signal_waybar = False
    settings.waybar.refresh_interval_seconds = 60.0
    settings.desktop_notify.notify_binary = str(notify_binary)
    if hyprland_socket is not None:
        settings.focus.socket_path = hyprland_socket

    return settings, waybar_path


def _make_fake_ollama_transport(
    response_content: str | dict[str, Any] | None = None,
    *,
    status_code: int = 200,
    model: str = "qwen3:14b",
    latency_ms: int = 145,
    side_effect: Exception | None = None,
) -> httpx.MockTransport:
    """Create httpx.MockTransport simulating Ollama chat completion."""
    def handler(request: httpx.Request) -> httpx.Response:
        if side_effect is not None:
            raise side_effect
        if isinstance(response_content, dict):
            content_str = json.dumps(response_content)
        elif response_content is not None:
            if "{" in response_content and "}" in response_content:
                content_str = response_content
            else:
                content_str = json.dumps({
                    "answer": response_content,
                    "suggested_action": None,
                    "confidence": 0.95,
                })
        else:
            content_str = json.dumps({
                "answer": "You have been sitting for 45 minutes. Time to stand up.",
                "suggested_action": "stand_up",
                "confidence": 0.95,
            })
        resp_payload = {
            "model": model,
            "created_at": "2026-11-04T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": content_str,
            },
            "done": True,
            "total_duration": latency_ms * 1_000_000,
        }
        return httpx.Response(status_code, json=resp_payload)

    return httpx.MockTransport(handler)


class _RecordingPlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []
        self.raise_error: Exception | None = None

    def is_available(self) -> tuple[bool, str | None]:
        return True, None

    async def play(self, audio_path: Path) -> None:
        if self.raise_error:
            raise self.raise_error
        self.played.append(audio_path)


class _RecordingSynthesizer:
    def __init__(self) -> None:
        self.synthesized: list[tuple[str, Path]] = []
        self.raise_error: Exception | None = None

    def is_available(self) -> tuple[bool, str | None]:
        return True, None

    async def synthesize(self, text: str, output_wav: Path) -> Path:
        if self.raise_error:
            raise self.raise_error
        self.synthesized.append((text, output_wav))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00faketts")
        return output_wav


# ==============================================================================
# Milestone 1 Vertical Slice (Preserved & Updated for M2 Defaults)
# ==============================================================================


async def test_vertical_slice_milestone_1_e2e(tmp_path: Path) -> None:
    """End-to-end slice across HTML, manual state, timers, policies, Waybar, and privacy."""
    notify_bin, notify_log = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_m1.sock"

    async with _fake_hyprland_server(sock_path):
        settings, waybar_path = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )

        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8000
        assert settings.enabled_modules.inputs == ["manual_control", "hyprland_ipc"]
        assert settings.enabled_modules.policies == [
            "posture_policy",
            "pomodoro_policy",
            "meeting_policy",
            "focus_policy",
        ]
        assert settings.enabled_modules.outputs == ["waybar_status", "desktop_notify"]

        # ------------------------------------------------------------------
        # Boot #1: Verify pages, manual cycle, Waybar output, and 25_5 timer
        # ------------------------------------------------------------------
        app1 = create_app(settings=settings)
        started_timer_id: int | None = None

        async with app1.router.lifespan_context(app1):
            transport1 = httpx.ASGITransport(app=app1)
            async with httpx.AsyncClient(
                transport=transport1,
                base_url="http://127.0.0.1:8000",
            ) as client1:
                # 1. Verify `/`, `/history`, `/settings`, and local static HTMX return 200 OK
                health_resp = await client1.get("/api/health")
                assert health_resp.status_code == 200
                health_data = health_resp.json()
                assert health_data["status"] == "ok"
                assert health_data["degraded_modules"] == []
                enabled_names = {
                    m["module"] for m in health_data["modules"] if m["enabled"]
                }
                assert enabled_names == {
                    "manual_control",
                    "hyprland_ipc",
                    "posture_policy",
                    "pomodoro_policy",
                    "meeting_policy",
                    "focus_policy",
                    "waybar_status",
                    "desktop_notify",
                }

                dash_resp = await client1.get("/")
                assert dash_resp.status_code == 200
                assert "text/html" in dash_resp.headers["content-type"]
                dash_html = dash_resp.text
                assert "/static/htmx.min.js" in dash_html
                assert "/static/app.js" in dash_html
                assert "/static/styles.css" in dash_html
                assert "https://" not in dash_html
                assert "http://" not in dash_html
                assert 'hx-trigger="every 3s' in dash_html
                for label in (
                    "Sitting",
                    "Standing",
                    "Walking",
                    "Away",
                    "Cycle",
                    "Start 25/5",
                    "Start 50/10",
                    "Stop",
                    "Privacy Pause",
                    "DND Mode",
                ):
                    assert label in dash_html

                history_resp = await client1.get("/history")
                assert history_resp.status_code == 200
                assert "text/html" in history_resp.headers["content-type"]
                assert "Recent Activity Spans" in history_resp.text
                assert "Recommendation History" in history_resp.text
                assert 'data-daily-activity="sitting"' in history_resp.text
                assert 'data-daily-activity="standing"' in history_resp.text
                assert 'data-daily-activity="walking"' in history_resp.text

                settings_resp = await client1.get("/settings")
                assert settings_resp.status_code == 200
                assert "text/html" in settings_resp.headers["content-type"]
                assert "Modules &amp; Runtime Health" in settings_resp.text
                assert "manual_control" in settings_resp.text
                assert "hyprland_ipc" in settings_resp.text
                assert "posture_policy" in settings_resp.text
                assert "focus_policy" in settings_resp.text
                assert "waybar_status" in settings_resp.text
                assert "desktop_notify" in settings_resp.text

                htmx_js_resp = await client1.get("/static/htmx.min.js")
                assert htmx_js_resp.status_code == 200
                assert "htmx" in htmx_js_resp.text

                # 2. Cycle manual state via `/api/activity/cycle` -> SQLite + Waybar JSON
                assert waybar_path.exists()
                initial_wb = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "UNKNOWN" in initial_wb["text"]

                cycle1 = await client1.post("/api/activity/cycle")
                assert cycle1.status_code == 200
                assert cycle1.json()["activity"] == "sitting"

                db1: Database = app1.state.db
                current_span = await db1.get_current_activity_span()
                assert current_span is not None
                assert current_span["activity"] == "sitting"
                assert current_span["source"] == "manual_control"
                assert current_span["ended_at"] is None

                wb_after_sit = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "SIT" in wb_after_sit["text"]
                assert wb_after_sit["class"] == "sitting"
                assert "Manual override" in wb_after_sit["tooltip"]

                # Complete the full cycle back to `sitting` (standing -> walking -> away -> sitting)
                for expected_act, expected_short in (
                    ("standing", "STAND"),
                    ("walking", "WALK"),
                    ("away", "AWAY"),
                    ("sitting", "SIT"),
                ):
                    resp = await client1.post("/api/activity/cycle")
                    assert resp.status_code == 200
                    assert resp.json()["activity"] == expected_act
                    wb_data = json.loads(waybar_path.read_text(encoding="utf-8"))
                    assert expected_short in wb_data["text"]

                # 3. Start `25_5` timer via `/api/timers/start` -> Waybar JSON & `/api/state`
                timer_start_resp = await client1.post(
                    "/api/timers/start",
                    json={"kind": "25_5"},
                )
                assert timer_start_resp.status_code == 200
                timer_payload = timer_start_resp.json()
                assert timer_payload["active"] is True
                assert timer_payload["kind"] == "25_5"
                assert timer_payload["phase"] == "work"
                started_timer_id = int(timer_payload["timer_id"])

                wb_with_timer = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "25/5" in wb_with_timer["text"]

                state_with_timer = (await client1.get("/api/state")).json()
                assert state_with_timer["timer"]["active"] is True
                assert state_with_timer["timer"]["kind"] == "25_5"
                assert state_with_timer["active_timer"]["id"] == started_timer_id

        # ------------------------------------------------------------------
        # Boot #2: Simulate app restart -> interrupted timer, posture policy,
        #          dedupe/cooldown, non-destructive accept, Privacy & DND
        # ------------------------------------------------------------------
        assert started_timer_id is not None
        app2 = create_app(settings=settings)

        async with app2.router.lifespan_context(app2):
            transport2 = httpx.ASGITransport(app=app2)
            async with httpx.AsyncClient(
                transport=transport2,
                base_url="http://127.0.0.1:8000",
            ) as client2:
                db2: Database = app2.state.db
                bus2: EventBus = app2.state.bus

                # Verify in-flight timer was marked `interrupted` and NOT silently resumed
                state_after_restart = (await client2.get("/api/state")).json()
                assert state_after_restart["timer"]["active"] is False
                assert state_after_restart["timer"]["phase"] == "idle"
                assert state_after_restart["active_timer"] is None

                wb_after_restart = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "25/5" not in wb_after_restart["text"]

                timer_rows = await db2.list_timer_spans(limit=10)
                assert len(timer_rows) == 1
                assert timer_rows[0]["id"] == started_timer_id
                assert timer_rows[0]["status"] == "interrupted"
                assert timer_rows[0]["ended_at"] is not None

                history_after_restart = await client2.get("/history")
                assert history_after_restart.status_code == 200
                assert 'data-timer-status="interrupted"' in history_after_restart.text
                assert "interrupted" in history_after_restart.text

                # 4. Backdate open `sitting` span past `max_sitting_minutes` (50m) -> 65m ago
                open_sit = await db2.get_current_activity_span()
                assert open_sit is not None
                assert open_sit["activity"] == "sitting"
                open_sit_id = int(open_sit["id"])

                backdated_start = now_local() - timedelta(
                    minutes=settings.posture.max_sitting_minutes + 15
                )
                await db2.connection.execute(
                    "UPDATE activity_spans SET started_at = ? WHERE id = ?",
                    (backdated_start.isoformat(), open_sit_id),
                )
                await db2.connection.commit()
                await app2.state.resolver.start()

                posture_mod = app2.state.registry.get_module("posture_policy")
                assert isinstance(posture_mod, PosturePolicyModule)

                created_rec = await posture_mod.evaluate()
                assert created_rec is not None
                rec_id = int(created_rec["id"])
                assert created_rec["dedupe_key"] == "posture:sitting"
                assert created_rec["status"] == "pending"

                # Appears in `/api/recommendations`, `/` HTML, Waybar JSON, and desktop_notify
                recs_resp = await client2.get("/api/recommendations")
                assert recs_resp.status_code == 200
                recs_json = recs_resp.json()
                assert recs_json["count"] == 1
                assert recs_json["recommendations"][0]["id"] == rec_id
                assert recs_json["recommendations"][0]["dedupe_key"] == "posture:sitting"

                dash_with_rec = await client2.get("/")
                assert dash_with_rec.status_code == 200
                assert 'data-dedupe-key="posture:sitting"' in dash_with_rec.text
                assert created_rec["message"] in dash_with_rec.text
                assert "1 pending" in dash_with_rec.text

                wb_with_rec = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "Pending recommendations: 1" in wb_with_rec["tooltip"]

                notifications_after_rec = _read_notify_log(notify_log)
                assert len(notifications_after_rec) == 1
                assert any("Posture Recommendation" in arg for arg in notifications_after_rec[0])

                # Re-evaluating while pending creates NO duplicate
                assert await posture_mod.evaluate() is None
                assert (await client2.get("/api/recommendations")).json()["count"] == 1

                # Accepting marks `accepted` WITHOUT starting a timer or changing posture state
                spans_before_accept = await db2.list_activity_spans()
                accept_resp = await client2.post(f"/api/recommendations/{rec_id}/accept")
                assert accept_resp.status_code == 200
                assert accept_resp.json()["recommendation"]["status"] == "accepted"

                state_after_accept = (await client2.get("/api/state")).json()
                assert state_after_accept["timer"]["active"] is False
                assert state_after_accept["active_timer"] is None
                assert state_after_accept["activity"] == "sitting"
                assert state_after_accept["pending_recommendation_count"] == 0
                assert len(await db2.list_activity_spans()) == len(spans_before_accept)

                # Re-evaluating within cooldown after resolution creates NO duplicate
                assert await posture_mod.evaluate() is None
                assert (await client2.get("/api/recommendations")).json()["count"] == 0
                all_recs = await db2.list_recommendations()
                assert len(all_recs) == 1
                assert all_recs[0]["status"] == "accepted"

                # 5. Toggle `DND` and `Privacy Pause` -> suppress notifications & state updates
                dnd_on_resp = await client2.post("/api/privacy/dnd", json={"dnd_mode": True})
                assert dnd_on_resp.status_code == 200
                assert dnd_on_resp.json()["dnd_mode"] is True
                assert "DND" in json.loads(waybar_path.read_text(encoding="utf-8"))["text"]

                # Emit a recommendation while DND is ON -> desktop_notify MUST suppress it
                notify_count_before_dnd = len(_read_notify_log(notify_log))
                await bus2.publish(
                    PlatformEvent(
                        type="recommendation_created",
                        source="pomodoro_policy",
                        payload={
                            "id": 9001,
                            "policy": "pomodoro_policy",
                            "message": "Start 50/10 block",
                            "status": "pending",
                        },
                    )
                )
                assert len(_read_notify_log(notify_log)) == notify_count_before_dnd

                await client2.post("/api/privacy/dnd", json={"dnd_mode": False})

                # Enable Privacy Pause -> suppresses external events & sensor updates
                pause_resp = await client2.post("/api/privacy/pause")
                assert pause_resp.status_code == 200
                assert pause_resp.json()["privacy_paused"] is True

                wb_paused = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "PAUSED" in wb_paused["text"]
                assert wb_paused["class"] == "paused"

                spans_before_pause_events = len(await db2.list_activity_spans())
                ingest_paused = await client2.post(
                    "/api/events",
                    json={
                        "type": "activity_reported",
                        "source": "ble_swalk",
                        "payload": {"activity": "walking", "confidence": 0.99},
                    },
                )
                assert ingest_paused.status_code == 200
                assert ingest_paused.json()["dispatched"] is False
                assert ingest_paused.json()["reason"] == "global_pause"

                # Direct internal report on EventBus is also ignored by StateResolver during pause
                await bus2.publish(
                    PlatformEvent(
                        type="activity_reported",
                        source="manual_control",
                        payload={"activity": "walking", "confidence": 1.0, "ttl_minutes": 30},
                    )
                )
                assert (await client2.get("/api/state")).json()["activity"] == "sitting"
                assert len(await db2.list_activity_spans()) == spans_before_pause_events

                # Notifications are also suppressed during Privacy Pause
                await bus2.publish(
                    PlatformEvent(
                        type="recommendation_created",
                        source="meeting_policy",
                        payload={
                            "id": 9002,
                            "policy": "meeting_policy",
                            "message": "Standup in 5 minutes",
                            "status": "pending",
                        },
                    )
                )
                assert len(_read_notify_log(notify_log)) == notify_count_before_dnd

                # Resume Privacy Pause -> live posture returns
                resume_resp = await client2.post("/api/privacy/resume")
                assert resume_resp.status_code == 200
                assert resume_resp.json()["privacy_paused"] is False
                wb_resumed = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "SIT" in wb_resumed["text"]


async def test_htmx_partials_and_hx_request_actions(tmp_path: Path) -> None:
    """Verify HTMX partial endpoints and `HX-Request: true` HTML + `HX-Trigger` responses."""
    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_htmx.sock"

    async with _fake_hyprland_server(sock_path):
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)
        hx_headers = {"HX-Request": "true"}

        async with app.router.lifespan_context(app):
            db: Database = app.state.db
            now_utc = datetime.now(tz=UTC)
            await db.upsert_calendar_event(
                event_id="evt_review",
                title="Design Review",
                start_time=now_utc + timedelta(minutes=12),
                end_time=now_utc + timedelta(minutes=42),
                task_type="reading",
                suitable_for_walking=True,
            )

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # All 9 HTMX polling partials return 200 HTML fragments
                for partial_url, expected_snippet in (
                    ("/partials/posture", "Effective Posture"),
                    ("/partials/timer", "Local Pomodoro Timer"),
                    ("/partials/calendar", "Design Review"),
                    ("/partials/recommendations", "Pending Recommendations"),
                    ("/partials/privacy", "Privacy &amp; Mode Switches"),
                    ("/partials/focus", "Desktop Focus"),
                    ("/partials/ask", "Local AI Assistant"),
                    ("/partials/history", "Recent Activity Spans"),
                    ("/partials/modules", "Modules &amp; Runtime Health"),
                ):
                    part_resp = await client.get(partial_url, headers=hx_headers)
                    assert part_resp.status_code == 200
                    assert "text/html" in part_resp.headers["content-type"]
                    assert "<!DOCTYPE html>" not in part_resp.text
                    assert expected_snippet in part_resp.text

                # HTMX form POST to `/api/activity` returns posture partial + HX-Trigger
                act_hx = await client.post(
                    "/api/activity",
                    data={"activity": "walking"},
                    headers=hx_headers,
                )
                assert act_hx.status_code == 200
                assert "text/html" in act_hx.headers["content-type"]
                assert act_hx.headers.get("HX-Trigger") == "activityChanged"
                assert "Walking" in act_hx.text
                assert "Manual override" in act_hx.text

                # HTMX form POST to `/api/timers/start` & `/api/timers/stop`
                timer_start_hx = await client.post(
                    "/api/timers/start",
                    data={"kind": "50_10"},
                    headers=hx_headers,
                )
                assert timer_start_hx.status_code == 200
                assert timer_start_hx.headers.get("HX-Trigger") == "timerChanged"
                assert "50/10" in timer_start_hx.text

                timer_stop_hx = await client.post("/api/timers/stop", headers=hx_headers)
                assert timer_stop_hx.status_code == 200
                assert timer_stop_hx.headers.get("HX-Trigger") == "timerChanged"
                assert "Idle" in timer_stop_hx.text

                # HTMX recommendation accept / snooze / dismiss partials
                rec1 = await db.create_recommendation(
                    policy="posture_policy",
                    message="Time to stretch",
                    suggested_action="stand",
                    dedupe_key="posture:test1",
                )
                rec2 = await db.create_recommendation(
                    policy="pomodoro_policy",
                    message="Try 25/5 focus block",
                    suggested_action="start_25_5",
                    dedupe_key="pomodoro:test2",
                )
                rec3 = await db.create_recommendation(
                    policy="meeting_policy",
                    message="Meeting in 8 minutes",
                    suggested_action="return_to_desk",
                    dedupe_key="meeting:test3",
                )

                acc_hx = await client.post(
                    f"/api/recommendations/{rec1['id']}/accept",
                    headers=hx_headers,
                )
                assert acc_hx.status_code == 200
                assert acc_hx.headers.get("HX-Trigger") == "recommendationChanged"
                assert "accepted" in acc_hx.text
                assert "2 pending" in acc_hx.text

                snz_hx = await client.post(
                    f"/api/recommendations/{rec2['id']}/snooze",
                    data={"snooze_minutes": "15"},
                    headers=hx_headers,
                )
                assert snz_hx.status_code == 200
                assert snz_hx.headers.get("HX-Trigger") == "recommendationChanged"
                assert "snoozed" in snz_hx.text
                assert "1 pending" in snz_hx.text

                dis_hx = await client.post(
                    f"/api/recommendations/{rec3['id']}/dismiss",
                    headers=hx_headers,
                )
                assert dis_hx.status_code == 200
                assert dis_hx.headers.get("HX-Trigger") == "recommendationChanged"
                assert "dismissed" in dis_hx.text
                assert "0 pending" in dis_hx.text

                # HTMX privacy switches + live module disable/enable on `/settings`
                dnd_hx = await client.post(
                    "/api/privacy/dnd",
                    data={"dnd_mode": "true"},
                    headers=hx_headers,
                )
                assert dnd_hx.status_code == 200
                assert dnd_hx.headers.get("HX-Trigger") == "privacyChanged"
                assert "DND On" in dnd_hx.text

                pause_hx = await client.post("/api/privacy/pause", headers=hx_headers)
                assert pause_hx.status_code == 200
                assert pause_hx.headers.get("HX-Trigger") == "privacyChanged"
                assert "Privacy Paused" in pause_hx.text

                resume_hx = await client.post("/api/privacy/resume", headers=hx_headers)
                assert resume_hx.status_code == 200
                assert resume_hx.headers.get("HX-Trigger") == "privacyChanged"
                assert "Monitoring Active" in resume_hx.text

                mod_dis_hx = await client.post(
                    "/api/modules/meeting_policy/disable",
                    headers=hx_headers,
                )
                assert mod_dis_hx.status_code == 200
                assert mod_dis_hx.headers.get("HX-Trigger") == "modulesChanged"
                assert "/api/modules/meeting_policy/enable" in mod_dis_hx.text

                mod_en_hx = await client.post(
                    "/api/modules/meeting_policy/enable",
                    headers=hx_headers,
                )
                assert mod_en_hx.status_code == 200
                assert mod_en_hx.headers.get("HX-Trigger") == "modulesChanged"
                assert "/api/modules/meeting_policy/disable" in mod_en_hx.text


async def test_dashboard_refresh_non_interrupting_typing_and_selection(tmp_path: Path) -> None:
    """Verify dashboard polling updates do not interrupt active inputs, forms, or text selection."""
    import shutil
    import subprocess

    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_non_interrupt.sock"

    async with _fake_hyprland_server(sock_path):
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                dash_resp = await client.get("/")
                assert dash_resp.status_code == 200
                html = dash_resp.text

                # 1. Base layout loads non-interrupt client controller
                assert '<script src="/static/app.js" defer></script>' in html

                # 2. Ask panel does NOT poll every 3s (which previously wiped out input & responses)
                # Instead, it responds to explicit module/privacy change events
                assert 'id="ask-panel"' in html
                assert 'hx-trigger="modulesChanged from:body, privacyChanged from:body"' in html
                assert 'every 3s, modulesChanged' not in html

                # 3. Dynamic metric panels retain their periodic 3s polling
                assert 'id="posture-panel"' in html
                assert 'id="timer-panel"' in html
                assert 'id="calendar-panel"' in html
                assert 'id="recommendations-panel"' in html
                assert 'hx-trigger="every 3s' in html

                # 4. Ask partial form and response containers use hx-preserve when LLM is enabled
                await client.post("/api/modules/llm_ollama/enable")
                part_resp = await client.get("/partials/ask", headers={"HX-Request": "true"})
                assert part_resp.status_code == 200
                part_html = part_resp.text
                assert 'id="ask-form"' in part_html
                assert 'hx-preserve="true"' in part_html
                assert 'id="ask-question-input"' in part_html
                assert '<div id="ask-response" hx-preserve="true"' in part_html

                # 5. Static app.js is served and contains non-interrupting logic
                app_js_resp = await client.get("/static/app.js")
                assert app_js_resp.status_code == 200
                js_content = app_js_resp.text
                assert "hasSelectionIntersecting" in js_content
                assert "hasActiveInput" in js_content
                assert "shouldBlockUpdate" in js_content
                assert "htmx:beforeRequest" in js_content
                assert "htmx:beforeSwap" in js_content

                # 6. Verify JavaScript controller behavior if Node.js is present
                node_bin = shutil.which("node")
                if node_bin:
                    node_script = """
const fs = require('fs');
const code = fs.readFileSync('src/jarvis/web/static/app.js', 'utf8');

const listeners = {};
const windowListeners = {};

global.window = {
  addEventListener(evt, cb) {
    windowListeners[evt] = windowListeners[evt] || [];
    windowListeners[evt].push(cb);
  },
  getSelection() { return global.mockSelection; }
};

global.document = {
  activeElement: null,
  addEventListener(evt, cb) {
    listeners[evt] = listeners[evt] || [];
    listeners[evt].push(cb);
  },
  getElementById(id) { return global.mockElements[id] || null; }
};

global.mockSelection = { isCollapsed: true, rangeCount: 0, toString: () => '' };
global.mockElements = {};

eval(code);

// Test A: Idle state -> Polling request allowed
let prevented = false;
let fakeEvent = {
  detail: {
    target: { contains: () => false, querySelectorAll: () => [] },
    elt: {},
    requestConfig: {}
  },
  preventDefault() { prevented = true; }
};
listeners['htmx:beforeRequest'][0](fakeEvent);
if (prevented) process.exit(1);

// Test B: Active text selection -> Polling request blocked
prevented = false;
global.mockSelection = {
  isCollapsed: false,
  rangeCount: 1,
  toString: () => 'Sitting for 45 minutes',
  getRangeAt: () => ({ intersectsNode: () => true })
};
listeners['htmx:beforeRequest'][0](fakeEvent);
if (!prevented) process.exit(2);

// Test C: User click action while selection exists -> Allowed
prevented = false;
let clickEvent = {
  detail: {
    target: fakeEvent.detail.target,
    elt: {},
    requestConfig: { triggeringEvent: { type: 'click' } }
  },
  preventDefault() { prevented = true; }
};
listeners['htmx:beforeRequest'][0](clickEvent);
if (prevented) process.exit(3);

// Test D: Mouse down (dragging) -> Polling blocked
global.mockSelection = { isCollapsed: true, rangeCount: 0, toString: () => '' };
windowListeners['mousedown'][0]({ button: 0 });
prevented = false;
listeners['htmx:beforeRequest'][0](fakeEvent);
if (!prevented) process.exit(4);

windowListeners['mouseup'][0]();
prevented = false;
listeners['htmx:beforeRequest'][0](fakeEvent);
if (prevented) process.exit(5);

// Test E: Active focused input -> Polling blocked
const mockInput = { tagName: 'INPUT', type: 'text', value: 'Draft question' };
document.activeElement = mockInput;
const mockTarget = { contains: (el) => el === mockInput, querySelectorAll: () => [mockInput] };
let inputEvent = {
  detail: { target: mockTarget, elt: mockTarget, requestConfig: {} },
  preventDefault() { prevented = true; }
};
prevented = false;
listeners['htmx:beforeRequest'][0](inputEvent);
if (!prevented) process.exit(6);

// Test F: beforeSwap shouldSwap set to false
let swapEvent = {
  detail: { target: mockTarget, elt: mockTarget, requestConfig: {}, shouldSwap: true },
  preventDefault() { prevented = true; }
};
listeners['htmx:beforeSwap'][0](swapEvent);
if (!prevented || swapEvent.detail.shouldSwap !== false) process.exit(7);

process.exit(0);
"""
                    res = subprocess.run(
                        [node_bin, "-e", node_script], capture_output=True, text=True
                    )
                    assert res.returncode == 0, f"Node controller test failed: {res.stderr}"



# ==============================================================================
# Milestone 2 Integration & Control Plane E2E Tests
# ==============================================================================


async def test_fresh_default_boot_and_state_api(tmp_path: Path) -> None:
    """Verify fresh boot has zero degraded modules and GET /api/state returns all M2 fields."""
    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_fresh.sock"

    async with _fake_hyprland_server(sock_path):
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # 1. Health endpoint invariant: zero degraded modules
                health = (await client.get("/api/health")).json()
                assert health["status"] == "ok"
                assert health["degraded_modules"] == []
                active_modules = {m["module"] for m in health["modules"] if m["enabled"]}
                assert active_modules == {
                    "manual_control",
                    "hyprland_ipc",
                    "posture_policy",
                    "pomodoro_policy",
                    "meeting_policy",
                    "focus_policy",
                    "waybar_status",
                    "desktop_notify",
                }

                # 2. State endpoint invariant: rich desktop, calendar, LLM, voice & privacy fields
                state = (await client.get("/api/state")).json()
                assert "focus_context" in state
                assert "window_category" in state
                assert "call_window" in state
                assert state["call_window"] is False
                assert state["window_category"] in ("neutral", "unknown")
                assert state["llm_enabled"] is False
                assert "voice_status" in state
                assert state["privacy_paused"] is False
                assert state["dnd_mode"] is False
                assert state["privacy"]["allow_tts"] is False
                assert state["privacy"]["hyprland_ipc_mode"] == "observe_only"

                # 3. Dashboard HTML: Observe-only badge, link to settings for LLM, timer framing
                dash_html = (await client.get("/")).text
                assert "observe_only" in dash_html or "Observe Only" in dash_html
                assert "Local Pomodoro Timer · Desktop Countdown" in dash_html
                assert "/settings" in dash_html
                # Confirm voice hint is not a hardcoded keybind
                assert "configurable in Hyprland" in dash_html or "jarvis-ptt.sh" in dash_html


async def test_simulated_hyprland_window_transitions_observe_only_and_history(
    tmp_path: Path,
) -> None:
    """Verify window focus stream, classify, update /api/state, and never alert."""
    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_windows.sock"

    async with _fake_hyprland_server(sock_path) as writers:
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # Wait until Hyprland IPC connects
                await _wait_until(lambda: len(writers) > 0)
                writer = writers[0]

                # 1. Focus VS Code -> work / focused
                writer.write(b"activewindow>>code,src/jarvis/web/router.py - jarvis - Code\n")
                await writer.drain()

                async def _focused_updated() -> bool:
                    st = (await client.get("/api/state")).json()
                    fc = st.get("focus_context") or {}
                    return fc.get("app_class") == "code" and st.get("window_category") == "focused"

                await _wait_until(_focused_updated)

                state_work = (await client.get("/api/state")).json()
                assert state_work["focus_context"]["app_class"] == "code"
                assert state_work["window_category"] == "focused"
                assert state_work["call_window"] is False

                # Dashboard partial displays code & focused badge
                focus_partial = (
                    await client.get("/partials/focus", headers={"HX-Request": "true"})
                ).text
                assert "code" in focus_partial
                assert "focused" in focus_partial
                assert "Observe Only" in focus_partial

                # 2. Focus distraction window (Reddit on Firefox)
                writer.write(
                    b"activewindow>>firefox,Reddit - Dive into anything - Mozilla Firefox\n"
                )
                await writer.drain()

                async def _distracted_updated() -> bool:
                    st = (await client.get("/api/state")).json()
                    fc = st.get("focus_context") or {}
                    return (
                        fc.get("app_class") == "firefox"
                        and st.get("window_category") == "distracted"
                    )

                await _wait_until(_distracted_updated)

                state_distracted = (await client.get("/api/state")).json()
                assert state_distracted["window_category"] == "distracted"
                assert state_distracted["focus_context"]["app_class"] == "firefox"

                # 3. Observe-only invariant: NO distraction recommendations or warnings created
                recs = (await client.get("/api/recommendations")).json()
                assert recs["count"] == 0

                # 4. Switch to empty desktop -> closes current window span cleanly
                writer.write(b"activewindow>>,\n")
                await writer.drain()

                db: Database = app.state.db

                async def _span_closed() -> bool:
                    return (await db.get_current_window_span()) is None

                await _wait_until(_span_closed)

                # 5. History endpoint displays focus categories and durations
                history_html = (await client.get("/history")).text
                assert "Focus Category Durations" in history_html
                assert "focused" in history_html
                assert "distracted" in history_html


async def test_calendar_task_classification_recommendations_and_call_suppression(
    tmp_path: Path,
) -> None:
    """Verify calendar classification, countdown presets, and call window suppression."""
    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_cal.sock"

    async with _fake_hyprland_server(sock_path) as writers:
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            db: Database = app.state.db
            now_utc = datetime.now(tz=UTC)

            # Insert current Deep Work event and upcoming Meeting event
            await db.upsert_calendar_event(
                event_id="evt_coding",
                title="Refactor Web Control Plane",
                start_time=now_utc - timedelta(minutes=20),
                end_time=now_utc + timedelta(minutes=40),
                task_type="deep_work",
                suitable_for_walking=False,
            )
            await db.upsert_calendar_event(
                event_id="evt_standup",
                title="Daily Engineering Sync",
                start_time=now_utc + timedelta(minutes=45),
                end_time=now_utc + timedelta(minutes=75),
                task_type="meeting",
                suitable_for_walking=False,
            )

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # 1. State returns current and next calendar events
                state = (await client.get("/api/state")).json()
                assert state["calendar"]["current_event"] is not None
                assert state["calendar"]["current_event"]["title"] == "Refactor Web Control Plane"
                assert state["calendar"]["current_event"]["task_type"] == "deep_work"
                assert state["calendar"]["next_event"]["title"] == "Daily Engineering Sync"

                # 2. Calendar partial shows badges and recommended countdown preset (50/10)
                cal_html = (
                    await client.get("/partials/calendar", headers={"HX-Request": "true"})
                ).text
                assert "Deep Work" in cal_html
                assert "50/10" in cal_html

                # 3. Simulate ad-hoc Zoom call window
                await _wait_until(lambda: len(writers) > 0)
                writer = writers[0]

                writer.write(b"activewindow>>zoom,Zoom Meeting\n")
                await writer.drain()

                async def _call_window_active() -> bool:
                    st = (await client.get("/api/state")).json()
                    return bool(st.get("call_window"))

                await _wait_until(_call_window_active)

                state_call = (await client.get("/api/state")).json()
                assert state_call["call_window"] is True
                assert state_call["focus_context"]["is_call_window"] is True

                # Posture policy evaluates during call window -> posture nudges suppressed
                posture_mod = app.state.registry.get_module("posture_policy")
                assert isinstance(posture_mod, PosturePolicyModule)
                rec = await posture_mod.evaluate()
                assert rec is None, "Posture alerts must be suppressed during active call window"

                # Switch away from call -> call window cleared
                writer.write(b"activewindow>>code,main.py\n")
                await writer.drain()

                async def _call_window_cleared() -> bool:
                    st = (await client.get("/api/state")).json()
                    return not bool(st.get("call_window"))

                await _wait_until(_call_window_cleared)


async def test_llm_ask_reasoning_stripping_and_history_persistence(tmp_path: Path) -> None:
    """Verify enabling Ollama, stripping Qwen3 <think> reasoning traces, and history persistence."""
    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_llm.sock"

    async with _fake_hyprland_server(sock_path):
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # 1. Asking before enabling llm_ollama returns 503
                ask_disabled = await client.post(
                    "/api/ask",
                    json={"question": "Should I stand up?", "source": "text", "speak": False},
                )
                assert ask_disabled.status_code == 503

                # 2. Enable llm_ollama
                enable_resp = await client.post("/api/modules/llm_ollama/enable")
                assert enable_resp.status_code == 200

                # 3. Inject mock transport with Qwen3 reasoning trace
                fake_content = (
                    "<think>\n"
                    "User has been sitting for 55 minutes. Context indicates coding task.\n"
                    "Recommend standing.\n"
                    "</think>\n"
                    + json.dumps({
                        "answer": (
                            "You have been seated for 55 minutes. You should stand up and stretch."
                        ),
                        "suggested_action": "stand_up",
                        "confidence": 0.95,
                    })
                )
                fake_transport = _make_fake_ollama_transport(
                    fake_content,
                    model="qwen3:14b",
                    latency_ms=130,
                )
                llm_mod = app.state.registry.get_module("llm_ollama")
                assert isinstance(llm_mod, LlmOllamaModule)
                llm_mod.client = OllamaClient(
                    base_url="http://127.0.0.1:0",
                    model="qwen3:14b",
                    keep_alive="5m",
                    transport=fake_transport,
                )

                # 4. JSON request to POST /api/ask
                ask_resp = await client.post(
                    "/api/ask",
                    json={"question": "Should I stand?", "source": "text", "speak": False},
                )
                assert ask_resp.status_code == 200
                data = ask_resp.json()
                assert "<think>" not in data["answer"]
                assert "</think>" not in data["answer"]
                assert "You have been seated for 55 minutes" in data["answer"]
                assert data["model"] == "qwen3:14b"
                assert isinstance(data["latency_ms"], int)
                assert data["latency_ms"] >= 0
                assert data["interaction_id"] is not None

                # 5. HTMX request to POST /api/ask
                hx_resp = await client.post(
                    "/api/ask",
                    data={"question": "Next action?", "source": "text", "speak": "false"},
                    headers={"HX-Request": "true"},
                )
                assert hx_resp.status_code == 200
                assert hx_resp.headers.get("HX-Trigger") == "assistantAnswered"
                hx_text = hx_resp.text
                assert "<think>" not in hx_text
                assert "You have been seated for 55 minutes" in hx_text
                assert "Latency:" in hx_text
                assert "qwen3:14b" in hx_text

                # 6. Verify interaction is persisted in DB and visible in history
                history_html = (await client.get("/history")).text
                assert "Recent Local AI" in history_html
                assert "Should I stand?" in history_html

                history_api = (await client.get("/api/ask/history")).json()
                assert history_api["count"] >= 2
                assert history_api["interactions"][0]["model"] == "qwen3:14b"


async def test_voice_ptt_transcript_tts_and_dnd_modes(tmp_path: Path) -> None:
    """Verify voice requests, privacy.allow_tts gate, Piper playback, and DND suppression."""
    notify_bin, _ = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_voice.sock"

    async with _fake_hyprland_server(sock_path):
        settings, _ = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # Enable llm_ollama and tts_piper
                await client.post("/api/modules/llm_ollama/enable")
                await client.post("/api/modules/tts_piper/enable")

                # Setup mock LLM and mock TTS
                fake_transport = _make_fake_ollama_transport("Time to take a walking break.")
                llm_mod = app.state.registry.get_module("llm_ollama")
                assert isinstance(llm_mod, LlmOllamaModule)
                llm_mod.client = OllamaClient(
                    base_url="http://127.0.0.1:0",
                    model="qwen3:14b",
                    transport=fake_transport,
                )

                tts_mod = app.state.registry.get_module("tts_piper")
                assert isinstance(tts_mod, TtsPiperModule)
                fake_syn = _RecordingSynthesizer()
                fake_player = _RecordingPlayer()
                tts_mod._synthesizer = fake_syn
                tts_mod._player = fake_player

                # 1. Voice ask with speak=True when allow_tts=False (default) -> no speech playback
                ask1 = await client.post(
                    "/api/ask",
                    json={"question": "What is next?", "source": "voice", "speak": True},
                )
                assert ask1.status_code == 200
                await asyncio.sleep(0.05)
                assert len(fake_player.played) == 0

                # 2. Enable allow_tts via POST /api/privacy/tts
                allow_resp = await client.post("/api/privacy/tts", json={"allow_tts": True})
                assert allow_resp.status_code == 200
                assert allow_resp.json()["allow_tts"] is True

                # Voice ask with speak=True when allow_tts=True -> playback triggers
                ask2 = await client.post(
                    "/api/ask",
                    json={"question": "What is next?", "source": "voice", "speak": True},
                )
                assert ask2.status_code == 200
                await _wait_until(lambda: len(fake_player.played) == 1)
                assert len(fake_syn.synthesized) == 1
                assert fake_syn.synthesized[0][0] == "Time to take a walking break."

                # 3. Enable DND mode -> voice ask with speak=True suppresses speech
                await client.post("/api/privacy/dnd", json={"dnd_mode": True})
                fake_player.played.clear()

                ask3 = await client.post(
                    "/api/ask",
                    json={"question": "What is next?", "source": "voice", "speak": True},
                )
                assert ask3.status_code == 200
                await asyncio.sleep(0.05)
                assert len(fake_player.played) == 0, "TTS speech must be suppressed during DND"

                # 4. Turn off DND, set microphone to disabled -> voice ask returns 409 Conflict
                await client.post("/api/privacy/dnd", json={"dnd_mode": False})
                mic_resp = await client.post(
                    "/api/privacy/microphone",
                    json={"microphone_mode": "disabled"},
                )
                assert mic_resp.status_code == 200

                ask_mic_disabled = await client.post(
                    "/api/ask",
                    json={"question": "Voice question", "source": "voice", "speak": False},
                )
                assert ask_mic_disabled.status_code == 409
                assert "Microphone is disabled" in ask_mic_disabled.json()["detail"]


async def test_failure_isolation_across_optional_modules(tmp_path: Path) -> None:
    """Verify failures in optional modules leave core posture & timers functional."""
    notify_bin, notify_log = _create_recording_notify_binary(tmp_path)
    sock_path = tmp_path / "s2_fail.sock"

    async with _fake_hyprland_server(sock_path) as writers:
        settings, waybar_path = _build_e2e_settings(
            tmp_path,
            notify_binary=notify_bin,
            hyprland_socket=sock_path,
        )
        app = create_app(settings=settings)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:8000",
            ) as client:
                # Wait for Hyprland to connect, then close writer to simulate socket death
                await _wait_until(lambda: len(writers) > 0)
                writers[0].close()
                await writers[0].wait_closed()

                # Enable optional modules and inject failure transports
                await client.post("/api/modules/llm_ollama/enable")
                await client.post("/api/modules/tts_piper/enable")

                llm_mod = app.state.registry.get_module("llm_ollama")
                assert isinstance(llm_mod, LlmOllamaModule)
                llm_mod.client = OllamaClient(
                    base_url="http://127.0.0.1:0",
                    transport=_make_fake_ollama_transport(
                        side_effect=httpx.ConnectError("Ollama connection failed")
                    ),
                )

                tts_mod = app.state.registry.get_module("tts_piper")
                assert isinstance(tts_mod, TtsPiperModule)
                fail_player = _RecordingPlayer()
                fail_player.raise_error = AudioPlaybackError("PipeWire sink error")
                tts_mod._player = fail_player

                # Trigger failed LLM ask -> returns 503 without crashing
                failed_ask = await client.post(
                    "/api/ask",
                    json={"question": "Fail?", "source": "text", "speak": False},
                )
                assert failed_ask.status_code == 503

                # --------------------------------------------------------------
                # Core platform verification despite all optional module failures:
                # --------------------------------------------------------------

                # 1. Manual posture cycling continues to function and update SQLite
                cycle_resp = await client.post("/api/activity/cycle")
                assert cycle_resp.status_code == 200
                assert cycle_resp.json()["activity"] == "sitting"

                cycle_resp2 = await client.post("/api/activity/cycle")
                assert cycle_resp2.status_code == 200
                assert cycle_resp2.json()["activity"] == "standing"

                db: Database = app.state.db
                span = await db.get_current_activity_span()
                assert span is not None
                assert span["activity"] == "standing"

                # 2. Desktop countdown timer starts and updates state
                timer_resp = await client.post("/api/timers/start", json={"kind": "25_5"})
                assert timer_resp.status_code == 200
                assert timer_resp.json()["active"] is True

                state = (await client.get("/api/state")).json()
                assert state["activity"] == "standing"
                assert state["timer"]["active"] is True
                assert state["timer"]["kind"] == "25_5"

                # 3. Waybar status file continues updating
                wb = json.loads(waybar_path.read_text(encoding="utf-8"))
                assert "STAND" in wb["text"]
                assert "25/5" in wb["text"]

                # 4. Notifications continue to fire
                await app.state.bus.publish(
                    PlatformEvent(
                        type="recommendation_created",
                        source="posture_policy",
                        payload={
                            "id": 9999,
                            "policy": "posture_policy",
                            "message": "Time to stand",
                            "status": "pending",
                        },
                    )
                )
                await _wait_until(lambda: len(_read_notify_log(notify_log)) >= 1)
                notifications = _read_notify_log(notify_log)
                assert len(notifications) >= 1
