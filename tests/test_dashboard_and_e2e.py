"""Full vertical-slice Milestone 1 E2E and HTMX control-plane integration tests."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, now_local
from jarvis.core.settings import Settings, load_settings
from jarvis.modules.policies.posture_policy import PosturePolicyModule


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


def _build_e2e_settings(
    tmp_path: Path,
    *,
    notify_binary: Path,
) -> tuple[Settings, Path]:
    """Load real `config.yaml` defaults and isolate DB, Waybar JSON, and notify binary."""
    settings = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    waybar_path = tmp_path / "posture_waybar.json"

    settings.database.path = tmp_path / "e2e_posture.db"
    settings.waybar.status_file = waybar_path
    settings.waybar.signal_waybar = False
    settings.waybar.refresh_interval_seconds = 60.0
    settings.desktop_notify.notify_binary = str(notify_binary)

    return settings, waybar_path


async def test_vertical_slice_milestone_1_e2e(tmp_path: Path) -> None:
    """End-to-end slice across HTML, manual state, timers, policies, Waybar, and privacy."""
    notify_bin, notify_log = _create_recording_notify_binary(tmp_path)
    settings, waybar_path = _build_e2e_settings(tmp_path, notify_binary=notify_bin)

    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8000
    assert settings.enabled_modules.inputs == ["manual_control"]
    assert settings.enabled_modules.policies == [
        "posture_policy",
        "pomodoro_policy",
        "meeting_policy",
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
                "posture_policy",
                "pomodoro_policy",
                "meeting_policy",
                "waybar_status",
                "desktop_notify",
            }

            dash_resp = await client1.get("/")
            assert dash_resp.status_code == 200
            assert "text/html" in dash_resp.headers["content-type"]
            dash_html = dash_resp.text
            assert "/static/htmx.min.js" in dash_html
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
            assert "posture_policy" in settings_resp.text
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

            # Enable Privacy Pause -> suppresses external `/api/events`, sensor updates, & alerts
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
    settings, _ = _build_e2e_settings(tmp_path, notify_binary=notify_bin)
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
            # All 7 HTMX polling partials return 200 HTML fragments
            for partial_url, expected_snippet in (
                ("/partials/posture", "Effective Posture"),
                ("/partials/timer", "Local Pomodoro Timer"),
                ("/partials/calendar", "Design Review"),
                ("/partials/recommendations", "Pending Recommendations"),
                ("/partials/privacy", "Privacy &amp; Mode Switches"),
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
