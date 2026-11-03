# Posture Platform — Architecture and Implementation

This document describes the implementation architecture for the local posture,
focus, and calendar copilot.

The system is intentionally modular:

```text
Inputs → Event Bus → State/SQLite → Policies → Outputs
```

No module calls another module directly. Modules communicate through typed events,
the core REST/WebSocket API, or clearly defined local interfaces.

## 1. Runtime topology

```text
┌──────────────────────────────────────────────────────────────┐
│ CONTROL AND OUTPUT SURFACES                                  │
│ -  Waybar custom module                                        │
│ -  HTMX dashboard at http://127.0.0.1:8000                    │
│ -  mako / dunst notifications                                  │
│ -  Optional Piper TTS                                          │
│ -  Optional phone / watch / Boosted bridge                     │
└───────────────▲──────────────────────────┬───────────────────┘
                │ REST / WebSocket         │ Manual state / webhooks
┌───────────────┴──────────────────────────▼───────────────────┐
│ posture-core.service                                          │
│ FastAPI + SQLite + EventBus + Resolver + Policy Engine        │
└───────▲───────────────▲───────────────▲───────────────▲──────┘
        │ HTTP          │ HTTP          │ Unix socket   │ HTTP
┌───────┴──────┐ ┌──────┴────────┐ ┌────┴─────────┐ ┌───┴────────────┐
│ Manual input │ │ Google Cal    │ │ Hyprland IPC │ │ Optional       │
│ Waybar/HTMX  │ │ sync/writer   │ │ observer     │ │ sensor workers │
└──────────────┘ └───────────────┘ └──────────────┘ └────────────────┘
```

### Services

| Service | Responsibility | Required |
|---|---|---|
| `posture-core.service` | FastAPI API, event bus, resolver, SQLite, policies, HTMX dashboard | Yes |
| `posture-vision.service` | Optional webcam pose/walking worker | No |
| `posture-voice.service` | Optional push-to-talk transcription worker | No |
| `ollama.service` | Optional local LLM/VLM server | No |
| `posture-bridge.service` | Optional phone/watch/Boosted bridge | No |

Heavy work—computer vision, Whisper, Ollama, screenshot analysis—must run outside
the FastAPI request path. Workers communicate with the core through authenticated
local HTTP webhooks or a Unix socket.

## 2. Technology choices

| Area | Choice | Reason |
|---|---|---|
| Language | Python 3.12+ | Strong ecosystem for FastAPI, SQLite, OpenCV, MediaPipe, BLE |
| API | FastAPI | Async REST, WebSockets, Pydantic models, lifespan management |
| UI | HTMX + Jinja2 | No Node/build pipeline; server-rendered partial updates |
| Database | SQLite in WAL mode | Local, zero-ops, supports concurrent readers with one writer |
| Desktop integration | Waybar, Hyprland IPC, `mako`/`dunst` | Native Hyprland workflow |
| Config | YAML + Pydantic settings | Human-readable and validated |
| Background workers | Separate systemd user services | Fault isolation |
| LLM | Ollama with small local text model | Optional; not required for core behavior |
| VLM | Ollama vision model | Optional; screenshot-only |
| STT | faster-whisper or whisper.cpp | Optional local speech-to-text |
| TTS | Piper | Optional local speech output |
| BLE | `bleak` | Optional sWalk Lite Plus integration |

FastAPI’s lifespan context manager is used to initialize and cleanly shut down the
database, event bus, module registry, and background tasks. [107]

## 3. Repository layout

```text
~/projects/posture-platform/
├── README.md
├── SPEC.md
├── ARCHITECTURE.md
├── pyproject.toml
├── config.yaml
├── main.py
├── core/
│   ├── __init__.py
│   ├── app.py
│   ├── bus.py
│   ├── schema.py
│   ├── database.py
│   ├── migrations.py
│   ├── resolver.py
│   ├── registry.py
│   └── settings.py
├── modules/
│   ├── base.py
│   ├── inputs/
│   │   ├── manual_control.py
│   │   ├── hyprland_ipc.py
│   │   ├── google_calendar.py
│   │   ├── s24_webhook.py
│   │   ├── swalk_ble.py
│   │   └── webcam_pose.py
│   ├── policies/
│   │   ├── posture_policy.py
│   │   ├── pomodoro_policy.py
│   │   ├── focus_policy.py
│   │   └── meeting_policy.py
│   └── outputs/
│       ├── waybar_status.py
│       ├── desktop_notify.py
│       ├── tts_piper.py
│       ├── phone_bridge.py
│       ├── calendar_writer.py
│       └── llm_ollama.py
├── web/
│   ├── router.py
│   ├── api.py
│   ├── ws.py
│   └── templates/
│       ├── base.html
│       ├── dashboard.html
│       ├── calibration.html
│       ├── history.html
│       └── settings.html
├── workers/
│   ├── webcam_pose_worker.py
│   ├── ptt_voice.sh
│   └── swalk_ble_worker.py
├── migrations/
│   ├── 0001_initial.sql
│   └── 0002_add_timer_spans.sql
├── systemd/
│   ├── posture-core.service
│   ├── posture-vision.service
│   └── posture-bridge.service
└── tests/
    ├── test_resolver.py
    ├── test_posture_policy.py
    ├── test_pomodoro_policy.py
    └── test_focus_policy.py
```

## 4. Event contract

All modules emit `PlatformEvent` objects. Modules must not import or invoke other
modules directly.

```python
# core/schema.py
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

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
]

ActivityState = Literal[
    "sitting",
    "standing",
    "walking",
    "away",
    "unknown",
]

class PlatformEvent(BaseModel):
    type: EventType
    source: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    dedupe_key: str | None = None
```

### Example: manual activity event

```json
{
  "type": "activity_reported",
  "source": "manual_control",
  "confidence": 1.0,
  "payload": {
    "activity": "walking",
    "ttl_minutes": 30
  },
  "dedupe_key": "manual:walking:2026-11-03T16:42:00"
}
```

### Example: webcam activity event

```json
{
  "type": "activity_reported",
  "source": "webcam_pose",
  "confidence": 0.78,
  "payload": {
    "activity": "standing",
    "samples_agreeing": 5,
    "samples_total": 5
  }
}
```

### Example: Hyprland window event

```json
{
  "type": "window_focus_changed",
  "source": "hyprland_ipc",
  "confidence": 1.0,
  "payload": {
    "app_class": "firefox",
    "title": "Reddit - /r/archlinux",
    "category": "distraction"
  }
}
```

## 5. State resolver

The resolver converts raw sensor reports into one effective activity state.

### Priority rules

```text
1. Privacy pause / DND mode
2. Manual override within TTL
3. Timer-derived constraint, e.g. meeting imminent
4. BLE walking-pad evidence
5. Phone pocket/activity-recognition evidence
6. Webcam pose evidence
7. Previous state retained until enough contrary evidence arrives
```

### Consensus rules

- Manual reports always win within their TTL.
- A sensor must report a different state for a configurable persistence period before it can override the current state.
- Webcam walking detection requires multiple agreeing samples.
- Conflicting low-confidence reports do not immediately change state.
- Every effective transition stores a `reason`.

### Example resolver output

```json
{
  "type": "activity_state_changed",
  "source": "resolver",
  "confidence": 1.0,
  "payload": {
    "previous_activity": "sitting",
    "activity": "walking",
    "reason": "manual_override",
    "override_expires_at": "2026-11-03T17:12:00+01:00"
  }
}
```

## 6. SQLite schema

Use SQLite with WAL mode. WAL improves concurrency for the common case of the
dashboard reading while the core writes; SQLite still permits only one writer at a
time. [106]

```sql
-- migrations/0001_initial.sql

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  activity TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0,
  reason TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_activity_started
  ON activity_spans(started_at);

CREATE TABLE IF NOT EXISTS window_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  app_class TEXT NOT NULL,
  window_title TEXT NOT NULL,
  category TEXT NOT NULL,
  classifier TEXT NOT NULL,
  calendar_event_id TEXT,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_window_started
  ON window_spans(started_at);

CREATE TABLE IF NOT EXISTS calendar_cache (
  event_id TEXT PRIMARY KEY,
  calendar_id TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  color_id TEXT,
  task_type TEXT NOT NULL DEFAULT 'unknown',
  suitable_for_walking INTEGER,
  recommended_pomodoro TEXT,
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL,
  raw_json TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timer_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  calendar_event_id TEXT,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  planned_duration_seconds INTEGER,
  actual_duration_seconds INTEGER,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  policy TEXT NOT NULL,
  message TEXT NOT NULL,
  suggested_action TEXT,
  action_payload_json TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  resolution_metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS calendar_drafts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation TEXT NOT NULL,
  event_id TEXT,
  proposed_changes_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  decided_at TEXT,
  decision_metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS module_health (
  module TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  last_event_at TEXT,
  last_error TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

## 7. Module interface

```python
# modules/base.py
from abc import ABC, abstractmethod
from typing import Any

from core.bus import EventBus
from core.database import Database


class BaseModule(ABC):
    name: str = "unnamed_module"

    def __init__(self, bus: EventBus, db: Database, config: dict[str, Any]):
        self.bus = bus
        self.db = db
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        """Subscribe to events and start background tasks."""

    @abstractmethod
    async def stop(self) -> None:
        """Cancel tasks and release resources."""

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Optional live-reconfiguration hook."""

    async def health(self) -> dict[str, Any]:
        return {
            "module": self.name,
            "status": "ok",
        }
```

### Module rules

- Inputs may only emit events or serve local webhooks.
- Policies may only consume events and emit recommendations/timer events.
- Outputs may only consume events and perform user-visible actions.
- No policy may directly access the webcam, microphone, or Google API.
- No input may directly trigger TTS, notifications, or calendar writes.

## 8. Core API

### REST endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Core health and version |
| `GET` | `/api/state` | Current activity, timer, calendar context, privacy mode |
| `POST` | `/api/activity` | Set manual activity |
| `POST` | `/api/activity/cycle` | Cycle manual activity |
| `POST` | `/api/events` | Ingest event from worker/webhook |
| `GET` | `/api/recommendations` | List pending recommendations |
| `POST` | `/api/recommendations/{id}/accept` | Accept recommendation |
| `POST` | `/api/recommendations/{id}/snooze` | Snooze recommendation |
| `POST` | `/api/recommendations/{id}/dismiss` | Dismiss recommendation |
| `POST` | `/api/timers/start` | Start local Pomodoro |
| `POST` | `/api/timers/stop` | Stop active timer |
| `GET` | `/api/history/activity` | Activity history and aggregates |
| `GET` | `/api/history/windows` | Window/focus history |
| `GET` | `/api/modules` | Module health and enabled state |
| `POST` | `/api/modules/{name}/enable` | Enable module |
| `POST` | `/api/modules/{name}/disable` | Disable module |
| `POST` | `/api/privacy/pause` | Enable global privacy pause |
| `POST` | `/api/privacy/resume` | Resume normal operation |
| `POST` | `/api/webhooks/s24` | Optional phone webhook |
| `POST` | `/api/calendar/drafts/{id}/approve` | Approve calendar draft |
| `POST` | `/api/calendar/drafts/{id}/reject` | Reject calendar draft |

### WebSocket channel

`/ws` streams:

- activity changes
- timer changes
- recommendations
- calendar context changes
- module health updates
- privacy mode changes

The HTMX dashboard can use WebSocket-driven DOM updates or periodic HTMX polling.
For v1, HTMX polling every 2–5 seconds is acceptable; WebSockets can be added once
the core is stable.

## 9. Configuration

```yaml
# config.yaml

server:
  host: "127.0.0.1"
  port: 8000

database:
  path: "~/.local/share/posture-platform/posture.db"
  wal_mode: true

enabled_modules:
  inputs:
    - manual_control
    - hyprland_ipc
    - google_calendar
    # - s24_webhook
    # - swalk_ble
    # - webcam_pose

  policies:
    - posture_policy
    - pomodoro_policy
    - meeting_policy
    - focus_policy

  outputs:
    - waybar_status
    - desktop_notify
    # - tts_piper
    # - phone_bridge
    # - calendar_writer
    # - llm_ollama

privacy:
  global_pause: false
  dnd_mode: false
  hyprland_ipc_mode: "observe_only"  # observe_only | notify_on_distraction
  allow_phone_webhook: false
  allow_vlm_screenshots: false
  allow_tts: false
  allow_calendar_writes: false
  dashboard_bind_local_only: true

posture:
  manual_override_ttl_minutes: 30
  max_sitting_minutes: 50
  max_walking_minutes: 45
  pre_meeting_warning_minutes: 10
  sensor_consensus_samples: 4
  sensor_consensus_window_samples: 5

pomodoro:
  deep_work: "50_10"
  coding: "50_10"
  reading: "50_10"
  admin: "25_5"
  email: "25_5"
  planning: "25_5"
  unknown: null
  auto_start: false

focus:
  distraction_grace_seconds: 180
  phone_unlock_limit_seconds: 60
  max_prompts_per_hour: 4
  work_app_classes:
    - "code"
    - "Alacritty"
    - "kitty"
    - "obsidian"
    - "Zotero"
  distraction_title_patterns:
    - "YouTube"
    - "Reddit"
    - "Twitch"
    - "Twitter"
    - "X"

google_calendar:
  calendar_id: "primary"
  sync_interval_minutes: 15
  oauth_token_path: "~/.local/share/posture-platform/google-token.json"
  read_only: true

webcam:
  device: "/dev/video0"
  width: 320
  height: 240
  sample_interval_seconds: 5
  walking_burst_frames: 8
  walking_burst_interval_seconds: 1.5

swalk_ble:
  enabled: false
  device_name: "sWalk"
  scan_timeout_seconds: 10

phone_bridge:
  transport: "none"  # none | kdeconnect | ntfy | webhook
  endpoint: ""

llm:
  provider: "ollama"
  base_url: "http://127.0.0.1:11434"
  text_model: "qwen2.5:3b"
  vision_model: "qwen2.5vl:7b"
  keep_alive_minutes: 15
```

## 10. Waybar integration

### Waybar module

```jsonc
{
  "custom/posture": {
    "exec": "~/.config/waybar/scripts/posture-status.py",
    "return-type": "json",
    "signal": 8,
    "interval": 0,
    "on-click": "curl -s -X POST http://127.0.0.1:8000/api/activity/cycle",
    "on-click-right": "xdg-open http://127.0.0.1:8000",
    "on-click-middle": "curl -s -X POST http://127.0.0.1:8000/api/recommendations/snooze"
  }
}
```

Waybar custom modules can execute a script, emit JSON, and refresh when they receive
the configured realtime signal. [92][108]

### Status script output

```json
{
  "text": "󰗉 WALK 12m · 50/10 22m",
  "tooltip": "Walking\nCurrent: Backend refactor\nNext: Meeting in 38m\nManual override: none",
  "class": "walking"
}
```

The output module writes this JSON to `/dev/shm/posture_waybar.json`, then sends
`SIGRTMIN+8` to Waybar.

## 11. HTMX dashboard

The dashboard is a local administrative control plane, not the primary daily interface.

### Pages

| Page | Purpose |
|---|---|
| Dashboard | Current state, timer, next calendar block, pending recommendations |
| Calibration | Capture sitting/standing webcam baselines; test BLE and webcam modules |
| History | Daily/weekly activity, focus categories, Pomodoro completion |
| Modules | Enable/disable modules and inspect health/errors |
| Privacy | Pause monitoring, toggle camera/mic/screen/VLM/calendar-write permissions |
| Calendar | Preview today’s schedule, task classifications, and pending drafts |

### Example dashboard actions

```html
<button hx-post="/api/activity"
        hx-vals='{"activity": "walking"}'
        hx-target="#current-state">
  Walking
</button>

<button hx-post="/api/timers/start"
        hx-vals='{"kind": "50_10"}'
        hx-target="#timer">
  Start 50/10
</button>

<button hx-post="/api/privacy/pause"
        hx-target="#privacy-status">
  Pause everything
</button>
```

## 12. Worker design

### Webcam pose worker

```text
every 5 seconds:
  capture one 320x240 frame
  run lightweight pose landmark detection
  extract shoulder landmarks
  compare against calibrated sitting/standing baselines
  append result to 5-sample ring buffer
  if at least 4 of 5 samples agree:
      POST /api/events with activity_reported
  release camera resources
```

Rules:

- Never retain frames.
- Never run continuous video inference.
- Manual override always wins during its TTL.
- Walking detection is experimental and disabled by default.

### sWalk BLE worker

```text
scan for configured device
connect if known GATT characteristics are found
poll/subscribe to available telemetry
map telemetry to walking state and speed
POST /api/events
reconnect with exponential backoff on failure
```

If the BLE protocol cannot be reliably reverse-engineered, disable this module and
use manual state, phone activity recognition, or webcam fallback.

### Hyprland IPC module

```text
connect to Hyprland socket2
listen for activewindow and relevant input/idle events
classify app/title
emit window_focus_changed / input_idle_changed
store spans through core
```

This module must support `observe_only` mode. In that mode it records data but sends
no distraction prompts.

## 13. Policy engine

### Posture policy

```python
def evaluate(state, calendar_context, settings):
    if state.activity == "sitting":
        if state.duration_minutes >= settings.max_sitting_minutes:
            return recommendation(
                message="You have been sitting for {minutes} minutes.",
                action="stand_or_walk",
            )

    if state.activity == "walking":
        if state.duration_minutes >= settings.max_walking_minutes:
            return recommendation(
                message="You have been walking for {minutes} minutes.",
                action="take_seated_break",
            )

    return None
```

### Pomodoro policy

```python
def select_pomodoro(task_type, settings):
    return settings.pomodoro.get(task_type)
```

The policy does not auto-start timers unless `pomodoro.auto_start` is enabled and the
task type is explicitly whitelisted.

### Focus policy

```text
Stage 1: Observe
  Log app/title/category and duration.

Stage 2: Nudge
  If distraction persists beyond grace period during a productivity block:
      send one soft notification.

Stage 3: Optional inspect
  If still unresolved and VLM screenshots are enabled:
      ask for explicit approval before capturing one screenshot.
```

## 14. Google Calendar integration

### Read-only mode

Use:

```text
[https://www.googleapis.com/auth/calendar.readonly](https://www.googleapis.com/auth/calendar.readonly)
```

This supports the initial calendar sync and recommendations.

### Write-enabled mode

Only enable this after the read-only version is stable:

```text
[https://www.googleapis.com/auth/calendar.events](https://www.googleapis.com/auth/calendar.events)
```

The Google Calendar API supports event creation and partial updates through
`events.patch`; full replacement uses `events.update`. [120][122]

### Write flow

```text
1. Policy or user request creates a CalendarDraft.
2. Draft is stored locally with status = pending.
3. Desktop notification shows:
   - current event/time
   - proposed event/time
   - Approve / Reject actions
4. On Approve:
   - core calls Google Calendar API
   - draft status becomes approved
   - result and API response metadata are logged
5. On Reject:
   - draft status becomes rejected
   - no Google API write occurs
```

## 15. Systemd user services

### `posture-core.service`

```ini
[Unit]
Description=Posture Platform core daemon
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.local/bin/posture-platform-core
WorkingDirectory=%h/projects/posture-platform
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=graphical-session.target
```

### `posture-vision.service`

```ini
[Unit]
Description=Posture Platform webcam worker
After=posture-core.service
Requires=posture-core.service

[Service]
Type=simple
ExecStart=%h/.local/bin/posture-platform-vision
WorkingDirectory=%h/projects/posture-platform
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=graphical-session.target
```

Enable with:

```bash
systemctl --user daemon-reload
systemctl --user enable --now posture-core.service
```

## 16. Implementation stages

| Stage | Deliverable | Verification |
|---|---|---|
| 1 | Core skeleton | FastAPI starts, SQLite migrations run, `/api/health` works |
| 2 | Manual control + Waybar | Waybar cycles and persists activity states |
| 3 | HTMX dashboard | Dashboard shows state, history, and module switches |
| 4 | Posture + Pomodoro policies | Sitting/walking thresholds and local timer work |
| 5 | Hyprland observe-only IPC | Window spans and input-idle data appear in SQLite/dashboard |
| 6 | Google Calendar read-only | Current/next events appear; posture/Pomodoro recommendations use them |
| 7 | Notification outputs | Desktop prompts, snooze, accept, dismiss work |
| 8 | sWalk BLE worker | Walking state is ingested if BLE telemetry is accessible |
| 9 | Webcam sit/stand worker | Calibration and 0.2 FPS sit/stand detection work |
| 10 | Phone bridge | Optional webhook input and notification output work |
| 11 | Text LLM + PTT voice | Voice Q&A and ambiguous task classification work |
| 12 | VLM screenshots + calendar writes | Both are opt-in, approval-gated, and fully auditable |

## 17. Testing strategy

### Unit tests

- Resolver priority and TTL behavior.
- Activity-span open/close logic.
- Pomodoro selection.
- Posture thresholds.
- Focus classification precedence.
- Calendar draft approval flow.

### Integration tests

- Fake Hyprland IPC event stream.
- Fake Google Calendar API.
- Fake webcam frame generator.
- Fake BLE characteristic.
- Core API webhook ingestion.

### Manual acceptance tests

- Restart core during an active timer.
- Disable webcam while walking.
- Disconnect phone during a focus block.
- Enable DND during a meeting.
- Verify no screenshot or calendar write occurs without approval.
- Measure idle CPU and GPU with modules enabled/disabled.

## 18. Definition of done

A module is done when:

- It has a typed event contract.
- It can be enabled/disabled through configuration.
- It fails safely when its dependency is unavailable.
- It writes useful health/status information.
- It has unit or integration tests.
- It does not violate the privacy defaults.
- Its removal does not break the core daemon.