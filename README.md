# 🛡️ Jarvis — Local-First Posture, Focus & Calendar Copilot

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Compositor](https://img.shields.io/badge/compositor-Hyprland%20%2F%20Wayland-brightgreen.svg)](https://hyprland.org/)
[![Status Bar](https://img.shields.io/badge/status%20bar-Waybar-orange.svg)](https://github.com/Alexays/Waybar)
[![Tests](https://img.shields.io/badge/tests-255%20passed-success.svg)](https://github.com/pytest-dev/pytest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Privacy](https://img.shields.io/badge/privacy-100%25%20local--first-purple.svg)](#privacy--security-by-design)

A modular, local-first productivity, ergonomics, and calendar daemon engineered specifically for **Arch Linux** and the **Hyprland** Wayland compositor. 

Jarvis bridges the gap between physical health, deep work, and scheduled commitments. It continuously tracks posture (sitting, standing, walking), window focus context, and calendar blocks, issuing deterministic, non-destructive health breaks and adaptive Pomodoro timers without relying on third-party cloud trackers.

---

## 📑 Table of Contents

- [Overview & Core Philosophy](#-overview--core-philosophy)
- [Architecture & Runtime Topology](#-architecture--runtime-topology)
- [Key Features](#-key-features)
  - [Posture & Ergonomic Health](#1-posture--ergonomic-health)
  - [Adaptive Pomodoro Engine](#2-adaptive-pomodoro-engine)
  - [Zero-Overhead Hyprland Focus Tracking](#3-zero-overhead-hyprland-focus-tracking)
  - [Google Calendar & Meeting Guardrails](#4-google-calendar--meeting-guardrails)
  - [Native Waybar & Desktop Integration](#5-native-waybar--desktop-integration)
  - [Zero-Build HTMX Web Dashboard](#6-zero-build-htmx-web-dashboard)
  - [Local Voice Copilot & Piper TTS](#7-local-voice-copilot--piper-tts)
  - [Privacy & Security by Design](#8-privacy--security-by-design)
- [Prerequisites & System Requirements](#-prerequisites--system-requirements)
- [Installation & Quickstart](#-installation--quickstart)
- [Desktop Integration Setup](#-desktop-integration-setup)
  - [Waybar Status Module](#waybar-status-module)
  - [Waybar Styling](#waybar-styling)
  - [Hyprland Keybindings](#hyprland-keybindings)
  - [Hypridle Idle Detection](#hypridle-idle-detection)
- [Running as a Systemd Service](#-running-as-a-systemd-service)
- [Configuration Reference](#-configuration-reference)
  - [config.yaml](#configyaml)
  - [focus_rules.yaml](#focus_rulesyaml)
- [REST API Reference](#-rest-api-reference)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 💡 Overview & Core Philosophy

Modern desktop workstations encourage prolonged sedentary behavior, frequent context-switching, and missed breaks. Most existing productivity solutions either:
1. Demand intrusive cloud tracking or subscriptions.
2. Rely on rigid, dumb timers that interrupt you during a critical debugging flow or right before an important video meeting.
3. Don't integrate seamlessly into tiling Wayland desktop environments.

**Jarvis solves this natively on Linux:**
- **Local-First & Offline:** All decisions, timers, state resolution, and classifications execute entirely on your machine.
- **Event-Driven & Modular:** Modules communicate exclusively via typed events over an internal async `EventBus`. Components can be enabled or disabled cleanly in `config.yaml`.
- **Non-Destructive Recommendations:** Jarvis never forces window minimization, blocks screens, or mutates your calendar without explicit consent.
- **Desktop Native:** Built to live inside Waybar, `mako`/`dunst` notifications, and Hyprland IPC sockets.

---

## 🏛️ Architecture & Runtime Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SURFACES & CONTROLS                             │
│  - Waybar Custom Module (SIGRTMIN+8)  - Desktop Notifications (mako)   │
│  - HTMX Dashboard (127.0.0.1:8000)    - Push-to-Talk (SUPER+V)         │
└───────────────────▲───────────────────────────────┬────────────────────┘
                    │ REST / SSE                    │ Webhooks / Clicks
┌───────────────────┴───────────────────────────────▼────────────────────┐
│                        JARVIS CORE DAEMON                              │
│                                                                        │
│   ┌───────────────┐     ┌────────────────┐     ┌───────────────────┐   │
│   │   EventBus    │────▶│ State Resolver │────▶│  Policy Engine    │   │
│   │ (Typed Events)│     │  & SQLite WAL  │     │ (Posture/Pomodoro)│   │
│   └───────────────┘     └────────────────┘     └───────────────────┘   │
└───────▲─────────────────▲─────────────────▲─────────────────────▲──────┘
        │                 │                 │                     │
┌───────┴──────┐   ┌──────┴────────┐ ┌──────┴─────────┐   ┌───────┴──────┐
│ Hyprland IPC │   │ Google Cal    │ │ Manual Input   │   │ Optional     │
│ Socket (Live)│   │ Sync (OAuth)  │ │ (Waybar / UI)  │   │ Sensors/BLE  │
└──────────────┘   └───────────────┘ └────────────────┘   └──────────────┘
```

---

## 🚀 Key Features

### 1. Posture & Ergonomic Health
- **State Tracking:** Tracks continuous spans for `sitting`, `standing`, `walking`, and `away`.
- **Health Limits:** Enforces configurable thresholds (default: maximum 50 minutes continuous sitting, 45 minutes walking).
- **Multi-Sensor Consensus & Manual Overrides:** Resolves posture across manual overrides (with configurable TTL), desk height inputs, BLE walking pads (e.g., Sportstech sWalk Lite Plus), and optional low-overhead webcam pose estimation.
- **Stepped Cooldown Backoff:** Intelligently backs off notification frequency when recommendations are repeatedly dismissed.
- **Crash & Reboot Resilience:** Automatically reconciles and closes open spans on shutdown or unexpected reboot without corrupting historical statistics.

### 2. Adaptive Pomodoro Engine
- **Cognitive Effort & Context Aware:** Recommends optimal focus cycles based on active tasks and cognitive load:
  - **50 / 10:** Deep work, coding, research, writing, thesis study.
  - **25 / 5:** Administration, email triage, sprint planning.
  - **No Timer / Suppressed:** Active video meetings, lunch breaks.
- **Flow State Respect:** Recognizes tags like `#flow` or `#notimer` in calendar blocks and respects prolonged coding flow.
- **Pre-Meeting Guardrail:** Suppresses break suggestions within 10 minutes of an upcoming meeting so you aren't stranded mid-break when a call starts.

### 3. Zero-Overhead Hyprland Focus Tracking
- **Socket-Based Observation:** Hooks directly into the Hyprland UNIX IPC socket (`$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket2.sock`) to react instantaneously to window focus events without polling.
- **Three-Tier Classification Hierarchy:**
  1. **Deterministic Rules (`focus_rules.yaml`):** Extensive out-of-the-box catalog of developer IDEs (VS Code, Cursor, Zed, JetBrains, Neovim), terminals (Kitty, Alacritty, Ghostty, Foot), and academic thesis tools (Zotero, Sioyek, Obsidian, LaTeX editors, PDF viewers).
  2. **Regex Pattern Matcher:** Fast title-based heuristics for ambiguous windows.
  3. **Local LLM Fallback:** Ultra-fast, lightweight classification via local Ollama (`qwen2.5:0.5b`, ~390MB, ~40ms inference) for unknown or obfuscated window titles.
- **Distraction Grace Period:** Configurable grace threshold (default: 180 seconds) prevents spamming alerts when quickly switching to a browser or reference page.

### 4. Google Calendar & Meeting Guardrails
- **Read-Only Sync:** Periodically pulls and caches upcoming events locally in SQLite via Google Calendar OAuth.
- **Smart Category Mapping:** Categorizes calendar entries using explicit mappings, color-coding, or regex rules.
- **Treadmill Step-Down Warning:** Alerts you 10 minutes prior to meetings so you can step off your walking pad, lower your desk, and test your audio before joining.
- **Safe Calendar Writes:** Optional calendar write features are strictly approval-gated and require manual confirmation before modifying any remote event.

### 5. Native Waybar & Desktop Integration
- **Atomic State Updates:** Writes formatted Waybar JSON to `/dev/shm/posture_waybar.json` (in RAM) and triggers instant bar re-renders using POSIX real-time signals (`SIGRTMIN+8`).
- **Interactive Mouse Controls:**
  - **Left-Click:** Cycle manual posture state (`sitting` → `standing` → `walking`).
  - **Middle-Click:** Snooze pending break recommendations.
  - **Right-Click:** Open the local HTMX web dashboard in your browser.
- **Desktop Notifications:** Dispatches clean, actionable alerts via `notify-send` compatible with `mako`, `dunst`, or `swaync`.

### 6. Zero-Build HTMX Web Dashboard
- **Accessible Local Web UI:** Hosted at `http://127.0.0.1:8000`.
- **Zero NPM / Node Pipeline:** Server-rendered with FastAPI and Jinja2, utilizing HTMX for real-time partial DOM swaps.
- **Interactive Control Panels:**
  - **Posture & Session Card:** Live session duration, current activity badge, and manual override buttons.
  - **Adaptive Timer Card:** Active Pomodoro status, countdown, and manual start/stop presets.
  - **Agenda & Calendar Card:** Current and upcoming schedule blocks with classified task badges.
  - **Focus & History Analytics:** Interactive daily breakdown of productive vs distracted application usage.
  - **Copilot Ask Card:** Natural-language conversational interface to query your agenda, focus statistics, and posture history.

### 7. Local Voice Copilot & Piper TTS
- **Push-to-Talk (`SUPER+V`):** Hold or toggle voice recording powered by PipeWire (`pw-record`) and transcribed locally using `faster-whisper` (int8 on CPU/GPU).
- **Conversational Queries:** Asks natural language questions to local Ollama LLMs via `/api/ask` (e.g., *"What is my next meeting and how long have I been sitting?"*).
- **Bilingual Piper Speech:** Speaks responses using local ONNX neural voices in English or German, equipped with natural time and number formatting (e.g., *"von 14 bis 15 Uhr"*).

### 8. Privacy & Security by Design
- **100% Offline by Default:** Strictly binds to `127.0.0.1`.
- **No Cloud Telemetry:** Zero analytics, ping-backs, or third-party dependencies.
- **Volatile Audio & Transient Processing:** All microphone recordings and temporary files are stored exclusively in tmpfs (`/dev/shm`) and deleted immediately after transcription.
- **Instant Controls:** One-click **Global Pause** switch and **Do Not Disturb (DND)** mode to silence all audio, notifications, and prompts instantly during presentations or gaming sessions.

---

## 📦 Prerequisites & System Requirements

| Component | Requirement | Role |
|---|---|---|
| **Operating System** | Linux (Arch Linux recommended) | System environment |
| **Compositor** | Hyprland (Wayland) | IPC window focus events and keybindings |
| **Status Bar** | Waybar | Desktop status display and interactive clicks |
| **Python** | 3.12, 3.13, or 3.14 | Runtime environment |
| **Package Manager** | [`uv`](https://docs.astral.sh/uv/) | Fast dependency and virtual environment management |
| **Notifications** | `mako`, `dunst`, or `swaync` + `notify-send` | Desktop notifications |
| **Audio (Optional)** | `pipewire`, `pw-record`, `pw-play` | Push-to-Talk voice input and Piper TTS playback |
| **Local LLM (Optional)** | [Ollama](https://ollama.com/) | Local AI classifications (`qwen2.5:0.5b`) and Q&A (`qwen3:14b`) |

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/jarvis.git ~/Projects/jarvis
cd ~/Projects/jarvis
```

### 2. Install Dependencies with `uv`
Jarvis manages dependencies using `uv` with optional dependency groups:

```bash
# Install core daemon and development packages
uv sync --all-groups
```

To install only specific capability groups:
```bash
uv sync --group dev        # Core + Pytest + Ruff
uv sync --group calendar   # Google Calendar API client
uv sync --group llm        # Local Ollama client
uv sync --group voice      # faster-whisper, piper-tts, av
```

### 3. Initialize Configuration
Copy and review the default configuration files:
```bash
cp config.yaml ~/.config/jarvis/config.yaml # or keep config.yaml in project root
```

### 4. (Optional) Download Piper TTS Voices
If you wish to enable local voice audio alerts:
```bash
mkdir -p ~/.local/share/posture-platform/voices
# English voice model:
curl -L -o ~/.local/share/posture-platform/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o ~/.local/share/posture-platform/voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# German voice model:
curl -L -o ~/.local/share/posture-platform/voices/de_DE-lila_hokuspokus-high.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/lila_hokuspokus/high/de_DE-lila_hokuspokus-high.onnx
curl -L -o ~/.local/share/posture-platform/voices/de_DE-lila_hokuspokus-high.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/lila_hokuspokus/high/de_DE-lila_hokuspokus-high.onnx.json
```

### 5. (Optional) Pull Local Ollama Models
```bash
# Ultra-fast window classification fallback model (~390MB):
ollama pull qwen2.5:0.5b

# Conversational assistant model:
ollama pull qwen3:14b # or lightweight qwen2.5:3b
```

### 6. Run the Daemon
```bash
uv run jarvis
```
The server will boot at `http://127.0.0.1:8000`. Navigate to that URL in your browser to inspect the HTMX dashboard.

---

## 🖥️ Desktop Integration Setup

### Waybar Status Module
Add the custom posture module to `~/.config/waybar/config.jsonc`:

```jsonc
"custom/posture": {
  "exec": "/home/steff/Projects/jarvis/scripts/posture-status.py",
  "return-type": "json",
  "signal": 8,
  "interval": 30,
  "max-length": 40,
  "escape": true,
  "tooltip": true,
  "on-click": "curl -fsS -X POST http://127.0.0.1:8000/api/activity/cycle >/dev/null",
  "on-click-right": "xdg-open http://127.0.0.1:8000",
  "on-click-middle": "curl -fsS -X POST http://127.0.0.1:8000/api/recommendations/snooze >/dev/null"
}
```
Add `"custom/posture"` to your `modules-right` or `modules-center` array.

### Waybar Styling
Incorporate status styling in `~/.config/waybar/style.css`:

```css
#custom-posture {
  padding: 0 10px;
  border-radius: 6px;
  background-color: #1e1e2e;
  color: #cdd6f4;
}

#custom-posture.sitting {
  color: #89b4fa;
}

#custom-posture.standing {
  color: #a6e3a1;
}

#custom-posture.walking {
  color: #f9e2af;
}

#custom-posture.offline {
  color: #6c7086;
}
```

### Hyprland Keybindings
Add custom shortcuts in `~/.config/hypr/hyprland.conf`:

```ini
# Push-to-Talk Voice Query
bind = SUPER, V, exec, /home/steff/Projects/jarvis/scripts/jarvis-ptt.sh

# Quick Posture State Cycle (Sitting -> Standing -> Walking)
bind = SUPER_ALT, P, exec, curl -fsS -X POST http://127.0.0.1:8000/api/activity/cycle >/dev/null

# Open Jarvis Dashboard
bind = SUPER_ALT, J, exec, xdg-open http://127.0.0.1:8000
```

### Hypridle Idle Detection
Merge the following listener into `~/.config/hypr/hypridle.conf` to automatically alert Jarvis when you step away:

```ini
listener {
    timeout = 300
    on-timeout = curl -fsS -X POST http://127.0.0.1:8000/api/events -H 'Content-Type: application/json' -d '{"type":"input_idle_changed","source":"hypridle","payload":{"idle":true,"idle_seconds":300}}' >/dev/null 2>&1 || true
    on-resume = curl -fsS -X POST http://127.0.0.1:8000/api/events -H 'Content-Type: application/json' -d '{"type":"input_idle_changed","source":"hypridle","payload":{"idle":false,"idle_seconds":0}}' >/dev/null 2>&1 || true
}
```

---

## ⚙️ Running as a Systemd Service

Jarvis includes a pre-configured user service unit for seamless startup with your graphical session.

1. Symlink or copy the service file to your systemd user directory:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp systemd/posture-core.service ~/.config/systemd/user/
   ```

2. Reload systemd and enable the service:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now posture-core.service
   ```

3. View real-time logs:
   ```bash
   journalctl --user -u posture-core.service -f
   ```

---

## 🔧 Configuration Reference

### `config.yaml`
Central configuration for server settings, modules, timers, and privacy:

```yaml
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
  policies:
    - posture_policy
    - pomodoro_policy
    - meeting_policy
    - focus_policy
  outputs:
    - waybar_status
    - desktop_notify

posture:
  max_sitting_minutes: 50
  max_walking_minutes: 45
  manual_override_ttl_minutes: 30
  recommendation_cooldown_minutes: 30

pomodoro:
  deep_work: "50_10"
  coding: "50_10"
  reading: "50_10"
  admin: "25_5"
  email: "25_5"
  planning: "25_5"

privacy:
  global_pause: false
  dnd_mode: false
  allow_tts: false
  allow_webcam: false
  microphone_mode: "push_to_talk"
  retention_days: 90
```

### `focus_rules.yaml`
Easily customizable rules defining how windows and application classes are categorized:
- **`work_app_classes`:** IDEs, terminals, Git clients, LaTeX tools, and academic/thesis software (Zotero, Sioyek, Obsidian).
- **`distraction_app_classes`:** Games, streaming media (Steam, Lutris, Netflix).
- **`neutral_app_classes`:** File managers, password managers, system settings.

---

## 🔌 REST API Reference

The daemon exposes an async REST API on `http://127.0.0.1:8000/api`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/state` | Returns the complete current state (posture, timer, calendar, privacy). |
| `POST` | `/api/activity` | Sets manual activity state (`sitting`, `standing`, `walking`, `away`). |
| `POST` | `/api/activity/cycle` | Cycles through active posture states (`sitting` → `standing` → `walking`). |
| `POST` | `/api/events` | Ingests a raw `PlatformEvent` onto the internal `EventBus`. |
| `GET` | `/api/recommendations` | Lists active recommendations. |
| `POST` | `/api/recommendations/snooze`| Snoozes active recommendations for a period. |
| `POST` | `/api/recommendations/{id}/accept` | Accepts and executes a recommendation. |
| `POST` | `/api/recommendations/{id}/dismiss`| Dismisses a recommendation. |
| `POST` | `/api/timers/start` | Starts a Pomodoro timer (`50_10`, `25_5`, or custom minutes). |
| `POST` | `/api/timers/stop` | Stops the running timer. |
| `GET` | `/api/history/activity` | Fetches activity spans for a given time window. |
| `GET` | `/api/history/windows` | Fetches logged window focus spans. |
| `POST` | `/api/privacy/pause` | Enables Global Privacy Pause. |
| `POST` | `/api/privacy/resume` | Resumes normal tracking operations. |
| `POST` | `/api/privacy/dnd` | Toggles Do Not Disturb mode. |
| `POST` | `/api/ask` | Natural language query endpoint to the local LLM copilot. |
| `DELETE` | `/api/ask/session/{id}` | Resets an active ephemeral conversation session (e.g. `voice`). |
| `DELETE` | `/api/ask/session` | Resets all active conversation sessions. |

---

## 🧪 Development & Testing

Jarvis includes a comprehensive test suite with over **250+ unit and integration tests** ensuring policy correctness, calendar classification accuracy, and concurrency safety:

```bash
# Run all tests using uv
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run code style and lint checks
uv run ruff check .

# Format codebase
uv run ruff format .
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
