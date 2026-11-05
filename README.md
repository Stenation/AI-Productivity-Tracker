# 🛡️ Jarvis

> A personal, vibe-coded posture, focus, and calendar copilot for Arch Linux + Hyprland.

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Compositor](https://img.shields.io/badge/compositor-Hyprland-brightgreen.svg)](https://hyprland.org/)
[![Status Bar](https://img.shields.io/badge/status%20bar-Waybar-orange.svg)](https://github.com/Alexays/Waybar)
[![Tests](https://img.shields.io/badge/tests-255%20passed-success.svg)](https://github.com/pytest-dev/pytest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Privacy](https://img.shields.io/badge/privacy-local--first-purple.svg)](#-privacy--local-first)

---

## ☕ What is this?

This is a personal, vibe-coded background daemon I put together to keep myself from sitting in a chair for 8 hours straight while coding or studying on my Arch Linux + Hyprland setup.

It tracks how long you've been sitting, standing, or walking at your desk, suggests Pomodoro breaks based on what you're actually doing, hooks right into Waybar, and pulls your Google Calendar so you get a heads-up to step off your treadmill or lower your desk before a meeting starts. It also includes optional push-to-talk voice and local LLM helpers for quick questions.

It's not a commercial product, SaaS startup, or trying to sell anything—just a daily driver tool built with FastAPI, SQLite, and HTMX that works well for my workflow. If you run Arch and Hyprland and want something similar, feel free to use it, tweak it, or fork it.

---

## 📑 Table of Contents

- [How It Works](#-how-it-works)
- [Features](#-features)
  - [Posture & Desk Breaks](#1-posture--desk-breaks)
  - [Context-Aware Pomodoro](#2-context-aware-pomodoro)
  - [Hyprland Window Focus Tracking](#3-hyprland-window-focus-tracking)
  - [Google Calendar & Pre-Meeting Heads-Up](#4-google-calendar--pre-meeting-heads-up)
  - [Waybar & Desktop Controls](#5-waybar--desktop-controls)
  - [HTMX Web Dashboard](#6-htmx-web-dashboard)
  - [Optional Voice & Local AI](#7-optional-voice--local-ai)
  - [Privacy & Local-First](#8-privacy--local-first)
- [Prerequisites](#-prerequisites)
- [Quickstart & Setup](#-quickstart--setup)
- [Desktop Integration](#-desktop-integration)
  - [Waybar Module](#waybar-module)
  - [Waybar Styling](#waybar-styling)
  - [Hyprland Keybindings](#hyprland-keybindings)
  - [Hypridle (Idle Detection)](#hypridle-idle-detection)
- [Running as a User Service](#-running-as-a-user-service)
- [Configuration](#-configuration)
  - [config.yaml](#configyaml)
  - [focus_rules.yaml](#focus_rulesyaml)
- [API Endpoints](#-api-endpoints)
- [Testing & Hacking](#-testing--hacking)
- [License](#-license)

---

## ⚙️ How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DESKTOP SURFACES                                │
│   Waybar Module (SIGRTMIN+8)  │  Desktop Notifications (mako/dunst)    │
│   HTMX Dashboard (localhost)  │  Push-to-Talk Voice (SUPER+V)          │
└───────────────────▲───────────────────────────────┬────────────────────┘
                    │ JSON / SSE                    │ Webhooks / Clicks
┌───────────────────┴───────────────────────────────▼────────────────────┐
│                        JARVIS DAEMON (FastAPI)                         │
│                                                                        │
│   ┌───────────────┐     ┌────────────────┐     ┌───────────────────┐   │
│   │   EventBus    │────▶│ State Resolver │────▶│  Policy Engine    │   │
│   │ (Typed Events)│     │  & SQLite WAL  │     │ (Posture/Pomodoro)│   │
│   └───────────────┘     └────────────────┘     └───────────────────┘   │
└───────▲─────────────────▲─────────────────▲─────────────────────▲──────┘
        │                 │                 │                     │
┌───────┴──────┐   ┌──────┴────────┐ ┌──────┴─────────┐   ┌───────┴──────┐
│ Hyprland IPC │   │ Google Cal    │ │ Manual Input   │   │ Optional     │
│ Socket       │   │ Sync (OAuth)  │ │ (Waybar clicks)│   │ Sensors/BLE  │
└──────────────┘   └───────────────┘ └────────────────┘   └──────────────┘
```

- **Local Daemon:** Runs on `127.0.0.1:8000` with SQLite in WAL mode. Posture tracking, window logging, timers, and local AI all stay local.
- **External Cloud Boundary:** Google Calendar is the only thing that talks to an external network (using your own OAuth credentials to pull upcoming events). If you don't configure it or disable it in `config.yaml`, the daemon stays completely offline.
- **Hyprland IPC:** Listens directly to the Hyprland UNIX socket to detect active windows without polling.
- **Waybar Updates:** Writes a small JSON file to `/dev/shm` and signals Waybar using `SIGRTMIN+8` for instant bar updates on clicks.

---

## ✨ Features

### 1. Posture & Desk Breaks
- **Activity tracking:** Keeps tabs on time spent `sitting`, `standing`, `walking`, or `away`.
- **Healthy limits:** Configurable reminders when sitting too long (default 50 mins) or walking too long (default 45 mins).
- **Manual or sensor inputs:** Easily toggle posture manually (via Waybar click or shortcut), or let it read from BLE walking pads, desk height sensors, or webcam pose estimation.
- **Stepped cooldowns:** If you snooze or dismiss an alert, it backs off gracefully instead of constantly nagging you.
- **Reboot safety:** Doesn't lose or corrupt open spans if you reboot or shut down abruptly.

### 2. Context-Aware Pomodoro
- **Task-based presets:**
  - **50 / 10:** Deep work, coding, reading, thesis research.
  - **25 / 5:** Admin tasks, emails, planning.
  - **None / Suppressed:** Active video calls or scheduled breaks.
- **Flow friendly:** Recognizes `#flow` or `#notimer` in calendar blocks and respects your focus.
- **Meeting guardrail:** Suppresses break suggestions when a meeting is less than 10 minutes away, so you don't walk away right before a sync.

### 3. Hyprland Window Focus Tracking
- **Zero-polling socket listener:** Connects to `$XDG_RUNTIME_DIR/hypr/.../.socket2.sock`.
- **Three-tier classification:**
  1. `focus_rules.yaml`: Simple, editable list of work tools (editors, terminals, PDF viewers, Zotero, Obsidian, LaTeX), distraction apps, and neutral utilities.
  2. Regex rules: Matches titles for quick categorization.
  3. Optional local LLM: Tiny local Ollama model (`qwen2.5:0.5b`, ~40ms) as a fallback for ambiguous titles.
- **Grace period:** 3-minute grace period by default so quickly looking something up in a browser doesn't immediately flag you as distracted.

### 4. Google Calendar & Pre-Meeting Heads-Up
- **Local event cache:** Syncs upcoming schedule blocks into SQLite via OAuth.
- **Heads-up warning:** Alerts you 10 minutes before a meeting so you have time to lower your standing desk, step off your treadmill, and grab water.
- **Approval-gated writes:** Any potential calendar writing requires explicit manual approval.

### 5. Waybar & Desktop Controls
- **One-click actions:**
  - **Left-Click:** Cycle posture (`sitting` → `standing` → `walking`).
  - **Right-Click:** Open the local HTMX dashboard in your default browser.
  - **Middle-Click:** Snooze the current recommendation.
- **Desktop notifications:** Gentle notifications sent via `notify-send` (works with `mako`, `dunst`, `swaync`).

### 6. HTMX Web Dashboard
- **Runs locally at `http://127.0.0.1:8000`**.
- **No Node/npm build steps:** Built simply with FastAPI, Jinja2 templates, and HTMX for partial swaps.
- **Overview cards:** Real-time posture status, timer countdowns, upcoming calendar agenda, daily focus history, and privacy toggles.

### 7. Optional Voice & Local AI
- **Push-to-Talk (`SUPER+V`):** Hold or press a hotkey to record audio with PipeWire (`pw-record`) and transcribe locally with `faster-whisper`.
- **Bilingual TTS (Piper):** Speaks responses with local ONNX voices in English or German, with normalized time pronunciation.
- **Ollama Q&A (`/api/ask`):** Ask natural-language questions about your schedule or posture history.

### 8. Privacy & Local-First
- Binds strictly to `127.0.0.1`.
- Zero telemetry, analytics, or third-party tracking.
- Audio recordings stay in volatile RAM (`/dev/shm`) and get deleted right after transcription.
- Quick **Global Pause** and **Do Not Disturb (DND)** buttons for gaming, watching videos, or screen sharing.

---

## 📦 Prerequisites

- **OS:** Linux (built on Arch Linux)
- **Compositor:** Hyprland
- **Status Bar:** Waybar
- **Python:** 3.12+
- **Tooling:** [`uv`](https://docs.astral.sh/uv/) for running the daemon and dependencies
- **Optional:**
  - `pipewire`, `pw-record`, `pw-play` (for voice and TTS)
  - [Ollama](https://ollama.com/) (for local text classification & `/api/ask`)
  - Piper ONNX voice models (for voice responses)

---

## 🚀 Quickstart & Setup

### 1. Clone & install dependencies
```bash
git clone https://github.com/your-username/jarvis.git ~/Projects/jarvis
cd ~/Projects/jarvis

# Install core packages and dev tools with uv
uv sync --all-groups
```

### 2. Configuration
The daemon looks for `config.yaml` in the working directory (or via `--config`):
```bash
cp config.yaml ~/.config/jarvis/config.yaml # optional custom path
```

### 3. (Optional) Local Models
If you want to use the optional local LLM features:
```bash
# Lightweight model for window classification fallback (~390MB):
ollama pull qwen2.5:0.5b

# Model for general /api/ask queries:
ollama pull qwen3:14b # or qwen2.5:3b
```

### 4. Run the daemon
```bash
uv run jarvis
```
Open `http://127.0.0.1:8000` to view the dashboard.

---

## 🖥️ Desktop Integration

### Waybar Module
Add this custom module to your `~/.config/waybar/config.jsonc`:

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

### Waybar Styling
In your `~/.config/waybar/style.css`:

```css
#custom-posture {
  padding: 0 10px;
  border-radius: 6px;
  background-color: #1e1e2e;
  color: #cdd6f4;
}

#custom-posture.sitting { color: #89b4fa; }
#custom-posture.standing { color: #a6e3a1; }
#custom-posture.walking { color: #f9e2af; }
#custom-posture.offline { color: #6c7086; }
```

### Hyprland Keybindings
In your `~/.config/hypr/hyprland.conf`:

```ini
# Push-to-Talk voice query
bind = SUPER, V, exec, /home/steff/Projects/jarvis/scripts/jarvis-ptt.sh

# Quick cycle posture (sitting -> standing -> walking)
bind = SUPER_ALT, P, exec, curl -fsS -X POST http://127.0.0.1:8000/api/activity/cycle >/dev/null

# Open dashboard
bind = SUPER_ALT, J, exec, xdg-open http://127.0.0.1:8000
```

### Hypridle (Idle Detection)
In `~/.config/hypr/hypridle.conf` to notify Jarvis when you walk away:

```ini
listener {
    timeout = 300
    on-timeout = curl -fsS -X POST http://127.0.0.1:8000/api/events -H 'Content-Type: application/json' -d '{"type":"input_idle_changed","source":"hypridle","payload":{"idle":true,"idle_seconds":300}}' >/dev/null 2>&1 || true
    on-resume = curl -fsS -X POST http://127.0.0.1:8000/api/events -H 'Content-Type: application/json' -d '{"type":"input_idle_changed","source":"hypridle","payload":{"idle":false,"idle_seconds":0}}' >/dev/null 2>&1 || true
}
```

---

## 🛠️ Running as a User Service

You can manage Jarvis with systemd user services so it starts automatically with your Hyprland session:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/posture-core.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now posture-core.service
```

View logs with:
```bash
journalctl --user -u posture-core.service -f
```

---

## 🔧 Configuration

### `config.yaml`
Main configuration for timers, posture thresholds, enabled modules, and privacy:

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
    # - google_calendar
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
```

### `focus_rules.yaml`
List of application classes and how they should be classified:
- **`work_app_classes`**: IDEs, terminals, Git tools, document & PDF viewers, LaTeX tools, reference managers.
- **`distraction_app_classes`**: Games, video streaming.
- **`neutral_app_classes`**: Audio volume mixers, settings, password managers.

---

## 🔌 API Endpoints

A quick overview of handy local endpoints on `http://127.0.0.1:8000/api`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/state` | Full state dump (posture, timer, calendar, privacy). |
| `POST` | `/api/activity` | Set posture manually (`sitting`, `standing`, `walking`, `away`). |
| `POST` | `/api/activity/cycle` | Cycle through active posture states. |
| `GET` | `/api/recommendations` | View active recommendations. |
| `POST` | `/api/recommendations/snooze` | Snooze current recommendation. |
| `POST` | `/api/timers/start` | Start a timer (`50_10`, `25_5`, or custom duration). |
| `POST` | `/api/timers/stop` | Stop the running timer. |
| `POST` | `/api/privacy/pause` | Toggle Global Pause. |
| `POST` | `/api/privacy/dnd` | Toggle Do Not Disturb mode. |
| `POST` | `/api/ask` | Send a question to the local LLM assistant. |

---

## 🧪 Testing & Hacking

The codebase has a comprehensive test suite covering the event bus, policies, calendar matching, and state resolver:

```bash
# Run tests
uv run pytest

# Check code formatting & linting
uv run ruff check .
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
