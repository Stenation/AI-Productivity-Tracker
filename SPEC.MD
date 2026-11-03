# Posture Platform — Specification

A local-first, modular posture, focus, and calendar copilot for Arch Linux + Hyprland.

It tracks sitting, standing, walking, focus context, and calendar blocks; then provides
low-friction recommendations and timers. All routine decisions are deterministic.
Local LLM/VLM features are optional plugins.

## 1. Goals

### Primary goals

- Reduce uninterrupted sitting and encourage healthy standing/walking breaks.
- Recommend suitable activities based on the current Google Calendar block.
- Suggest 25/5 or 50/10 Pomodoro cycles based on task type.
- Provide Hyprland-native controls through Waybar and desktop notifications.
- Keep all sensitive processing local by default.
- Remain useful when optional sensors, phone integrations, voice, or LLMs are disabled.

### Non-goals

- Not a medical posture-correction or ergonomic-assessment device.
- Not a surveillance or employee-monitoring system.
- Not a cloud-based activity tracker.
- Not a replacement for Google Calendar, Google Tasks, Obsidian, or Boosted.
- Not guaranteed to detect phone use, desk height, or walking with high accuracy.

## 2. Environment

| Component | Baseline | Role |
|---|---|---|
| OS / compositor | Arch Linux + Hyprland | Hyprland IPC, Waybar, notifications, keybinds |
| GPU | AMD RX 9070 XT, 16 GB VRAM | Optional local LLM/VLM inference |
| RAM | 32 GB DDR5 | Local models, browser dashboard, workers |
| Desk | Flexispot E7 Pro | Sitting/standing context; optional future height sensor |
| Walking pad | Sportstech sWalk Lite Plus | Optional BLE walking detection; manual fallback |
| Webcam | Existing low-resolution Logitech webcam | Optional low-frequency posture detection |
| Phone | Galaxy S24 Plus | Optional webhook input and notification bridge |
| Watch | Galaxy Watch 7 | Optional notification target through phone bridge |
| Timer app | Boosted | Optional external timer; local Waybar timer is primary |

## 3. Core concepts

### Activity states

- `sitting`
- `standing`
- `walking`
- `away`
- `unknown`

### Focus states

- `focused`
- `neutral`
- `distracted`
- `unknown`

### Calendar task types

| Task type | Default Pomodoro | Walking suitable |
|---|---|---|
| `deep_work` | 50/10 | Usually no |
| `coding` | 50/10 | Usually no |
| `reading` | 50/10 | Sometimes |
| `admin` | 25/5 | Yes |
| `email` | 25/5 | Yes |
| `planning` | 25/5 | Yes |
| `meeting` | None | No |
| `break` | None | Yes |
| `unknown` | None by default | Unknown |

### Recommendation lifecycle

```text
pending → accepted | snoozed | dismissed | expired
```

Recommendations are non-destructive. They may suggest an action, but they do not
automatically change calendar events, start Boosted, or capture the screen unless the
relevant module is explicitly enabled and approval requirements are met.

## 4. Feature matrix

### P1 — Must have

| ID | Feature | Description |
|---|---|---|
| FR-01 | Activity tracking | Track `sitting`, `standing`, `walking`, `away`, and `unknown` states in SQLite with source, confidence, reason, start/end timestamps. |
| FR-02 | Manual override | Waybar, HTMX dashboard, CLI, or API can set activity manually. Manual state overrides sensors for a configurable TTL. |
| FR-03 | Calendar read sync | Read and cache current/upcoming Google Calendar events. |
| FR-04 | Local Pomodoro timer | Run 25/5, 50/10, or custom timers locally and display them in Waybar. |
| FR-05 | Posture policy | Recommend standing/walking after excessive sitting and seated breaks after excessive walking. |
| FR-06 | Pomodoro policy | Recommend 25/5 or 50/10 based on calendar task type and user preferences. |
| FR-07 | Meeting guardrail | Warn before meetings so the user can stop walking and return to the desk. |
| FR-08 | Hyprland observation | Passively observe active window class/title and input-idle duration through Hyprland IPC. |
| FR-09 | Focus logging | Store window spans and focus categories for later analysis. |
| FR-10 | Desktop notifications | Show recommendations, timer transitions, and approval prompts through `mako` or `dunst`. |
| FR-11 | Waybar status | Display activity, duration, timer state, and current calendar context. |
| FR-12 | HTMX dashboard | Provide local calibration, analytics, module health, privacy switches, and manual controls. |
| FR-13 | Modular configuration | Enable/disable input, policy, and output modules through `config.yaml` without changing core code. |
| FR-14 | Privacy pause | Globally pause camera, microphone, screen observation, phone webhooks, and external notifications. |
| FR-15 | DND / gaming mode | Suppress TTS, phone alerts, distraction prompts, and VLM checks while optionally continuing background logging. |

### P2 — Should have

| ID | Feature | Description |
|---|---|---|
| FR-16 | Calendar task classification | Map event titles, descriptions, colors, or user rules to task types. |
| FR-17 | Walking suitability | Determine whether the current task is suitable for walking. |
| FR-18 | sWalk BLE input | Read walking state/speed from the sWalk Lite Plus if its BLE interface is accessible. |
| FR-19 | Webcam sit/stand detection | Detect gross sitting versus standing at 0.2 FPS using calibrated shoulder-height baselines. |
| FR-20 | Phone distraction prompt | Receive an optional webhook when the phone remains unlocked beyond a threshold during a focus block, then issue a soft prompt. |
| FR-21 | Phone notification bridge | Send selected timer/posture notifications to the phone through KDE Connect, ntfy, or another configurable bridge. |
| FR-22 | Approval-gated calendar writes | Draft calendar changes and require explicit user approval before writing. |
| FR-23 | Local text LLM | Optionally use a local text model for ambiguous task classification, natural-language Q&A, and calendar drafting. |
| FR-24 | Push-to-talk voice | Provide optional `SUPER+V` voice input using local speech-to-text. |
| FR-25 | Local TTS | Optionally speak selected alerts using Piper. |

### P3 — Could have / experimental

| ID | Feature | Description |
|---|---|---|
| FR-26 | Webcam walking detection | Infer walking from shoulder oscillation when BLE and phone-step inputs are unavailable. |
| FR-27 | VLM screenshot check | Optionally classify one user-approved screenshot using a local vision-language model. |
| FR-28 | Contextual roast mode | Optional, rate-limited humorous feedback after an approved screen check. |
| FR-29 | Boosted automation | Optionally trigger or prefill Boosted through an Android automation bridge. |
| FR-30 | Desk-height input | Add a camera, ToF, ultrasonic, or controller-based desk-height module. |
| FR-31 | Wearable integration | Send selected events to Galaxy Watch 7 through the phone bridge. |

## 5. Detailed requirements

### Activity and overrides

- **FR-01:** The system shall persist every effective activity span with:
  - activity state
  - source
  - confidence
  - transition reason
  - start timestamp
  - end timestamp
- **FR-02:** A manual override shall take precedence over automated sensors until its TTL expires, the user changes state again, or the current timer ends if configured.
- **FR-02.1:** The system shall support manual states: `sitting`, `standing`, `walking`, and `away`.
- **FR-02.2:** Manual overrides shall be recorded with `confidence = 1.0` and `reason = manual_override`.

### Calendar

- **FR-03:** The system shall cache calendar events locally and refresh on a configurable interval.
- **FR-03.1:** Calendar access shall use the minimum OAuth scope required for the enabled feature:
  - `calendar.readonly` for read-only mode
  - `calendar.events` only when calendar writing is enabled
- **FR-16:** The system shall classify calendar events into task types using, in order:
  1. explicit user mapping
  2. event color
  3. title/description rules
  4. optional local LLM classification
- **FR-22:** The system shall not create, move, or edit a Google Calendar event without explicit user approval.
- **FR-22.1:** Calendar changes shall use the Google Calendar API. Partial updates should use `events.patch`; full replacement should use `events.update`. [120][122]

### Timers and posture

- **FR-04:** The system shall support local timers independent of Boosted.
- **FR-04.1:** Timer state shall be visible in Waybar.
- **FR-05:** Default posture rules shall be configurable:
  - maximum continuous sitting: 50 minutes
  - maximum continuous walking: 45 minutes
  - pre-meeting warning: 10 minutes
- **FR-06:** The default Pomodoro selector shall be:
  - `deep_work`, `coding`, intensive `reading`: 50/10
  - `admin`, `email`, `planning`: 25/5
  - `meeting`: no Pomodoro
  - `unknown`: suggest, do not auto-start
- **FR-07:** If a meeting begins within the configured warning window, the system shall recommend leaving the walking pad and returning to the desk.

### Screen and focus

- **FR-08:** The Hyprland module shall listen to IPC events rather than continuously poll where possible.
- **FR-09:** The system shall store window spans with app class, title, category, start, and end time.
- **FR-09.1:** Focus classification shall use this precedence:
  1. app-class rules
  2. title rules
  3. calendar-task mappings
  4. optional local LLM classification
- **FR-09.2:** The default screen mode shall be `observe_only`.
- **FR-09.3:** Distraction prompts require:
  - an active productivity block
  - a classified distraction or unknown window
  - a configurable grace period, default 180 seconds
  - DND mode disabled
- **FR-27:** VLM screenshot checks are disabled by default and require per-incident approval.
- **FR-27.1:** If enabled, screenshots shall be written only to `tmpfs`/`/dev/shm`, processed locally, and deleted immediately after inference.
- **FR-28:** Roast mode is disabled by default, rate-limited, and must be separately enabled.

### Phone and wearables

- **FR-20:** Phone distraction detection is optional and must degrade gracefully if the phone is offline.
- **FR-20.1:** A phone-unlocked event shall produce a soft prompt, not an automatic punishment.
- **FR-21:** The phone bridge shall support at least one transport:
  - KDE Connect
  - ntfy
  - generic HTTP webhook
- **FR-29:** Boosted automation is optional and must never be required for the local timer to work.
- **FR-31:** Watch notifications are optional and depend on the phone bridge.

### Voice, LLM, and VLM

- **FR-24:** Voice input shall be push-to-talk by default; always-on microphone listening is prohibited.
- **FR-23:** The local text LLM is optional. Core posture, timer, calendar, and focus rules must work without it.
- **FR-27:** The VLM is optional and used only for explicitly approved screenshot checks or user-initiated screen questions.
- **FR-25:** TTS is optional and must be disabled in DND mode.

## 6. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | Modularity | Every input, policy, and output module can be enabled/disabled through configuration. |
| NFR-02 | Replaceability | A sensor module can be replaced without modifying policy or output modules. |
| NFR-03 | Local-first | Camera, microphone, screen, OCR, and LLM/VLM processing remain local unless the user explicitly enables an external service. |
| NFR-04 | Core resilience | Core timer, state tracking, and notifications continue if webcam, BLE, phone, voice, or LLM modules fail. |
| NFR-05 | Performance | Core daemon target: under 1% average CPU over 60 seconds when idle. Vision worker target: under 5% average CPU at 0.2 FPS. GPU target: 0% when LLM/VLM features are disabled. |
| NFR-06 | Privacy | No continuous audio recording, no continuous video retention, no persistent screenshots, no automatic calendar writes. |
| NFR-07 | Data minimization | Store derived states and metadata; do not store raw camera frames, raw audio, or screenshots by default. |
| NFR-08 | Observability | Provide module health, recent events, errors, and recommendation history in the HTMX dashboard. |
| NFR-09 | Configurability | Thresholds, grace periods, scopes, module toggles, task mappings, and privacy modes are user-configurable. |
| NFR-10 | Upgrade safety | SQLite schema changes use versioned migrations. |

## 7. Privacy policy

### Defaults

| Capability | Default |
|---|---|
| Webcam | Disabled until explicitly enabled |
| Microphone | Push-to-talk only |
| Screen observation | Observe-only |
| Screenshots | Disabled |
| VLM analysis | Disabled |
| Calendar writes | Disabled |
| Phone webhook | Disabled |
| External notifications | Disabled |
| Raw video/audio retention | Never |
| Screenshot retention | Never, unless user explicitly saves it |

### Required controls

- Global privacy pause.
- DND / gaming mode.
- Per-module enable/disable switches.
- Camera and microphone status indicators.
- Calendar write approval prompts.
- Screenshot approval prompts.
- Configurable data-retention cleanup.
- Local-only binding for the dashboard, e.g. `127.0.0.1`.

## 8. Use cases

### UC-01 — Manual posture tracking

**Actor:** Steffen  
**Trigger:** User clicks the Waybar posture module.

**Flow:**

1. User left-clicks Waybar.
2. Core cycles `sitting → standing → walking → away`.
3. Core closes the previous activity span and opens a new one.
4. Waybar updates activity and elapsed duration.
5. HTMX dashboard and SQLite reflect the change.

**Acceptance criteria:**

- Manual state survives webcam disagreement for the configured TTL.
- Activity durations are correctly calculated after restart.
- The manual override is visible with its source and expiry time.

### UC-02 — Pre-block posture recommendation

**Actor:** Posture policy  
**Trigger:** Calendar event starts in 5 minutes.

**Flow:**

1. Calendar module emits `calendar_context_changed`.
2. Posture policy reads recent activity spans.
3. If the user has sat too long and the next task is not a meeting, it recommends standing.
4. If the task is walking-suitable, it recommends a walking block.
5. User accepts, snoozes, or dismisses.

**Example:**

> “Backend refactor starts in 5 minutes. You have been sitting for 65 minutes. Consider standing for the first part. Recommended timer: 50/10.”

### UC-03 — Walking-compatible admin block

**Actor:** Pomodoro policy  
**Trigger:** Calendar block `Email triage` starts.

**Flow:**

1. Calendar module classifies the event as `admin`.
2. Policy determines that walking is suitable.
3. System suggests a 25/5 walking block.
4. User accepts.
5. Local Waybar timer starts.
6. Optional phone bridge notifies the user to start Boosted manually or through automation.

### UC-04 — Meeting guardrail

**Actor:** Meeting policy  
**Trigger:** Meeting starts in 10 minutes while user is walking.

**Flow:**

1. Calendar module detects the upcoming meeting.
2. Policy emits a recommendation to stop walking.
3. Waybar and desktop notification display the warning.
4. Optional phone/watch bridge sends a reminder.

### UC-05 — Phone distraction prompt

**Actor:** Focus policy  
**Trigger:** S24 Plus webhook reports phone unlocked for 60 seconds during a focus block.

**Flow:**

1. Phone webhook emits `phone_state_changed`.
2. Focus policy verifies:
   - active focus timer
   - not a break
   - not a meeting
   - DND disabled
3. System sends a soft desktop and/or phone notification.
4. User can dismiss or resume focus.

**Example:**

> “Phone unlocked for 1 minute during focus block. Return to task or take a scheduled break?”

### UC-06 — Off-task window during focus

**Actor:** Focus policy  
**Trigger:** Reddit remains active for 3 minutes during a study block.

**Flow:**

1. Hyprland IPC emits `window_focus_changed`.
2. Focus classifier labels the window as `distraction`.
3. After the grace period, system sends a refocus prompt.
4. User may refocus, snooze, mark the block as changed, or request a screen check.
5. If VLM screenshots are enabled and user approves, one screenshot is processed locally and deleted.

### UC-07 — Push-to-talk question

**Actor:** Steffen  
**Trigger:** User presses `SUPER+V`.

**Flow:**

1. System records audio until the keybind is pressed again.
2. Local speech-to-text transcribes the request.
3. Core adds current calendar and activity context.
4. Optional local text model answers.
5. Piper speaks the response if TTS is enabled.

### UC-08 — Calendar reschedule proposal

**Actor:** Calendar writer  
**Trigger:** User says, “Push my next task back by 30 minutes,” or policy detects a schedule conflict.

**Flow:**

1. System creates a calendar-change draft.
2. Desktop notification shows old and proposed times.
3. User selects Approve or Reject.
4. Only on Approve does the system call the Google Calendar API.
5. The decision and result are logged.

## 9. Acceptance criteria for v1

v1 is complete when:

- The core daemon starts as a systemd user service.
- SQLite stores activity spans, window spans, calendar cache, recommendations, and settings.
- Waybar displays activity, duration, and timer state.
- Manual activity control works from Waybar and HTMX.
- Google Calendar events are read and cached.
- Posture and Pomodoro recommendations appear as desktop notifications.
- Hyprland window/idle observation works in observe-only mode.
- The HTMX dashboard shows live state, history, module health, and privacy controls.
- The system continues functioning with webcam, BLE, phone, voice, LLM, and VLM modules disabled.