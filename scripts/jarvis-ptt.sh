#!/usr/bin/env bash
#
# scripts/jarvis-ptt.sh - Push-to-Talk Toggle for Posture Platform (Jarvis)
#
# Toggles voice recording on/off using the local push-to-talk pipeline.
#
# Optional Hyprland Configuration:
# To bind this script to a key combination in your ~/.config/hypr/hyprland.conf,
# you can define a variable and bind it. For example:
#
#   $voiceKeybind = SUPER, V
#
#   bind = $voiceKeybind, exec, /home/steff/Projects/jarvis/scripts/jarvis-ptt.sh
#
# Note: You may change $voiceKeybind to any Hyprland-compatible key combination
# (e.g. SUPER, SPACE or SUPER_ALT, V). Do not bind automatically; configure
# according to your personal keybinding preferences.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Run jarvis-ptt via uv or virtual environment
if command -v uv >/dev/null 2>&1; then
    exec uv run --project "${PROJECT_ROOT}" jarvis-ptt "$@"
elif [ -x "${HOME}/.local/bin/uv" ]; then
    exec "${HOME}/.local/bin/uv" run --project "${PROJECT_ROOT}" jarvis-ptt "$@"
elif [ -x "${PROJECT_ROOT}/.venv/bin/jarvis-ptt" ]; then
    exec "${PROJECT_ROOT}/.venv/bin/jarvis-ptt" "$@"
else
    exec jarvis-ptt "$@"
fi

