#!/usr/bin/env python3
"""Minimal stdlib-only Waybar status reader for custom/posture."""

from __future__ import annotations

import json
import os
import sys

DEFAULT_STATUS_FILE = "/dev/shm/posture_waybar.json"
ENV_VAR_NAME = "POSTURE_WAYBAR_FILE"

OFFLINE_PAYLOAD: dict[str, str] = {
    "text": "POSTURE OFFLINE",
    "tooltip": "posture-core.service is not running",
    "class": "offline",
}


def read_waybar_payload(file_path: str | None = None) -> dict[str, str]:
    """Read and validate the Waybar JSON status file, falling back to offline JSON."""
    configured = (
        file_path
        if file_path is not None
        else os.environ.get(ENV_VAR_NAME, "").strip()
    )
    target = os.path.expanduser(configured or DEFAULT_STATUS_FILE)

    try:
        with open(target, encoding="utf-8") as handle:
            raw = handle.read()
        data = json.loads(raw)
    except (OSError, ValueError):
        return dict(OFFLINE_PAYLOAD)

    if not isinstance(data, dict):
        return dict(OFFLINE_PAYLOAD)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return dict(OFFLINE_PAYLOAD)

    tooltip_raw = data.get("tooltip", "")
    class_raw = data.get("class", "unknown")
    tooltip = tooltip_raw if isinstance(tooltip_raw, str) else str(tooltip_raw)
    css_class = (
        class_raw
        if isinstance(class_raw, str) and class_raw.strip()
        else "unknown"
    )

    return {
        "text": text,
        "tooltip": tooltip,
        "class": css_class,
    }


def main() -> int:
    """Print single-line Waybar JSON to stdout."""
    payload = read_waybar_payload()
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
