"""Shared pytest fixtures for Posture Platform tests."""

from __future__ import annotations

import stat
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jarvis.core.database import Database
from jarvis.core.settings import (
    DEFAULT_MIGRATIONS_DIR,
    DatabaseSettings,
    DesktopNotifySettings,
    LlmSettings,
    Settings,
    TtsSettings,
    VoiceSettings,
    WaybarSettings,
)


@pytest.fixture(autouse=True)
def _isolate_host_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent ambient host environment from leaking into tests.

    Isolates:
    - Hyprland socket & instance signature (never connect to host Hyprland socket)
    - Host Waybar status file
    - Host Ollama URL & port
    - PipeWire & voice/TTS paths
    """
    # 1. Waybar isolation
    monkeypatch.delenv("POSTURE_WAYBAR_FILE", raising=False)

    # 2. Hyprland socket & compositor environment isolation
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("HYPRLAND_CMD", raising=False)
    fake_runtime = tmp_path / "fake_run"
    fake_runtime.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(fake_runtime))

    # 3. LLM base URL isolation
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:0")

    # 4. PipeWire / voice isolation env
    monkeypatch.setenv("PIPEWIRE_REMOTE", "none")


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return isolated Settings pointing to temporary SQLite database and fake services."""
    db_file = tmp_path / "test_posture.db"

    noop_notify = tmp_path / "noop-notify-send"
    noop_notify.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    noop_notify.chmod(noop_notify.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    noop_recorder = tmp_path / "noop-pw-record"
    noop_recorder.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    noop_recorder.chmod(noop_recorder.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    noop_player = tmp_path / "noop-pw-play"
    noop_player.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    noop_player.chmod(noop_player.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return Settings(
        database=DatabaseSettings(
            path=db_file,
            wal_mode=True,
            migrations_dir=DEFAULT_MIGRATIONS_DIR,
        ),
        waybar=WaybarSettings(
            status_file=tmp_path / "waybar.json",
            signal_waybar=False,
        ),
        desktop_notify=DesktopNotifySettings(
            notify_binary=str(noop_notify),
        ),
        llm=LlmSettings(
            base_url="http://127.0.0.1:0",
            text_model="qwen3:14b",
            keep_alive="5m",
        ),
        voice=VoiceSettings(
            ptt_state_file=tmp_path / "jarvis_ptt.json",
            ptt_audio_file=tmp_path / "jarvis_ptt.wav",
            recorder_binary=str(noop_recorder),
        ),
        tts=TtsSettings(
            voice_model_path=tmp_path / "voices" / "en_US-lessac-medium.onnx",
            temp_dir=str(tmp_path),
            player_binary=str(noop_player),
        ),
    )


@pytest.fixture
async def db(test_settings: Settings) -> AsyncIterator[Database]:
    """Provide an initialized async SQLite WAL Database backed by tmp_path."""
    database = Database(
        test_settings.database.path,
        migrations_dir=test_settings.migrations_dir,
        wal_mode=test_settings.database.wal_mode,
        busy_timeout_ms=test_settings.database.busy_timeout_ms,
    )
    await database.connect(run_migrations=True)
    try:
        yield database
    finally:
        await database.close()


calendar_db = db

