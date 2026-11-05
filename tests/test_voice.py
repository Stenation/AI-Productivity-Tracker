"""Comprehensive tests for Push-to-Talk voice worker, Piper TTS module, and privacy gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from jarvis.core.app import create_app
from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent
from jarvis.core.settings import (
    PrivacySettings,
    Settings,
    VoiceSettings,
)
from jarvis.modules.outputs.tts_piper import (
    AudioPlaybackError,
    TtsPiperModule,
)
from jarvis.workers.ptt_voice import (
    FasterWhisperTranscriber,
    PttVoiceWorker,
    RecorderError,
    TranscriptionError,
    run_watchdog,
)


class FakeRecorder:
    """In-memory fake AudioRecorder for testing lifecycle without PipeWire."""

    def __init__(self) -> None:
        self.recorded_files: list[Path] = []
        self.running_pids: set[int] = set()
        self._next_pid = 4200
        self.start_error: Exception | None = None
        self.stop_error: Exception | None = None

    def start_recording(self, output_path: Path, max_duration_seconds: float) -> int:
        if self.start_error:
            raise self.start_error
        self.recorded_files.append(output_path)
        # Create a non-empty fake wav file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00fakeaudio")
        pid = self._next_pid
        self._next_pid += 1
        self.running_pids.add(pid)
        return pid

    def stop_recording(self, pid: int) -> None:
        if self.stop_error:
            raise self.stop_error
        self.running_pids.discard(pid)

    def is_recording(self, pid: int) -> bool:
        return pid in self.running_pids


class FakeTranscriber:
    """In-memory fake AudioTranscriber returning fixed or configured transcripts."""

    def __init__(self, transcript: str = "What is my next calendar block?") -> None:
        self.transcript = transcript
        self.calls: list[Path] = []
        self.error: Exception | None = None

    def transcribe(self, audio_path: Path) -> str:
        self.calls.append(audio_path)
        if self.error:
            raise self.error
        return self.transcript


class FakeAskClient:
    """In-memory fake AskApiClient tracking questions and returning mock responses."""

    def __init__(self, state_dict: dict[str, Any] | None = None) -> None:
        self.state_dict = state_dict
        self.questions: list[dict[str, Any]] = []
        self.health_reports: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def get_state(self) -> dict[str, Any] | None:
        return self.state_dict

    def ask(
        self,
        question: str,
        source: str = "voice",
        speak: bool = True,
        language: str | None = None,
    ) -> dict[str, Any]:
        if self.error:
            raise self.error
        req = {"question": question, "source": source, "speak": speak, "language": language}
        self.questions.append(req)
        return {
            "answer": f"Your next block is Deep Work. (re: {question})",
            "suggested_action": "start_timer",
            "confidence": 0.95,
            "model": "fake_model",
            "interaction_id": 1,
        }

    def report_health(self, status: str, last_error: str | None = None) -> None:
        self.health_reports.append({"status": status, "last_error": last_error})


class FakeSynthesizer:
    """In-memory fake SpeechSynthesizer for TTS testing."""

    def __init__(self, available: bool = True, error: str | None = None) -> None:
        self.available = available
        self.available_error = error
        self.synthesize_calls: list[tuple[str, Path]] = []
        self.raise_on_synthesize: Exception | None = None

    def is_available(self) -> tuple[bool, str | None]:
        return self.available, self.available_error

    async def synthesize(self, text: str, output_wav: Path) -> Path:
        if self.raise_on_synthesize:
            raise self.raise_on_synthesize
        self.synthesize_calls.append((text, output_wav))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00faketts")
        return output_wav


class FakePlayer:
    """In-memory fake AudioPlayer for TTS testing."""

    def __init__(self, available: bool = True, error: str | None = None) -> None:
        self.available = available
        self.available_error = error
        self.played_files: list[Path] = []
        self.raise_on_play: Exception | None = None

    def is_available(self) -> tuple[bool, str | None]:
        return self.available, self.available_error

    async def play(self, audio_path: Path) -> None:
        if self.raise_on_play:
            raise self.raise_on_play
        self.played_files.append(audio_path)


# ---------------------------------------------------------------------------
# Test 1: Push-to-talk start/stop lifecycle and RAM file cleanup
# ---------------------------------------------------------------------------
def test_ptt_start_stop_lifecycle_and_ram_cleanup(tmp_path: Path) -> None:
    state_file = tmp_path / "jarvis_ptt.json"
    audio_file = tmp_path / "jarvis_ptt.wav"

    recorder = FakeRecorder()
    transcriber = FakeTranscriber(transcript="Start a pomodoro timer")
    ask_client = FakeAskClient()

    settings = Settings(
        voice=VoiceSettings(
            ptt_state_file=state_file,
            ptt_audio_file=audio_file,
            max_duration_seconds=30.0,
        )
    )

    worker = PttVoiceWorker(
        settings=settings,
        recorder=recorder,
        transcriber=transcriber,
        ask_client=ask_client,
        state_file=state_file,
        audio_file=audio_file,
    )

    # 1. First invocation: Starts recording
    res1 = worker.toggle()
    assert res1["status"] == "recording_started"
    assert "pid" in res1
    assert worker.is_recording()
    assert state_file.exists()
    assert audio_file.exists()

    # 2. Second invocation: Stops recording, transcribes, calls /api/ask
    res2 = worker.toggle()
    assert res2["status"] == "asked"
    assert res2["transcript"] == "Start a pomodoro timer"
    assert "response" in res2
    assert len(transcriber.calls) == 1
    assert len(ask_client.questions) == 1
    assert ask_client.questions[0]["question"] == "Start a pomodoro timer"
    assert ask_client.questions[0]["source"] == "voice"
    assert ask_client.questions[0]["speak"] is True

    # 3. Verify RAM files are strictly deleted in finally block
    assert not worker.is_recording()
    assert not state_file.exists(), "State file must be deleted after toggle stop"
    assert not audio_file.exists(), "WAV file must be deleted after transcription"


# ---------------------------------------------------------------------------
# Test 2: Enforce 30-second recording cap (Watchdog)
# ---------------------------------------------------------------------------
def test_ptt_watchdog_enforces_recording_cap(tmp_path: Path) -> None:
    import subprocess
    from contextlib import suppress

    state_file = tmp_path / "jarvis_ptt.json"
    audio_file = tmp_path / "jarvis_ptt.wav"

    proc = subprocess.Popen(["sleep", "10"])
    try:
        state_file.write_text(f'{{"pid": {proc.pid}, "audio_path": "{audio_file}"}}')
        audio_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")

        assert state_file.exists()
        assert audio_file.exists()

        # Run watchdog with short timeout against the live PID
        run_watchdog(
            pid=proc.pid,
            timeout_seconds=0.01,
            audio_path_str=str(audio_file),
            state_file_str=str(state_file),
        )

        # Watchdog must terminate PID and clean up state and audio files
        assert not state_file.exists()
        assert not audio_file.exists()
        proc.wait(timeout=1.0)
        assert proc.poll() is not None
    finally:
        with suppress(Exception):
            proc.kill()


# ---------------------------------------------------------------------------
# Test 3: WAV cleanup in finally block even when transcription fails
# ---------------------------------------------------------------------------
def test_ptt_wav_cleanup_on_transcription_error(tmp_path: Path) -> None:
    state_file = tmp_path / "jarvis_ptt.json"
    audio_file = tmp_path / "jarvis_ptt.wav"

    recorder = FakeRecorder()
    transcriber = FakeTranscriber()
    transcriber.error = TranscriptionError("Whisper failed on corrupted audio")
    ask_client = FakeAskClient()

    worker = PttVoiceWorker(
        recorder=recorder,
        transcriber=transcriber,
        ask_client=ask_client,
        state_file=state_file,
        audio_file=audio_file,
    )

    # Start recording
    worker.toggle()
    assert audio_file.exists()
    assert state_file.exists()

    # Stop recording - transcription raises error
    res = worker.toggle()
    assert res["status"] == "degraded"
    assert "Whisper failed" in res["error"]

    # Verify both state and audio file are deleted in finally block
    assert not audio_file.exists(), "WAV file must be deleted despite transcription error"
    assert not state_file.exists(), "State file must be deleted despite transcription error"
    assert len(ask_client.health_reports) == 1
    assert ask_client.health_reports[0]["status"] == "degraded"


# ---------------------------------------------------------------------------
# Test 4: WAV cleanup in finally block when recorder fails on start
# ---------------------------------------------------------------------------
def test_ptt_cleanup_on_recorder_error(tmp_path: Path) -> None:
    state_file = tmp_path / "jarvis_ptt.json"
    audio_file = tmp_path / "jarvis_ptt.wav"

    recorder = FakeRecorder()
    recorder.start_error = RecorderError("PipeWire daemon not responding")
    ask_client = FakeAskClient()

    worker = PttVoiceWorker(
        recorder=recorder,
        ask_client=ask_client,
        state_file=state_file,
        audio_file=audio_file,
    )

    res = worker.toggle()
    assert res["status"] == "error"
    assert "PipeWire" in res["error"]
    assert not audio_file.exists()
    assert not state_file.exists()
    assert len(ask_client.health_reports) == 1
    assert ask_client.health_reports[0]["status"] == "degraded"


# ---------------------------------------------------------------------------
# Test 5: Rejection when microphone_mode == "disabled"
# ---------------------------------------------------------------------------
def test_ptt_aborts_when_microphone_mode_disabled(tmp_path: Path) -> None:
    state_file = tmp_path / "jarvis_ptt.json"
    audio_file = tmp_path / "jarvis_ptt.wav"

    settings = Settings(
        privacy=PrivacySettings(microphone_mode="disabled"),
    )
    recorder = FakeRecorder()
    worker = PttVoiceWorker(
        settings=settings,
        recorder=recorder,
        state_file=state_file,
        audio_file=audio_file,
    )

    res = worker.toggle()
    assert res["status"] == "aborted"
    assert "Microphone is disabled" in res["reason"]
    assert len(recorder.recorded_files) == 0
    assert not state_file.exists()
    assert not audio_file.exists()


# ---------------------------------------------------------------------------
# Test 6: Rejection when privacy.global_pause is active
# ---------------------------------------------------------------------------
def test_ptt_aborts_when_global_pause_active(tmp_path: Path) -> None:
    state_file = tmp_path / "jarvis_ptt.json"
    audio_file = tmp_path / "jarvis_ptt.wav"

    settings = Settings(
        privacy=PrivacySettings(global_pause=True),
    )
    recorder = FakeRecorder()
    worker = PttVoiceWorker(
        settings=settings,
        recorder=recorder,
        state_file=state_file,
        audio_file=audio_file,
    )

    res = worker.toggle()
    assert res["status"] == "aborted"
    assert "global pause" in res["reason"].lower()
    assert len(recorder.recorded_files) == 0
    assert not state_file.exists()
    assert not audio_file.exists()


# ---------------------------------------------------------------------------
# Test 7: API /api/ask rejects voice when microphone_mode is disabled
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_api_ask_rejects_voice_when_mic_disabled(
    test_settings: Settings,
) -> None:
    test_settings.privacy.microphone_mode = "disabled"
    app = create_app(settings=test_settings)

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/ask",
                json={"question": "Hello", "source": "voice", "speak": True},
            )
            assert resp.status_code == 409
            body = resp.json()
            assert body["error"] == "microphone_disabled"


# ---------------------------------------------------------------------------
# Test 8: API /api/ask rejects when global pause is active
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_api_ask_rejects_when_global_pause(
    test_settings: Settings,
) -> None:
    test_settings.privacy.global_pause = True
    app = create_app(settings=test_settings)

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/ask",
                json={"question": "Hello", "source": "voice", "speak": True},
            )
            assert resp.status_code == 409
            body = resp.json()
            assert body["error"] == "privacy_paused"


# ---------------------------------------------------------------------------
# Test 9: TTS Piper synthesis and playback with WAV cleanup
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_piper_synthesis_and_playback(
    db: Database,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    temp_dir = tmp_path / "shm"
    fake_syn = FakeSynthesizer()
    fake_player = FakePlayer()

    config = {
        "allow_tts": True,
        "global_pause": False,
        "dnd_mode": False,
        "temp_dir": str(temp_dir),
    }

    tts_mod = TtsPiperModule(
        bus=bus,
        db=db,
        config=config,
        synthesizer=fake_syn,
        player=fake_player,
        temp_dir=temp_dir,
    )
    await tts_mod.start()

    try:
        # Publish tts_requested
        await bus.publish(
            PlatformEvent(
                type="tts_requested",
                source="test",
                payload={"text": "Stand up and stretch."},
            )
        )

        assert len(fake_syn.synthesize_calls) == 1
        assert fake_syn.synthesize_calls[0][0] == "Stand up and stretch."
        assert len(fake_player.played_files) == 1

        # Verify WAV was deleted after playback
        played_wav = fake_player.played_files[0]
        assert not played_wav.exists(), "TTS WAV must be deleted immediately after playback"
        assert list(temp_dir.glob("*.wav")) == []
    finally:
        await tts_mod.stop()


# ---------------------------------------------------------------------------
# Test 10: TTS suppression when allow_tts=false, global_pause=true, or dnd_mode=true
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_suppression_gates(db: Database, tmp_path: Path) -> None:
    bus = EventBus()
    fake_syn = FakeSynthesizer()
    fake_player = FakePlayer()

    # Case A: allow_tts is False
    tts_mod = TtsPiperModule(
        bus=bus,
        db=db,
        config={"allow_tts": False, "global_pause": False, "dnd_mode": False},
        synthesizer=fake_syn,
        player=fake_player,
        temp_dir=tmp_path,
    )
    await tts_mod.start()
    try:
        await bus.publish(
            PlatformEvent(
                type="tts_requested", source="test", payload={"text": "Suppressed"}
            )
        )
        assert len(fake_syn.synthesize_calls) == 0

        # Case B: Update privacy_mode_changed to allow_tts=True, but global_pause=True
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                payload={"allow_tts": True, "global_pause": True, "dnd_mode": False},
            )
        )
        await bus.publish(
            PlatformEvent(
                type="tts_requested", source="test", payload={"text": "Suppressed"}
            )
        )
        assert len(fake_syn.synthesize_calls) == 0

        # Case C: global_pause=False, but dnd_mode=True
        await bus.publish(
            PlatformEvent(
                type="privacy_mode_changed",
                source="api",
                payload={"allow_tts": True, "global_pause": False, "dnd_mode": True},
            )
        )
        await bus.publish(
            PlatformEvent(
                type="tts_requested", source="test", payload={"text": "Suppressed"}
            )
        )
        assert len(fake_syn.synthesize_calls) == 0
    finally:
        await tts_mod.stop()


# ---------------------------------------------------------------------------
# Test 11: Missing Piper / Whisper degraded resilience
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_dependencies_degraded_resilience(
    db: Database,
    tmp_path: Path,
) -> None:
    bus = EventBus()

    # TTS missing model/binary -> reports degraded in health, doesn't crash daemon
    missing_syn = FakeSynthesizer(
        available=False,
        error="Piper voice model not found at ~/.local/share/.../en_US-lessac-medium.onnx",
    )
    fake_player = FakePlayer()

    tts_mod = TtsPiperModule(
        bus=bus,
        db=db,
        config={"allow_tts": True},
        synthesizer=missing_syn,
        player=fake_player,
        temp_dir=tmp_path,
    )
    await tts_mod.start()
    try:
        health = await tts_mod.health()
        assert health["status"] == "degraded"
        assert "Piper voice model not found" in health["last_error"]

        # An event shouldn't crash the loop
        await bus.publish(
            PlatformEvent(
                type="tts_requested", source="test", payload={"text": "Test"}
            )
        )
    finally:
        await tts_mod.stop()

    # Whisper missing -> FasterWhisperTranscriber raises cleanly
    transcriber = FasterWhisperTranscriber()
    # Force _model to None and test non-existent file check
    with pytest.raises(TranscriptionError, match="Audio file does not exist"):
        transcriber.transcribe(tmp_path / "nonexistent.wav")


# ---------------------------------------------------------------------------
# Test 12: PipeWire Player playback failure handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_player_failure_marks_degraded(
    db: Database,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    fake_syn = FakeSynthesizer()
    fake_player = FakePlayer()
    fake_player.raise_on_play = AudioPlaybackError("pw-play failed: device busy")

    temp_dir = tmp_path / "shm"
    tts_mod = TtsPiperModule(
        bus=bus,
        db=db,
        config={"allow_tts": True, "global_pause": False, "dnd_mode": False},
        synthesizer=fake_syn,
        player=fake_player,
        temp_dir=temp_dir,
    )
    await tts_mod.start()
    try:
        await bus.publish(
            PlatformEvent(
                type="tts_requested", source="test", payload={"text": "Will fail"}
            )
        )
        health = await tts_mod.health()
        assert health["status"] == "degraded"
        assert "pw-play failed" in health["last_error"]
        # Temp WAV should still be cleaned up!
        assert list(temp_dir.glob("*.wav")) == []
    finally:
        await tts_mod.stop()


# ---------------------------------------------------------------------------
# Test 13: Hyprland Script exists and includes non-hardcoded documentation
# ---------------------------------------------------------------------------
def test_hyprland_script_contents() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "jarvis-ptt.sh"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    assert "$voiceKeybind = SUPER, V" in content
    assert "bind = $voiceKeybind, exec," in content
    assert "jarvis-ptt" in content


# ---------------------------------------------------------------------------
# Test 14: Bilingual German/English Text Detection & TTS Routing
# ---------------------------------------------------------------------------
def test_is_german_text() -> None:
    from jarvis.modules.outputs.tts_piper import is_german_text

    # German sentences with umlauts
    assert is_german_text("Dein nächster Termin ist Training.") is True
    assert is_german_text("Schöne Pause!") is True
    # German sentence without umlauts (stopwords matching)
    assert is_german_text("Das ist ein Test für heute.") is True
    # English sentences
    assert is_german_text("Your next appointment is Training at 7 PM.") is False
    assert is_german_text("Start a pomodoro timer for coding.") is False


@pytest.mark.asyncio
async def test_tts_german_voice_routing(db: Database, tmp_path: Path) -> None:
    bus = EventBus()
    default_syn = FakeSynthesizer()
    german_syn = FakeSynthesizer()
    fake_player = FakePlayer()

    temp_dir = tmp_path / "shm"
    tts_mod = TtsPiperModule(
        bus=bus,
        db=db,
        config={"allow_tts": True, "global_pause": False, "dnd_mode": False},
        synthesizer=default_syn,
        player=fake_player,
        temp_dir=temp_dir,
    )
    # Inject fake German synthesizer
    tts_mod._german_synthesizer = german_syn

    await tts_mod.start()
    try:
        # 1. English request -> default synthesizer
        await bus.publish(
            PlatformEvent(
                type="tts_requested",
                source="test",
                payload={"text": "Your next meeting is in ten minutes.", "language": "en"},
            )
        )
        assert len(default_syn.synthesize_calls) == 1
        assert len(german_syn.synthesize_calls) == 0

        # 2. Explicit German request via language payload -> german synthesizer
        await bus.publish(
            PlatformEvent(
                type="tts_requested",
                source="test",
                payload={"text": "Dein nächster Termin ist Training um 19 Uhr.", "language": "de"},
            )
        )
        assert len(default_syn.synthesize_calls) == 1
        assert len(german_syn.synthesize_calls) == 1
        assert "Training" in german_syn.synthesize_calls[0][0]

        # 3. Implicit German text without explicit language payload -> routed to german synthesizer
        await bus.publish(
            PlatformEvent(
                type="tts_requested",
                source="test",
                payload={"text": "Du sitzt seit 30 Minuten. Zeit für eine Pause."},
            )
        )
        assert len(default_syn.synthesize_calls) == 1
        assert len(german_syn.synthesize_calls) == 2
    finally:
        await tts_mod.stop()


def test_clean_text_for_speech_german_natural_times_and_markdown() -> None:
    from jarvis.modules.outputs.tts_piper import clean_text_for_speech

    # 1. Full hours: 07:00 -> 7 Uhr (prevents reading "null sieben null null")
    assert clean_text_for_speech("Um 07:00 Uhr aufstehen.", is_german=True) == "Um 7 Uhr aufstehen."
    assert clean_text_for_speech("Termin um 08:00.", is_german=True) == "Termin um 8 Uhr."

    # 2. Time ranges: 07:00 – 07:15 -> von 7 Uhr bis 7 Uhr 15
    res_range = clean_text_for_speech("Aufstehen (07:00 – 07:15)", is_german=True)
    assert "von 7 Uhr bis 7 Uhr 15" in res_range

    # 3. Minutes with Uhr: 7:15 Uhr -> 7 Uhr 15 (prevents piper repeating 'uhr' twice)
    assert clean_text_for_speech("Treffen um 7:15 Uhr.", is_german=True) == "Treffen um 7 Uhr 15."

    # 4. Robotic markdown list removal:
    robotic_list = "1. **Aufstehen** (07:00 – 07:15) – Coding\n2. **Augenarzt** (08:30 – 09:30)"
    cleaned = clean_text_for_speech(robotic_list, is_german=True)
    assert "1." not in cleaned
    assert "2." not in cleaned
    assert "**" not in cleaned
    assert "07:00" not in cleaned
    assert "von 7 Uhr bis 7 Uhr 15" in cleaned
    assert "von 8 Uhr 30 bis 9 Uhr 30" in cleaned


def test_clean_text_for_speech_english_natural_times() -> None:
    from jarvis.modules.outputs.tts_piper import clean_text_for_speech

    # Full hours: 07:00 -> 7 o'clock (prevents reading "zero seven zero zero")
    assert clean_text_for_speech("Meeting at 07:00.", is_german=False) == "Meeting at 7 o'clock."
    assert clean_text_for_speech("Starts at 08:00 AM.", is_german=False) == "Starts at 8 AM."

    # Time ranges: 07:00 – 08:00 -> from 7 to 8 o'clock
    res_range = clean_text_for_speech("Doctor from 08:00 – 09:00", is_german=False)
    assert "from 8 to 9 o'clock" in res_range

    # Minutes: 07:15 -> 7:15 (removes leading zero)
    assert clean_text_for_speech("Alarm at 07:15.", is_german=False) == "Alarm at 7:15."

