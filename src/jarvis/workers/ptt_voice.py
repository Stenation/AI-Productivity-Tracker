"""Push-to-talk voice worker with local-only recording, Whisper STT, and /api/ask round-trip."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jarvis.core.settings import Settings, load_settings

logger = logging.getLogger(__name__)

DEFAULT_PTT_STATE_FILE = Path("/dev/shm/jarvis_ptt.json")
DEFAULT_PTT_AUDIO_FILE = Path("/dev/shm/jarvis_ptt.wav")
DEFAULT_MAX_RECORDING_DURATION_SECONDS = 30.0
DEFAULT_RECORDER_BINARY = "pw-record"
DEFAULT_RECORDER_ARGS = ("--rate", "16000", "--channels", "1")
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_API_URL = "http://127.0.0.1:8000"


class VoiceError(Exception):
    """Base exception for voice worker operations."""


class VoiceDependencyMissingError(VoiceError):
    """Raised when faster-whisper is not installed."""


class RecorderError(VoiceError):
    """Raised when PipeWire or recorder fails."""


class TranscriptionError(VoiceError):
    """Raised when transcription fails."""


class PrivacyAbortedError(VoiceError):
    """Raised when voice recording is disallowed by privacy settings."""


@runtime_checkable
class AudioRecorder(Protocol):
    """Protocol for starting and stopping push-to-talk audio recording."""

    def start_recording(self, output_path: Path, max_duration_seconds: float) -> int:
        """Start recording to output_path and return process PID."""
        ...

    def stop_recording(self, pid: int) -> None:
        """Terminate the recording process."""
        ...

    def is_recording(self, pid: int) -> bool:
        """Check if process pid is actively running."""
        ...


@runtime_checkable
class AudioTranscriber(Protocol):
    """Protocol for local speech-to-text transcription."""

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file locally and return text."""
        ...


@runtime_checkable
class AskApiClient(Protocol):
    """Protocol for interacting with Jarvis HTTP API."""

    def ask(
        self,
        question: str,
        source: str = "voice",
        speak: bool = True,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Send transcript to POST /api/ask."""
        ...

    def get_state(self) -> dict[str, Any] | None:
        """Get system state from GET /api/state."""
        ...

    def report_health(self, status: str, last_error: str | None = None) -> None:
        """Report worker health to POST /api/events."""
        ...


class PipeWireRecorder:
    """Default AudioRecorder using PipeWire's pw-record command."""

    def __init__(
        self,
        binary: str = DEFAULT_RECORDER_BINARY,
        args: Sequence[str] = DEFAULT_RECORDER_ARGS,
        watchdog_enabled: bool = True,
        state_file: Path | None = None,
    ) -> None:
        self.binary = binary
        self.args = list(args)
        self.watchdog_enabled = watchdog_enabled
        self.state_file = state_file or DEFAULT_PTT_STATE_FILE

    def start_recording(self, output_path: Path, max_duration_seconds: float) -> int:
        cmd = [self.binary, *self.args, str(output_path)]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RecorderError(
                f"Audio recorder binary '{self.binary}' not found on PATH. "
                "Ensure PipeWire (pw-record) is installed."
            ) from exc
        except Exception as exc:
            raise RecorderError(f"Failed to start recorder '{self.binary}': {exc}") from exc

        pid = proc.pid
        if self.watchdog_enabled:
            self._spawn_watchdog(pid, max_duration_seconds, output_path)

        return pid

    def _spawn_watchdog(self, pid: int, max_duration_seconds: float, output_path: Path) -> None:
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "jarvis.workers.ptt_voice",
                    "--watchdog",
                    str(pid),
                    str(max_duration_seconds),
                    str(output_path),
                    "--state-file",
                    str(self.state_file),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            logger.warning("Could not launch PTT watchdog process: %s", exc)

    def stop_recording(self, pid: int) -> None:
        if not self.is_recording(pid):
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        deadline = time.time() + 1.5
        while time.time() < deadline:
            if not self.is_recording(pid):
                return
            time.sleep(0.05)

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def is_recording(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


class FasterWhisperTranscriber:
    """Default AudioTranscriber using faster-whisper WhisperModel."""

    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        download_root: Path | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model: Any = None
        self.last_language: str | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise VoiceDependencyMissingError(
                "faster-whisper is not installed. Install with 'uv sync --group voice'."
            ) from exc
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root) if self.download_root else None,
            )
            return self._model
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load Whisper model '{self.model_size}': {exc}"
            ) from exc

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file does not exist: {audio_path}")
        model = self._get_model()
        try:
            segments, info = model.transcribe(str(audio_path))
            self.last_language = getattr(info, "language", None)
            text = " ".join(s.text.strip() for s in segments if s.text).strip()
            return text
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc


class HttpAskClient:
    """HTTP client communicating with Jarvis REST API."""

    def __init__(self, api_url: str = DEFAULT_API_URL, timeout_seconds: float = 30.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_state(self) -> dict[str, Any] | None:
        url = f"{self.api_url}/api/state"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("Failed to query %s: %s", url, exc)
            return None
        return None

    def ask(
        self,
        question: str,
        source: str = "voice",
        speak: bool = True,
        language: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_url}/api/ask"
        req_data: dict[str, Any] = {"question": question, "source": source, "speak": speak}
        if language:
            req_data["language"] = language
        payload = json.dumps(req_data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise VoiceError(f"POST /api/ask failed with HTTP {exc.code}: {err_body}") from exc
        except Exception as exc:
            raise VoiceError(f"Failed to send request to {url}: {exc}") from exc

    def report_health(self, status: str, last_error: str | None = None) -> None:
        url = f"{self.api_url}/api/events"
        event = {
            "type": "module_health_changed",
            "source": "ptt_voice",
            "payload": {
                "module": "ptt_voice",
                "category": "inputs",
                "status": status,
                "last_error": last_error,
            },
        }
        try:
            payload = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0):
                pass
        except Exception:
            pass


class PttVoiceWorker:
    """Push-to-talk toggle manager and lifecycle orchestrator."""

    def __init__(
        self,
        settings: Settings | None = None,
        recorder: AudioRecorder | None = None,
        transcriber: AudioTranscriber | None = None,
        ask_client: AskApiClient | None = None,
        state_file: Path | None = None,
        audio_file: Path | None = None,
        max_duration_seconds: float | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        voice_cfg = self.settings.voice

        self.state_file = state_file or voice_cfg.ptt_state_file
        self.audio_file = audio_file or voice_cfg.ptt_audio_file
        self.max_duration_seconds = (
            max_duration_seconds
            if max_duration_seconds is not None
            else voice_cfg.max_duration_seconds
        )

        self.recorder = recorder or PipeWireRecorder(
            binary=voice_cfg.recorder_binary,
            state_file=self.state_file,
        )
        self.transcriber = transcriber or FasterWhisperTranscriber(
            model_size=voice_cfg.model_size,
            device=voice_cfg.device,
            compute_type=voice_cfg.compute_type,
            download_root=voice_cfg.model_path,
        )
        self.ask_client = ask_client or HttpAskClient(
            api_url=voice_cfg.api_url,
        )

    def is_recording(self) -> bool:
        if not self.state_file.exists():
            return False
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            pid = int(state.get("pid", 0))
            return self.recorder.is_recording(pid)
        except Exception:
            return False

    def check_privacy(self) -> tuple[bool, str | None]:
        """Check if push-to-talk is allowed by privacy policy.

        Returns (allowed, reason_if_not).
        """
        api_state = self.ask_client.get_state()
        if api_state is not None:
            if api_state.get("privacy_paused") or api_state.get("global_pause"):
                return False, "Privacy global pause is active"
            priv = api_state.get("privacy", {})
            if priv.get("microphone_mode") == "disabled":
                return False, "Microphone is disabled in privacy settings"
            if priv.get("global_pause"):
                return False, "Privacy global pause is active"

        if self.settings.privacy.global_pause:
            return False, "Privacy global pause is active"
        if self.settings.privacy.microphone_mode == "disabled":
            return False, "Microphone is disabled in privacy settings"

        return True, None

    def toggle(self) -> dict[str, Any]:
        """Toggle push-to-talk: start recording on first call, stop/transcribe/ask on second."""
        allowed, reason = self.check_privacy()
        if not allowed:
            logger.warning("PTT voice aborted: %s", reason)
            self._cleanup_all()
            return {"status": "aborted", "reason": reason}

        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                pid = int(state.get("pid", 0))
            except Exception:
                pid = 0

            if pid > 0 and self.recorder.is_recording(pid):
                return self._stop_and_process(pid)
            else:
                self._cleanup_all()

        return self._start_recording()

    def _start_recording(self) -> dict[str, Any]:
        self._ensure_parent_dirs()
        self._cleanup_all()

        try:
            pid = self.recorder.start_recording(self.audio_file, self.max_duration_seconds)
            state_data = {
                "pid": pid,
                "started_at": time.time(),
                "audio_path": str(self.audio_file),
                "max_duration_seconds": self.max_duration_seconds,
            }
            tmp_state = self.state_file.with_suffix(".tmp")
            tmp_state.write_text(json.dumps(state_data), encoding="utf-8")
            tmp_state.replace(self.state_file)
            return {"status": "recording_started", "pid": pid}
        except Exception as exc:
            logger.exception("Failed to start voice recording: %s", exc)
            self._cleanup_all()
            self.ask_client.report_health("degraded", str(exc))
            return {"status": "error", "error": str(exc)}

    def _stop_and_process(self, pid: int) -> dict[str, Any]:
        try:
            self.recorder.stop_recording(pid)
        except Exception as exc:
            logger.warning("Error stopping recording PID %s: %s", pid, exc)

        transcript = ""
        error = None

        try:
            if not self.audio_file.exists() or self.audio_file.stat().st_size == 0:
                logger.info("Audio file empty or missing; skipping transcription")
                return {"status": "stopped", "transcript": ""}

            transcript = self.transcriber.transcribe(self.audio_file)
            detected_lang = getattr(self.transcriber, "last_language", None)
            if transcript:
                logger.info("Transcribed voice request (lang=%s): %s", detected_lang, transcript)
                response = self.ask_client.ask(
                    transcript,
                    source="voice",
                    speak=True,
                    language=detected_lang,
                )
                return {
                    "status": "asked",
                    "transcript": transcript,
                    "language": detected_lang,
                    "response": response,
                }
            return {"status": "stopped", "transcript": ""}
        except (VoiceDependencyMissingError, RecorderError, TranscriptionError) as exc:
            error = str(exc)
            logger.error("Voice processing error: %s", exc)
            self.ask_client.report_health("degraded", error)
            return {"status": "degraded", "error": error}
        except Exception as exc:
            error = str(exc)
            logger.exception("Unexpected error in PTT voice: %s", exc)
            self.ask_client.report_health("degraded", error)
            return {"status": "error", "error": error}
        finally:
            self._cleanup_all()

    def _ensure_parent_dirs(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.audio_file.parent.mkdir(parents=True, exist_ok=True)

    def _cleanup_all(self) -> None:
        try:
            if self.audio_file.exists():
                self.audio_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if self.state_file.exists():
                self.state_file.unlink(missing_ok=True)
        except Exception:
            pass


def run_watchdog(
    pid: int,
    timeout_seconds: float,
    audio_path_str: str,
    state_file_str: str | None = None,
) -> None:
    """Enforce hard maximum recording duration, cleaning up audio and state files on timeout."""
    time.sleep(timeout_seconds)
    audio_path = Path(audio_path_str)
    state_file = Path(state_file_str) if state_file_str else DEFAULT_PTT_STATE_FILE

    try:
        os.kill(pid, 0)
        logger.warning(
            "Push-to-talk recording exceeded %ss cap; terminating PID %s",
            timeout_seconds,
            pid,
        )
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.1)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    except OSError:
        pass

    try:
        if state_file.exists():
            try:
                state_data = json.loads(state_file.read_text(encoding="utf-8"))
                if int(state_data.get("pid", 0)) == pid:
                    state_file.unlink(missing_ok=True)
            except Exception:
                state_file.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
    except Exception:
        pass


def _send_desktop_notification(
    summary: str,
    body: str,
    urgency: str = "normal",
    expire_ms: int = 3000,
) -> None:
    """Best-effort notification to notify-send for desktop feedback."""
    try:
        cmd = [
            "notify-send",
            "-a",
            "Jarvis Voice",
            "-u",
            urgency,
            "-t",
            str(expire_ms),
            summary,
            body,
        ]
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def main() -> None:
    """CLI entry point for jarvis-ptt command."""
    parser = argparse.ArgumentParser(description="Jarvis Push-to-Talk Voice Toggle")
    parser.add_argument(
        "--watchdog",
        nargs=3,
        metavar=("PID", "TIMEOUT", "AUDIO_PATH"),
        help="Run timeout watchdog for PID",
    )
    parser.add_argument(
        "--state-file",
        help="Optional state file path",
    )
    args = parser.parse_args()

    if args.watchdog:
        pid = int(args.watchdog[0])
        timeout = float(args.watchdog[1])
        audio_path = args.watchdog[2]
        run_watchdog(pid, timeout, audio_path, state_file_str=args.state_file)
        sys.exit(0)

    worker = PttVoiceWorker()
    result = worker.toggle()
    status = result.get("status")
    if status == "aborted":
        reason = result.get("reason", "Unknown reason")
        _send_desktop_notification(
            "Jarvis Voice", f"Aborted: {reason}", urgency="critical", expire_ms=4000
        )
        print(f"Push-to-talk aborted: {reason}", file=sys.stderr)
        sys.exit(1)
    elif status == "recording_started":
        _send_desktop_notification(
            "Jarvis Voice", "Listening... (Press Super+I again when done)", expire_ms=3500
        )
        print(f"PTT: recording started (PID {result.get('pid')})")
    elif status == "asked":
        transcript = result.get("transcript", "")
        _send_desktop_notification(
            "Jarvis Voice", f'Processing: "{transcript}"', expire_ms=3000
        )
        print(f"PTT: {transcript}")
    elif status in ("error", "degraded"):
        err = result.get("error", "Unknown error")
        _send_desktop_notification(
            "Jarvis Voice", f"Error: {err}", urgency="critical", expire_ms=5000
        )
        print(f"PTT {status}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
