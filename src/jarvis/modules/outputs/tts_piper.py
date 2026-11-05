"""Piper TTS output module playing via PipeWire (pw-play) with strict privacy gating."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
import wave
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, now_local
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_VOICE_MODEL_PATH = Path("~/.local/share/posture-platform/voices/en_US-lessac-medium.onnx")
DEFAULT_GERMAN_VOICE_MODEL_PATH = Path(
    "~/.local/share/posture-platform/voices/de_DE-lila_hokuspokus-high.onnx"
)
DEFAULT_PLAYER_BINARY = "pw-play"
DEFAULT_TEMP_DIR = Path("/dev/shm")


def is_german_text(text: str) -> bool:
    """Fast heuristic to detect German language text for TTS synthesis."""
    if any(ch in text for ch in "äöüÄÖÜß"):
        return True
    de_words = {
        "und", "der", "die", "das", "ist", "nicht", "ein", "eine", "einer", "einem",
        "einen", "im", "in", "am", "an", "von", "zu", "mit", "für", "auf", "dem",
        "den", "des", "termin", "termine", "uhr", "heute", "morgen", "keine", "hast",
        "habe", "bist", "dein", "deine", "deinen", "deinem", "deiner", "mein", "meine",
        "jetzt", "zeit", "pause", "arbeiten", "arbeit", "minuten", "stunden", "hallo",
        "stehen", "woche", "tag", "tage", "wichtig", "kalender"
    }
    words = {w.strip(".,!?:;\"'()[]{}").lower() for w in text.split()}
    return len(words & de_words) >= 2


class TtsError(Exception):
    """Base exception for TTS operations."""


class TtsDependencyMissingError(TtsError):
    """Raised when neither piper Python package nor piper CLI binary is found."""


class TtsModelNotFoundError(TtsError):
    """Raised when the specified Piper ONNX model does not exist on disk."""


class AudioPlaybackError(TtsError):
    """Raised when pw-play audio playback fails."""


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Protocol isolating Piper text-to-speech synthesis."""

    async def synthesize(self, text: str, output_wav: Path) -> Path:
        """Synthesize text to output_wav."""
        ...

    def is_available(self) -> tuple[bool, str | None]:
        """Return (available, error_reason_if_not)."""
        ...


@runtime_checkable
class AudioPlayer(Protocol):
    """Protocol isolating audio playback."""

    async def play(self, audio_path: Path) -> None:
        """Play WAV file."""
        ...

    def is_available(self) -> tuple[bool, str | None]:
        """Return (available, error_reason_if_not)."""
        ...


class PiperPythonSynthesizer:
    """SpeechSynthesizer using the piper-tts Python package."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path | None = None,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
    ) -> None:
        self.model_path = model_path.expanduser()
        self.config_path = config_path.expanduser() if config_path else None
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self._voice: Any = None

    def _load_voice(self) -> Any:
        if self._voice is not None:
            return self._voice

        if not self.model_path.exists():
            raise TtsModelNotFoundError(
                f"Piper voice model not found at {self.model_path}. "
                "Download the ONNX voice model or configure tts.voice_model_path."
            )

        piper_voice_cls = None
        try:
            import piper

            if hasattr(piper, "PiperVoice"):
                piper_voice_cls = piper.PiperVoice
        except ImportError:
            pass

        if piper_voice_cls is None:
            try:
                from piper.voice import PiperVoice  # type: ignore[import-not-found]

                piper_voice_cls = PiperVoice
            except ImportError:
                pass

        if piper_voice_cls is None:
            raise TtsDependencyMissingError(
                "piper-tts Python library is not installed. Install with 'uv sync --group voice'."
            )

        try:
            cfg_path_str = str(self.config_path) if self.config_path else None
            self._voice = piper_voice_cls.load(str(self.model_path), config_path=cfg_path_str)
            return self._voice
        except Exception as exc:
            raise TtsError(f"Failed to load Piper voice model: {exc}") from exc

    def is_available(self) -> tuple[bool, str | None]:
        if not self.model_path.exists():
            return False, f"Piper voice model not found at {self.model_path}"
        try:
            import piper  # noqa: F401

            return True, None
        except ImportError:
            return False, "piper-tts Python package is not installed"

    async def synthesize(self, text: str, output_wav: Path) -> Path:
        voice = self._load_voice()
        loop = asyncio.get_running_loop()

        def _sync_synthesize() -> None:
            with wave.open(str(output_wav), "wb") as wav_file:
                # piper-tts >= 1.3: `synthesize_wav` sets the WAV format itself, while
                # `synthesize` now returns audio chunks instead of writing a WAV file.
                if hasattr(voice, "synthesize_wav"):
                    syn_config: Any = None
                    if self.speaker_id is not None or self.length_scale != 1.0:
                        from piper.config import (  # type: ignore[import-not-found]
                            SynthesisConfig,
                        )

                        syn_config = SynthesisConfig(
                            speaker_id=self.speaker_id,
                            length_scale=self.length_scale,
                        )
                    voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                    return

                # Legacy piper-tts < 1.3 API: synthesize(text, wav_file, **kwargs)
                kwargs: dict[str, Any] = {}
                if self.speaker_id is not None:
                    kwargs["speaker_id"] = self.speaker_id
                if self.length_scale != 1.0:
                    kwargs["length_scale"] = self.length_scale
                try:
                    voice.synthesize(text, wav_file, **kwargs)
                except TypeError:
                    voice.synthesize(text, wav_file)

        await loop.run_in_executor(None, _sync_synthesize)
        return output_wav


class PiperCliSynthesizer:
    """SpeechSynthesizer using an external piper CLI binary."""

    def __init__(
        self,
        model_path: Path,
        binary_path: str = "piper",
        config_path: Path | None = None,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
    ) -> None:
        self.model_path = model_path.expanduser()
        self.binary_path = binary_path
        self.config_path = config_path.expanduser() if config_path else None
        self.speaker_id = speaker_id
        self.length_scale = length_scale

    def is_available(self) -> tuple[bool, str | None]:
        bin_resolved = shutil.which(self.binary_path)
        if not bin_resolved:
            return False, f"Piper CLI binary '{self.binary_path}' not found on PATH"
        if not self.model_path.exists():
            return False, f"Piper voice model not found at {self.model_path}"
        return True, None

    async def synthesize(self, text: str, output_wav: Path) -> Path:
        available, reason = self.is_available()
        if not available:
            raise TtsError(reason)

        bin_resolved = shutil.which(self.binary_path) or self.binary_path
        cmd = [
            bin_resolved,
            "--model",
            str(self.model_path),
            "--output_file",
            str(output_wav),
        ]
        if self.config_path:
            cmd.extend(["--config", str(self.config_path)])
        if self.speaker_id is not None:
            cmd.extend(["--speaker", str(self.speaker_id)])
        if self.length_scale != 1.0:
            cmd.extend(["--length-scale", str(self.length_scale)])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(input=text.encode("utf-8"))
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise TtsError(f"Piper CLI failed (exit {proc.returncode}): {err_msg}")
        return output_wav


class AutoPiperSynthesizer:
    """Delegating synthesizer that prefers Python package and falls back to CLI binary."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path | None = None,
        piper_binary: str | None = None,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.piper_binary = piper_binary
        self.speaker_id = speaker_id
        self.length_scale = length_scale

        self._py_syn = PiperPythonSynthesizer(
            model_path=model_path,
            config_path=config_path,
            speaker_id=speaker_id,
            length_scale=length_scale,
        )
        cli_bin = piper_binary or "piper"
        self._cli_syn = PiperCliSynthesizer(
            model_path=model_path,
            binary_path=cli_bin,
            config_path=config_path,
            speaker_id=speaker_id,
            length_scale=length_scale,
        )

    def is_available(self) -> tuple[bool, str | None]:
        if not self.model_path.expanduser().exists():
            return False, (
                f"Piper voice model not found at {self.model_path.expanduser()}. "
                "Download the ONNX voice model or configure tts.voice_model_path."
            )
        py_ok, _ = self._py_syn.is_available()
        if py_ok:
            return True, None
        cli_ok, _ = self._cli_syn.is_available()
        if cli_ok:
            return True, None
        return False, (
            "Neither piper-tts Python package nor piper CLI binary found. "
            "Install piper-tts with 'uv sync --group voice' or install piper CLI binary."
        )

    async def synthesize(self, text: str, output_wav: Path) -> Path:
        py_ok, _ = self._py_syn.is_available()
        if py_ok:
            return await self._py_syn.synthesize(text, output_wav)
        cli_ok, _ = self._cli_syn.is_available()
        if cli_ok:
            return await self._cli_syn.synthesize(text, output_wav)
        available, reason = self.is_available()
        raise TtsDependencyMissingError(reason or "No Piper synthesizer available")


class PipeWirePlayer:
    """AudioPlayer using PipeWire's pw-play binary."""

    def __init__(self, binary: str = DEFAULT_PLAYER_BINARY) -> None:
        self.binary = binary

    def is_available(self) -> tuple[bool, str | None]:
        bin_resolved = shutil.which(self.binary)
        if not bin_resolved:
            return False, f"PipeWire player '{self.binary}' not found on PATH"
        return True, None

    async def play(self, audio_path: Path) -> None:
        bin_resolved = shutil.which(self.binary) or self.binary
        try:
            proc = await asyncio.create_subprocess_exec(
                bin_resolved,
                str(audio_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                raise AudioPlaybackError(f"pw-play failed with code {proc.returncode}: {err_msg}")
        except FileNotFoundError as exc:
            raise AudioPlaybackError(f"Audio player '{self.binary}' not found") from exc
        except Exception as exc:
            raise AudioPlaybackError(f"Audio playback error: {exc}") from exc


class TtsPiperModule(BaseModule):
    """Output module synthesizing speech via Piper and playing via PipeWire."""

    name: str = "tts_piper"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any] | None = None,
        *,
        synthesizer: SpeechSynthesizer | None = None,
        player: AudioPlayer | None = None,
        temp_dir: Path | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=dict(config) if config else {})
        tts_cfg = self.config.get("tts", {})
        if not isinstance(tts_cfg, dict):
            tts_cfg = {}
        # Merge section-specific config with top-level config
        model_path_raw = (
            self.config.get("voice_model_path")
            or tts_cfg.get("voice_model_path")
            or DEFAULT_VOICE_MODEL_PATH
        )
        self._model_path = Path(str(model_path_raw)).expanduser()

        cfg_path_raw = self.config.get("config_path") or tts_cfg.get("config_path")
        self._config_path = Path(str(cfg_path_raw)).expanduser() if cfg_path_raw else None

        piper_bin = self.config.get("piper_binary") or tts_cfg.get("piper_binary")
        player_bin = (
            self.config.get("player_binary")
            or tts_cfg.get("player_binary")
            or DEFAULT_PLAYER_BINARY
        )
        speaker_id = self.config.get("speaker_id") or tts_cfg.get("speaker_id")
        length_scale = float(self.config.get("length_scale") or tts_cfg.get("length_scale") or 1.0)

        temp_dir_raw = (
            temp_dir or self.config.get("temp_dir") or tts_cfg.get("temp_dir") or DEFAULT_TEMP_DIR
        )
        self._temp_dir = Path(str(temp_dir_raw)).expanduser()

        self._synthesizer = synthesizer or AutoPiperSynthesizer(
            model_path=self._model_path,
            config_path=self._config_path,
            piper_binary=piper_bin,
            speaker_id=speaker_id,
            length_scale=length_scale,
        )

        german_model_raw = (
            self.config.get("german_voice_model_path")
            or tts_cfg.get("german_voice_model_path")
            or DEFAULT_GERMAN_VOICE_MODEL_PATH
        )
        self._german_model_path = Path(str(german_model_raw)).expanduser()
        if self._german_model_path.exists():
            self._german_synthesizer: SpeechSynthesizer | None = AutoPiperSynthesizer(
                model_path=self._german_model_path,
                config_path=None,
                piper_binary=piper_bin,
                speaker_id=None,
                length_scale=length_scale,
            )
        else:
            self._german_synthesizer = None

        self._player = player or PipeWirePlayer(binary=player_bin)

        privacy_cfg = self.config.get("privacy", {})
        if not isinstance(privacy_cfg, dict):
            privacy_cfg = {}

        self._allow_tts: bool = bool(
            self.config.get("allow_tts", privacy_cfg.get("allow_tts", False))
        )
        self._global_pause: bool = bool(
            self.config.get("global_pause", privacy_cfg.get("global_pause", False))
        )
        self._dnd_mode: bool = bool(self.config.get("dnd_mode", privacy_cfg.get("dnd_mode", False)))

        self._running: bool = False
        self._last_error: str | None = None

    async def start(self) -> None:
        """Subscribe to events, sync persisted privacy settings, and report health."""
        if hasattr(self.db, "get_setting"):
            try:
                persisted_allow = await self.db.get_setting("privacy.allow_tts")
                if isinstance(persisted_allow, bool):
                    self._allow_tts = persisted_allow
                persisted_pause = await self.db.get_setting("privacy.global_pause")
                if isinstance(persisted_pause, bool):
                    self._global_pause = persisted_pause
                persisted_dnd = await self.db.get_setting("privacy.dnd_mode")
                if isinstance(persisted_dnd, bool):
                    self._dnd_mode = persisted_dnd
            except Exception as exc:
                logger.debug("Could not read privacy settings from DB: %s", exc)

        self.bus.subscribe("tts_requested", self._on_tts_requested)
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)

        self._running = True

        syn_ok, syn_err = self._synthesizer.is_available()
        play_ok, play_err = self._player.is_available()
        if not syn_ok or not play_ok:
            self._last_error = syn_err or play_err or "TTS unavailable"
            logger.warning("tts_piper starting in degraded mode: %s", self._last_error)
            await self._record_health(status="degraded", last_error=self._last_error)
        else:
            await self._record_health(status="ok", last_error=None)

    async def stop(self) -> None:
        """Unsubscribe cleanly from EventBus topics."""
        self._running = False
        self.bus.unsubscribe("tts_requested", self._on_tts_requested)
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)

    async def health(self) -> dict[str, Any]:
        syn_ok, syn_err = self._synthesizer.is_available()
        play_ok, play_err = self._player.is_available()
        if not syn_ok or not play_ok:
            err = syn_err or play_err or "TTS unavailable"
            return {
                "module": self.name,
                "status": "degraded",
                "last_error": err,
            }
        return {
            "module": self.name,
            "status": "ok" if self._last_error is None else "degraded",
            "last_error": self._last_error,
        }

    async def _on_privacy_mode_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        if "allow_tts" in payload:
            self._allow_tts = bool(payload["allow_tts"])
        if "privacy_paused" in payload or "global_pause" in payload:
            self._global_pause = bool(
                payload.get("privacy_paused", payload.get("global_pause", False))
            )
        if "dnd_mode" in payload:
            self._dnd_mode = bool(payload["dnd_mode"])
        logger.debug(
            "tts_piper privacy updated: allow_tts=%s, global_pause=%s, dnd=%s",
            self._allow_tts,
            self._global_pause,
            self._dnd_mode,
        )

    async def _on_tts_requested(self, event: PlatformEvent) -> None:
        # Check suppression rules:
        # Suppress speech whenever allow_tts is false, global_pause is true, or dnd_mode is true
        if not self._allow_tts:
            logger.info("Speech suppressed: privacy.allow_tts is false")
            return
        if self._global_pause:
            logger.info("Speech suppressed: privacy.global_pause is active")
            return
        if self._dnd_mode:
            logger.info("Speech suppressed: privacy.dnd_mode is active")
            return

        text = str(event.payload.get("text", "")).strip()
        if not text:
            return

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        temp_wav = self._temp_dir / f"jarvis_tts_{uuid.uuid4().hex[:8]}.wav"

        lang = str(event.payload.get("language") or "").strip().lower()
        use_german = (lang.startswith("de")) or (not lang and is_german_text(text))

        synthesizer = (
            self._german_synthesizer
            if use_german and self._german_synthesizer is not None
            else self._synthesizer
        )

        try:
            await synthesizer.synthesize(text, temp_wav)
            await self._player.play(temp_wav)
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("TTS failed: %s", exc)
            await self._record_health(status="degraded", last_error=self._last_error)
        finally:
            if temp_wav.exists():
                try:
                    temp_wav.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _record_health(self, status: str, last_error: str | None) -> None:
        if hasattr(self.db, "upsert_module_health"):
            try:
                await self.db.upsert_module_health(
                    module=self.name,
                    status=status,
                    last_error=last_error,
                )
            except Exception as exc:
                logger.debug("Failed to record health in DB: %s", exc)

        await self.bus.publish(
            PlatformEvent(
                type="module_health_changed",
                source=self.name,
                timestamp=now_local(),
                payload={
                    "module": self.name,
                    "category": "outputs",
                    "status": status,
                    "last_error": last_error,
                },
            )
        )
