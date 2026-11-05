"""Local Ollama LLM assistant output module (FR-23, SPEC §5, ARCHITECTURE §9)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.llm import (
    LlmConnectionError,
    LlmError,
    LlmTimeoutError,
    LlmValidationError,
    OllamaClient,
    build_assistant_context,
)
from jarvis.core.schema import PlatformEvent, now_local
from jarvis.core.settings import Settings
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)


class LlmOllamaModule(BaseModule):
    """Local Ollama assistant module responding to questions with compact platform context."""

    name: str = "llm_ollama"
    _dependency_override: bool | None = None

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any],
        client: OllamaClient | None = None,
    ) -> None:
        super().__init__(bus, db, config)
        self._status: str = "disabled"
        self._last_error: str | None = None
        self._running: bool = False

        llm_cfg = config.get("llm", config)
        base_url = str(llm_cfg.get("base_url") or "http://127.0.0.1:11434")
        text_model = str(llm_cfg.get("text_model") or "qwen3:14b")
        keep_alive = str(llm_cfg.get("keep_alive") or "5m")
        timeout_seconds = float(llm_cfg.get("timeout_seconds") or 30.0)

        self._client: OllamaClient | None = client or config.get("client")
        if self._client is None:
            self.client = OllamaClient(
                base_url=base_url,
                model=text_model,
                keep_alive=keep_alive,
                timeout_seconds=timeout_seconds,
            )
        else:
            self.client = self._client

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the optional 'llm' dependency group ('ollama') is installed."""
        if cls._dependency_override is not None:
            return cls._dependency_override
        return importlib.util.find_spec("ollama") is not None

    async def start(self) -> None:
        """Start the LLM module, ensuring dependencies are met."""
        if not self.is_available() and self._client is None:
            err_msg = (
                "Optional dependency group 'llm' is not installed. "
                "Run 'uv sync --group llm' to install 'ollama'."
            )
            self._status = "degraded"
            self._last_error = err_msg
            await self.db.upsert_module_health(
                module=self.name,
                status="degraded",
                last_error=err_msg,
            )
            raise RuntimeError(err_msg)

        self._running = True
        self._status = "ok"
        self._last_error = None
        await self.db.upsert_module_health(
            module=self.name,
            status="ok",
            last_error=None,
        )

    async def stop(self) -> None:
        """Stop the LLM module and close client resources."""
        self._running = False
        self._status = "disabled"
        if self.client is not None:
            await self.client.close()
        await self.db.upsert_module_health(
            module=self.name,
            status="disabled",
            last_error=None,
        )

    async def health(self) -> dict[str, Any]:
        """Return current module health dict."""
        return {
            "module": self.name,
            "status": self._status,
            "last_error": self._last_error,
        }

    async def ask(
        self,
        question: str,
        source: str = "dashboard",
        speak: bool = False,
        *,
        language: str | None = None,
        context: dict[str, Any] | None = None,
        resolver: Any = None,
        timer_manager: Any = None,
        settings: Settings | None = None,
        registry: Any = None,
    ) -> dict[str, Any]:
        """Ask a question, persist successful interaction, emit events, and return result."""
        if not self._running:
            raise LlmError("Module 'llm_ollama' is disabled. Enable llm_ollama in Settings.")

        # 1. Build compact snapshot context if not provided
        effective_context = context
        if effective_context is None:
            effective_context = await build_assistant_context(
                db=self.db,
                resolver=resolver,
                timer_manager=timer_manager,
                settings=settings,
                registry=registry,
            )

        # 2. Invoke local LLM with error isolation
        try:
            res = await self.client.ask(
                question=question,
                context=effective_context,
                language=language,
            )
        except (LlmConnectionError, LlmTimeoutError, LlmValidationError, LlmError) as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            self._status = "degraded"
            self._last_error = err_str
            await self.db.upsert_module_health(
                module=self.name,
                status="degraded",
                last_error=err_str,
            )
            await self.bus.publish(
                PlatformEvent(
                    type="module_health_changed",
                    source=self.name,
                    timestamp=now_local(),
                    payload={
                        "module": self.name,
                        "category": "outputs",
                        "enabled": True,
                        "status": "degraded",
                        "last_error": err_str,
                    },
                )
            )
            raise
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            self._status = "degraded"
            self._last_error = err_str
            await self.db.upsert_module_health(
                module=self.name,
                status="degraded",
                last_error=err_str,
            )
            raise LlmError(f"Unexpected error in LLM ask: {exc}") from exc

        # 3. Mark healthy on successful response
        self._status = "ok"
        self._last_error = None
        await self.db.upsert_module_health(
            module=self.name,
            status="ok",
            last_error=None,
        )

        # 4. Persist interaction in llm_interactions
        interaction = await self.db.record_llm_interaction(
            source=source,
            question=question,
            answer=res["answer"],
            suggested_action=res.get("suggested_action"),
            confidence=res["confidence"],
            model=res["model"],
            latency_ms=res.get("latency_ms"),
            metadata={"context": effective_context},
        )
        interaction_id = interaction["id"]

        # 5. Emit assistant_answered
        await self.bus.publish(
            PlatformEvent(
                type="assistant_answered",
                source=self.name,
                timestamp=now_local(),
                payload={
                    "question": question,
                    "answer": res["answer"],
                    "suggested_action": res.get("suggested_action"),
                    "confidence": res["confidence"],
                    "model": res["model"],
                    "interaction_id": interaction_id,
                    "source": source,
                },
            )
        )

        # 6. Emit tts_requested if speak=True, allow_tts=True, and not dnd_mode
        allow_tts = False
        dnd_mode = False
        if settings is not None:
            allow_tts = bool(settings.privacy.allow_tts)
            dnd_mode = bool(settings.privacy.dnd_mode)
        else:
            privacy_cfg = self.config.get("privacy", {})
            allow_tts = bool(privacy_cfg.get("allow_tts", False))
            dnd_mode = bool(privacy_cfg.get("dnd_mode", False))

        if speak and allow_tts and not dnd_mode:
            await self.bus.publish(
                PlatformEvent(
                    type="tts_requested",
                    source=self.name,
                    timestamp=now_local(),
                    payload={
                        "text": res["answer"],
                        "interaction_id": interaction_id,
                        "language": language,
                    },
                )
            )

        return {
            "answer": res["answer"],
            "suggested_action": res.get("suggested_action"),
            "confidence": res["confidence"],
            "model": res["model"],
            "interaction_id": interaction_id,
            "latency_ms": res.get("latency_ms"),
        }
