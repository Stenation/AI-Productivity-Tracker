"""Compact context builder, Ollama HTTP client, and safety-boundary assistant logic."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from jarvis.core.database import Database
from jarvis.core.schema import ensure_aware, now_local
from jarvis.core.settings import Settings

logger = logging.getLogger(__name__)

# JSON schema enforced for structured assistant responses
RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Natural language response to the user's question.",
        },
        "suggested_action": {
            "type": ["string", "null"],
            "description": "Optional recommended action identifier.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0.0 and 1.0.",
        },
    },
    "required": ["answer", "confidence", "suggested_action"],
}

CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": [
                "deep_work",
                "coding",
                "reading",
                "admin",
                "email",
                "planning",
                "meeting",
                "break",
            ],
            "description": "Classified calendar task category.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0.0 and 1.0.",
        },
    },
    "required": ["task_type", "confidence"],
}

_THINK_TAG_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


class LlmError(Exception):
    """Base exception for LLM operations."""


class LlmConnectionError(LlmError):
    """Raised when the Ollama server is unreachable or connection is refused."""


class LlmTimeoutError(LlmError):
    """Raised when an Ollama request times out."""


class LlmValidationError(LlmError):
    """Raised when Ollama returns an unparseable or schema-violating response."""


def clean_think_tags(raw_text: str) -> str:
    """Defensively strip <think> tags (closed or unclosed) and code fences."""
    cleaned = _THINK_TAG_PATTERN.sub("", raw_text)
    cleaned = _UNCLOSED_THINK_PATTERN.sub("", cleaned).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def validate_llm_response(parsed: Any) -> dict[str, Any]:
    """Validate parsed JSON dictionary against the assistant response schema."""
    if not isinstance(parsed, dict):
        raise LlmValidationError(f"Expected JSON object, got {type(parsed).__name__}")

    if (
        "answer" not in parsed
        or not isinstance(parsed["answer"], str)
        or not parsed["answer"].strip()
    ):
        raise LlmValidationError("Response missing required non-empty string field 'answer'")

    if "confidence" not in parsed:
        raise LlmValidationError("Response missing required field 'confidence'")

    try:
        confidence = float(parsed["confidence"])
    except (TypeError, ValueError) as exc:
        raise LlmValidationError(
            f"Field 'confidence' is not a number: {parsed.get('confidence')}"
        ) from exc

    if not (0.0 <= confidence <= 1.0):
        raise LlmValidationError(
            f"Field 'confidence' must be between 0.0 and 1.0, got {confidence}"
        )

    suggested_action = parsed.get("suggested_action")
    if suggested_action is not None and not isinstance(suggested_action, str):
        type_name = type(suggested_action).__name__
        raise LlmValidationError(
            f"Field 'suggested_action' must be string or null, got {type_name}"
        )

    action_clean = (
        suggested_action.strip()
        if isinstance(suggested_action, str) and suggested_action.strip()
        else None
    )

    return {
        "answer": parsed["answer"].strip(),
        "suggested_action": action_clean,
        "confidence": confidence,
    }


async def build_assistant_context(
    db: Database,
    resolver: Any | None = None,
    timer_manager: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Build a compact, privacy-safe JSON snapshot of platform state.

    Contains:
    - current activity, source, and elapsed duration
    - active desktop countdown timer status: kind, phase, remaining
    - current and next calendar event: title, task_type, start_time, end_time, suitable_for_walking
    - recent window categories and duration totals (never sensitive raw window titles)
    - pending recommendations
    """
    now = now_local()

    # 1. Effective activity state
    if resolver is not None and hasattr(resolver, "get_effective_state"):
        effective = resolver.get_effective_state()
        activity_data = {
            "activity": getattr(effective, "activity", "unknown"),
            "source": getattr(effective, "source", "system"),
            "elapsed_seconds": int(getattr(effective, "elapsed_seconds", 0)),
        }
    else:
        current_span = await db.get_current_activity_span()
        if current_span:
            start_dt = ensure_aware(datetime.fromisoformat(str(current_span["started_at"])))
            elapsed = max(0, int((now - start_dt).total_seconds()))
            activity_data = {
                "activity": current_span.get("activity", "unknown"),
                "source": current_span.get("source", "system"),
                "elapsed_seconds": elapsed,
            }
        else:
            activity_data = {
                "activity": "unknown",
                "source": "system",
                "elapsed_seconds": 0,
            }

    # 2. Active desktop countdown timer status
    if timer_manager is not None and hasattr(timer_manager, "get_status"):
        timer_status = timer_manager.get_status()
        timer_data = {
            "kind": timer_status.get("kind"),
            "phase": timer_status.get("phase", "idle"),
            "remaining_seconds": timer_status.get("remaining_seconds", 0),
        }
    else:
        active_timer_row = await db.get_active_timer()
        if active_timer_row is not None:
            started_dt = ensure_aware(datetime.fromisoformat(str(active_timer_row["started_at"])))
            planned = active_timer_row.get("planned_duration_seconds")
            elapsed = max(0, int((now - started_dt).total_seconds()))
            remaining = max(0, int(planned) - elapsed) if planned is not None else 0
            raw_kind = str(active_timer_row.get("kind", ""))
            is_break = raw_kind.endswith("_break") or raw_kind == "break"
            base_kind = raw_kind.removesuffix("_break") if is_break else raw_kind
            timer_data = {
                "kind": base_kind,
                "phase": "break" if is_break else "work",
                "remaining_seconds": remaining,
            }
        else:
            timer_data = {
                "kind": None,
                "phase": "idle",
                "remaining_seconds": 0,
            }

    # 3. Calendar: current and next events
    cur_row, next_row = await db.get_current_and_next_events(now=now)

    def _format_event(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "title": row.get("title", ""),
            "task_type": row.get("task_type", "unknown"),
            "start_time": str(row.get("start_time", "")),
            "end_time": str(row.get("end_time", "")),
            "suitable_for_walking": row.get("suitable_for_walking"),
        }

    calendar_data = {
        "current_event": _format_event(cur_row),
        "next_event": _format_event(next_row),
    }

    # 4. Recent window categories and duration totals (NO raw window titles)
    recent_spans = await db.list_window_spans(limit=25)
    category_totals: dict[str, int] = {}
    recent_categories: list[str] = []
    for span in recent_spans:
        cat = str(span.get("category", "unknown"))
        if cat not in recent_categories:
            recent_categories.append(cat)
        try:
            st = ensure_aware(datetime.fromisoformat(str(span["started_at"])))
            et = (
                ensure_aware(datetime.fromisoformat(str(span["ended_at"])))
                if span.get("ended_at")
                else now
            )
            duration = max(0, int((et - st).total_seconds()))
        except (ValueError, TypeError):
            duration = 0
        category_totals[cat] = category_totals.get(cat, 0) + duration

    window_focus_data = {
        "category_totals_seconds": category_totals,
        "recent_categories": recent_categories,
    }

    # 5. Pending recommendations
    pending_recs = await db.get_pending_recommendations()
    recs_data = [
        {
            "id": r["id"],
            "policy": r["policy"],
            "message": r["message"],
            "suggested_action": r.get("suggested_action"),
        }
        for r in pending_recs
    ]

    return {
        "activity": activity_data,
        "timer": timer_data,
        "calendar": calendar_data,
        "window_focus": window_focus_data,
        "pending_recommendations": recs_data,
    }


class OllamaClient:
    """Async client communicating with local Ollama daemon at /api/chat."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:14b",
        keep_alive: str = "5m",
        timeout_seconds: float = 30.0,
        *,
        http_client: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds
        self._external_client = http_client is not None
        self._transport = transport
        self._client = http_client

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if internally owned."""
        if not self._external_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        keep_alive: str | None = None,
        think: bool = False,
        format_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send chat messages to POST /api/chat and return validated response dictionary."""
        effective_model = model or self.model
        effective_keep_alive = keep_alive or self.keep_alive
        effective_format = format_schema or RESPONSE_JSON_SCHEMA

        payload = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
            "keep_alive": effective_keep_alive,
            "think": think,
            "format": effective_format,
        }

        client = await self._get_client()
        url = f"{self.base_url}/api/chat"
        start_t = time.perf_counter()

        try:
            resp = await client.post(url, json=payload)
        except Exception as exc:
            import httpx

            if isinstance(exc, (httpx.ConnectError, ConnectionRefusedError, OSError)):
                raise LlmConnectionError(f"Ollama connection refused at {url}: {exc}") from exc
            if isinstance(exc, httpx.TimeoutException):
                raise LlmTimeoutError(
                    f"Ollama request timed out after {self.timeout_seconds}s"
                ) from exc
            raise LlmError(f"Ollama HTTP request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start_t) * 1000)

        if resp.status_code != 200:
            raise LlmError(
                f"Ollama returned HTTP status {resp.status_code}: {resp.text[:200]}"
            )

        try:
            resp_data = resp.json()
        except Exception as exc:
            raise LlmValidationError(f"Ollama response was not valid JSON: {exc}") from exc

        content = resp_data.get("message", {}).get("content", "")
        cleaned_content = clean_think_tags(content)

        try:
            parsed = json.loads(cleaned_content)
        except Exception as exc:
            snippet = cleaned_content[:100]
            raise LlmValidationError(
                f"Ollama message content was not valid JSON: {exc}. Content snippet: {snippet!r}"
            ) from exc

        if format_schema is RESPONSE_JSON_SCHEMA or format_schema is None:
            validated = validate_llm_response(parsed)
            return {
                **validated,
                "model": resp_data.get("model", effective_model),
                "latency_ms": latency_ms,
                "raw": resp_data,
            }

        return {
            "parsed": parsed,
            "model": resp_data.get("model", effective_model),
            "latency_ms": latency_ms,
            "raw": resp_data,
        }

    async def ask(
        self,
        question: str,
        context: dict[str, Any] | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Ask a question given a platform context snapshot."""
        ctx_str = json.dumps(context or {}, indent=2)
        sys_prompt = system_prompt or (
            "You are Jarvis, a local posture, focus, and calendar assistant for Arch Linux.\n"
            "Answer the user's question accurately and concisely based on the context.\n"
            "Safety boundary: You may only answer questions and optionally suggest an action via "
            "'suggested_action' (e.g., 'start_timer', 'stand_up', 'take_break', or null). "
            "You must NEVER attempt to execute shell commands, directly change posture state, "
            "or modify settings.\n"
            f"Current platform context:\n{ctx_str}"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]

        return await self.chat(messages, model=model)


async def classify_event_with_llm(
    title: str = "",
    description: str | None = None,
    color_id: str | None = None,
    event: Any = None,
    *,
    client: OllamaClient | None = None,
    settings: Settings | None = None,
    db: Database | None = None,
    module_registry: Any = None,
) -> dict[str, Any] | None:
    """Implement optional LLM calendar event classification fallback (FR-16, FR-17)."""
    # 1. Check if classification fallback is enabled in settings
    if settings is not None:
        if not getattr(settings.llm, "enable_classification_fallback", False):
            return None

    # 2. Check if llm_ollama module is enabled if registry is provided
    if module_registry is not None and hasattr(module_registry, "is_enabled"):
        if not module_registry.is_enabled("llm_ollama"):
            return None

    effective_client = client or OllamaClient()
    own_client = client is None

    prompt = (
        "Classify the following calendar event into exactly one task type from: "
        "[deep_work, coding, reading, admin, email, planning, meeting, break].\n"
        f"Event title: {title}\n"
        f"Description: {description or 'None'}\n"
        f"Color ID: {color_id or 'None'}\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant categorizing calendar events into one of these task types: "
                "deep_work, coding, reading, admin, email, planning, meeting, break. "
                "Respond strictly with a JSON object containing 'task_type' and 'confidence'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        res = await effective_client.chat(messages, format_schema=CLASSIFICATION_JSON_SCHEMA)
        parsed = res.get("parsed") if isinstance(res.get("parsed"), dict) else res
        task_type = parsed.get("task_type")
        confidence = float(parsed.get("confidence", 0.8))
        if task_type in {
            "deep_work",
            "coding",
            "reading",
            "admin",
            "email",
            "planning",
            "meeting",
            "break",
        }:
            return {"task_type": task_type, "confidence": max(0.0, min(1.0, confidence))}
        return None
    except Exception as exc:
        logger.warning("classify_event_with_llm failed: %s", exc)
        return None
    finally:
        if own_client:
            await effective_client.close()
