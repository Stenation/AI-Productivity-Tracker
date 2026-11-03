"""Abstract BaseModule contract for inputs, policies, and outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.database import Database


class BaseModule(ABC):
    """Base lifecycle and health interface implemented by every platform module."""

    name: str = "unnamed_module"

    def __init__(self, bus: EventBus, db: Database, config: dict[str, Any]) -> None:
        self.bus = bus
        self.db = db
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        """Subscribe to events and start background tasks."""

    @abstractmethod
    async def stop(self) -> None:
        """Cancel tasks and release resources."""

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Optional live-reconfiguration hook."""
        self.config = config

    async def health(self) -> dict[str, Any]:
        """Return current module health payload."""
        return {
            "module": self.name,
            "status": "ok",
        }
