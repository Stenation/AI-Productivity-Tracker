"""Dynamic ModuleRegistry for loading, isolating, and toggling platform modules."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass
from typing import Any, Literal

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, now_local
from jarvis.core.settings import Settings
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

ModuleCategory = Literal["inputs", "policies", "outputs"]
MODULE_CATEGORIES: tuple[ModuleCategory, ...] = ("inputs", "policies", "outputs")
STARTUP_CATEGORY_ORDER: tuple[ModuleCategory, ...] = ("outputs", "policies", "inputs")
SHUTDOWN_CATEGORY_ORDER: tuple[ModuleCategory, ...] = ("inputs", "policies", "outputs")

KNOWN_MODULE_CATEGORIES: dict[str, ModuleCategory] = {
    "manual_control": "inputs",
    "hyprland_ipc": "inputs",
    "google_calendar": "inputs",
    "s24_webhook": "inputs",
    "swalk_ble": "inputs",
    "webcam_pose": "inputs",
    "posture_policy": "policies",
    "pomodoro_policy": "policies",
    "focus_policy": "policies",
    "meeting_policy": "policies",
    "waybar_status": "outputs",
    "desktop_notify": "outputs",
    "tts_piper": "outputs",
    "phone_bridge": "outputs",
    "calendar_writer": "outputs",
    "llm_ollama": "outputs",
}

MODULE_SETTINGS_SECTION: dict[str, str] = {
    "webcam_pose": "webcam",
    "swalk_ble": "swalk_ble",
    "google_calendar": "google_calendar",
    "calendar_writer": "google_calendar",
    "posture_policy": "posture",
    "meeting_policy": "posture",
    "pomodoro_policy": "pomodoro",
    "focus_policy": "focus",
    "hyprland_ipc": "focus",
    "waybar_status": "waybar",
    "desktop_notify": "desktop_notify",
    "phone_bridge": "phone_bridge",
    "s24_webhook": "phone_bridge",
    "llm_ollama": "llm",
    "tts_piper": "tts",
}


@dataclass(slots=True)
class _ModuleRecord:
    name: str
    category: ModuleCategory | None
    enabled: bool = False
    status: str = "disabled"
    last_error: str | None = None
    instance: BaseModule | None = None


class ModuleRegistry:
    """Manage dynamic import, lifecycle, and fault isolation of BaseModule plugins."""

    def __init__(self, bus: EventBus, db: Database, settings: Settings) -> None:
        self._bus = bus
        self._db = db
        self._settings = settings
        self._records: dict[str, _ModuleRecord] = {}

        for category in MODULE_CATEGORIES:
            configured: list[str] = getattr(settings.enabled_modules, category, [])
            for name in configured:
                self._records.setdefault(
                    name,
                    _ModuleRecord(name=name, category=category),
                )

    async def start_all(self) -> dict[str, str]:
        """Enable and start all modules configured in settings.enabled_modules."""
        statuses: dict[str, str] = {}
        for category in STARTUP_CATEGORY_ORDER:
            configured: list[str] = getattr(self._settings.enabled_modules, category, [])
            for name in configured:
                await self.enable_module(name, category=category)
                record = self._records.get(name)
                statuses[name] = record.status if record is not None else "degraded"
        return statuses

    async def start(self) -> dict[str, str]:
        """Alias for start_all()."""
        return await self.start_all()

    async def stop_all(self) -> None:
        """Stop all currently running modules cleanly in reverse startup order."""
        ordered_records: list[_ModuleRecord] = []
        for category in SHUTDOWN_CATEGORY_ORDER:
            for record in self._records.values():
                if record.category == category and record.instance is not None:
                    ordered_records.append(record)
        for record in self._records.values():
            if record.instance is not None and record not in ordered_records:
                ordered_records.append(record)

        for record in ordered_records:
            instance = record.instance
            if instance is None:
                continue
            record.instance = None
            record.enabled = False
            try:
                await instance.stop()
                record.status = "disabled"
                record.last_error = None
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                record.status = "degraded"
                record.last_error = error_msg
                logger.exception(
                    "Module '%s' failed while stopping during shutdown", record.name
                )

    async def stop(self) -> None:
        """Alias for stop_all()."""
        await self.stop_all()

    async def enable_module(
        self,
        name: str,
        *,
        category: ModuleCategory | None = None,
    ) -> bool:
        """Import, instantiate, and start `name`, recording degraded health on error."""
        existing = self._records.get(name)
        if existing is not None and existing.enabled and existing.instance is not None:
            return True

        resolved_category = (
            category
            or (existing.category if existing is not None else None)
            or self._resolve_category(name)
        )

        record = existing or _ModuleRecord(name=name, category=resolved_category)
        record.category = resolved_category
        self._records[name] = record

        if resolved_category is None:
            error_msg = (
                f"ModuleNotFoundError: Could not locate module '{name}' under "
                "jarvis.modules.{inputs,policies,outputs}"
            )
            logger.error("Failed to enable module '%s': %s", name, error_msg)
            await self._record_health(
                record,
                enabled=False,
                status="degraded",
                last_error=error_msg,
            )
            return False

        import_path = f"jarvis.modules.{resolved_category}.{name}"
        try:
            module_obj = importlib.import_module(import_path)
            module_cls = self._select_module_class(module_obj, name)
            config = self._build_module_config(name)
            instance = module_cls(bus=self._bus, db=self._db, config=config)
            if getattr(instance, "name", "unnamed_module") == "unnamed_module":
                instance.name = name
            await instance.start()
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Module '%s' (%s) failed to start; marking degraded",
                name,
                import_path,
            )
            record.instance = None
            await self._record_health(
                record,
                enabled=False,
                status="degraded",
                last_error=error_msg,
            )
            return False

        record.instance = instance
        await self._record_health(
            record,
            enabled=True,
            status="ok",
            last_error=None,
        )
        return True

    async def disable_module(self, name: str) -> bool:
        """Stop a running module and mark it disabled in memory and SQLite."""
        record = self._records.get(name)
        if record is None:
            resolved_category = self._resolve_category(name)
            record = _ModuleRecord(name=name, category=resolved_category)
            self._records[name] = record

        instance = record.instance
        record.instance = None

        if instance is not None:
            try:
                await instance.stop()
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("Module '%s' failed while stopping", name)
                await self._record_health(
                    record,
                    enabled=False,
                    status="degraded",
                    last_error=error_msg,
                )
                return False

        await self._record_health(
            record,
            enabled=False,
            status="disabled",
            last_error=None,
        )
        return True

    def is_enabled(self, name: str) -> bool:
        """Return True if `name` is currently active and running."""
        record = self._records.get(name)
        return bool(record is not None and record.enabled and record.instance is not None)

    def get_module(self, name: str) -> BaseModule | None:
        """Return the live BaseModule instance for `name`, or None if not active."""
        record = self._records.get(name)
        return record.instance if record is not None else None

    def get_status(self, name: str) -> dict[str, Any] | None:
        """Return runtime status dictionary for `name`, if known."""
        record = self._records.get(name)
        if record is None:
            return None
        return {
            "module": record.name,
            "category": record.category,
            "enabled": record.enabled,
            "status": record.status,
            "last_error": record.last_error,
        }

    def list_modules(self) -> list[dict[str, Any]]:
        """Return status summaries for all tracked modules sorted by name."""
        return [
            {
                "module": rec.name,
                "category": rec.category,
                "enabled": rec.enabled,
                "status": rec.status,
                "last_error": rec.last_error,
            }
            for rec in sorted(self._records.values(), key=lambda r: r.name)
        ]

    def _resolve_category(self, name: str) -> ModuleCategory | None:
        for category in MODULE_CATEGORIES:
            configured: list[str] = getattr(self._settings.enabled_modules, category, [])
            if name in configured:
                return category

        for category in MODULE_CATEGORIES:
            full_name = f"jarvis.modules.{category}.{name}"
            if full_name in sys.modules:
                return category

        for category in MODULE_CATEGORIES:
            full_name = f"jarvis.modules.{category}.{name}"
            try:
                if importlib.util.find_spec(full_name) is not None:
                    return category
            except Exception:
                continue

        return KNOWN_MODULE_CATEGORIES.get(name)

    @staticmethod
    def _select_module_class(module_obj: Any, expected_name: str) -> type[BaseModule]:
        candidates: list[type[BaseModule]] = [
            obj
            for _, obj in inspect.getmembers(module_obj, inspect.isclass)
            if issubclass(obj, BaseModule)
            and obj is not BaseModule
            and not inspect.isabstract(obj)
        ]
        if not candidates:
            raise TypeError(
                f"Module '{module_obj.__name__}' defines no concrete BaseModule subclass"
            )

        local_candidates = [
            cls for cls in candidates if cls.__module__ == module_obj.__name__
        ]
        pool = local_candidates or candidates

        for cls in pool:
            if getattr(cls, "name", None) == expected_name:
                return cls
        return pool[0]

    def _build_module_config(self, name: str) -> dict[str, Any]:
        full_dump = self._settings.model_dump()
        section_name = MODULE_SETTINGS_SECTION.get(name)
        if section_name and isinstance(full_dump.get(section_name), dict):
            return {**full_dump[section_name], **full_dump}
        return full_dump

    async def _record_health(
        self,
        record: _ModuleRecord,
        *,
        enabled: bool,
        status: str,
        last_error: str | None,
    ) -> None:
        record.enabled = enabled
        record.status = status
        record.last_error = last_error

        await self._db.upsert_module_health(
            module=record.name,
            status=status,
            last_error=last_error,
        )
        await self._bus.publish(
            PlatformEvent(
                type="module_health_changed",
                source="registry",
                timestamp=now_local(),
                payload={
                    "module": record.name,
                    "category": record.category,
                    "enabled": enabled,
                    "status": status,
                    "last_error": last_error,
                },
            )
        )
