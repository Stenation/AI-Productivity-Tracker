"""Core infrastructure for the Posture Platform."""

from typing import TYPE_CHECKING, Any

from jarvis.core.bus import EventBus
from jarvis.core.database import Database
from jarvis.core.migrations import apply_migrations, get_applied_migrations, run_migrations
from jarvis.core.resolver import StateResolver
from jarvis.core.schema import (
    MANUAL_ACTIVITY_CYCLE,
    ActivitySpan,
    ActivityState,
    CalendarEvent,
    EffectiveActivityState,
    EventType,
    FocusState,
    ManualActivityState,
    ModuleHealth,
    PlatformEvent,
    Recommendation,
    RecommendationStatus,
    SystemStateSnapshot,
    TaskType,
    TimerSpan,
    ensure_aware,
    now_local,
)
from jarvis.core.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIGRATIONS_DIR,
    DesktopNotifySettings,
    Settings,
    WaybarSettings,
    load_settings,
)
from jarvis.core.timer import TimerManager

if TYPE_CHECKING:
    from jarvis.core.app import create_app
    from jarvis.core.registry import ModuleRegistry


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from jarvis.core.app import create_app

        return create_app
    if name == "ModuleRegistry":
        from jarvis.core.registry import ModuleRegistry

        return ModuleRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MIGRATIONS_DIR",
    "MANUAL_ACTIVITY_CYCLE",
    "ActivitySpan",
    "ActivityState",
    "CalendarEvent",
    "Database",
    "DesktopNotifySettings",
    "EffectiveActivityState",
    "EventBus",
    "EventType",
    "FocusState",
    "ManualActivityState",
    "ModuleHealth",
    "ModuleRegistry",
    "PlatformEvent",
    "Recommendation",
    "RecommendationStatus",
    "Settings",
    "StateResolver",
    "SystemStateSnapshot",
    "TaskType",
    "TimerManager",
    "TimerSpan",
    "WaybarSettings",
    "apply_migrations",
    "create_app",
    "ensure_aware",
    "get_applied_migrations",
    "load_settings",
    "now_local",
    "run_migrations",
]
