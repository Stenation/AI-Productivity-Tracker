"""YAML + Pydantic v2 configuration settings for the Posture Platform."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_MIGRATIONS_DIR = PROJECT_ROOT / "migrations"

_ACTIVE_YAML_DATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "_ACTIVE_YAML_DATA", default=None
)


def _expand_path(value: Path | str) -> Path:
    raw = str(value)
    if raw == ":memory:" or raw.startswith("file:"):
        return Path(raw)
    return Path(raw).expanduser()


def _expand_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return _expand_path(value)


def _resolve_migrations_path(value: Path | str) -> Path:
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        return (PROJECT_ROOT / expanded).resolve()
    return expanded.resolve()


class _YamlContextSettingsSource(PydanticBaseSettingsSource):
    """Settings source reading parsed YAML data from the active context variable."""

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        data = _ACTIVE_YAML_DATA.get() or {}
        field_value = data.get(field_name)
        return field_value, field_name, False

    def __call__(self) -> dict[str, Any]:
        data = _ACTIVE_YAML_DATA.get()
        if not data:
            return {}
        return dict(data)


class ServerSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    path: Path = Path("~/.local/share/posture-platform/posture.db")
    wal_mode: bool = True
    busy_timeout_ms: int = Field(default=5000, ge=0)
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Path | str) -> Path:
        return _expand_path(value)

    @field_validator("migrations_dir", mode="before")
    @classmethod
    def _validate_migrations_dir(cls, value: Path | str) -> Path:
        return _resolve_migrations_path(value)


class EnabledModulesSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    inputs: list[str] = Field(default_factory=list)
    policies: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class PrivacySettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    global_pause: bool = False
    dnd_mode: bool = False
    hyprland_ipc_mode: Literal["observe_only", "notify_on_distraction"] = "observe_only"
    allow_webcam: bool = False
    microphone_mode: Literal["push_to_talk", "disabled"] = "push_to_talk"
    allow_screenshots: bool = False
    allow_vlm_screenshots: bool = False
    allow_roast_mode: bool = False
    allow_tts: bool = False
    allow_calendar_writes: bool = False
    allow_phone_webhook: bool = False
    allow_external_notifications: bool = False
    dashboard_bind_local_only: bool = True
    retention_days: int = Field(default=90, ge=1)


class PostureSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    manual_override_ttl_minutes: int = Field(default=30, ge=1)
    max_sitting_minutes: int = Field(default=50, ge=1)
    max_walking_minutes: int = Field(default=45, ge=1)
    pre_meeting_warning_minutes: int = Field(default=10, ge=1)
    recommendation_cooldown_minutes: int = Field(default=30, ge=0)
    sensor_consensus_samples: int = Field(default=4, ge=1)
    sensor_consensus_window_samples: int = Field(default=5, ge=1)


class PomodoroSettings(BaseModel):
    model_config = ConfigDict(
        validate_default=True,
        extra="ignore",
        populate_by_name=True,
    )

    deep_work: str | None = "50_10"
    coding: str | None = "50_10"
    reading: str | None = "50_10"
    admin: str | None = "25_5"
    email: str | None = "25_5"
    planning: str | None = "25_5"
    meeting: str | None = None
    break_: str | None = Field(default=None, alias="break")
    unknown: str | None = None
    auto_start: bool = False
    recommendation_cooldown_minutes: int = Field(default=30, ge=0)

    def get(self, task_type: str, default: str | None = None) -> str | None:
        """Return the configured Pomodoro preset for a calendar task type."""
        mapping: dict[str, str | None] = {
            "deep_work": self.deep_work,
            "coding": self.coding,
            "reading": self.reading,
            "admin": self.admin,
            "email": self.email,
            "planning": self.planning,
            "meeting": self.meeting,
            "break": self.break_,
            "unknown": self.unknown,
        }
        if task_type in mapping:
            return mapping[task_type]
        return default


class FocusSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    distraction_grace_seconds: int = Field(default=180, ge=0)
    phone_unlock_limit_seconds: int = Field(default=60, ge=0)
    max_prompts_per_hour: int = Field(default=4, ge=0)
    recommendation_cooldown_minutes: int = Field(default=15, ge=0)
    work_app_classes: list[str] = Field(
        default_factory=lambda: ["code", "Alacritty", "kitty", "obsidian", "Zotero"]
    )
    distraction_app_classes: list[str] = Field(default_factory=list)
    neutral_app_classes: list[str] = Field(default_factory=list)
    distraction_title_patterns: list[str] = Field(
        default_factory=lambda: ["YouTube", "Reddit", "Twitch", "Twitter", "X"]
    )
    work_title_patterns: list[str] = Field(default_factory=list)
    neutral_title_patterns: list[str] = Field(default_factory=list)
    call_app_classes: list[str] = Field(
        default_factory=lambda: [
            "zoom",
            "teams",
            "microsoft teams",
            "webex",
        ]
    )
    call_title_patterns: list[str] = Field(
        default_factory=lambda: [
            "Zoom Meeting",
            "Zoom Cloud Meetings",
            "Google Meet",
            "Meet -",
            "meet.google.com",
            "Microsoft Teams",
            "Teams Meeting",
            "Discord Voice",
            "Voice Connected",
            "Huddle",
        ]
    )
    app_class_rules: dict[str, str] = Field(default_factory=dict)


class CalendarClassificationSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    explicit_mappings: dict[str, str] = Field(default_factory=dict)
    color_mappings: dict[str, str] = Field(
        default_factory=lambda: {
            "1": "reading",
            "2": "deep_work",
            "3": "planning",
            "4": "break",
            "5": "email",
            "6": "admin",
            "7": "meeting",
            "8": "coding",
            "9": "reading",
            "10": "coding",
            "11": "deep_work",
        }
    )
    regex_rules: list[dict[str, str]] = Field(
        default_factory=lambda: [
            {
                "pattern": (
                    r"(?i)\b(1:1|one-on-one|sync|standup|meeting|interview"
                    r"|retro|retrospective|huddle|call|debrief|presentation|demo)\b"
                ),
                "task_type": "meeting",
            },
            {
                "pattern": (
                    r"(?i)\b(break|lunch|dinner|breakfast|coffee|snack|stretch|rest|pause)\b"
                ),
                "task_type": "break",
            },
            {
                "pattern": (
                    r"(?i)\b(deep\s*work|focus\s*block|focus|study|studying"
                    r"|research|thesis|writing)\b"
                ),
                "task_type": "deep_work",
            },
            {
                "pattern": (
                    r"(?i)\b(code|coding|programming|refactor|refactoring|pr\s*review"
                    r"|pull\s*request|debug|debugging|dev|frontend|backend|bug|bugs|bugfix)\b"
                ),
                "task_type": "coding",
            },
            {
                "pattern": r"(?i)\b(email|emails|inbox\s*zero|inbox|triage\s*inbox|mail)\b",
                "task_type": "email",
            },
            {
                "pattern": (
                    r"(?i)\b(admin|administration|paperwork|expense|expenses|invoices?"
                    r"|timesheet|compliance|hr)\b"
                ),
                "task_type": "admin",
            },
            {
                "pattern": (
                    r"(?i)\b(planning|plan|sprint\s*plan|sprint\s*planning|roadmap"
                    r"|backlog|prioritization|weekly\s*review|okr)\b"
                ),
                "task_type": "planning",
            },
            {
                "pattern": (
                    r"(?i)\b(read|reading|book|audiobook|article|docs"
                    r"|documentation|paper|spec|specs)\b"
                ),
                "task_type": "reading",
            },
        ]
    )

    @field_validator("regex_rules", mode="before")
    @classmethod
    def _validate_regex_rules(cls, value: Any) -> list[dict[str, str]]:
        if isinstance(value, dict):
            return [{"pattern": str(k), "task_type": str(v)} for k, v in value.items()]
        if isinstance(value, list):
            normalized: list[dict[str, str]] = []
            for item in value:
                if isinstance(item, dict):
                    normalized.append({
                        "pattern": str(item.get("pattern", "")),
                        "task_type": str(item.get("task_type", "unknown")),
                    })
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    normalized.append({
                        "pattern": str(item[0]),
                        "task_type": str(item[1]),
                    })
            return normalized
        return value


class ClassificationSettings(CalendarClassificationSettings):
    """Configuration settings for explicit event, color, regex, and window classification."""

    model_config = ConfigDict(validate_default=True, extra="ignore")

    work_app_classes: list[str] = Field(
        default_factory=lambda: ["code", "Alacritty", "kitty", "obsidian", "Zotero"]
    )
    distraction_app_classes: list[str] = Field(default_factory=list)
    neutral_app_classes: list[str] = Field(default_factory=list)
    distraction_title_patterns: list[str] = Field(
        default_factory=lambda: ["YouTube", "Reddit", "Twitch", "Twitter", "X"]
    )
    work_title_patterns: list[str] = Field(default_factory=list)
    neutral_title_patterns: list[str] = Field(default_factory=list)
    call_app_classes: list[str] = Field(
        default_factory=lambda: [
            "zoom",
            "teams",
            "microsoft teams",
            "webex",
        ]
    )
    call_title_patterns: list[str] = Field(
        default_factory=lambda: [
            "Zoom Meeting",
            "Zoom Cloud Meetings",
            "Google Meet",
            "Meet -",
            "meet.google.com",
            "Microsoft Teams",
            "Teams Meeting",
            "Discord Voice",
            "Voice Connected",
            "Huddle",
        ]
    )
    app_class_rules: dict[str, str] = Field(default_factory=dict)


class GoogleCalendarSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    calendar_id: str = "primary"
    sync_interval_minutes: int = Field(default=15, ge=1)
    oauth_token_path: Path = Path("~/.local/share/posture-platform/google-token.json")
    credentials_path: Path | None = None
    read_only: bool = True
    classification: CalendarClassificationSettings | None = None

    @field_validator("oauth_token_path", mode="before")
    @classmethod
    def _validate_oauth_token_path(cls, value: Path | str) -> Path:
        return _expand_path(value)

    @field_validator("credentials_path", mode="before")
    @classmethod
    def _validate_credentials_path(cls, value: Path | str | None) -> Path | None:
        return _expand_optional_path(value)


class WebcamSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    enabled: bool = False
    device: str = "/dev/video0"
    width: int = Field(default=320, ge=1)
    height: int = Field(default=240, ge=1)
    sample_interval_seconds: float = Field(default=5.0, gt=0)
    walking_burst_frames: int = Field(default=8, ge=1)
    walking_burst_interval_seconds: float = Field(default=1.5, gt=0)


class SwalkBleSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    enabled: bool = False
    device_name: str = "sWalk"
    scan_timeout_seconds: float = Field(default=10.0, gt=0)


class PhoneBridgeSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    transport: Literal["none", "kdeconnect", "ntfy", "webhook"] = "none"
    endpoint: str = ""
    token_path: Path | None = None

    @field_validator("token_path", mode="before")
    @classmethod
    def _validate_token_path(cls, value: Path | str | None) -> Path | None:
        return _expand_optional_path(value)


class LlmSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    text_model: str = "qwen3:14b"
    vision_model: str = "qwen2.5vl:7b"
    keep_alive: str = "5m"
    keep_alive_minutes: int = Field(default=5, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    enable_classification_fallback: bool = False


class WaybarSettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    status_file: Path = Path("/dev/shm/posture_waybar.json")
    refresh_interval_seconds: float = Field(default=15.0, gt=0)
    signal_waybar: bool = True
    waybar_signal: int = Field(default=8, ge=1)

    @field_validator("status_file", mode="before")
    @classmethod
    def _validate_status_file(cls, value: Path | str) -> Path:
        return _expand_path(value)


class DesktopNotifySettings(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    notify_binary: str = "notify-send"
    app_name: str = "Posture Platform"
    urgency: Literal["low", "normal", "critical"] = "normal"
    expire_time_ms: int | None = Field(default=None, ge=0)


class Settings(BaseSettings):
    """Top-level validated platform settings loaded from YAML and environment."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        validate_default=True,
        extra="ignore",
        populate_by_name=True,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR
    enabled_modules: EnabledModulesSettings = Field(
        default_factory=EnabledModulesSettings
    )
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    posture: PostureSettings = Field(default_factory=PostureSettings)
    pomodoro: PomodoroSettings = Field(default_factory=PomodoroSettings)
    focus: FocusSettings = Field(default_factory=FocusSettings)
    google_calendar: GoogleCalendarSettings = Field(
        default_factory=GoogleCalendarSettings
    )
    calendar_classification: CalendarClassificationSettings = Field(
        default_factory=CalendarClassificationSettings
    )
    classification: ClassificationSettings = Field(
        default_factory=ClassificationSettings
    )
    webcam: WebcamSettings = Field(default_factory=WebcamSettings)
    swalk_ble: SwalkBleSettings = Field(default_factory=SwalkBleSettings)
    waybar: WaybarSettings = Field(default_factory=WaybarSettings)
    desktop_notify: DesktopNotifySettings = Field(
        default_factory=DesktopNotifySettings
    )
    phone_bridge: PhoneBridgeSettings = Field(default_factory=PhoneBridgeSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)

    @field_validator("migrations_dir", mode="before")
    @classmethod
    def _validate_migrations_dir(cls, value: Path | str) -> Path:
        return _resolve_migrations_path(value)

    @model_validator(mode="after")
    def _validate_invariants(self) -> Settings:
        if (
            "migrations_dir" in self.model_fields_set
            and "migrations_dir" not in self.database.model_fields_set
        ):
            self.database.migrations_dir = self.migrations_dir
        elif "migrations_dir" in self.database.model_fields_set:
            self.migrations_dir = self.database.migrations_dir

        if "focus" in self.model_fields_set and "classification" not in self.model_fields_set:
            for field in (
                "work_app_classes",
                "distraction_app_classes",
                "neutral_app_classes",
                "distraction_title_patterns",
                "work_title_patterns",
                "neutral_title_patterns",
                "call_app_classes",
                "call_title_patterns",
                "app_class_rules",
            ):
                if field in self.focus.model_fields_set:
                    setattr(self.classification, field, getattr(self.focus, field))
        elif "classification" in self.model_fields_set and "focus" not in self.model_fields_set:
            for field in (
                "work_app_classes",
                "distraction_app_classes",
                "neutral_app_classes",
                "distraction_title_patterns",
                "work_title_patterns",
                "neutral_title_patterns",
                "call_app_classes",
                "call_title_patterns",
                "app_class_rules",
            ):
                if field in self.classification.model_fields_set:
                    setattr(self.focus, field, getattr(self.classification, field))

        if (
            not self.classification.explicit_mappings
            and self.calendar_classification.explicit_mappings
        ):
            self.classification.explicit_mappings = (
                self.calendar_classification.explicit_mappings
            )
        elif (
            self.classification.explicit_mappings
            and not self.calendar_classification.explicit_mappings
        ):
            self.calendar_classification.explicit_mappings = (
                self.classification.explicit_mappings
            )

        if self.google_calendar.classification is None:
            self.google_calendar.classification = self.calendar_classification
        elif (
            not self.calendar_classification.explicit_mappings
            and self.google_calendar.classification.explicit_mappings
        ):
            self.calendar_classification = self.google_calendar.classification
            self.classification.explicit_mappings = (
                self.google_calendar.classification.explicit_mappings
            )

        if self.privacy.dashboard_bind_local_only and self.server.host not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "server.host must be loopback ('127.0.0.1', 'localhost', or '::1') "
                "when privacy.dashboard_bind_local_only is true"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _YamlContextSettingsSource(settings_cls),
            file_secret_settings,
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str, **overrides: Any) -> Settings:
        """Load settings from an explicit YAML file, applying env and keyword overrides."""
        resolved = _expand_path(config_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved}")
        raw_text = resolved.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw_text)
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected YAML mapping at root of {resolved}, got {type(parsed).__name__}"
            )
        token = _ACTIVE_YAML_DATA.set(parsed)
        try:
            return cls(**overrides)
        finally:
            _ACTIVE_YAML_DATA.reset(token)


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load platform settings from config_path, JARVIS_CONFIG, or default config.yaml."""
    env_path = os.environ.get("JARVIS_CONFIG")
    if config_path is not None:
        return Settings.from_yaml(config_path)
    if env_path:
        return Settings.from_yaml(env_path)
    if DEFAULT_CONFIG_PATH.exists():
        return Settings.from_yaml(DEFAULT_CONFIG_PATH)
    return Settings()

