"""Google Calendar read-only synchronization input module (FR-03, FR-14, FR-16, FR-17).

Synchronizes calendar events within [now - 1h, now + 24h], parses timed and all-day
events, prunes stale/declined/cancelled events, performs deterministic task
classification, and emits calendar_context_changed on event boundary transitions.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from jarvis.core.bus import EventBus
from jarvis.core.classification import classify_calendar_event
from jarvis.core.database import Database
from jarvis.core.schema import PlatformEvent, ensure_aware, now_local
from jarvis.core.settings import load_settings
from jarvis.modules.base import BaseModule

logger = logging.getLogger(__name__)

READONLY_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/calendar.events.readonly",)

DEFAULT_SYNC_INTERVAL_MINUTES = 15
DEFAULT_BOUNDARY_CHECK_SECONDS = 60.0
DEFAULT_OAUTH_TOKEN_PATH = Path("~/.local/share/posture-platform/google-token.json")
DEFAULT_CREDENTIALS_PATH = Path("~/.config/posture-platform/google-credentials.json")


class GoogleLibrariesMissingError(ImportError):
    """Raised when Google Calendar dependencies are not installed."""


@dataclass(slots=True)
class _GoogleLibraries:
    build: Any
    Credentials: Any
    InstalledAppFlow: Any
    Request: Any
    HttpError: type[Exception]
    RefreshError: type[Exception]


_GOOGLE_LIBS: _GoogleLibraries | None = None
_GOOGLE_LIBS_LOADED: bool = False
_GOOGLE_LIBS_OVERRIDE: _GoogleLibraries | None = None


def get_google_libraries() -> _GoogleLibraries | None:
    """Lazy-import Google API and OAuth libraries, returning None if missing."""
    global _GOOGLE_LIBS, _GOOGLE_LIBS_LOADED
    if _GOOGLE_LIBS_OVERRIDE is not None:
        return _GOOGLE_LIBS_OVERRIDE
    if _GOOGLE_LIBS_LOADED:
        return _GOOGLE_LIBS

    try:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        _GOOGLE_LIBS = _GoogleLibraries(
            build=build,
            Credentials=Credentials,
            InstalledAppFlow=InstalledAppFlow,
            Request=Request,
            HttpError=HttpError,
            RefreshError=RefreshError,
        )
    except ImportError:
        _GOOGLE_LIBS = None

    _GOOGLE_LIBS_LOADED = True
    return _GOOGLE_LIBS


def save_token_file(token_path: Path | str, token_json: str) -> Path:
    """Save token JSON with strict 0600 file permissions."""
    path = Path(token_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(token_json)
    except Exception:
        os.close(fd)
        raise
    path.chmod(0o600)
    return path


def run_auth_flow(
    credentials_path: Path | str,
    oauth_token_path: Path | str,
    *,
    no_browser: bool = False,
    port: int = 0,
    flow_factory: Any | None = None,
) -> Path:
    """Execute standalone OAuth bootstrap flow and persist token with 0600 permissions."""
    cred_p = Path(credentials_path).expanduser().resolve()
    token_p = Path(oauth_token_path).expanduser().resolve()

    libs = get_google_libraries()
    if libs is None and flow_factory is None:
        raise GoogleLibrariesMissingError(
            "Google Calendar dependencies are not installed. Please run: uv sync --group calendar"
        )

    if not cred_p.exists() and flow_factory is None:
        raise FileNotFoundError(
            f"OAuth credentials client secrets file not found at: {cred_p}\n"
            "Please download your OAuth 2.0 Client ID JSON from Google Cloud Console "
            "(APIs & Services > Credentials) and save it to that path, "
            "or specify --credentials <path>."
        )

    if flow_factory is not None:
        flow = flow_factory(str(cred_p), scopes=list(READONLY_SCOPES))
    else:
        assert libs is not None
        flow = libs.InstalledAppFlow.from_client_secrets_file(
            str(cred_p),
            scopes=list(READONLY_SCOPES),
        )

    creds = flow.run_local_server(port=port, open_browser=not no_browser)
    token_json = creds.to_json() if hasattr(creds, "to_json") else json.dumps(creds)
    return save_token_file(token_p, token_json)


class GoogleCalendarModule(BaseModule):
    """Input module for Google Calendar read-only sync and context diffing."""

    name: str = "google_calendar"

    def __init__(
        self,
        bus: EventBus,
        db: Database,
        config: dict[str, Any],
        *,
        service: Any | None = None,
        fetcher: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(bus=bus, db=db, config=config)
        self._injected_service = service
        self._injected_fetcher = fetcher
        self._custom_clock = clock

        self._running: bool = False
        self._last_error: str | None = None
        self._last_event_at: datetime | None = None
        self._last_sync_at: datetime | None = None
        self._last_context_signature: tuple[Any, ...] = (None, None, None, None)

        self._sync_task: asyncio.Task[None] | None = None
        self._boundary_task: asyncio.Task[None] | None = None

        self._privacy_paused: bool = self._extract_privacy_paused(self.config)
        self._dnd_mode: bool = self._extract_dnd_mode(self.config)

    @property
    def calendar_id(self) -> str:
        """Return configured Google Calendar ID (default: 'primary')."""
        val = self.config.get("calendar_id")
        return str(val).strip() if val else "primary"

    @property
    def sync_interval_seconds(self) -> float:
        """Return sync interval in seconds (default: 15 minutes)."""
        raw = self.config.get("sync_interval_minutes")
        if raw is not None:
            return max(1.0, float(raw) * 60.0)
        return DEFAULT_SYNC_INTERVAL_MINUTES * 60.0

    @property
    def boundary_check_interval_seconds(self) -> float:
        """Return interval in seconds for lightweight boundary check loop (default: 60s)."""
        raw = self.config.get("boundary_check_interval_seconds")
        if raw is not None:
            return max(1.0, float(raw))
        return DEFAULT_BOUNDARY_CHECK_SECONDS

    @property
    def oauth_token_path(self) -> Path:
        """Return resolved path to stored user OAuth token JSON."""
        val = self.config.get("oauth_token_path")
        if val is not None:
            return Path(str(val)).expanduser()
        return DEFAULT_OAUTH_TOKEN_PATH.expanduser()

    @property
    def credentials_path(self) -> Path | None:
        """Return resolved path to client secrets JSON if configured."""
        val = self.config.get("credentials_path")
        if val is not None:
            return Path(str(val)).expanduser()
        return None

    @property
    def read_only(self) -> bool:
        """Confirm module runs in read-only scope mode."""
        return bool(self.config.get("read_only", True))

    async def start(self) -> None:
        """Subscribe to events, verify dependencies, and start background sync loops."""
        self.bus.subscribe("privacy_mode_changed", self._on_privacy_mode_changed)

        if self._db_connected():
            persisted_pause = await self.db.get_setting("privacy.global_pause")
            if persisted_pause is not None:
                self._privacy_paused = bool(persisted_pause)

        self._running = True
        self._last_error = None

        # Check dependency availability when neither mock service nor fetcher was injected
        libs = get_google_libraries()
        if libs is None and self._injected_service is None and self._injected_fetcher is None:
            error_msg = (
                "Google Calendar libraries not installed. "
                "Install optional calendar dependencies: uv sync --group calendar"
            )
            logger.warning("%s: %s", self.name, error_msg)
            await self._record_failure(error_msg, event_time=self._now())
            return

        # Start periodic background tasks
        self._sync_task = asyncio.create_task(
            self._periodic_sync_loop(),
            name=f"{self.name}_sync_loop",
        )
        self._boundary_task = asyncio.create_task(
            self._periodic_boundary_loop(),
            name=f"{self.name}_boundary_loop",
        )

        # Run initial sync immediately
        await self.sync_calendar()

    async def stop(self) -> None:
        """Cancel background loops and release resources."""
        self._running = False
        self.bus.unsubscribe("privacy_mode_changed", self._on_privacy_mode_changed)

        for task_ref in (self._sync_task, self._boundary_task):
            if task_ref is not None and not task_ref.done():
                task_ref.cancel()
                with suppress(asyncio.CancelledError):
                    await task_ref

        self._sync_task = None
        self._boundary_task = None

    async def reload_config(self, config: dict[str, Any]) -> None:
        """Refresh configuration parameters and privacy mode."""
        await super().reload_config(dict(config))
        was_paused = self._privacy_paused
        self._privacy_paused = self._extract_privacy_paused(self.config)
        self._dnd_mode = self._extract_dnd_mode(self.config)

        if was_paused and not self._privacy_paused:
            await self._check_and_emit_context_changed(self._now())

    async def health(self) -> dict[str, Any]:
        """Return module health payload."""
        status = (
            "degraded" if self._last_error is not None else ("ok" if self._running else "disabled")
        )
        return {
            "module": self.name,
            "status": status,
            "calendar_id": self.calendar_id,
            "read_only": self.read_only,
            "last_event_at": (
                self._last_event_at.isoformat() if self._last_event_at is not None else None
            ),
            "last_sync_at": (
                self._last_sync_at.isoformat() if self._last_sync_at is not None else None
            ),
            "last_error": self._last_error,
        }

    async def sync_calendar(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Synchronize calendar events within [now - 1h, now + 24h]."""
        ref_now = ensure_aware(now) if now is not None else self._now()

        if self._privacy_paused:
            logger.debug("%s: Sync skipped due to privacy.global_pause", self.name)
            return []

        start_of_today = datetime.combine(ref_now.date(), time.min, tzinfo=ref_now.tzinfo)
        end_of_today = datetime.combine(
            ref_now.date() + timedelta(days=1), time.min, tzinfo=ref_now.tzinfo
        )
        time_min = min(ref_now - timedelta(hours=1), start_of_today)
        time_max = max(ref_now + timedelta(hours=24), end_of_today)

        all_items: list[dict[str, Any]] = []
        cal_tz_name: str | None = None
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "calendarId": self.calendar_id,
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                page_data = await self._fetch_events_page(params)
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.warning("%s sync fetch failed: %s", self.name, error_msg)
                await self._record_failure(error_msg, event_time=ref_now)
                return []

            if not cal_tz_name and page_data.get("timeZone"):
                cal_tz_name = page_data.get("timeZone")

            items = page_data.get("items", [])
            all_items.extend(items)

            page_token = page_data.get("nextPageToken")
            if not page_token:
                break

        cal_tz = self._resolve_timezone(cal_tz_name)
        synced_ids: set[str] = set()
        persisted_events: list[dict[str, Any]] = []

        for item in all_items:
            event_id = item.get("id")
            if not event_id:
                continue

            if item.get("status") == "cancelled":
                continue

            if self._is_declined_by_user(item):
                continue

            start_info = item.get("start", {})
            end_info = item.get("end", {})

            start_dt: datetime | None = None
            end_dt: datetime | None = None

            if "dateTime" in start_info and start_info["dateTime"]:
                try:
                    start_dt = ensure_aware(datetime.fromisoformat(start_info["dateTime"]))
                except Exception:
                    pass
            elif "date" in start_info and start_info["date"]:
                try:
                    s_date = date.fromisoformat(start_info["date"])
                    start_dt = datetime.combine(s_date, time.min, tzinfo=cal_tz)
                except Exception:
                    pass

            if "dateTime" in end_info and end_info["dateTime"]:
                try:
                    end_dt = ensure_aware(datetime.fromisoformat(end_info["dateTime"]))
                except Exception:
                    pass
            elif "date" in end_info and end_info["date"]:
                try:
                    e_date = date.fromisoformat(end_info["date"])
                    end_dt = datetime.combine(e_date, time.min, tzinfo=cal_tz)
                except Exception:
                    pass

            if start_dt is None or end_dt is None:
                logger.debug("Skipping unparseable calendar event %s", event_id)
                continue

            title = str(item.get("summary") or "(No title)")
            description = item.get("description")
            color_id = str(item["colorId"]) if item.get("colorId") is not None else None

            classification = await classify_calendar_event(
                title=title,
                description=description,
                color_id=color_id,
                settings=self._get_classification_settings(),
                pomodoro_settings=self._get_pomodoro_settings(),
            )

            if self._db_connected():
                saved = await self.db.upsert_calendar_event(
                    event_id=event_id,
                    calendar_id=self.calendar_id,
                    title=title,
                    description=description,
                    color_id=color_id,
                    task_type=classification.task_type,
                    suitable_for_walking=classification.suitable_for_walking,
                    recommended_pomodoro=classification.recommended_pomodoro,
                    start_time=start_dt,
                    end_time=end_dt,
                    raw=item,
                )
                persisted_events.append(saved)

            synced_ids.add(event_id)

        # Prune stale or cancelled events that were cached in this window
        if self._db_connected():
            await self.db.prune_stale_calendar_events(
                time_min=time_min,
                time_max=time_max,
                keep_event_ids=synced_ids,
                calendar_id=self.calendar_id,
            )

        self._last_sync_at = ref_now
        await self._record_success(event_time=ref_now)
        await self._check_and_emit_context_changed(ref_now)
        return persisted_events

    async def _periodic_sync_loop(self) -> None:
        interval = self.sync_interval_seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                await self.sync_calendar()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Unexpected error in %s sync loop: %s", self.name, exc)

    async def _periodic_boundary_loop(self) -> None:
        interval = self.boundary_check_interval_seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                await self._check_and_emit_context_changed(self._now())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Unexpected error in %s boundary check: %s", self.name, exc)

    async def _check_and_emit_context_changed(
        self,
        now: datetime | None = None,
    ) -> PlatformEvent | None:
        if self._privacy_paused or not self._db_connected():
            return None

        ref_now = ensure_aware(now) if now is not None else self._now()
        cur_evt, next_evt = await self.db.get_current_and_next_events(now=ref_now)

        cur_id = str(cur_evt["event_id"]) if cur_evt and cur_evt.get("event_id") else None
        cur_task = str(cur_evt["task_type"]) if cur_evt and cur_evt.get("task_type") else None
        next_id = str(next_evt["event_id"]) if next_evt and next_evt.get("event_id") else None
        next_start = (
            str(next_evt["start_time"]) if next_evt and next_evt.get("start_time") else None
        )

        sig = (cur_id, cur_task, next_id, next_start)
        if self._last_context_signature == sig:
            return None

        self._last_context_signature = sig
        event = PlatformEvent(
            type="calendar_context_changed",
            source=self.name,
            confidence=1.0,
            timestamp=ref_now,
            payload={
                "current_event": cur_evt,
                "next_event": next_evt,
                "current_event_id": cur_id,
                "current_task_type": cur_task,
                "next_event_id": next_id,
                "next_start_time": next_start,
            },
        )
        await self.bus.publish(event)
        self._last_event_at = ref_now
        return event

    async def _fetch_events_page(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch a single page of events using injected fetcher or Google API client."""
        if self._injected_fetcher is not None:
            res = self._injected_fetcher(params)
            if inspect.isawaitable(res):
                res = await res
            return dict(res) if isinstance(res, Mapping) else {}

        service = await self._get_service()

        def _execute() -> dict[str, Any]:
            req = service.events().list(**params)
            res = req.execute()
            return dict(res) if isinstance(res, Mapping) else {}

        return await asyncio.to_thread(_execute)

    async def _get_service(self) -> Any:
        """Return cached or newly initialized Google Calendar Resource client."""
        if self._injected_service is not None:
            return self._injected_service

        token_path = self.oauth_token_path
        if not token_path.exists():
            raise FileNotFoundError(
                f"OAuth token file not found at: {token_path}\n"
                "Run 'python -m jarvis.modules.inputs.google_calendar --auth' to authenticate."
            )

        libs = get_google_libraries()
        if libs is None:
            raise GoogleLibrariesMissingError(
                "Google Calendar dependencies are not installed. "
                "Install optional calendar dependencies: uv sync --group calendar"
            )

        creds = await self._load_and_refresh_credentials(libs)

        def _build_client() -> Any:
            return libs.build(
                "calendar",
                "v3",
                credentials=creds,
                cache_discovery=False,
            )

        return await asyncio.to_thread(_build_client)

    async def _load_and_refresh_credentials(self, libs: _GoogleLibraries | None) -> Any:
        """Load OAuth token from disk and refresh silently if expired."""
        token_path = self.oauth_token_path
        if not token_path.exists():
            raise FileNotFoundError(
                f"OAuth token file not found at: {token_path}\n"
                "Run 'python -m jarvis.modules.inputs.google_calendar --auth' to authenticate."
            )

        if libs is None:
            raise GoogleLibrariesMissingError(
                "Google Calendar dependencies are not installed. "
                "Install optional calendar dependencies: uv sync --group calendar"
            )

        token_raw = token_path.read_text(encoding="utf-8")
        token_dict = json.loads(token_raw)
        creds = libs.Credentials.from_authorized_user_info(
            token_dict,
            scopes=list(READONLY_SCOPES),
        )

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                logger.info("%s: Refreshing expired OAuth access token", self.name)
                req = libs.Request()
                try:
                    await asyncio.to_thread(creds.refresh, req)
                except libs.RefreshError as exc:
                    raise libs.RefreshError(
                        f"OAuth token refresh failed (revoked or expired): {exc}\n"
                        "Run 'python -m jarvis.modules.inputs.google_calendar --auth' "
                        "to re-authenticate."
                    ) from exc

                save_token_file(token_path, creds.to_json())
            else:
                raise PermissionError(
                    f"OAuth credentials invalid and lack refresh token at: {token_path}\n"
                    "Run 'python -m jarvis.modules.inputs.google_calendar --auth' to authenticate."
                )

        return creds

    @staticmethod
    def _is_declined_by_user(item: dict[str, Any]) -> bool:
        attendees = item.get("attendees")
        if not isinstance(attendees, Sequence) or isinstance(attendees, (str, bytes)):
            return False
        for attendee in attendees:
            if isinstance(attendee, Mapping):
                if attendee.get("self") is True and attendee.get("responseStatus") == "declined":
                    return True
        return False

    def _resolve_timezone(self, cal_tz_name: str | None) -> tzinfo:
        # Check explicit config overrides first
        cfg_tz = self.config.get("timezone") or self.config.get("calendar_timezone")
        effective = str(cfg_tz).strip() if cfg_tz else (cal_tz_name or "")
        if effective:
            try:
                return ZoneInfo(effective)
            except Exception:
                pass
        local_tz = datetime.now().astimezone().tzinfo
        return local_tz if local_tz is not None else UTC

    def _get_classification_settings(self) -> Any:
        return self.config.get("classification") or self.config.get("calendar_classification")

    def _get_pomodoro_settings(self) -> Any:
        return self.config.get("pomodoro")

    def _now(self) -> datetime:
        if self._custom_clock is not None:
            return ensure_aware(self._custom_clock())
        return now_local()

    def _db_connected(self) -> bool:
        return getattr(self.db, "_conn", None) is not None

    async def _record_success(self, *, event_time: datetime) -> None:
        self._last_error = None
        self._last_event_at = event_time
        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "ok",
                last_event_at=event_time,
                last_error=None,
            )

    async def _record_failure(self, error_msg: str, *, event_time: datetime) -> None:
        self._last_error = error_msg
        self._last_event_at = event_time
        if self._db_connected():
            await self.db.upsert_module_health(
                self.name,
                "degraded",
                last_event_at=event_time,
                last_error=error_msg,
            )

    def _on_privacy_mode_changed(self, event: PlatformEvent) -> None:
        payload = event.payload
        was_paused = self._privacy_paused
        for key in ("global_pause", "privacy_paused", "paused"):
            if key in payload and payload[key] is not None:
                self._privacy_paused = bool(payload[key])
                self._sync_privacy_in_config("global_pause", self._privacy_paused)
                break

        if "dnd_mode" in payload and payload["dnd_mode"] is not None:
            self._dnd_mode = bool(payload["dnd_mode"])
            self._sync_privacy_in_config("dnd_mode", self._dnd_mode)

        if was_paused and not self._privacy_paused:
            asyncio.create_task(self._check_and_emit_context_changed(self._now()))

    @staticmethod
    def _extract_privacy_paused(config: dict[str, Any]) -> bool:
        for key in ("global_pause", "privacy_paused"):
            if key in config and config[key] is not None:
                return bool(config[key])
        privacy_cfg = config.get("privacy")
        if isinstance(privacy_cfg, Mapping) and privacy_cfg.get("global_pause") is not None:
            return bool(privacy_cfg["global_pause"])
        return False

    @staticmethod
    def _extract_dnd_mode(config: dict[str, Any]) -> bool:
        if "dnd_mode" in config and config["dnd_mode"] is not None:
            return bool(config["dnd_mode"])
        privacy_cfg = config.get("privacy")
        if isinstance(privacy_cfg, Mapping) and privacy_cfg.get("dnd_mode") is not None:
            return bool(privacy_cfg["dnd_mode"])
        return False

    def _sync_privacy_in_config(self, key: str, value: bool) -> None:
        privacy_cfg = self.config.get("privacy")
        if isinstance(privacy_cfg, dict):
            privacy_cfg[key] = value
        if key in self.config:
            self.config[key] = value


def cli(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for standalone Google Calendar OAuth authentication."""
    parser = argparse.ArgumentParser(
        description="Posture Platform Google Calendar OAuth setup utility",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Run interactive OAuth authorization flow and save token with 0600 permissions",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=None,
        help="Path to OAuth client secrets JSON (overrides config credentials_path)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Path to destination token JSON (overrides config oauth_token_path)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open local browser automatically (display URL in console)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local HTTP server port for OAuth callback (default: 0 for random port)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.auth:
        parser.print_help()
        return 0

    try:
        settings = load_settings(args.config)
        cfg_cred = settings.google_calendar.credentials_path
        cfg_token = settings.google_calendar.oauth_token_path
    except Exception:
        cfg_cred = None
        cfg_token = None

    cred_path = args.credentials or cfg_cred or DEFAULT_CREDENTIALS_PATH
    token_path = args.token or cfg_token or DEFAULT_OAUTH_TOKEN_PATH

    try:
        saved_path = run_auth_flow(
            credentials_path=cred_path,
            oauth_token_path=token_path,
            no_browser=args.no_browser,
            port=args.port,
        )
        print(f"Authentication successful! Token saved to: {saved_path} (mode 0600)")
        return 0
    except Exception as exc:
        print(f"Error during OAuth authentication: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
