"""Web layer (REST API, HTMX views, and static assets)."""

from jarvis.web.api import router as api_router
from jarvis.web.router import router as web_router

__all__ = ["api_router", "web_router"]
