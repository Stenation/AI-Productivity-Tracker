"""CLI entrypoint running the Jarvis FastAPI daemon via Uvicorn."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from jarvis.core.app import create_app
from jarvis.core.settings import load_settings


def main(argv: Sequence[str] | None = None) -> None:
    """Load platform settings and run the Uvicorn ASGI server."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Local-first posture, focus, and calendar copilot daemon.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.yaml (defaults to JARVIS_CONFIG or ./config.yaml).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    settings = load_settings(args.config)
    app = create_app(args.config, settings=settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
    )


if __name__ == "__main__":
    main()
