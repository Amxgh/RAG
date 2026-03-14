"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger.jsonlogger import JsonFormatter

from waam_rag.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure root logging once for the application."""

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(
            JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        )
    root.addHandler(handler)
