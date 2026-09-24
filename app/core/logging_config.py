"""Centralised logging configuration.

A single :func:`configure_logging` call wires up both console and rotating
file handlers so that every module can simply do::

    from app.core.logging_config import get_logger
    logger = get_logger(__name__)

Sensitive values (passwords, API keys, connection strings with credentials)
must never be passed to the logger by callers; helpers here also redact the
credentials portion of database URLs before they are ever emitted.
"""

from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Matches "scheme://user:password@host" and keeps only "scheme://***@host".
_CREDENTIALS_RE = re.compile(r"(?P<scheme>\w+://)[^:/@\s]+:[^@/\s]+@")


def redact_url(url: str) -> str:
    """Remove ``user:password`` credentials from a connection URL.

    >>> redact_url("mysql+pymysql://root:secret@localhost/db")
    'mysql+pymysql://***@localhost/db'
    """

    return _CREDENTIALS_RE.sub(lambda m: f"{m.group('scheme')}***@", url)


def configure_logging() -> None:
    """Configure the root logger with console + rotating file handlers.

    Idempotent: safe to call multiple times (e.g. from the app factory and
    from CLI entry points) without duplicating handlers.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Replace any handlers a previous (e.g. uvicorn) config installed.
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Keep noisy third-party loggers at WARNING unless we're debugging.
    if level > logging.DEBUG:
        for noisy in ("sqlalchemy.engine", "matplotlib", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.info(
        "Logging configured (level=%s, file=%s)",
        settings.log_level.upper(),
        os.fspath(log_file),
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging has been configured first."""

    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
