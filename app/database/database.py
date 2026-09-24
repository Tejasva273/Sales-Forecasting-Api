"""Database engine and session management.

The engine is created from ``DATABASE_URL`` so that SQLite (the default) can be
swapped for MySQL purely through configuration — no code changes required. The
``connect_args`` / pooling tweaks are applied conditionally depending on the
backend detected in the URL.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings
from app.core.logging_config import get_logger, redact_url

logger = get_logger(__name__)

_settings = get_settings()

# SQLite needs `check_same_thread=False` to be used across FastAPI's threads.
_is_sqlite = _settings.database_url.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=not _is_sqlite,  # validate pooled connections for MySQL
    future=True,
)

logger.info("Database engine created for %s", redact_url(_settings.database_url))

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


class Base(DeclarativeBase):
    """Declarative base class shared by all ORM models."""


def init_db() -> None:
    """Create all tables. Safe to call repeatedly (``checkfirst=True``)."""

    # Import here so all models are registered on ``Base.metadata`` before
    # ``create_all`` runs, while avoiding a circular import at module load.
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured (create_all).")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a session and always closes it."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
