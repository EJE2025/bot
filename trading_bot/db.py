"""Database helpers for the dashboard and shared services."""

from __future__ import annotations

import logging
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base

from trading_bot import config

logger = logging.getLogger(__name__)

engine = create_engine(
    config.DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
)

Base = declarative_base()


def get_session() -> Iterator[scoped_session]:
    """Return the scoped session factory for convenience."""

    return SessionLocal


def remove_session() -> None:
    """Dispose the current scoped session (used on app teardown)."""

    SessionLocal.remove()


def init_db() -> None:
    """Create database tables for registered models."""

    # Import models that need to be registered with SQLAlchemy metadata.
    from trading_bot import authentication  # noqa: F401  # pylint: disable=unused-import

    Base.metadata.create_all(bind=engine)


__all__ = ["Base", "SessionLocal", "engine", "init_db", "remove_session", "get_session"]

