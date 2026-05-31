from __future__ import annotations

import os

from sqlalchemy import create_engine


DEFAULT_SQLITE_URL = "sqlite:///./sports_analytics.db"


def get_database_url(default: str = DEFAULT_SQLITE_URL) -> str:
    """Return the configured database URL, falling back to local SQLite."""
    return os.getenv("DATABASE_URL") or default


def create_database_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine that works for both Postgres and SQLite."""
    url = database_url or get_database_url()
    engine_kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **engine_kwargs)