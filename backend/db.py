from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts.db_utils import normalize_database_url

DEFAULT_SQLITE_URL = "sqlite:///./sports_analytics.db"

# Read at import time but do NOT raise — the engine is built lazily so the
# module can be imported without DATABASE_URL (e.g. during local dev or tests).
DATABASE_URL: str = normalize_database_url(os.environ.get("DATABASE_URL") or DEFAULT_SQLITE_URL)

_engine_kwargs: dict = {"pool_pre_ping": True, "future": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def require_engine() -> None:
    """No-op kept for backward compatibility; engine is always initialised."""


@contextmanager
def get_db_session() -> Generator:
    require_engine()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
