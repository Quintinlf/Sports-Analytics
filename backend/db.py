from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import DATABASE_URL as _RAW_DATABASE_URL
from scripts.db_utils import normalize_database_url

DEFAULT_SQLITE_URL = "sqlite:///./sports_analytics.db"

DATABASE_URL: str = normalize_database_url(_RAW_DATABASE_URL or DEFAULT_SQLITE_URL)


class Base(DeclarativeBase):
    pass


_engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
elif DATABASE_URL.startswith("postgresql"):
    _engine_kwargs.update(
        {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_recycle": 1800,
        }
    )

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def require_engine() -> None:
    """No-op kept for backward compatibility; engine is always initialised."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Backward-compatible context manager for existing route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
