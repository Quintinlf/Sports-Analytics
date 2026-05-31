from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


DATABASE_URL = _require_env("DATABASE_URL")

engine = None
if DATABASE_URL:
    engine_kwargs = {"pool_pre_ping": True, "future": True}
    if DATABASE_URL.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(
        DATABASE_URL,
        **engine_kwargs,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def require_engine() -> None:
    if engine is None:
        raise RuntimeError("DATABASE_URL is not set. Set it to your PostgreSQL connection string.")


@contextmanager
def get_db_session() -> Generator:
    require_engine()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
