from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "")

engine = None
if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
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
