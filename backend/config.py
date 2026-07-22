from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.db_utils import DEFAULT_SQLITE_URL, resolve_database_url

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

DATABASE_URL: str = resolve_database_url(default=DEFAULT_SQLITE_URL)

_LOCAL_DEV_ORIGINS = (
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


def get_cors_origins() -> list[str]:
    """Browser origins allowed to call the API.

    Priority:
    1. CORS_ALLOWED_ORIGINS (comma-separated) when set
    2. localhost defaults + FEEDBACK_BASE_URL (production frontend origin)
    """
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    origins = list(_LOCAL_DEV_ORIGINS)
    base = (os.getenv("FEEDBACK_BASE_URL") or "").strip().rstrip("/")
    if base and base not in origins:
        origins.append(base)
    return origins
