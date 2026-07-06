from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from scripts.db_utils import DEFAULT_SQLITE_URL, resolve_database_url

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

DATABASE_URL: str = resolve_database_url(default=DEFAULT_SQLITE_URL)
