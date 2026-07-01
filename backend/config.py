from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

DEFAULT_SQLITE_URL = "sqlite:///./sports_analytics.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)
