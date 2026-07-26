"""Initialize all database tables for CI/automation (no FastAPI required).

Usage:
  DATABASE_URL=postgresql://... python -m scripts.init_database
  # Also accepts SUPABASE_DATABASE_URL or SUPERBASE_DATABASE_URL
  # (same priority as FastAPI via scripts.db_utils.resolve_database_url).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.config  # noqa: F401 — load .env when present

from sqlalchemy import inspect

from backend.db import Base, engine
import backend.models  # noqa: F401 — register every ORM model with Base.metadata
from scripts.db_utils import (
    ensure_default_reviewers,
    ensure_reviewer_email_unique_index,
    ensure_unified_schema,
    resolve_database_url,
)

DEBUG_LOG = ROOT / "debug-ca0755.log"

REQUIRED_TABLES = frozenset(
    {
        "analyst_answers",
        "analyst_case_studies",
        "analyst_comments",
        "analyst_feedback",
        "analyst_questions",
        "feature_suggestions",
        "predictions",
        "prediction_options",
        "prediction_reviews",
        "review_outcomes",
        "reviewer_custom_sections",
        "reviewer_preferences",
        "reviewers",
    }
)


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    entry = {
        "sessionId": "ca0755",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # #endregion


def initialize_database() -> list[str]:
    """Create ORM tables and unified prediction schema. Returns sorted table names."""
    # Deploy-time path must always apply DDL, even on PostgreSQL where the web
    # process defaults SCHEMA_AUTO_MIGRATE=false.
    os.environ["SCHEMA_AUTO_MIGRATE"] = "true"

    orm_tables = sorted(Base.metadata.tables.keys())
    _agent_log(
        "B",
        "init_database.py:initialize_database:orm",
        "ORM models registered on Base.metadata",
        {"orm_tables": orm_tables, "reviewers_registered": "reviewers" in Base.metadata.tables},
    )

    inspector = inspect(engine)
    before = sorted(inspector.get_table_names())
    _agent_log(
        "C",
        "init_database.py:initialize_database:before",
        "tables present before create_all",
        {"tables": before, "reviewers_exists_before": "reviewers" in before},
    )

    Base.metadata.create_all(bind=engine)
    ensure_unified_schema(engine)
    ensure_default_reviewers(engine)
    ensure_reviewer_email_unique_index(engine)

    # Keep question catalogs in sync on deploy (same UPSERTs as web init_platform).
    try:
        from backend.routes.feedback import (
            _seed_onboarding_questions,
            _seed_research_questions,
        )

        _seed_onboarding_questions(engine)
        _seed_research_questions(engine)
    except Exception as exc:
        print(f"WARNING: question seed skipped: {exc}")

    after = sorted(inspect(engine).get_table_names())
    missing = sorted(REQUIRED_TABLES - set(after))
    _agent_log(
        "A",
        "init_database.py:initialize_database:after",
        "tables present after initialization",
        {
            "tables": after,
            "reviewers_exists_after": "reviewers" in after,
            "missing_required": missing,
        },
    )
    return after


def main() -> None:
    try:
        # Same priority as FastAPI / cron: SUPABASE > SUPERBASE > DATABASE_URL.
        resolve_database_url(default=None, required=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    tables = initialize_database()
    missing = sorted(REQUIRED_TABLES - set(tables))
    if missing:
        print(f"ERROR: missing required tables: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Database initialized successfully ({len(tables)} tables).")
    print("Required tables present:", ", ".join(sorted(REQUIRED_TABLES)))


if __name__ == "__main__":
    main()
