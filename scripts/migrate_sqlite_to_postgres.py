"""One-time migration: local SQLite -> Railway PostgreSQL.

Usage:
  SQLITE_DATABASE_URL=sqlite:///./sports_analytics.db \\
  DATABASE_URL=postgresql+psycopg2://... \\
  python scripts/migrate_sqlite_to_postgres.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text

from backend.models import Base
from scripts.db_utils import create_database_engine, ensure_unified_schema

SQLITE_DEFAULT = "sqlite:///./sports_analytics.db"

TABLES_IN_ORDER = [
    "reviewers",
    "reviewer_preferences",
    "predictions",
    "prediction_reviews",
    "review_outcomes",
    "reviewer_custom_sections",
]

BOOL_COLUMNS: Dict[str, Set[str]] = {
    "prediction_reviews": {"agree_with_model"},
    "review_outcomes": {
        "model_correct",
        "reviewer_correct",
        "reviewer_beat_model",
        "should_be_feature",
    },
    "reviewer_preferences": {
        "emails_enabled",
        "wants_betting_section",
        "wants_explanations",
        "wants_postgame_reviews",
    },
    "reviewer_custom_sections": {"active"},
}

QUINTIN_EMAIL = "quintinlf7@gmail.com"


def _coerce_bool(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _row_to_params(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for col in BOOL_COLUMNS.get(table, set()):
        if col in out:
            out[col] = _coerce_bool(out[col])
    return out


def _table_exists(engine, table: str) -> bool:
    return inspect(engine).has_table(table)


def _fetch_rows(engine, table: str) -> List[Dict[str, Any]]:
    if not _table_exists(engine, table):
        return []
    with engine.begin() as conn:
        rows = conn.execute(text(f"SELECT * FROM {table}")).mappings().all()
    return [dict(r) for r in rows]


def _clear_target_tables(pg_engine) -> None:
    with pg_engine.begin() as conn:
        for table in reversed(TABLES_IN_ORDER):
            if _table_exists(pg_engine, table):
                conn.execute(text(f"DELETE FROM {table}"))


def _insert_rows(pg_engine, table: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})")
    inserted = 0
    with pg_engine.begin() as conn:
        for row in rows:
            conn.execute(sql, _row_to_params(table, row))
            inserted += 1
    return inserted


def _reset_prediction_sequence(pg_engine) -> None:
    with pg_engine.begin() as conn:
        conn.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence('predictions', 'prediction_id'),
                    COALESCE((SELECT MAX(prediction_id) FROM predictions), 1)
                )
                """
            )
        )


def _upsert_quintin(pg_engine) -> None:
    with pg_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO reviewers (reviewer_id, name, email, created_at)
                VALUES ('quintin', 'Quintin', :email, NOW())
                ON CONFLICT (reviewer_id) DO UPDATE SET
                    email = EXCLUDED.email
                """
            ),
            {"email": QUINTIN_EMAIL},
        )
        conn.execute(
            text(
                """
                INSERT INTO reviewer_preferences
                    (reviewer_id, favorite_sports, emails_enabled, wants_betting_section,
                     wants_explanations, wants_postgame_reviews, email_frequency, updated_at)
                VALUES
                    ('quintin', :sports, TRUE, TRUE, TRUE, TRUE, 'weekly', NOW())
                ON CONFLICT (reviewer_id) DO UPDATE SET
                    emails_enabled = TRUE,
                    updated_at = NOW()
                """
            ),
            {"sports": '["MLB", "NBA"]'},
        )


def main() -> int:
    sqlite_url = os.getenv("SQLITE_DATABASE_URL", SQLITE_DEFAULT)
    pg_url = os.getenv("DATABASE_URL")
    if not pg_url:
        print("ERROR: DATABASE_URL (PostgreSQL target) is required.")
        return 1
    if pg_url.startswith("sqlite"):
        print("ERROR: DATABASE_URL must be a PostgreSQL URL, not SQLite.")
        return 1

    sqlite_engine = create_database_engine(sqlite_url)
    pg_engine = create_database_engine(pg_url)

    print("Creating PostgreSQL schema...")
    Base.metadata.create_all(bind=pg_engine)
    ensure_unified_schema(pg_engine)

    print("Clearing target tables...")
    _clear_target_tables(pg_engine)

    total = 0
    for table in TABLES_IN_ORDER:
        rows = _fetch_rows(sqlite_engine, table)
        count = _insert_rows(pg_engine, table, rows)
        total += count
        print(f"  {table}: {count} rows")

    if _table_exists(pg_engine, "predictions"):
        _reset_prediction_sequence(pg_engine)
    ensure_unified_schema(pg_engine)
    _upsert_quintin(pg_engine)

    print(f"Migration complete. {total} rows copied.")
    print(f"Quintin reviewer email set to {QUINTIN_EMAIL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
