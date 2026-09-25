"""One-shot: promote Quintin to leader and opt out of analyst digests.

Usage (loads .env via backend.config):
    python -m scripts.promote_leader
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.config  # noqa: F401
from sqlalchemy import text

from scripts.db_utils import (
    DEFAULT_REVIEWER_ID,
    create_database_engine,
    database_url_source,
    ensure_default_reviewers,
    resolve_database_url,
)


def main() -> int:
    url = resolve_database_url()
    print(f"DB source: {database_url_source()}")
    engine = create_database_engine(url)
    ensure_default_reviewers(engine)
    with engine.connect() as conn:
        role = conn.execute(
            text("SELECT analyst_role FROM reviewers WHERE reviewer_id = :r"),
            {"r": DEFAULT_REVIEWER_ID},
        ).scalar()
        emails = conn.execute(
            text(
                "SELECT emails_enabled FROM reviewer_preferences WHERE reviewer_id = :r"
            ),
            {"r": DEFAULT_REVIEWER_ID},
        ).scalar()
    print(f"{DEFAULT_REVIEWER_ID}: role={role} emails_enabled={emails}")
    if str(role or "").lower() != "leader":
        print("WARNING: leader role not applied", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
