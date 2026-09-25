"""Advance live SQLAlchemy prediction lifecycle (not SportsAnalyticsDB).

Rules:
  - Rows with both actual scores → game_status=FINAL, prediction_status=SETTLED
  - Rows past the Pacific horizon with no scores → prediction_status=VOID
    (keeps history; excludes them from upcoming dashboard)

Usage:
    python -m scripts.settle_live_predictions
    python -m scripts.settle_live_predictions --void-after-days 3 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.config  # noqa: F401
from sqlalchemy import text

from data.prediction_time import (
    GAME_FINAL,
    PRED_SETTLED,
    PRED_VOID,
    pacific_today,
)
from scripts.db_utils import create_database_engine, ensure_unified_schema, resolve_database_url
from src.utils.timezone_utils import convert_utc_to_pst
from datetime import datetime, timezone

logger = logging.getLogger("settle_live_predictions")


def _column_set(engine) -> set[str]:
    from scripts.db_utils import _column_names

    return set(_column_names(engine, "predictions") or set())


def settle(engine, void_after_days: int = 3, dry_run: bool = False) -> dict:
    ensure_unified_schema(engine)
    cols = _column_set(engine)
    has_game_status = "game_status" in cols
    has_pacific = "game_date_pacific" in cols

    today = pacific_today()
    void_cutoff = (
        convert_utc_to_pst(datetime.now(timezone.utc)).date() - timedelta(days=void_after_days)
    ).isoformat()

    stats = {"settled": 0, "voided": 0, "dry_run": dry_run}

    with engine.begin() as conn:
        # Normalize legacy FINAL → SETTLED
        if dry_run:
            n = conn.execute(
                text(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE UPPER(COALESCE(prediction_status, '')) = 'FINAL'"
                )
            ).scalar()
            stats["legacy_final"] = int(n or 0)
        else:
            result = conn.execute(
                text(
                    "UPDATE predictions SET prediction_status = :settled "
                    "WHERE UPPER(COALESCE(prediction_status, '')) = 'FINAL'"
                ),
                {"settled": PRED_SETTLED},
            )
            stats["legacy_final"] = int(result.rowcount or 0)

        # Settle rows that already have scores
        set_bits = ["prediction_status = :settled"]
        if has_game_status:
            set_bits.append("game_status = :game_final")
        settle_sql = f"""
            UPDATE predictions
            SET {', '.join(set_bits)},
                correct = CASE
                    WHEN actual_winner IS NOT NULL AND predicted_winner IS NOT NULL
                         AND UPPER(predicted_winner) = UPPER(actual_winner)
                    THEN 1
                    WHEN actual_winner IS NOT NULL AND predicted_winner IS NOT NULL
                    THEN 0
                    ELSE correct
                END
            WHERE actual_home_score IS NOT NULL
              AND actual_away_score IS NOT NULL
              AND UPPER(COALESCE(prediction_status, '')) NOT IN ('SETTLED', 'VOID')
        """
        params = {"settled": PRED_SETTLED, "game_final": GAME_FINAL}
        if dry_run:
            n = conn.execute(
                text(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE actual_home_score IS NOT NULL "
                    "AND actual_away_score IS NOT NULL "
                    "AND UPPER(COALESCE(prediction_status, '')) NOT IN ('SETTLED', 'VOID')"
                )
            ).scalar()
            stats["settled"] = int(n or 0)
        else:
            result = conn.execute(text(settle_sql), params)
            stats["settled"] = int(result.rowcount or 0)

        # Void stale upcoming rows with no scores past cutoff
        date_col = "game_date_pacific" if has_pacific else "game_date"
        void_sql = f"""
            UPDATE predictions
            SET prediction_status = :void
            WHERE UPPER(COALESCE(prediction_status, '')) IN ('UPCOMING', 'ACTIVE', '')
              AND actual_home_score IS NULL
              AND {date_col} IS NOT NULL
              AND CAST({date_col} AS TEXT) < :cutoff
        """
        if dry_run:
            n = conn.execute(
                text(
                    f"""
                    SELECT COUNT(*) FROM predictions
                    WHERE UPPER(COALESCE(prediction_status, '')) IN ('UPCOMING', 'ACTIVE', '')
                      AND actual_home_score IS NULL
                      AND {date_col} IS NOT NULL
                      AND CAST({date_col} AS TEXT) < :cutoff
                    """
                ),
                {"cutoff": void_cutoff},
            ).scalar()
            stats["voided"] = int(n or 0)
        else:
            result = conn.execute(
                text(void_sql), {"void": PRED_VOID, "cutoff": void_cutoff}
            )
            stats["voided"] = int(result.rowcount or 0)

    stats["pacific_today"] = today
    stats["void_cutoff"] = void_cutoff
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--void-after-days", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    url = resolve_database_url()
    engine = create_database_engine(url)
    stats = settle(engine, void_after_days=args.void_after_days, dry_run=args.dry_run)
    logger.info("Settlement complete: %s", stats)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
