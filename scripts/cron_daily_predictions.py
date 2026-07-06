"""Production Daily Prediction Ingestion Pipeline.

Standalone automation script that:
- Initializes UnifiedPredictionService
- Fetches live predictions from NBA, MLB, FIFA
- UPSERTs predictions permanently (no cache delete)
- Supports dry-run mode for testing
- Can be run independently of web server

Usage:
  python scripts/cron_daily_predictions.py           # Normal execution
  python scripts/cron_daily_predictions.py --dry-run # Preview without DB writes
  python -m scripts.cron_daily_predictions           # Module execution (GitHub Actions)
"""
from __future__ import annotations

import os
import sys
import argparse
import logging

# Path alignment logic if called directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.config  # noqa: F401 — load .env before database URL resolution

from sqlalchemy import text

from scripts.db_utils import (
    create_database_engine,
    database_url_source,
    resolve_database_url,
    _parse_database_host_and_name,
)
from scripts.prediction_runner import run_live_prediction_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cron_daily_predictions")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Production Daily Prediction Ingestion Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Fetch records without writing to the DB")
    return parser.parse_args()


def _verify_usa_belgium(engine) -> None:
    """Confirm USA vs Belgium World Cup prediction exists after ingest."""
    sql = text(
        """
        SELECT prediction_id, sport, league, home_team, away_team, game_date,
               predicted_winner, win_probability, confidence_level, prediction_status
        FROM predictions
        WHERE home_team = 'USA' AND away_team = 'Belgium'
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql).mappings().first()
    if not row:
        logger.warning("Post-ingest check: USA vs Belgium row NOT found in predictions.")
        return
    logger.info(
        "Post-ingest check: USA vs Belgium found — id=%s league=%s date=%s winner=%s "
        "win_prob=%s confidence=%s status=%s",
        row["prediction_id"],
        row["league"],
        row["game_date"],
        row["predicted_winner"],
        row["win_probability"],
        row["confidence_level"],
        row["prediction_status"],
    )


def main() -> None:
    """Execute the daily prediction ingestion pipeline."""
    args = parse_arguments()
    logger.info("=" * 80)
    logger.info("Starting scheduled prediction collection process...")
    logger.info("=" * 80)

    database_url = resolve_database_url()
    source = database_url_source()
    host, dbname = _parse_database_host_and_name(database_url)
    logger.info("Database source: %s", source)
    logger.info("host: %s", host)
    logger.info("db: %s", dbname)
    if database_url.startswith("sqlite"):
        logger.warning("Using local SQLite — set SUPERBASE_DATABASE_URL or SUPABASE_DATABASE_URL for Supabase.")

    try:
        engine = create_database_engine(database_url)
    except Exception as e:
        logger.error(f"Failed to initialize database engine: {e}", exc_info=True)
        sys.exit(1)

    if args.dry_run:
        from scripts.prediction_runner import fetch_live_predictions

        live_payloads = fetch_live_predictions()
        logger.info(f"Retrieved a total of {len(live_payloads)} games across all pipelines.")
        logger.info("[DRY RUN MODE] Data processed successfully. Skipping database modification.")
        logger.info("=" * 80)
        sys.exit(0)

    logger.info("Running live prediction pipeline...")
    success = run_live_prediction_pipeline(engine)

    if success:
        _verify_usa_belgium(engine)

    logger.info("=" * 80)
    if success:
        logger.info("Automation task completed cleanly.")
        sys.exit(0)
    else:
        logger.error("Automation run finished with execution conflicts.")
        sys.exit(1)


if __name__ == "__main__":
    main()
