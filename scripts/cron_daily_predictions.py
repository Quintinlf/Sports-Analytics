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
from sqlalchemy import create_engine

# Path alignment logic if called directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main() -> None:
    """Execute the daily prediction ingestion pipeline."""
    args = parse_arguments()
    logger.info("=" * 80)
    logger.info("Starting scheduled prediction collection process...")
    logger.info("=" * 80)

    database_url = os.getenv("DATABASE_URL", "sqlite:///./sports_analytics.db")
    logger.info(f"Database URL: {database_url}")

    try:
        engine = create_engine(database_url)
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

    logger.info("=" * 80)
    if success:
        logger.info("Automation task completed cleanly.")
        sys.exit(0)
    else:
        logger.error("Automation run finished with execution conflicts.")
        sys.exit(1)


if __name__ == "__main__":
    main()
