"""Daily MLB prediction runner.

Generates predictions, stores to PostgreSQL, logs feature usage and model outputs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine, text

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"daily_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _ensure_predictions_table(engine) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS mlb_predictions (
        prediction_id TEXT PRIMARY KEY,
        game_date DATE NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        win_probability DOUBLE PRECISION NOT NULL,
        predicted_spread DOUBLE PRECISION NOT NULL,
        confidence_level TEXT NOT NULL,
        model_version TEXT NOT NULL,
        feature_snapshot JSONB NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        actual_home_score INTEGER,
        actual_away_score INTEGER
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


def _fetch_mlb_predictions() -> List[Dict[str, Any]]:
    """Return prediction rows for today's games.

    TODO: Replace with real MLB pipeline.
    """
    today = datetime.utcnow().date().isoformat()
    logger.warning("Using stub predictions. Replace _fetch_mlb_predictions with real pipeline.")
    return [
        {
            "prediction_id": f"stub-{today}-nyy-bos",
            "game_date": today,
            "home_team": "NYY",
            "away_team": "BOS",
            "win_probability": 0.62,
            "predicted_spread": -1.2,
            "confidence_level": "MEDIUM",
            "model_version": "mlb-stub-0.1",
            "feature_snapshot": {"rest_diff": 1, "bullpen_fatigue": 0.25},
        }
    ]


def _insert_predictions(engine, rows: List[Dict[str, Any]]) -> int:
    insert_sql = """
    INSERT INTO mlb_predictions (
        prediction_id,
        game_date,
        home_team,
        away_team,
        win_probability,
        predicted_spread,
        confidence_level,
        model_version,
        feature_snapshot
    )
    VALUES (
        :prediction_id,
        :game_date,
        :home_team,
        :away_team,
        :win_probability,
        :predicted_spread,
        :confidence_level,
        :model_version,
        :feature_snapshot
    )
    ON CONFLICT (prediction_id) DO NOTHING;
    """
    with engine.begin() as conn:
        result = conn.execute(text(insert_sql), rows)
        return result.rowcount or 0


def main() -> int:
    try:
        database_url = _require_env("DATABASE_URL")
        engine = create_engine(database_url, pool_pre_ping=True)
        _ensure_predictions_table(engine)

        predictions = _fetch_mlb_predictions()
        if not predictions:
            logger.info("No predictions returned. Exiting.")
            return 0

        for row in predictions:
            logger.info("Prediction: %s", json.dumps(row, default=str))

        inserted = _insert_predictions(engine, predictions)
        logger.info("Inserted %s predictions", inserted)
        return 0
    except Exception as exc:
        logger.exception("Daily prediction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
