"""Daily MLB prediction runner.

Generates predictions, stores to PostgreSQL/SQLite, logs feature usage and model outputs.
Writes to mlb_predictions (legacy) and unified predictions + prediction_options tables.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Path alignment when invoked as `python scripts/run_daily_predictions.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.config  # noqa: F401 — load .env before database URL resolution

from sqlalchemy import text

from scripts.db_utils import (
    create_database_engine,
    ensure_unified_schema,
    insert_prediction,
    insert_prediction_options,
    resolve_database_url,
)
from data.sport_config import (
    binary_home_win_probabilities,
    build_outcome_options,
    get_config,
    get_league_default,
)

SPORT = "MLB"

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


def _ensure_mlb_predictions_table(engine) -> None:
    # Use TEXT for feature_snapshot so the table works on both SQLite and
    # PostgreSQL. JSON serialisation is handled by the application layer.
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
        feature_snapshot TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
    sport_cfg = get_config(SPORT)
    logger.warning("Using stub predictions. Replace _fetch_mlb_predictions with real pipeline.")
    return [
        {
            "prediction_id": f"stub-{today}-nyy-bos",
            "sport": SPORT,
            "game_date": today,
            "home_team": "NYY",
            "away_team": "BOS",
            "win_probability": 0.62,
            "predicted_spread": -1.2,
            "confidence_level": "MEDIUM",
            "model_version": f"{sport_cfg['model_type']}-stub-0.1",
            "feature_snapshot": {
                "rest_diff": 1,
                "bullpen_fatigue": 0.25,
            },
        }
    ]


def _build_prediction_options(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build outcome options from sport config."""
    sport = row.get("sport", SPORT)
    probabilities = binary_home_win_probabilities(float(row["win_probability"]))
    return build_outcome_options(
        sport,
        row["home_team"],
        row["away_team"],
        probabilities,
    )


def _build_unified_prediction(row: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map a stub row to the unified predictions schema."""
    sport = row.get("sport", SPORT)
    top = options[0]
    return {
        "sport": sport,
        "league": get_league_default(sport),
        "game_date": row["game_date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "predicted_winner": top["option_name"],
        "confidence_level": row["confidence_level"],
        "feature_snapshot": row.get("feature_snapshot"),
        "win_probability": float(row["win_probability"]),
        "predicted_spread": float(row.get("predicted_spread") or 0.0),
        "confidence_score": top["probability"],
    }


def _insert_mlb_predictions(engine, rows: List[Dict[str, Any]]) -> int:
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
    serialized = []
    for row in rows:
        r = dict(row)
        if isinstance(r.get("feature_snapshot"), dict):
            r["feature_snapshot"] = json.dumps(r["feature_snapshot"])
        serialized.append(r)

    with engine.begin() as conn:
        result = conn.execute(text(insert_sql), serialized)
        return result.rowcount or 0


def _insert_unified_predictions(engine, rows: List[Dict[str, Any]]) -> int:
    """Write rows to unified predictions + prediction_options tables."""
    ensure_unified_schema(engine)
    inserted = 0

    for row in rows:
        options = _build_prediction_options(row)
        # insert_prediction_options uses option_name, probability, rank only
        option_rows = [
            {
                "option_name": o["option_name"],
                "probability": o["probability"],
                "rank": o["rank"],
            }
            for o in options
        ]
        unified = _build_unified_prediction(row, options)
        prediction_id = insert_prediction(engine, unified)
        insert_prediction_options(engine, prediction_id, option_rows)
        inserted += 1
        logger.info(
            "Unified prediction stored (id=%s): %s vs %s -> %s",
            prediction_id,
            unified["home_team"],
            unified["away_team"],
            unified["predicted_winner"],
        )

    return inserted


def main() -> int:
    try:
        database_url = resolve_database_url()
        engine = create_database_engine(database_url)
        _ensure_mlb_predictions_table(engine)
        ensure_unified_schema(engine)

        predictions = _fetch_mlb_predictions()
        if not predictions:
            logger.info("No predictions returned. Exiting.")
            return 0

        for row in predictions:
            logger.info("Prediction: %s", json.dumps(row, default=str))

        mlb_inserted = _insert_mlb_predictions(engine, predictions)
        unified_inserted = _insert_unified_predictions(engine, predictions)
        logger.info(
            "Inserted %s mlb_predictions row(s), %s unified prediction(s)",
            mlb_inserted,
            unified_inserted,
        )
        return 0
    except Exception as exc:
        logger.exception("Daily prediction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
