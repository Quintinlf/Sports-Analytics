from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


DEFAULT_SQLITE_URL = "sqlite:///./sports_analytics.db"

UNIFIED_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    league TEXT,
    game_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    confidence_level TEXT NOT NULL,
    feature_snapshot TEXT,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_winner TEXT,
    correct INTEGER,
    created_at TEXT NOT NULL
);
"""

PREDICTION_OPTIONS_DDL = """
CREATE TABLE IF NOT EXISTS prediction_options (
    option_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    option_name TEXT NOT NULL,
    probability REAL NOT NULL,
    rank INTEGER NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
    UNIQUE(prediction_id, option_name)
);
"""

# Postgres-compatible DDL (used when engine dialect is postgresql)
UNIFIED_PREDICTIONS_DDL_PG = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    sport TEXT NOT NULL,
    league TEXT,
    game_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    confidence_level TEXT NOT NULL,
    feature_snapshot TEXT,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_winner TEXT,
    correct INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PREDICTION_OPTIONS_DDL_PG = """
CREATE TABLE IF NOT EXISTS prediction_options (
    option_id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
    option_name TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    UNIQUE(prediction_id, option_name)
);
"""

UNIFIED_PREDICTION_COLUMNS = [
    ("sport", "TEXT"),
    ("league", "TEXT"),
    ("feature_snapshot", "TEXT"),
    ("actual_home_score", "INTEGER"),
    ("actual_away_score", "INTEGER"),
    ("created_at", "TEXT"),
]


def get_database_url(default: str = DEFAULT_SQLITE_URL) -> str:
    """Return the configured database URL, falling back to local SQLite."""
    return os.getenv("DATABASE_URL") or default


def create_database_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine that works for both Postgres and SQLite."""
    url = database_url or get_database_url()
    engine_kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **engine_kwargs)


def _is_postgresql(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def _table_exists(engine: Engine, table_name: str) -> bool:
    return inspect(engine).has_table(table_name)


def _column_exists(engine: Engine, table_name: str, column_name: str) -> bool:
    if not _table_exists(engine, table_name):
        return False
    return column_name in {col["name"] for col in inspect(engine).get_columns(table_name)}


def _predictions_is_legacy(engine: Engine) -> bool:
    """True when predictions uses the legacy NBA schema (requires spread/probability cols)."""
    return _table_exists(engine, "predictions") and _column_exists(
        engine, "predictions", "predicted_spread"
    )


def _serialize_feature_snapshot(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def ensure_unified_schema(engine: Engine) -> None:
    """Create or migrate unified predictions and prediction_options tables."""
    is_pg = _is_postgresql(engine)

    with engine.begin() as conn:
        if not _table_exists(engine, "predictions"):
            ddl = UNIFIED_PREDICTIONS_DDL_PG if is_pg else UNIFIED_PREDICTIONS_DDL
            conn.execute(text(ddl))
        elif _predictions_is_legacy(engine):
            for col_name, col_type in UNIFIED_PREDICTION_COLUMNS:
                if not _column_exists(engine, "predictions", col_name):
                    conn.execute(
                        text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
                    )

        options_ddl = PREDICTION_OPTIONS_DDL_PG if is_pg else PREDICTION_OPTIONS_DDL
        conn.execute(text(options_ddl))

        if is_pg:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_sport_date "
                    "ON predictions (sport, game_date)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_options_prediction "
                    "ON prediction_options (prediction_id)"
                )
            )
        else:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_sport_date "
                    "ON predictions (sport, game_date)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_options_prediction "
                    "ON prediction_options (prediction_id)"
                )
            )


def insert_prediction(engine: Engine, prediction_data: Dict[str, Any]) -> int:
    """Insert a unified prediction row and return its integer prediction_id."""
    ensure_unified_schema(engine)

    created_at = prediction_data.get("created_at") or datetime.utcnow().isoformat()
    feature_snapshot = _serialize_feature_snapshot(prediction_data.get("feature_snapshot"))

    is_legacy = _predictions_is_legacy(engine)

    with engine.begin() as conn:
        if is_legacy:
            # Legacy NBA table: fill required NOT NULL columns with safe defaults.
            win_prob = float(prediction_data.get("win_probability") or 0.5)
            sql = """
                INSERT INTO predictions (
                    sport, league, game_date, home_team, away_team,
                    predicted_winner, confidence_level, feature_snapshot,
                    actual_home_score, actual_away_score, actual_winner, correct,
                    created_at,
                    predicted_spread, win_probability, confidence_score,
                    prediction_timestamp
                ) VALUES (
                    :sport, :league, :game_date, :home_team, :away_team,
                    :predicted_winner, :confidence_level, :feature_snapshot,
                    :actual_home_score, :actual_away_score, :actual_winner, :correct,
                    :created_at,
                    :predicted_spread, :win_probability, :confidence_score,
                    :prediction_timestamp
                )
            """
            params = {
                "sport": prediction_data["sport"],
                "league": prediction_data.get("league"),
                "game_date": prediction_data["game_date"],
                "home_team": prediction_data["home_team"],
                "away_team": prediction_data["away_team"],
                "predicted_winner": prediction_data["predicted_winner"],
                "confidence_level": prediction_data["confidence_level"],
                "feature_snapshot": feature_snapshot,
                "actual_home_score": prediction_data.get("actual_home_score"),
                "actual_away_score": prediction_data.get("actual_away_score"),
                "actual_winner": prediction_data.get("actual_winner"),
                "correct": prediction_data.get("correct"),
                "created_at": created_at,
                "predicted_spread": float(prediction_data.get("predicted_spread") or 0.0),
                "win_probability": win_prob,
                "confidence_score": float(prediction_data.get("confidence_score") or win_prob),
                "prediction_timestamp": created_at,
            }
        else:
            sql = """
                INSERT INTO predictions (
                    sport, league, game_date, home_team, away_team,
                    predicted_winner, confidence_level, feature_snapshot,
                    actual_home_score, actual_away_score, actual_winner, correct,
                    created_at
                ) VALUES (
                    :sport, :league, :game_date, :home_team, :away_team,
                    :predicted_winner, :confidence_level, :feature_snapshot,
                    :actual_home_score, :actual_away_score, :actual_winner, :correct,
                    :created_at
                )
            """
            params = {
                "sport": prediction_data["sport"],
                "league": prediction_data.get("league"),
                "game_date": prediction_data["game_date"],
                "home_team": prediction_data["home_team"],
                "away_team": prediction_data["away_team"],
                "predicted_winner": prediction_data["predicted_winner"],
                "confidence_level": prediction_data["confidence_level"],
                "feature_snapshot": feature_snapshot,
                "actual_home_score": prediction_data.get("actual_home_score"),
                "actual_away_score": prediction_data.get("actual_away_score"),
                "actual_winner": prediction_data.get("actual_winner"),
                "correct": prediction_data.get("correct"),
                "created_at": created_at,
            }

        if _is_postgresql(engine):
            result = conn.execute(text(sql + " RETURNING prediction_id"), params)
            row = result.fetchone()
            return int(row[0])

        conn.execute(text(sql), params)
        result = conn.execute(text("SELECT last_insert_rowid()"))
        return int(result.scalar())


def insert_prediction_options(
    engine: Engine,
    prediction_id: int,
    options: List[Dict[str, Any]],
) -> int:
    """Insert prediction options for a prediction. Returns number of rows inserted."""
    ensure_unified_schema(engine)

    if not options:
        return 0

    insert_sql = """
        INSERT INTO prediction_options (prediction_id, option_name, probability, rank)
        VALUES (:prediction_id, :option_name, :probability, :rank)
    """

    rows = []
    for idx, opt in enumerate(options, start=1):
        rows.append(
            {
                "prediction_id": prediction_id,
                "option_name": opt["option_name"],
                "probability": float(opt["probability"]),
                "rank": int(opt.get("rank") or idx),
            }
        )

    with engine.begin() as conn:
        conn.execute(text(insert_sql), rows)
    return len(rows)


def get_predictions_by_date(
    engine: Engine,
    start_date: str,
    end_date: str,
    sport: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return unified predictions within a date range, optionally filtered by sport."""
    if not _table_exists(engine, "predictions"):
        return []

    sql = """
        SELECT
            prediction_id, sport, league, game_date,
            home_team, away_team, predicted_winner, confidence_level,
            feature_snapshot, actual_home_score, actual_away_score,
            actual_winner, correct, created_at
        FROM predictions
        WHERE game_date >= :start_date AND game_date <= :end_date
    """
    params: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}

    if sport:
        sql += " AND sport = :sport"
        params["sport"] = sport

    sql += " ORDER BY game_date, prediction_id"

    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(row) for row in rows]


def get_prediction_options(engine: Engine, prediction_id: int) -> List[Dict[str, Any]]:
    """Return all options for a prediction ordered by rank."""
    if not _table_exists(engine, "prediction_options"):
        return []

    sql = """
        SELECT option_id, prediction_id, option_name, probability, rank
        FROM prediction_options
        WHERE prediction_id = :prediction_id
        ORDER BY rank, option_id
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"prediction_id": prediction_id}).mappings().all()
    return [dict(row) for row in rows]
