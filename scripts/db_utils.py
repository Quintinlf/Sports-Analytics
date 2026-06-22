from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


DEFAULT_SQLITE_URL = "sqlite:///./sports_analytics.db"

UNIFIED_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_game_id TEXT,
    game_signature TEXT,
    sport TEXT NOT NULL,
    league TEXT,
    game_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    win_probability REAL,
    confidence_level TEXT NOT NULL,
    bet_type TEXT,
    bet_units REAL,
    bet_recommendation TEXT,
    feature_snapshot TEXT,
    model_name TEXT,
    prediction_status TEXT,
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
    provider_game_id TEXT,
    game_signature TEXT,
    sport TEXT NOT NULL,
    league TEXT,
    game_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    win_probability DOUBLE PRECISION,
    confidence_level TEXT NOT NULL,
    bet_type TEXT,
    bet_units DOUBLE PRECISION,
    bet_recommendation TEXT,
    feature_snapshot TEXT,
    model_name TEXT,
    prediction_status TEXT,
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
    ("provider_game_id", "TEXT"),
    ("game_signature", "TEXT"),
    ("sport", "TEXT"),
    ("league", "TEXT"),
    ("win_probability", "REAL"),
    ("feature_snapshot", "TEXT"),
    ("bet_type", "TEXT"),
    ("bet_units", "REAL"),
    ("bet_recommendation", "TEXT"),
    ("model_name", "TEXT"),
    ("prediction_status", "TEXT"),
    ("actual_home_score", "INTEGER"),
    ("actual_away_score", "INTEGER"),
    ("actual_winner", "TEXT"),
    ("correct", "INTEGER"),
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


def _compute_game_signature(prediction_data: Dict[str, Any]) -> str:
    sport = str(prediction_data.get("sport", "")).upper().strip()
    game_date = str(prediction_data.get("game_date", "")).strip()
    home = str(prediction_data.get("home_team", "")).upper().strip()
    away = str(prediction_data.get("away_team", "")).upper().strip()
    raw = f"{sport}|{game_date}|{home}|{away}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_unified_schema(engine: Engine) -> None:
    """Create or migrate unified predictions and prediction_options tables."""
    is_pg = _is_postgresql(engine)

    with engine.begin() as conn:
        if not _table_exists(engine, "predictions"):
            ddl = UNIFIED_PREDICTIONS_DDL_PG if is_pg else UNIFIED_PREDICTIONS_DDL
            conn.execute(text(ddl))
        else:
            for col_name, col_type in UNIFIED_PREDICTION_COLUMNS:
                if not _column_exists(engine, "predictions", col_name):
                    conn.execute(
                        text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
                    )

            if _column_exists(engine, "predictions", "game_signature"):
                conn.execute(
                    text(
                        """
                        DELETE FROM predictions
                        WHERE prediction_id NOT IN (
                            SELECT MAX(prediction_id)
                            FROM predictions
                            GROUP BY sport, game_date, home_team, away_team
                        )
                        """
                    )
                )

                rows = conn.execute(
                    text(
                        """
                        SELECT prediction_id, sport, game_date, home_team, away_team
                        FROM predictions
                        WHERE game_signature IS NULL
                        """
                    )
                ).mappings().all()
                for row in rows:
                    sig = _compute_game_signature(dict(row))
                    conn.execute(
                        text(
                            "UPDATE predictions SET game_signature = :sig WHERE prediction_id = :pid"
                        ),
                        {"sig": sig, "pid": row["prediction_id"]},
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
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_game_signature "
                    "ON predictions (game_signature)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_provider_game_id "
                    "ON predictions (provider_game_id)"
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
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_game_signature "
                    "ON predictions (game_signature)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_provider_game_id "
                    "ON predictions (provider_game_id)"
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
    provider_game_id = prediction_data.get("provider_game_id")
    game_signature = prediction_data.get("game_signature") or _compute_game_signature(prediction_data)

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
            # Optional columns added via schema migration.
            optional_cols = []
            optional_vals = []
            for col_name, param_name in [
                ("bet_type", "bet_type"),
                ("bet_units", "bet_units"),
                ("bet_recommendation", "bet_recommendation"),
                ("model_name", "model_name"),
                ("prediction_status", "prediction_status"),
            ]:
                if _column_exists(engine, "predictions", col_name):
                    optional_cols.append(col_name)
                    optional_vals.append(f":{param_name}")
                    params[param_name] = prediction_data.get(param_name)

            if optional_cols:
                sql = sql.replace(
                    "prediction_timestamp\n                ) VALUES (",
                    "prediction_timestamp, " + ", ".join(optional_cols) + "\n                ) VALUES (",
                ).replace(
                    ":prediction_timestamp\n                )",
                    ":prediction_timestamp, " + ", ".join(optional_vals) + "\n                )",
                )
        else:
            params = {
                "provider_game_id": provider_game_id,
                "game_signature": game_signature,
                "sport": prediction_data["sport"],
                "league": prediction_data.get("league"),
                "game_date": prediction_data["game_date"],
                "home_team": prediction_data["home_team"],
                "away_team": prediction_data["away_team"],
                "predicted_winner": prediction_data["predicted_winner"],
                "win_probability": float(prediction_data.get("win_probability") or 0.5),
                "confidence_level": prediction_data["confidence_level"],
                "bet_type": prediction_data.get("bet_type"),
                "bet_units": prediction_data.get("bet_units"),
                "bet_recommendation": prediction_data.get("bet_recommendation"),
                "feature_snapshot": feature_snapshot,
                "model_name": prediction_data.get("model_name"),
                "prediction_status": prediction_data.get("prediction_status", "UPCOMING"),
                "actual_home_score": prediction_data.get("actual_home_score"),
                "actual_away_score": prediction_data.get("actual_away_score"),
                "actual_winner": prediction_data.get("actual_winner"),
                "correct": prediction_data.get("correct"),
                "created_at": created_at,
            }
            existing = None
            if provider_game_id:
                existing = conn.execute(
                    text(
                        "SELECT prediction_id FROM predictions "
                        "WHERE provider_game_id = :provider_game_id LIMIT 1"
                    ),
                    {"provider_game_id": provider_game_id},
                ).fetchone()
            if not existing:
                existing = conn.execute(
                    text(
                        "SELECT prediction_id FROM predictions "
                        "WHERE game_signature = :game_signature LIMIT 1"
                    ),
                    {"game_signature": game_signature},
                ).fetchone()

            if existing:
                prediction_id = int(existing[0])
                conn.execute(
                    text(
                        """
                        UPDATE predictions
                        SET provider_game_id = :provider_game_id,
                            game_signature = :game_signature,
                            sport = :sport,
                            league = :league,
                            game_date = :game_date,
                            home_team = :home_team,
                            away_team = :away_team,
                            predicted_winner = :predicted_winner,
                            win_probability = :win_probability,
                            confidence_level = :confidence_level,
                            bet_type = :bet_type,
                            bet_units = :bet_units,
                            bet_recommendation = :bet_recommendation,
                            feature_snapshot = :feature_snapshot,
                            model_name = :model_name,
                            prediction_status = :prediction_status,
                            actual_home_score = :actual_home_score,
                            actual_away_score = :actual_away_score,
                            actual_winner = :actual_winner,
                            correct = :correct
                        WHERE prediction_id = :prediction_id
                        """
                    ),
                    {**params, "prediction_id": prediction_id},
                )
                return prediction_id

            insert_sql = """
                INSERT INTO predictions (
                    provider_game_id, game_signature, sport, league, game_date, home_team, away_team,
                    predicted_winner, win_probability, confidence_level,
                    bet_type, bet_units, bet_recommendation,
                    feature_snapshot, model_name, prediction_status,
                    actual_home_score, actual_away_score, actual_winner, correct,
                    created_at
                ) VALUES (
                    :provider_game_id, :game_signature, :sport, :league, :game_date, :home_team, :away_team,
                    :predicted_winner, :win_probability, :confidence_level,
                    :bet_type, :bet_units, :bet_recommendation,
                    :feature_snapshot, :model_name, :prediction_status,
                    :actual_home_score, :actual_away_score, :actual_winner, :correct,
                    :created_at
                )
            """
            if _is_postgresql(engine):
                result = conn.execute(text(insert_sql + " RETURNING prediction_id"), params)
                row = result.fetchone()
                return int(row[0])
            conn.execute(text(insert_sql), params)
            result = conn.execute(text("SELECT last_insert_rowid()"))
            return int(result.scalar())

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
