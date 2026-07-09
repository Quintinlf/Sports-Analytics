from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

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


def normalize_database_url(url: str) -> str:
    """Normalize Render/Heroku postgres URLs for SQLAlchemy + psycopg2."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://") and "+psycopg2" not in url:
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def database_url_source() -> str:
    """Return which env var supplied the active database URL."""
    if os.getenv("SUPABASE_DATABASE_URL"):
        return "SUPABASE_DATABASE_URL"
    if os.getenv("SUPERBASE_DATABASE_URL"):
        return "SUPERBASE_DATABASE_URL"
    if os.getenv("DATABASE_URL"):
        return "DATABASE_URL"
    return "default"


def resolve_database_url(
    *,
    default: str | None = DEFAULT_SQLITE_URL,
    required: bool = False,
) -> str:
    """Prefer Supabase env vars, then DATABASE_URL, then optional default."""
    raw = (
        os.getenv("SUPABASE_DATABASE_URL")
        or os.getenv("SUPERBASE_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not raw:
        if required:
            raise RuntimeError(
                "Missing SUPABASE_DATABASE_URL, SUPERBASE_DATABASE_URL, or DATABASE_URL."
            )
        if default is not None:
            raw = default
        else:
            raise RuntimeError(
                "Missing SUPABASE_DATABASE_URL, SUPERBASE_DATABASE_URL, or DATABASE_URL."
            )
    return normalize_database_url(raw)


def format_database_target(url: str) -> str:
    """Return host and database name for logging (credentials masked)."""
    host, dbname = _parse_database_host_and_name(url)
    if url.startswith("sqlite"):
        return f"driver=sqlite path={dbname}"
    return f"host={host} dbname={dbname}"


def _parse_database_host_and_name(url: str) -> tuple[str, str]:
    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1] if "///" in url else url
        return "sqlite", path

    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    port = parsed.port
    dbname = unquote((parsed.path or "/").lstrip("/") or "postgres")
    host_part = f"{host}:{port}" if port else host
    return host_part, dbname


def log_startup_database_diagnostics(engine: Engine, database_url: str) -> None:
    """Log resolved DB target and prediction table snapshot (no credentials)."""
    import logging

    logger = logging.getLogger("startup.database")
    source = database_url_source()
    host, dbname = _parse_database_host_and_name(database_url)
    display_host = host.split(":")[0] if not database_url.startswith("sqlite") else host

    print("STARTUP DATABASE DIAGNOSTIC", flush=True)
    print(f"Database source: {source}", flush=True)
    print(f"Host: {display_host}", flush=True)
    print(f"DB: {dbname}", flush=True)

    logger.info("Database source: %s", source)
    logger.info("Host: %s", display_host)
    logger.info("DB: %s", dbname)

    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
            max_id = conn.execute(text("SELECT MAX(prediction_id) FROM predictions")).scalar()
            usa_row = conn.execute(
                text(
                    """
                    SELECT 1 FROM predictions
                    WHERE home_team = 'USA' AND away_team = 'Belgium'
                    LIMIT 1
                    """
                )
            ).first()
    except Exception as exc:
        print(f"Prediction count: N/A (query failed: {exc})", flush=True)
        print("Max prediction id: N/A", flush=True)
        print("USA vs Belgium found: N/A", flush=True)
        logger.info("Prediction count: N/A (query failed: %s)", exc)
        logger.info("Max prediction id: N/A")
        logger.info("USA vs Belgium found: N/A")
        return

    found = "YES" if usa_row else "NO"
    print(f"Prediction count: {count}", flush=True)
    print(f"Max prediction id: {max_id if max_id is not None else 'N/A'}", flush=True)
    print(f"USA vs Belgium found: {found}", flush=True)
    logger.info("Prediction count: %s", count)
    logger.info("Max prediction id: %s", max_id if max_id is not None else "N/A")
    logger.info("USA vs Belgium found: %s", found)


def get_database_url(default: str = DEFAULT_SQLITE_URL) -> str:
    """Return the configured database URL, falling back to local SQLite."""
    return resolve_database_url(default=default)


def create_database_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine that works for both Postgres and SQLite."""
    url = normalize_database_url(database_url or get_database_url())
    engine_kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **engine_kwargs)


def sql_bool_true(column: str, engine: Engine) -> str:
    """Dialect-safe SQL fragment for boolean true comparison."""
    if _is_postgresql(engine):
        return f"{column} IS TRUE"
    return f"{column} = 1"


def sql_case_bool_true(column: str, engine: Engine) -> str:
    """Dialect-safe CASE expression counting boolean true rows."""
    if _is_postgresql(engine):
        return f"CASE WHEN {column} IS TRUE THEN 1 ELSE 0 END"
    return f"CASE WHEN {column} = 1 THEN 1 ELSE 0 END"


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


DEFAULT_REVIEWER_ID = "quintin"
DEFAULT_REVIEWER_NAME = "Quintin"
DEFAULT_REVIEWER_EMAIL = "quintinlf7@gmail.com"

REVIEWER_PROFILE_COLUMNS = [
    ("first_name", "TEXT"),
    ("last_name", "TEXT"),
    ("bio", "TEXT"),
    ("analyst_role", "TEXT DEFAULT 'analyst'"),
    ("profile_public", "BOOLEAN DEFAULT 0"),
    ("onboarding_completed_at", "TIMESTAMP"),
]

TRUSTED_ANALYSTS = [
    {
        "reviewer_id": "lamar",
        "name": "Lamar",
        "first_name": "Lamar",
        "last_name": "",
        "bio": "Trusted sports logic analyst helping train prediction models.",
        "analyst_role": "trusted_analyst",
        "profile_public": True,
    },
    {
        "reviewer_id": "melissa",
        "name": "Melissa",
        "first_name": "Melissa",
        "last_name": "",
        "bio": "Trusted analyst focused on identifying what the model misses.",
        "analyst_role": "trusted_analyst",
        "profile_public": True,
    },
    {
        "reviewer_id": "alex",
        "name": "Alex",
        "first_name": "Alex",
        "last_name": "",
        "bio": "Trusted betting logic analyst contributing structured reasoning.",
        "analyst_role": "trusted_analyst",
        "profile_public": True,
    },
]


def _split_display_name(name: str) -> tuple[str, str]:
    parts = (name or "").strip().split(None, 1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _ensure_reviewer_profile_columns(conn, engine: Engine) -> set[str]:
    cols = {c["name"] for c in inspect(engine).get_columns("reviewers")}
    for col_name, ddl in REVIEWER_PROFILE_COLUMNS:
        if col_name not in cols:
            conn.execute(text(f"ALTER TABLE reviewers ADD COLUMN {col_name} {ddl}"))
            cols.add(col_name)
    return cols


def _backfill_reviewer_names(conn) -> None:
    rows = conn.execute(
        text(
            """
            SELECT reviewer_id, name, first_name
            FROM reviewers
            WHERE first_name IS NULL OR first_name = ''
            """
        )
    ).mappings().all()
    for row in rows:
        first, last = _split_display_name(row["name"] or "")
        conn.execute(
            text(
                """
                UPDATE reviewers
                SET first_name = :first_name,
                    last_name = :last_name
                WHERE reviewer_id = :rid
                """
            ),
            {"first_name": first, "last_name": last, "rid": row["reviewer_id"]},
        )


def _seed_trusted_analysts(conn, engine: Engine, ts: str) -> None:
    for analyst in TRUSTED_ANALYSTS:
        existing = conn.execute(
            text("SELECT reviewer_id FROM reviewers WHERE reviewer_id = :rid"),
            {"rid": analyst["reviewer_id"]},
        ).first()
        if existing:
            conn.execute(
                text(
                    """
                    UPDATE reviewers
                    SET name = :name,
                        first_name = :first_name,
                        last_name = :last_name,
                        bio = :bio,
                        analyst_role = :analyst_role,
                        profile_public = :profile_public
                    WHERE reviewer_id = :rid
                    """
                ),
                {
                    "rid": analyst["reviewer_id"],
                    "name": analyst["name"],
                    "first_name": analyst["first_name"],
                    "last_name": analyst["last_name"],
                    "bio": analyst["bio"],
                    "analyst_role": analyst["analyst_role"],
                    "profile_public": analyst["profile_public"],
                },
            )
        else:
            conn.execute(
                text(
                    """
                    INSERT INTO reviewers
                        (reviewer_id, name, email, first_name, last_name, bio,
                         analyst_role, profile_public, created_at)
                    VALUES
                        (:rid, :name, NULL, :first_name, :last_name, :bio,
                         :analyst_role, :profile_public, :ts)
                    """
                ),
                {
                    "rid": analyst["reviewer_id"],
                    "name": analyst["name"],
                    "first_name": analyst["first_name"],
                    "last_name": analyst["last_name"],
                    "bio": analyst["bio"],
                    "analyst_role": analyst["analyst_role"],
                    "profile_public": analyst["profile_public"],
                    "ts": ts,
                },
            )

        if _table_exists(engine, "reviewer_preferences"):
            conn.execute(
                text(
                    """
                    INSERT INTO reviewer_preferences
                        (reviewer_id, favorite_sports, emails_enabled, wants_betting_section,
                         wants_explanations, wants_postgame_reviews, email_frequency, updated_at)
                    VALUES
                        (:rid, :sports, 1, 1, 1, 1, 'weekly', :ts)
                    ON CONFLICT(reviewer_id) DO NOTHING
                    """
                ),
                {"rid": analyst["reviewer_id"], "sports": json.dumps(["MLB", "NBA"]), "ts": ts},
            )


def ensure_default_reviewers(engine: Engine) -> None:
    """Seed beta reviewer rows for automation paths that skip API startup."""
    import logging

    logger = logging.getLogger(__name__)
    if not _table_exists(engine, "reviewers"):
        logger.warning("reviewers table missing — skipping default reviewer seed")
        return

    ts = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        cols = _ensure_reviewer_profile_columns(conn, engine)
        if "email" not in cols:
            conn.execute(text("ALTER TABLE reviewers ADD COLUMN email TEXT"))

        existing = conn.execute(
            text(
                """
                SELECT reviewer_id
                FROM reviewers
                WHERE reviewer_id = :rid OR lower(name) = lower(:name)
                LIMIT 1
                """
            ),
            {"rid": DEFAULT_REVIEWER_ID, "name": DEFAULT_REVIEWER_NAME},
        ).first()

        if existing:
            conn.execute(
                text(
                    """
                    UPDATE reviewers
                    SET email = :email
                    WHERE reviewer_id = :rid
                      AND (email IS NULL OR email = 'quintin@example.com')
                    """
                ),
                {"rid": existing[0], "email": DEFAULT_REVIEWER_EMAIL},
            )
            reviewer_id = existing[0]
        else:
            first, last = _split_display_name(DEFAULT_REVIEWER_NAME)
            conn.execute(
                text(
                    """
                    INSERT INTO reviewers
                        (reviewer_id, name, email, first_name, last_name, analyst_role, profile_public, created_at)
                    VALUES (:rid, :name, :email, :first_name, :last_name, 'analyst', 0, :ts)
                    """
                ),
                {
                    "rid": DEFAULT_REVIEWER_ID,
                    "name": DEFAULT_REVIEWER_NAME,
                    "email": DEFAULT_REVIEWER_EMAIL,
                    "first_name": first,
                    "last_name": last,
                    "ts": ts,
                },
            )
            reviewer_id = DEFAULT_REVIEWER_ID

        if _table_exists(engine, "reviewer_preferences"):
            conn.execute(
                text(
                    """
                    INSERT INTO reviewer_preferences
                        (reviewer_id, favorite_sports, emails_enabled, wants_betting_section,
                         wants_explanations, wants_postgame_reviews, email_frequency, updated_at)
                    VALUES
                        (:rid, :sports, :emails_enabled, :wants_betting_section,
                         :wants_explanations, :wants_postgame_reviews, 'weekly', :ts)
                    ON CONFLICT(reviewer_id) DO NOTHING
                    """
                ),
                {
                    "rid": reviewer_id,
                    "sports": json.dumps(["MLB", "NBA"]),
                    "emails_enabled": True,
                    "wants_betting_section": True,
                    "wants_explanations": True,
                    "wants_postgame_reviews": True,
                    "ts": ts,
                },
            )

        _backfill_reviewer_names(conn)
        _seed_trusted_analysts(conn, engine, ts)

    logger.info("Default reviewer seed ensured (reviewer_id=%s)", reviewer_id)


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
