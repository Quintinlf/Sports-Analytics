"""AI Sports Analyst Feedback Platform — API router.

Endpoints under /api/feedback/  (all additive; existing /api/v1/* untouched).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import inspect as sa_inspect, text

from backend.db import engine, get_db_session
from backend.models import (
    Base,
    Reviewer,
    PredictionReview,
    ReviewOutcome,
    ReviewerPreference,
    ReviewerCustomSection,
)
from scripts.db_utils import (
    ensure_unified_schema,
    insert_prediction,
    sql_bool_true,
    sql_case_bool_true,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["feedback-platform"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPORT_MAP_UI_TO_DB = {"FIFA": "SOCCER"}
_SPORT_MAP_DB_TO_UI = {"SOCCER": "FIFA"}

LEAGUES: Dict[str, List[str]] = {
    "MLB":  ["MLB", "AL", "NL", "Spring Training"],
    "NBA":  ["NBA", "WNBA", "G League"],
    "FIFA": [
        "Premier League", "La Liga", "Bundesliga",
        "Serie A", "Ligue 1", "MLS", "Champions League", "World Cup",
    ],
}

MISSING_FACTORS: Dict[str, List[str]] = {
    "MLB":  ["Starting pitcher", "Bullpen fatigue", "Weather", "Park factors", "Injuries"],
    "NBA":  ["Injuries", "Rest days", "Back-to-back games", "Minutes restrictions", "Coaching"],
    "FIFA": ["Injuries", "Formation", "Possession", "Red cards", "Travel", "Home advantage"],
}

POSTGAME_FACTORS = [
    "Injuries", "Matchups", "Coaching", "Momentum",
    "Betting market", "Team chemistry", "Other",
]

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_today = date.today().isoformat()
_yesterday = (date.today() - timedelta(days=1)).isoformat()
_twodaysago = (date.today() - timedelta(days=2)).isoformat()
_threedaysago = (date.today() - timedelta(days=3)).isoformat()

SEED_PREDICTIONS: List[Dict[str, Any]] = [
    # --- MLB (5) ---
    {
        "sport": "MLB", "league": "AL",
        "game_date": _today,
        "home_team": "Yankees", "away_team": "Red Sox",
        "predicted_winner": "Yankees", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.83,
            "explanations": [
                {"label": "Starting pitcher", "weight": 0.83},
                {"label": "Recent form",       "weight": 0.62},
                {"label": "Home advantage",    "weight": 0.41},
                {"label": "Bullpen fatigue",   "weight": 0.22},
            ],
        }),
        "actual_home_score": None, "actual_away_score": None,
        "actual_winner": None, "correct": None,
    },
    {
        "sport": "MLB", "league": "NL",
        "game_date": _yesterday,
        "home_team": "Dodgers", "away_team": "Giants",
        "predicted_winner": "Dodgers", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.79,
            "explanations": [
                {"label": "Starting pitcher", "weight": 0.79},
                {"label": "Bullpen depth",    "weight": 0.65},
                {"label": "Park factors",     "weight": 0.38},
                {"label": "Injuries",         "weight": 0.15},
            ],
        }),
        "actual_home_score": 5, "actual_away_score": 3,
        "actual_winner": "Dodgers", "correct": 1,
    },
    {
        "sport": "MLB", "league": "AL",
        "game_date": _twodaysago,
        "home_team": "Astros", "away_team": "Mariners",
        "predicted_winner": "Astros", "confidence_level": "MEDIUM",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.61,
            "explanations": [
                {"label": "Starting pitcher", "weight": 0.61},
                {"label": "Home advantage",   "weight": 0.50},
                {"label": "Weather",          "weight": 0.30},
            ],
        }),
        "actual_home_score": 2, "actual_away_score": 4,
        "actual_winner": "Mariners", "correct": 0,
    },
    {
        "sport": "MLB", "league": "NL",
        "game_date": _today,
        "home_team": "Braves", "away_team": "Mets",
        "predicted_winner": "Braves", "confidence_level": "MEDIUM",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.60,
            "explanations": [
                {"label": "Recent form",    "weight": 0.60},
                {"label": "Home advantage", "weight": 0.52},
                {"label": "Bullpen fatigue","weight": 0.25},
            ],
        }),
        "actual_home_score": None, "actual_away_score": None,
        "actual_winner": None, "correct": None,
    },
    {
        "sport": "MLB", "league": "NL",
        "game_date": _threedaysago,
        "home_team": "Cubs", "away_team": "Cardinals",
        "predicted_winner": "Cardinals", "confidence_level": "LOW",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.42,
            "explanations": [
                {"label": "Away form",      "weight": 0.42},
                {"label": "Park factors",   "weight": 0.35},
            ],
        }),
        "actual_home_score": 6, "actual_away_score": 4,
        "actual_winner": "Cubs", "correct": 0,
    },
    # --- NBA (5) ---
    {
        "sport": "NBA", "league": "NBA",
        "game_date": _today,
        "home_team": "Lakers", "away_team": "Celtics",
        "predicted_winner": "Celtics", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.77,
            "explanations": [
                {"label": "Injuries",         "weight": 0.77},
                {"label": "Away form",        "weight": 0.64},
                {"label": "Rest days",        "weight": 0.45},
                {"label": "Back-to-back",     "weight": 0.20},
            ],
        }),
        "actual_home_score": None, "actual_away_score": None,
        "actual_winner": None, "correct": None,
    },
    {
        "sport": "NBA", "league": "NBA",
        "game_date": _yesterday,
        "home_team": "Warriors", "away_team": "Nuggets",
        "predicted_winner": "Warriors", "confidence_level": "MEDIUM",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.58,
            "explanations": [
                {"label": "Home advantage",   "weight": 0.58},
                {"label": "Pace differential","weight": 0.47},
                {"label": "Injuries",         "weight": 0.30},
            ],
        }),
        "actual_home_score": 110, "actual_away_score": 118,
        "actual_winner": "Nuggets", "correct": 0,
    },
    {
        "sport": "NBA", "league": "NBA",
        "game_date": _twodaysago,
        "home_team": "Bucks", "away_team": "Heat",
        "predicted_winner": "Bucks", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.81,
            "explanations": [
                {"label": "Injuries",          "weight": 0.81},
                {"label": "Rest days",         "weight": 0.60},
                {"label": "Minutes restrictions","weight": 0.35},
            ],
        }),
        "actual_home_score": 121, "actual_away_score": 109,
        "actual_winner": "Bucks", "correct": 1,
    },
    {
        "sport": "NBA", "league": "NBA",
        "game_date": _today,
        "home_team": "Knicks", "away_team": "76ers",
        "predicted_winner": "Knicks", "confidence_level": "LOW",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.44,
            "explanations": [
                {"label": "Home advantage",   "weight": 0.44},
                {"label": "Coaching",         "weight": 0.38},
            ],
        }),
        "actual_home_score": None, "actual_away_score": None,
        "actual_winner": None, "correct": None,
    },
    {
        "sport": "NBA", "league": "NBA",
        "game_date": _threedaysago,
        "home_team": "Suns", "away_team": "Mavericks",
        "predicted_winner": "Mavericks", "confidence_level": "MEDIUM",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.65,
            "explanations": [
                {"label": "Away form",          "weight": 0.65},
                {"label": "Back-to-back",       "weight": 0.50},
                {"label": "Minutes restrictions","weight": 0.28},
            ],
        }),
        "actual_home_score": 102, "actual_away_score": 99,
        "actual_winner": "Suns", "correct": 0,
    },
    # --- FIFA / SOCCER (5) ---
    {
        "sport": "SOCCER", "league": "Premier League",
        "game_date": _today,
        "home_team": "Man City", "away_team": "Arsenal",
        "predicted_winner": "Man City", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.80,
            "explanations": [
                {"label": "Home advantage",  "weight": 0.80},
                {"label": "Recent form",     "weight": 0.70},
                {"label": "Injuries",        "weight": 0.40},
                {"label": "Possession",      "weight": 0.35},
            ],
        }),
        "actual_home_score": None, "actual_away_score": None,
        "actual_winner": None, "correct": None,
    },
    {
        "sport": "SOCCER", "league": "La Liga",
        "game_date": _yesterday,
        "home_team": "Real Madrid", "away_team": "Barcelona",
        "predicted_winner": "Real Madrid", "confidence_level": "MEDIUM",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.57,
            "explanations": [
                {"label": "Home advantage",  "weight": 0.57},
                {"label": "Formation",       "weight": 0.48},
                {"label": "Red cards",       "weight": 0.20},
            ],
        }),
        "actual_home_score": 2, "actual_away_score": 2,
        "actual_winner": "Draw", "correct": 0,
    },
    {
        "sport": "SOCCER", "league": "Bundesliga",
        "game_date": _twodaysago,
        "home_team": "Bayern", "away_team": "Dortmund",
        "predicted_winner": "Bayern", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.76,
            "explanations": [
                {"label": "Home advantage",  "weight": 0.76},
                {"label": "Possession",      "weight": 0.65},
                {"label": "Injuries",        "weight": 0.30},
                {"label": "Travel",          "weight": 0.15},
            ],
        }),
        "actual_home_score": 3, "actual_away_score": 1,
        "actual_winner": "Bayern", "correct": 1,
    },
    {
        "sport": "SOCCER", "league": "Ligue 1",
        "game_date": _today,
        "home_team": "PSG", "away_team": "Marseille",
        "predicted_winner": "PSG", "confidence_level": "HIGH",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.84,
            "explanations": [
                {"label": "Recent form",     "weight": 0.84},
                {"label": "Home advantage",  "weight": 0.70},
                {"label": "Formation",       "weight": 0.45},
            ],
        }),
        "actual_home_score": None, "actual_away_score": None,
        "actual_winner": None, "correct": None,
    },
    {
        "sport": "SOCCER", "league": "Serie A",
        "game_date": _threedaysago,
        "home_team": "Inter", "away_team": "Milan",
        "predicted_winner": "Inter", "confidence_level": "MEDIUM",
        "feature_snapshot": json.dumps({
            "confidence_score": 0.63,
            "explanations": [
                {"label": "Home advantage",  "weight": 0.63},
                {"label": "Possession",      "weight": 0.55},
                {"label": "Red cards",       "weight": 0.25},
            ],
        }),
        "actual_home_score": 1, "actual_away_score": 2,
        "actual_winner": "Milan", "correct": 0,
    },
]

# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _ensure_seed_predictions(db_engine) -> None:
    """Insert demo predictions for MLB/NBA/FIFA when missing (idempotent by matchup)."""
    for row in SEED_PREDICTIONS:
        with db_engine.connect() as conn:
            exists = conn.execute(
                text("""
                    SELECT 1 FROM predictions
                    WHERE sport = :sport
                      AND home_team = :home
                      AND away_team = :away
                      AND game_date = :game_date
                    LIMIT 1
                """),
                {
                    "sport": row["sport"],
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "game_date": row["game_date"],
                },
            ).first()
        if not exists:
            insert_prediction(db_engine, row)


def init_platform(db_engine) -> None:
    """Initialize platform: create tables, run migrations, seed demo data if empty.

    Live prediction ingestion is handled separately by scripts/cron_daily_predictions.py
    so the web service does not require ML dependencies at startup.
    """
    logger.info("Initializing feedback platform...")
    
    Base.metadata.create_all(bind=db_engine)
    ensure_unified_schema(db_engine)

    # Migrate reviewers table: add email column if missing
    with db_engine.begin() as conn:
        cols = [c["name"] for c in sa_inspect(db_engine).get_columns("reviewers")]
        if "email" not in cols:
            conn.execute(text("ALTER TABLE reviewers ADD COLUMN email TEXT"))
        existing_quintin = conn.execute(
            text(
                """
                SELECT reviewer_id
                FROM reviewers
                WHERE reviewer_id = :rid OR lower(name) = lower(:name)
                LIMIT 1
                """
            ),
            {"rid": "quintin", "name": "Quintin"},
        ).first()
        if existing_quintin:
            conn.execute(
                text(
                    """
                    UPDATE reviewers
                    SET email = :email
                    WHERE reviewer_id = :rid
                      AND (email IS NULL OR email = 'quintin@example.com')
                    """
                ),
                {"rid": existing_quintin[0], "email": "quintinlf7@gmail.com"},
            )
        else:
            conn.execute(
                text(
                    """
                    INSERT INTO reviewers (reviewer_id, name, email, created_at)
                    VALUES (:rid, :name, :email, :ts)
                    """
                ),
                {
                    "rid": "quintin",
                    "name": "Quintin",
                    "email": "quintinlf7@gmail.com",
                    "ts": datetime.utcnow().isoformat(),
                },
            )
        outcome_cols = [c["name"] for c in sa_inspect(db_engine).get_columns("review_outcomes")]
        for ddl in [
            ("structured_explanation", "TEXT"),
            ("factor_tags", "TEXT"),
            ("should_be_feature", "BOOLEAN"),
            ("importance", "INTEGER"),
        ]:
            if ddl[0] not in outcome_cols:
                conn.execute(text(f"ALTER TABLE review_outcomes ADD COLUMN {ddl[0]} {ddl[1]}"))

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
                "rid": "quintin",
                "sports": json.dumps(["MLB", "NBA"]),
                "emails_enabled": True,
                "wants_betting_section": True,
                "wants_explanations": True,
                "wants_postgame_reviews": True,
                "ts": datetime.utcnow().isoformat(),
            },
        )

    # Seed demo predictions when the table is empty (live ingest runs via cron/Actions)
    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()

    if count == 0:
        logger.info("Predictions table empty — seeding demo data.")
        for row in SEED_PREDICTIONS:
            try:
                insert_prediction(db_engine, row)
            except Exception as e:
                logger.error(f"Failed to populate seed prediction: {e}")
    else:
        _ensure_seed_predictions(db_engine)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ReviewerRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    reviewer_id: Optional[str] = Field(default=None, max_length=100)
    email: Optional[str] = Field(default=None, max_length=200)


class PregameReviewRequest(BaseModel):
    prediction_id: int
    reviewer_id: str
    reviewer_pick: str
    reviewer_confidence: int = Field(..., ge=1, le=5)
    would_bet: str = Field(default="no")
    agree_with_model: bool = Field(default=True)
    missing_factors: List[str] = Field(default_factory=list)
    pregame_notes: Optional[str] = None


class PostgameOutcomeRequest(BaseModel):
    review_id: str
    followup_missing_factors: List[str] = Field(default_factory=list)
    followup_reason: Optional[str] = None
    structured_explanation: Optional[str] = None
    factor_tags: List[str] = Field(default_factory=list)
    should_be_feature: Optional[bool] = None
    importance: Optional[int] = Field(default=None, ge=1, le=5)


class ReviewerPreferenceRequest(BaseModel):
    favorite_sports: List[str] = Field(default_factory=list)
    emails_enabled: bool = True
    wants_betting_section: bool = True
    wants_explanations: bool = True
    wants_postgame_reviews: bool = True
    email_frequency: str = Field(default="weekly")


class CustomSectionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    content: str = Field(..., min_length=1, max_length=1000)
    active: bool = True


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_CONF_SCORE = {"LOW": 0.40, "MEDIUM": 0.60, "HIGH": 0.80}


def _db_sport(ui_sport: str) -> str:
    return _SPORT_MAP_UI_TO_DB.get(ui_sport.upper(), ui_sport.upper())


def _ui_sport(db_sport: Optional[str]) -> str:
    if not db_sport:
        return "Unknown"
    return _SPORT_MAP_DB_TO_UI.get(db_sport.upper(), db_sport.upper())


def _parse_snapshot(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _confidence_pct(snap: Dict[str, Any], conf_level: str) -> float:
    if "confidence_score" in snap:
        return float(snap["confidence_score"])
    return _CONF_SCORE.get(conf_level.upper(), 0.5)


def _explanations(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return [{label, weight}] explanations, normalized to 0–1."""
    if "explanations" in snap:
        return snap["explanations"]
    # Flat numeric dict fallback
    numeric = {k: v for k, v in snap.items()
               if isinstance(v, (int, float)) and k != "confidence_score"}
    if numeric:
        max_v = max(numeric.values()) or 1.0
        return [{"label": k.replace("_", " ").title(), "weight": round(v / max_v, 3)}
                for k, v in sorted(numeric.items(), key=lambda x: -x[1])]
    return []


def _reviewer_stats(session, reviewer_id: str) -> Dict[str, Any]:
    """Compute reviewer stats via raw SQL."""
    eng = session.get_bind()
    beat_true = sql_bool_true("ro.reviewer_beat_model", eng)
    agree_true = sql_bool_true("pr.agree_with_model", eng)
    reviewer_correct_true = sql_bool_true("ro.reviewer_correct", eng)
    case_correct = sql_case_bool_true("ro.reviewer_correct", eng)
    case_beat = sql_case_bool_true("ro.reviewer_beat_model", eng)

    total_sql = text("""
        SELECT COUNT(*) FROM prediction_reviews WHERE reviewer_id = :rid
    """)
    total = session.execute(total_sql, {"rid": reviewer_id}).scalar() or 0

    beat_sql = text(f"""
        SELECT COUNT(*) FROM review_outcomes ro
        JOIN prediction_reviews pr ON ro.review_id = pr.review_id
        WHERE pr.reviewer_id = :rid AND {beat_true}
    """)
    beat_count = session.execute(beat_sql, {"rid": reviewer_id}).scalar() or 0

    agree_sql = text(f"""
        SELECT COUNT(*) FROM prediction_reviews pr
        JOIN predictions p ON pr.prediction_id = p.prediction_id
        WHERE pr.reviewer_id = :rid AND {agree_true}
    """)
    agree_count = session.execute(agree_sql, {"rid": reviewer_id}).scalar() or 0
    agree_pct = round((agree_count / total * 100) if total else 0, 1)

    reviewer_correct_sql = text(f"""
        SELECT COUNT(*) FROM review_outcomes ro
        JOIN prediction_reviews pr ON ro.review_id = pr.review_id
        WHERE pr.reviewer_id = :rid AND {reviewer_correct_true}
    """)
    reviewer_correct = session.execute(reviewer_correct_sql, {"rid": reviewer_id}).scalar() or 0

    settled_sql = text("""
        SELECT COUNT(*) FROM review_outcomes ro
        JOIN prediction_reviews pr ON ro.review_id = pr.review_id
        WHERE pr.reviewer_id = :rid
    """)
    settled = session.execute(settled_sql, {"rid": reviewer_id}).scalar() or 0
    reviewer_acc = round((reviewer_correct / settled * 100) if settled else 0, 1)

    sport_sql = text(f"""
        SELECT p.sport, COUNT(*) as cnt,
               SUM({case_correct}) as correct,
               SUM({case_beat}) as beat
        FROM prediction_reviews pr
        JOIN predictions p ON pr.prediction_id = p.prediction_id
        LEFT JOIN review_outcomes ro ON pr.review_id = ro.review_id
        WHERE pr.reviewer_id = :rid
        GROUP BY p.sport
    """)
    sport_rows = session.execute(sport_sql, {"rid": reviewer_id}).mappings().all()
    by_sport = {}
    for row in sport_rows:
        label = _ui_sport(row["sport"])
        by_sport[label] = {
            "reviews": row["cnt"],
            "correct": row["correct"] or 0,
            "beat_ai": row["beat"] or 0,
        }

    return {
        "total_reviews": total,
        "agree_pct": agree_pct,
        "beat_ai": beat_count,
        "reviewer_accuracy": reviewer_acc,
        "by_sport": by_sport,
    }


def _reviewer_history(session, reviewer_id: str) -> List[Dict[str, Any]]:
    sql = text("""
        SELECT
            pr.review_id,
            pr.prediction_id,
            p.game_date,
            p.sport,
            p.home_team,
            p.away_team,
            p.predicted_winner  AS ai_pick,
            pr.reviewer_pick,
            p.actual_winner,
            ro.model_correct,
            ro.reviewer_correct,
            ro.reviewer_beat_model
        FROM prediction_reviews pr
        JOIN predictions p ON pr.prediction_id = p.prediction_id
        LEFT JOIN review_outcomes ro ON pr.review_id = ro.review_id
        WHERE pr.reviewer_id = :rid
        ORDER BY p.game_date DESC, pr.created_at DESC
    """)
    rows = session.execute(sql, {"rid": reviewer_id}).mappings().all()
    result = []
    for row in rows:
        sport_ui = _ui_sport(row["sport"])
        if row["model_correct"] is None:
            badge = "pending"
        elif row["reviewer_beat_model"]:
            badge = "beat_ai"
        elif row["model_correct"] and row["reviewer_correct"]:
            badge = "both_correct"
        elif row["model_correct"]:
            badge = "ai_right"
        elif row["reviewer_correct"]:
            badge = "reviewer_right"
        else:
            badge = "both_wrong"
        result.append({
            "review_id": row["review_id"],
            "prediction_id": row["prediction_id"],
            "game_date": row["game_date"],
            "sport": sport_ui,
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "ai_pick": row["ai_pick"],
            "reviewer_pick": row["reviewer_pick"],
            "actual_winner": row["actual_winner"],
            "badge": badge,
        })
    return result


def _resolve_reviewer(session, reviewer_ref: str) -> Optional[Dict[str, str]]:
    """Resolve reviewer by exact id first, then by case-insensitive name."""
    row = session.execute(
        text(
            """
            SELECT reviewer_id, name
            FROM reviewers
            WHERE reviewer_id = :ref
               OR lower(name) = lower(:ref)
            ORDER BY CASE WHEN reviewer_id = :ref THEN 0 ELSE 1 END
            LIMIT 1
            """
        ),
        {"ref": reviewer_ref},
    ).mappings().first()
    if not row:
        return None
    return {"reviewer_id": row["reviewer_id"], "name": row["name"]}


def _load_reviewer_preferences(session, reviewer_id: str) -> Dict[str, Any]:
    row = session.execute(
        text(
            """
            SELECT reviewer_id, favorite_sports, emails_enabled, wants_betting_section,
                   wants_explanations, wants_postgame_reviews, email_frequency
            FROM reviewer_preferences
            WHERE reviewer_id = :rid
            """
        ),
        {"rid": reviewer_id},
    ).mappings().first()
    if not row:
        return {
            "reviewer_id": reviewer_id,
            "favorite_sports": [],
            "emails_enabled": True,
            "wants_betting_section": True,
            "wants_explanations": True,
            "wants_postgame_reviews": True,
            "email_frequency": "weekly",
        }
    return {
        "reviewer_id": row["reviewer_id"],
        "favorite_sports": json.loads(row["favorite_sports"] or "[]"),
        "emails_enabled": bool(row["emails_enabled"]),
        "wants_betting_section": bool(row["wants_betting_section"]),
        "wants_explanations": bool(row["wants_explanations"]),
        "wants_postgame_reviews": bool(row["wants_postgame_reviews"]),
        "email_frequency": row["email_frequency"] or "weekly",
    }


def _load_custom_sections(session, reviewer_id: str) -> List[Dict[str, Any]]:
    eng = session.get_bind()
    active_true = sql_bool_true("active", eng)
    rows = session.execute(
        text(
            f"""
            SELECT section_id, reviewer_id, title, content, active
            FROM reviewer_custom_sections
            WHERE reviewer_id = :rid AND {active_true}
            ORDER BY created_at DESC
            """
        ),
        {"rid": reviewer_id},
    ).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/missing-factors/{sport}")
def get_missing_factors(sport: str) -> Dict[str, Any]:
    sport_ui = sport.upper()
    if sport_ui == "SOCCER":
        sport_ui = "FIFA"
    factors = MISSING_FACTORS.get(sport_ui, [])
    return {"sport": sport_ui, "factors": factors, "postgame_factors": POSTGAME_FACTORS}


@router.get("/leagues/{sport}")
def get_leagues(sport: str) -> Dict[str, Any]:
    sport_ui = sport.upper()
    if sport_ui == "SOCCER":
        sport_ui = "FIFA"
    return {"sport": sport_ui, "leagues": LEAGUES.get(sport_ui, [])}


@router.get("/predictions")
def list_predictions(sport: Optional[str] = None) -> List[Dict[str, Any]]:
    db_sport = _db_sport(sport) if sport else None
    with get_db_session() as session:
        if db_sport:
            rows = session.execute(
                text("SELECT prediction_id, sport, league, game_date, home_team, away_team, "
                     "predicted_winner, confidence_level, actual_home_score, actual_away_score, "
                     "actual_winner, correct, prediction_status FROM predictions WHERE sport = :s ORDER BY game_date DESC"),
                {"s": db_sport},
            ).mappings().all()
        else:
            rows = session.execute(
                text("SELECT prediction_id, sport, league, game_date, home_team, away_team, "
                     "predicted_winner, confidence_level, actual_home_score, actual_away_score, "
                     "actual_winner, correct, prediction_status FROM predictions "
                     "WHERE sport IS NOT NULL AND predicted_winner IS NOT NULL "
                     "ORDER BY game_date DESC"),
            ).mappings().all()
    result = []
    for r in rows:
        d = dict(r)
        d["sport_ui"] = _ui_sport(d["sport"])
        d["settled"] = d["actual_home_score"] is not None
        result.append(d)
    return result


@router.get("/debug/predictions")
def debug_predictions() -> Dict[str, Any]:
    """Return diagnostic counts and latest prediction rows by sport/status."""
    with get_db_session() as session:
        cols = [c["name"] for c in sa_inspect(engine).get_columns("predictions")]
        has_prediction_status = "prediction_status" in cols

        rows = session.execute(
            text("SELECT sport, COUNT(*) AS cnt FROM predictions GROUP BY sport ORDER BY sport")
        ).mappings().all()
        counts_by_sport = {"MLB": 0, "NBA": 0, "FIFA": 0, "SOCCER": 0}
        for row in rows:
            sport = row["sport"] or "UNKNOWN"
            counts_by_sport[sport] = int(row["cnt"] or 0)

        if has_prediction_status:
            status_rows = session.execute(
                text(
                    """
                    SELECT sport, prediction_status AS status, COUNT(*) AS cnt
                    FROM predictions
                    GROUP BY sport, prediction_status
                    """
                )
            ).mappings().all()
        else:
            status_rows = session.execute(
                text(
                    """
                    SELECT
                        sport,
                        CASE
                            WHEN actual_winner IS NOT NULL THEN 'FINAL'
                            ELSE 'UPCOMING'
                        END AS status,
                        COUNT(*) AS cnt
                    FROM predictions
                    GROUP BY sport, status
                    """
                )
            ).mappings().all()

        upcoming_counts: Dict[str, int] = {}
        final_counts: Dict[str, int] = {}
        for row in status_rows:
            sport = row["sport"] or "UNKNOWN"
            status = (row["status"] or "UNKNOWN").upper()
            cnt = int(row["cnt"] or 0)
            if status == "UPCOMING":
                upcoming_counts[sport] = cnt
            elif status == "FINAL":
                final_counts[sport] = cnt

        latest = session.execute(
            text(
                """
                SELECT prediction_id, sport, league, game_date, home_team, away_team,
                       predicted_winner, confidence_level, actual_home_score, actual_away_score,
                       actual_winner, correct,
                       COALESCE(
                           prediction_status,
                           CASE WHEN actual_winner IS NOT NULL THEN 'FINAL' ELSE 'UPCOMING' END
                       ) AS prediction_status,
                       created_at
                FROM predictions
                ORDER BY prediction_id DESC
                LIMIT 25
                """
            )
        ).mappings().all()

    latest_predictions = []
    for row in latest:
        d = dict(row)
        d["sport_ui"] = _ui_sport(d.get("sport"))
        latest_predictions.append(d)

    return {
        "counts_by_sport": counts_by_sport,
        "latest_predictions": latest_predictions,
        "upcoming_counts": upcoming_counts,
        "final_counts": final_counts,
    }


@router.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int) -> Dict[str, Any]:
    with get_db_session() as session:
        row = session.execute(
            text("SELECT * FROM predictions WHERE prediction_id = :pid"),
            {"pid": prediction_id},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")

        opts = session.execute(
            text("SELECT option_name, probability, rank FROM prediction_options "
                 "WHERE prediction_id = :pid ORDER BY rank"),
            {"pid": prediction_id},
        ).mappings().all()

    d = dict(row)
    snap = _parse_snapshot(d.get("feature_snapshot"))
    d["sport_ui"] = _ui_sport(d["sport"])
    d["settled"] = d["actual_home_score"] is not None
    d["confidence_pct"] = _confidence_pct(snap, d.get("confidence_level") or "LOW")
    d["explanations"] = _explanations(snap)
    d["metrics"] = snap.get("metrics", {}) if isinstance(snap, dict) else {}
    d["data_source"] = snap.get("data_source") if isinstance(snap, dict) else None
    d["is_fallback"] = bool(snap.get("is_fallback")) if isinstance(snap, dict) else False
    d["offseason_notice"] = (d["metrics"] or {}).get("offseason_notice")
    d["experimental_betting"] = {
        "predicted_winner": d.get("predicted_winner"),
        "win_probability": d.get("win_probability"),
        "confidence": d.get("confidence_level"),
        "disclaimer": (
            "Experimental betting signals. Feedback is being collected "
            "before official recommendations."
        ),
    }
    d["prediction_options"] = [dict(o) for o in opts]
    return d


@router.post("/reviewers")
def get_or_create_reviewer(payload: ReviewerRequest) -> Dict[str, Any]:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    email = (payload.email or "").strip() or None
    custom_id = (payload.reviewer_id or "").strip() or None

    with get_db_session() as session:
        # If a custom reviewer_id is provided, upsert by that ID
        if custom_id:
            session.execute(
                text("""
                    INSERT INTO reviewers (reviewer_id, name, email, created_at)
                    VALUES (:rid, :name, :email, :ts)
                    ON CONFLICT(reviewer_id) DO UPDATE SET
                        name  = excluded.name,
                        email = COALESCE(excluded.email, reviewers.email)
                """),
                {"rid": custom_id, "name": name, "email": email,
                 "ts": datetime.utcnow().isoformat()},
            )
            session.commit()
            reviewer_id = custom_id
            created = False  # upsert — may have been created or updated
        else:
            # Classic get-or-create by name
            existing = session.execute(
                text("SELECT reviewer_id FROM reviewers WHERE name = :n"),
                {"n": name},
            ).mappings().first()

            if existing:
                reviewer_id = existing["reviewer_id"]
                # Update email if now provided and not yet stored
                if email:
                    session.execute(
                        text("UPDATE reviewers SET email = :email "
                             "WHERE reviewer_id = :rid AND email IS NULL"),
                        {"email": email, "rid": reviewer_id},
                    )
                    session.commit()
                created = False
            else:
                reviewer_id = str(uuid.uuid4())
                session.execute(
                    text("INSERT INTO reviewers (reviewer_id, name, email, created_at) "
                         "VALUES (:rid, :name, :email, :ts)"),
                    {"rid": reviewer_id, "name": name, "email": email,
                     "ts": datetime.utcnow().isoformat()},
                )
                session.commit()
                created = True

        stats = _reviewer_stats(session, reviewer_id)
        history = _reviewer_history(session, reviewer_id)
        preferences = _load_reviewer_preferences(session, reviewer_id)
        custom_sections = _load_custom_sections(session, reviewer_id)

    return {
        "reviewer_id": reviewer_id,
        "name": name,
        "created": created,
        "stats": stats,
        "history": history,
        "preferences": preferences,
        "custom_sections": custom_sections,
    }


@router.get("/reviewers/{reviewer_id}/stats")
def reviewer_stats(reviewer_id: str) -> Dict[str, Any]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        rid = resolved["reviewer_id"]
        stats = _reviewer_stats(session, rid)
    return {"reviewer_id": rid, "name": resolved["name"], **stats}


@router.get("/reviewers/{reviewer_id}/history")
def reviewer_history(reviewer_id: str) -> List[Dict[str, Any]]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        return _reviewer_history(session, resolved["reviewer_id"])


@router.get("/reviewers/{reviewer_id}/preferences")
def get_reviewer_preferences(reviewer_id: str) -> Dict[str, Any]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        return _load_reviewer_preferences(session, resolved["reviewer_id"])


@router.put("/reviewers/{reviewer_id}/preferences")
def update_reviewer_preferences(reviewer_id: str, payload: ReviewerPreferenceRequest) -> Dict[str, Any]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        rid = resolved["reviewer_id"]
        session.execute(
            text(
                """
                INSERT INTO reviewer_preferences
                    (reviewer_id, favorite_sports, emails_enabled, wants_betting_section,
                     wants_explanations, wants_postgame_reviews, email_frequency, updated_at)
                VALUES
                    (:rid, :sports, :emails_enabled, :wants_betting_section,
                     :wants_explanations, :wants_postgame_reviews, :email_frequency, :ts)
                ON CONFLICT(reviewer_id) DO UPDATE SET
                    favorite_sports = excluded.favorite_sports,
                    emails_enabled = excluded.emails_enabled,
                    wants_betting_section = excluded.wants_betting_section,
                    wants_explanations = excluded.wants_explanations,
                    wants_postgame_reviews = excluded.wants_postgame_reviews,
                    email_frequency = excluded.email_frequency,
                    updated_at = excluded.updated_at
                """
            ),
            {
                "rid": rid,
                "sports": json.dumps(payload.favorite_sports),
                "emails_enabled": payload.emails_enabled,
                "wants_betting_section": payload.wants_betting_section,
                "wants_explanations": payload.wants_explanations,
                "wants_postgame_reviews": payload.wants_postgame_reviews,
                "email_frequency": payload.email_frequency,
                "ts": datetime.utcnow().isoformat(),
            },
        )
        session.commit()
        return _load_reviewer_preferences(session, rid)


@router.get("/reviewers/{reviewer_id}/custom-sections")
def list_custom_sections(reviewer_id: str) -> List[Dict[str, Any]]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        return _load_custom_sections(session, resolved["reviewer_id"])


@router.post("/reviewers/{reviewer_id}/custom-sections")
def create_custom_section(reviewer_id: str, payload: CustomSectionRequest) -> Dict[str, Any]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        section_id = str(uuid.uuid4())
        session.execute(
            text(
                """
                INSERT INTO reviewer_custom_sections
                    (section_id, reviewer_id, title, content, active, created_at)
                VALUES
                    (:sid, :rid, :title, :content, :active, :ts)
                """
            ),
            {
                "sid": section_id,
                "rid": resolved["reviewer_id"],
                "title": payload.title.strip(),
                "content": payload.content.strip(),
                "active": payload.active,
                "ts": datetime.utcnow().isoformat(),
            },
        )
        session.commit()
        return {
            "section_id": section_id,
            "reviewer_id": resolved["reviewer_id"],
            "title": payload.title.strip(),
            "content": payload.content.strip(),
            "active": payload.active,
        }


@router.post("/prediction-reviews")
def submit_pregame_review(payload: PregameReviewRequest) -> Dict[str, Any]:
    with get_db_session() as session:
        # Check reviewer exists
        rev = session.execute(
            text("SELECT reviewer_id FROM reviewers WHERE reviewer_id = :rid"),
            {"rid": payload.reviewer_id},
        ).mappings().first()
        if not rev:
            raise HTTPException(status_code=404, detail="Reviewer not found")

        # Check prediction exists
        pred = session.execute(
            text("SELECT prediction_id, predicted_winner FROM predictions WHERE prediction_id = :pid"),
            {"pid": payload.prediction_id},
        ).mappings().first()
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Check for existing review
        existing = session.execute(
            text("SELECT review_id FROM prediction_reviews "
                 "WHERE prediction_id = :pid AND reviewer_id = :rid"),
            {"pid": payload.prediction_id, "rid": payload.reviewer_id},
        ).mappings().first()
        if existing:
            raise HTTPException(status_code=409, detail="Review already submitted for this prediction")

        review_id = str(uuid.uuid4())
        agree_with_model = payload.agree_with_model
        # Auto-derive agree_with_model from pick vs predicted_winner if not explicit
        factors_json = json.dumps(payload.missing_factors)
        session.execute(
            text("""
                INSERT INTO prediction_reviews
                    (review_id, prediction_id, reviewer_id, reviewer_pick,
                     reviewer_confidence, would_bet, agree_with_model,
                     missing_factors, pregame_notes, created_at)
                VALUES
                    (:rid, :pid, :rvid, :pick,
                     :conf, :bet, :agree,
                     :factors, :notes, :ts)
            """),
            {
                "rid": review_id,
                "pid": payload.prediction_id,
                "rvid": payload.reviewer_id,
                "pick": payload.reviewer_pick,
                "conf": payload.reviewer_confidence,
                "bet": payload.would_bet,
                "agree": agree_with_model,
                "factors": factors_json,
                "notes": payload.pregame_notes,
                "ts": datetime.utcnow().isoformat(),
            },
        )
        session.commit()

    return {"review_id": review_id, "status": "saved"}


@router.post("/review-outcomes")
def submit_postgame_outcome(payload: PostgameOutcomeRequest) -> Dict[str, Any]:
    with get_db_session() as session:
        # Load pregame review
        review = session.execute(
            text("SELECT review_id, prediction_id, reviewer_id, reviewer_pick, agree_with_model "
                 "FROM prediction_reviews WHERE review_id = :rid"),
            {"rid": payload.review_id},
        ).mappings().first()
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")

        # Check outcome not already saved
        existing = session.execute(
            text("SELECT review_id FROM review_outcomes WHERE review_id = :rid"),
            {"rid": payload.review_id},
        ).mappings().first()
        if existing:
            raise HTTPException(status_code=409, detail="Outcome already submitted")

        # Load prediction actuals
        pred = session.execute(
            text("SELECT predicted_winner, actual_winner, correct "
                 "FROM predictions WHERE prediction_id = :pid"),
            {"pid": review["prediction_id"]},
        ).mappings().first()
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")
        if pred["actual_winner"] is None:
            raise HTTPException(status_code=400, detail="Game has not settled yet")

        model_correct = bool(pred["correct"]) if pred["correct"] is not None else (
            pred["predicted_winner"] == pred["actual_winner"]
        )
        reviewer_correct = (
            review["reviewer_pick"].strip().lower() == pred["actual_winner"].strip().lower()
        )
        reviewer_beat = (
            reviewer_correct
            and not model_correct
            and not bool(review["agree_with_model"])
        )

        structured_explanation = payload.structured_explanation
        factor_tags_json = json.dumps(payload.factor_tags or [])
        should_be_feature = payload.should_be_feature
        importance = payload.importance
        if not reviewer_beat:
            structured_explanation = None
            factor_tags_json = json.dumps([])
            should_be_feature = None
            importance = None

        session.execute(
            text("""
                INSERT INTO review_outcomes
                    (review_id, model_correct, reviewer_correct,
                     reviewer_beat_model, followup_missing_factors,
                     followup_reason, structured_explanation, factor_tags,
                     should_be_feature, importance, resolved_at)
                VALUES
                    (:rid, :mc, :rc, :beat, :factors, :reason, :structured_explanation,
                     :factor_tags, :should_be_feature, :importance, :ts)
            """),
            {
                "rid": payload.review_id,
                "mc": model_correct,
                "rc": reviewer_correct,
                "beat": reviewer_beat,
                "factors": json.dumps(payload.followup_missing_factors),
                "reason": payload.followup_reason,
                "structured_explanation": structured_explanation,
                "factor_tags": factor_tags_json,
                "should_be_feature": should_be_feature,
                "importance": importance,
                "ts": datetime.utcnow().isoformat(),
            },
        )
        session.commit()

    return {
        "review_id": payload.review_id,
        "model_correct": model_correct,
        "reviewer_correct": reviewer_correct,
        "reviewer_beat_model": reviewer_beat,
        "deep_analysis_unlocked": reviewer_beat,
        "status": "saved",
    }


@router.get("/pending-postgame")
def pending_postgame(reviewer_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return pregame reviews whose prediction has settled but no outcome has been filed.

    Optionally filtered by reviewer_id.  Returns enough context to render the
    postgame reflection form without an extra round-trip.
    """
    base_sql = """
        SELECT
            pr.review_id,
            pr.reviewer_id,
            pr.prediction_id,
            pr.reviewer_pick,
            pr.reviewer_confidence,
            pr.agree_with_model,
            pr.created_at  AS reviewed_at,
            p.sport,
            p.league,
            p.home_team,
            p.away_team,
            p.game_date,
            p.predicted_winner,
            p.actual_winner,
            p.actual_home_score,
            p.actual_away_score,
            p.correct        AS model_correct_flag
        FROM prediction_reviews pr
        JOIN predictions p ON pr.prediction_id = p.prediction_id
        LEFT JOIN review_outcomes ro ON pr.review_id = ro.review_id
        WHERE p.actual_winner IS NOT NULL
          AND ro.review_id IS NULL
        {reviewer_filter}
        ORDER BY p.game_date DESC, pr.created_at DESC
    """
    if reviewer_id:
        sql = text(base_sql.format(reviewer_filter="AND pr.reviewer_id = :rid"))
        params: Dict[str, Any] = {"rid": reviewer_id}
    else:
        sql = text(base_sql.format(reviewer_filter=""))
        params = {}

    with get_db_session() as session:
        rows = session.execute(sql, params).mappings().all()

    result = []
    for row in rows:
        d = dict(row)
        d["sport_ui"] = _ui_sport(d["sport"])
        d["matchup"] = f"{d['away_team']} @ {d['home_team']}"
        result.append(d)
    return result
