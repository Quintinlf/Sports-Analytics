"""AI Sports Analyst Feedback Platform — API router.

Endpoints under /api/feedback/  (all additive; existing /api/v1/* untouched).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
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
    AnalystQuestion,
    AnalystAnswer,
    AnalystCaseStudy,
    AnalystComment,
)
from scripts.db_utils import (
    ensure_default_reviewers,
    ensure_unified_schema,
    insert_prediction,
    schema_auto_migrate,
    sql_bool_true,
    sql_case_bool_true,
    _column_names,
    _invalidate_schema_cache,
    _split_display_name,
    _ensure_reviewer_profile_columns,
    _backfill_reviewer_names,
)
from data.demo_data import demo_predictions_enabled
from backend.analyst_challenge import (
    OVERRIDE_FOLLOWUP_PROMPT,
    compose_analyst_reasoning,
    evaluate_challenge,
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
        "Serie A", "Ligue 1", "MLS", "Champions League", "World Cup", "FIFA World Cup",
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

PRIMARY_DECISION_VARIABLES = [
    "starting_pitcher", "bullpen", "lineup", "injuries", "weather",
    "home_field", "recent_form", "travel", "coaching", "other",
]

KNOWLEDGE_AREAS = [
    "Analyst Profile",
    "Statistics",
    "Game Theory",
    "UX",  # Future: e.g. "What information would make you trust or distrust this prediction?"
    "MLB",
    "NBA",
    "FIFA",
    "Bullpens",
]

ONBOARDING_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question_id": "onboard-eval-games",
        "title": "How you evaluate games",
        "body_markdown": "This helps the platform understand your analyst style.",
        "prompts_json": json.dumps([
            {
                "prompt": "How do you normally evaluate games?",
                "placeholder": "e.g. I start with recent form and injuries, then compare my read to the model.",
                "example": "Mention what you check first, what you weigh most, and when you disagree with the AI.",
            }
        ]),
        "knowledge_area": "Analyst Profile",
        "sort_order": 1,
    },
    {
        "question_id": "onboard-trust-stats",
        "title": "Statistics you trust",
        "body_markdown": "Tell us which metrics you rely on when challenging the model.",
        "prompts_json": json.dumps([
            {
                "prompt": "What statistics do you trust?",
                "placeholder": "e.g. ELO, run differential, rest days, xG, defensive rating — and which you ignore.",
                "example": "Separate stats you use for pregame picks vs. postgame reflection.",
            }
        ]),
        "knowledge_area": "Statistics",
        "sort_order": 2,
    },
    {
        "question_id": "onboard-sports",
        "title": "Sports expertise",
        "body_markdown": "Your sport focus helps route the right predictions to you.",
        "prompts_json": json.dumps([
            {
                "prompt": "Which sports/leagues can you analyze confidently?",
                "placeholder": "e.g. MLB deeply (AL/NL), NBA casually, FIFA — Premier League and Champions League.",
                "example": "Note depth vs. casual watching; league-level detail helps us route the right games.",
            }
        ]),
        "knowledge_area": "Analyst Profile",
        "sort_order": 3,
    },
    {
        "question_id": "onboard-risk",
        "title": "Risk tolerance",
        "body_markdown": "Understanding your risk profile improves training data quality.",
        "prompts_json": json.dumps([
            {
                "prompt": "How much risk do you typically take?",
                "placeholder": "e.g. I avoid heavy underdogs unless there is a clear injury or matchup edge.",
                "example": "Describe when you would bet vs. pass, and how that differs from the model's picks.",
            }
        ]),
        "knowledge_area": "Analyst Profile",
        "sort_order": 4,
    },
]

RESEARCH_SEED_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question_id": "research-nash-equilibrium",
        "title": "Can modeling opponents' decisions improve sports prediction?",
        "body_markdown": (
            "A Nash equilibrium describes a situation where no participant can improve "
            "their outcome by changing their strategy alone.\n\n"
            "In sports:\n"
            "- A team chooses a strategy.\n"
            "- Opponents react.\n"
            "- Coaches, players, and managers adjust.\n\n"
            "Could modeling these interactions improve predictions?\n\n"
            "The equilibrium condition is:\n\n"
            "$$u_i(s_i^*, s_{-i}^*) \\ge u_i(s_i, s_{-i})$$\n\n"
            "**Variables:**\n"
            "- $u_i$: payoff\n"
            "- $s_i$: strategy\n"
            "- $s_{-i}$: opponents' strategies"
        ),
        "prompts_json": json.dumps([
            "How would you apply this across MLB, NBA, and FIFA?",
            "Which variables already exist in our data?",
            "Which variables are missing?",
            "Where could we collect them?",
        ]),
        "knowledge_area": "Game Theory",
        "sort_order": 1,
        "featured": True,
    },
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


def _welcome_name(row: Dict[str, Any]) -> str:
    first = (row.get("first_name") or "").strip()
    last = (row.get("last_name") or "").strip()
    if first and last:
        return f"{first} {last}"
    if first:
        return first
    return (row.get("name") or "").strip()


def _profile_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reviewer_id": row["reviewer_id"],
        "name": row["name"],
        "first_name": row.get("first_name") or "",
        "last_name": row.get("last_name") or "",
        "display_name": _welcome_name(row),
        "bio": row.get("bio"),
        "analyst_role": row.get("analyst_role") or "analyst",
        "profile_public": bool(row.get("profile_public")),
        "onboarding_completed_at": row.get("onboarding_completed_at"),
    }


def _normalize_prompts(raw: Any) -> List[Dict[str, Any]]:
    """Normalize prompts_json to structured objects for the frontend."""
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            normalized.append({"prompt": item, "placeholder": None, "example": None})
        elif isinstance(item, dict):
            normalized.append({
                "prompt": str(item.get("prompt") or ""),
                "placeholder": item.get("placeholder"),
                "example": item.get("example"),
            })
    return normalized


def _question_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    try:
        raw_prompts = json.loads(d.pop("prompts_json", None) or "[]")
    except json.JSONDecodeError:
        raw_prompts = []
    d["prompts"] = _normalize_prompts(raw_prompts)
    return d


def _require_admin(x_admin_key: Optional[str]) -> None:
    expected = os.getenv("ADMIN_API_KEY", "").strip()
    if not expected or (x_admin_key or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Admin API key required")


def _migrate_platform_columns(db_engine) -> None:
    with db_engine.begin() as conn:
        q_cols = _column_names(db_engine, "analyst_questions")
        for col_name, ddl in [
            ("knowledge_area", "TEXT"),
            ("featured", "BOOLEAN DEFAULT FALSE"),
        ]:
            if col_name not in q_cols:
                conn.execute(text(f"ALTER TABLE analyst_questions ADD COLUMN {col_name} {ddl}"))

        a_cols = _column_names(db_engine, "analyst_answers")
        if "knowledge_area" not in a_cols:
            conn.execute(text("ALTER TABLE analyst_answers ADD COLUMN knowledge_area TEXT"))

        pr_cols = _column_names(db_engine, "prediction_reviews")
        if "primary_decision_variable" not in pr_cols:
            conn.execute(text("ALTER TABLE prediction_reviews ADD COLUMN primary_decision_variable TEXT"))

        pref_cols = _column_names(db_engine, "reviewer_preferences")
        if "email_days" not in pref_cols:
            conn.execute(text("ALTER TABLE reviewer_preferences ADD COLUMN email_days TEXT"))


def _upsert_question(conn, question: Dict[str, Any], context: str, ts: str) -> None:
    """Upsert question metadata only — never touches analyst_answers or onboarding_completed_at."""
    conn.execute(
        text(
            """
            INSERT INTO analyst_questions
                (question_id, context, title, body_markdown, prompts_json,
                 knowledge_area, sort_order, active, featured, created_at)
            VALUES
                (:qid, :ctx, :title, :body, :prompts, :area, :sort_order, :active, :featured, :ts)
            ON CONFLICT(question_id) DO UPDATE SET
                title = excluded.title,
                body_markdown = excluded.body_markdown,
                prompts_json = excluded.prompts_json,
                knowledge_area = excluded.knowledge_area,
                sort_order = excluded.sort_order
            """
        ),
        {
            "qid": question["question_id"],
            "ctx": context,
            "title": question["title"],
            "body": question["body_markdown"],
            "prompts": question["prompts_json"],
            "area": question.get("knowledge_area"),
            "sort_order": question.get("sort_order", 0),
            "active": True,
            "featured": bool(question.get("featured")),
            "ts": ts,
        },
    )


def _seed_onboarding_questions(db_engine) -> None:
    ts = datetime.utcnow().isoformat()
    with db_engine.begin() as conn:
        for question in ONBOARDING_QUESTIONS:
            _upsert_question(conn, question, "onboarding", ts)


def _seed_research_questions(db_engine) -> None:
    """Refresh research question copy on deploy; metadata only, answers preserved."""
    ts = datetime.utcnow().isoformat()
    with db_engine.begin() as conn:
        for question in RESEARCH_SEED_QUESTIONS:
            _upsert_question(conn, question, "research", ts)


def init_platform(db_engine) -> None:
    """Initialize platform tables/seeds when SCHEMA_AUTO_MIGRATE allows it.

    On PostgreSQL production (default), schema DDL is skipped at web startup —
    apply schema via ``python -m scripts.init_database`` or SQL migrations.
    SQLite / SCHEMA_AUTO_MIGRATE=true keep local auto-migrate behavior.

    Live prediction ingestion is handled separately by scripts/cron_daily_predictions.py
    so the web service does not require ML dependencies at startup.
    """
    logger.info("Initializing feedback platform...")
    # #region agent log
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    _dbg_path = _Path(__file__).resolve().parents[2] / "debug-968447.log"

    def _dbg(hid: str, msg: str, data: dict | None = None) -> None:
        try:
            with open(_dbg_path, "a", encoding="utf-8") as fh:
                fh.write(
                    _json.dumps(
                        {
                            "sessionId": "968447",
                            "runId": "post-fix",
                            "hypothesisId": hid,
                            "location": "feedback.py:init_platform",
                            "message": msg,
                            "data": data or {},
                            "timestamp": int(_time.time() * 1000),
                        },
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            pass

    _t0 = _time.perf_counter()
    _auto = schema_auto_migrate(db_engine)
    _dbg(
        "B",
        "init_platform enter",
        {"dialect": db_engine.dialect.name, "auto_migrate": _auto},
    )
    # #endregion

    if _auto:
        Base.metadata.create_all(bind=db_engine)
        # #region agent log
        _dbg("B", "after create_all", {"ms": round((_time.perf_counter() - _t0) * 1000, 2)})
        # #endregion
        ensure_unified_schema(db_engine)
        # #region agent log
        _dbg(
            "B",
            "after ensure_unified_schema",
            {"ms": round((_time.perf_counter() - _t0) * 1000, 2)},
        )
        # #endregion

        with db_engine.begin() as conn:
            _ensure_reviewer_profile_columns(conn, db_engine)
            _backfill_reviewer_names(conn)

        ensure_default_reviewers(db_engine)
        # #region agent log
        _t_mig = _time.perf_counter()
        # #endregion
        _migrate_platform_columns(db_engine)
        # #region agent log
        _dbg(
            "B",
            "after _migrate_platform_columns",
            {"ms": round((_time.perf_counter() - _t_mig) * 1000, 2)},
        )
        # #endregion

        with db_engine.begin() as conn:
            # #region agent log
            _t_out = _time.perf_counter()
            # #endregion
            outcome_cols = _column_names(db_engine, "review_outcomes")
            # #region agent log
            _dbg(
                "B",
                "review_outcomes column lookup",
                {
                    "ms": round((_time.perf_counter() - _t_out) * 1000, 2),
                    "col_count": len(outcome_cols),
                },
            )
            # #endregion
            for ddl in [
                ("structured_explanation", "TEXT"),
                ("factor_tags", "TEXT"),
                ("should_be_feature", "BOOLEAN"),
                ("importance", "INTEGER"),
                ("final_result", "TEXT"),
            ]:
                if ddl[0] not in outcome_cols:
                    conn.execute(text(f"ALTER TABLE review_outcomes ADD COLUMN {ddl[0]} {ddl[1]}"))
            _invalidate_schema_cache(db_engine, "review_outcomes")

        _seed_onboarding_questions(db_engine)
        _seed_research_questions(db_engine)
    else:
        logger.info(
            "SCHEMA_AUTO_MIGRATE disabled for %s — skipping runtime schema DDL "
            "(use python -m scripts.init_database or migrations/).",
            db_engine.dialect.name,
        )
        # #region agent log
        _dbg("B", "skipped schema DDL (production path)", {})
        # #endregion

    # #region agent log
    _dbg(
        "B",
        "init_platform schema phase done",
        {
            "total_ms": round((_time.perf_counter() - _t0) * 1000, 2),
            "auto_migrate": _auto,
        },
    )
    # #endregion

    # Demo predictions are dev-only: gated behind ENABLE_DEMO_PREDICTIONS so a
    # production deploy never has synthetic Yankees/Lakers/Man City rows
    # mixed into the real predictions table it serves to the dashboard.
    if not demo_predictions_enabled():
        logger.info("ENABLE_DEMO_PREDICTIONS not set — skipping demo prediction seed.")
        return

    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()

    if count == 0:
        logger.info("Predictions table empty — seeding demo data (ENABLE_DEMO_PREDICTIONS=true).")
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
    primary_decision_variable: Optional[str] = None


class CaseStudyRequest(BaseModel):
    review_id: str
    reviewer_id: str
    ai_missed: str = Field(..., min_length=1)
    decision_factors: str = Field(..., min_length=1)
    missing_variables: str = Field(..., min_length=1)
    data_sources: str = Field(..., min_length=1)
    confidence_rating: int = Field(..., ge=1, le=5)


class ResearchAnswerItem(BaseModel):
    question_id: str
    answer: str = Field(..., min_length=1)
    knowledge_area: Optional[str] = None


class ResearchAnswersRequest(BaseModel):
    reviewer_id: str
    answers: List[ResearchAnswerItem] = Field(..., min_length=1)


class AdminQuestionRequest(BaseModel):
    question_id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=200)
    body_markdown: str = Field(..., min_length=1)
    prompts: List[str] = Field(default_factory=list)
    knowledge_area: Optional[str] = None
    featured: bool = False
    active: bool = True


class CommentRequest(BaseModel):
    reviewer_id: str
    target_type: str = Field(..., pattern="^(case_study|research_question)$")
    target_id: str
    body: str = Field(..., min_length=1)


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


class OnboardingAnswerItem(BaseModel):
    question_id: str = Field(..., min_length=1, max_length=64)
    answer: str = Field(..., min_length=1)


class OnboardingAnswersRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=1, max_length=100)
    answers: List[OnboardingAnswerItem] = Field(..., min_length=1)


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
    """Return [{label, weight, value}] for UI — prefer why_factors when present."""
    why = snap.get("why_factors") if isinstance(snap, dict) else None
    if isinstance(why, list) and why:
        return [
            {
                "label": f.get("label", "Factor"),
                "weight": float(f.get("strength", 0.0) or 0.0),
                "value": f.get("detail") or f.get("label"),
            }
            for f in why
            if isinstance(f, dict)
        ]
    if isinstance(snap, dict) and "explanations" in snap:
        # Legacy demo/NBA/FIFA rows often omit "value" (label+weight only).
        # Normalize so mobile clients that filter on value still render.
        out: List[Dict[str, Any]] = []
        for e in snap["explanations"] or []:
            if not isinstance(e, dict):
                continue
            label = e.get("label", "Factor")
            out.append(
                {
                    "label": label,
                    "weight": float(e.get("weight", 0.0) or 0.0),
                    "value": e.get("value") if e.get("value") is not None else label,
                }
            )
        return out
    # Flat numeric dict fallback
    numeric = {k: v for k, v in snap.items()
               if isinstance(v, (int, float)) and k != "confidence_score"}
    if numeric:
        max_v = max(numeric.values()) or 1.0
        return [{"label": k.replace("_", " ").title(), "weight": round(v / max_v, 3)}
                for k, v in sorted(numeric.items(), key=lambda x: -x[1])]
    return []


def _why_factors(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(snap, dict):
        return []
    why = snap.get("why_factors")
    if isinstance(why, list) and why:
        return why
    # Synthesize from legacy explanations for older rows
    return [
        {
            "label": e.get("label", "Factor"),
            "detail": e.get("value") or e.get("label"),
            "side": "neutral",
            "strength": float(e.get("weight", 0.0) or 0.0),
            "source_feature": None,
        }
        for e in _explanations(snap)
        if isinstance(e, dict)
    ]


def _risk_factors(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(snap, dict):
        return []
    risks = snap.get("risk_factors")
    return risks if isinstance(risks, list) else []


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
            SELECT reviewer_id, name, first_name, last_name, bio, analyst_role,
                   profile_public, onboarding_completed_at
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
    return _profile_from_row(dict(row))


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


_FIFA_PREDICTIONS_BASE_SQL = """
    SELECT prediction_id, sport, league, game_date, home_team, away_team,
           predicted_winner, confidence_level, actual_home_score, actual_away_score,
           actual_winner, correct, prediction_status, feature_snapshot,
           model_name, created_at, data_source, is_fallback
    FROM predictions
    WHERE sport IN ('SOCCER', 'FIFA')
"""

_PREDICTION_SELECT_COLS = """
    prediction_id, sport, league, game_date, home_team, away_team,
    predicted_winner, confidence_level, actual_home_score, actual_away_score,
    actual_winner, correct, prediction_status, feature_snapshot,
    model_name, created_at, data_source, is_fallback
"""


def _dashboard_pred_limits() -> Dict[str, int]:
    """Per-sport dashboard limits (env overrides)."""
    defaults = {"MLB": 6, "NBA": 6, "FIFA": 6}
    for key, default in defaults.items():
        raw = os.getenv(f"DASHBOARD_PRED_LIMIT_{key}", "").strip()
        if raw.isdigit():
            defaults[key] = max(1, int(raw))
    return defaults


def _prediction_priority_order_sql() -> str:
    """ORDER BY clause: confidence, data richness, upcoming — extensible for priority_score column."""
    conf_rank = """
        CASE UPPER(COALESCE(confidence_level, 'LOW'))
            WHEN 'HIGH' THEN 3
            WHEN 'MEDIUM' THEN 2
            ELSE 1
        END
    """
    upcoming_rank = """
        CASE
            WHEN COALESCE(prediction_status, '') = 'UPCOMING' THEN 1
            WHEN actual_winner IS NULL THEN 1
            ELSE 0
        END
    """
    data_rank = """
        CASE
            WHEN feature_snapshot IS NOT NULL
             AND LENGTH(TRIM(feature_snapshot)) > 80
            THEN 1
            ELSE 0
        END
    """
    # Weighted score until a dedicated priority_score column exists
    return f"""
        ({conf_rank}) * 10 + ({data_rank}) * 5 + ({upcoming_rank}) * 3 DESC,
        game_date DESC,
        prediction_id DESC
    """


def _fetch_predictions_for_sport(
    session,
    db_sport: str,
    limit: int,
) -> List[Dict[str, Any]]:
    order = _prediction_priority_order_sql()
    if db_sport == "SOCCER":
        sql = f"{_FIFA_PREDICTIONS_BASE_SQL} ORDER BY {order} LIMIT :limit"
        rows = session.execute(text(sql), {"limit": limit}).mappings().all()
    else:
        sql = f"""
            SELECT {_PREDICTION_SELECT_COLS}
            FROM predictions
            WHERE sport = :s
              AND predicted_winner IS NOT NULL
            ORDER BY {order}
            LIMIT :limit
        """
        rows = session.execute(text(sql), {"s": db_sport, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def _serialize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    snap = _parse_snapshot(d.pop("feature_snapshot", None))
    d["sport_ui"] = _ui_sport(d["sport"])
    d["settled"] = d.get("actual_home_score") is not None
    if d.get("game_date") is not None:
        d["game_date"] = str(d["game_date"])
    if d.get("created_at") is not None:
        d["created_at"] = str(d["created_at"])
    # Prefer first-class columns; fall back to snapshot for older rows.
    if d.get("data_source") is None and isinstance(snap, dict):
        d["data_source"] = snap.get("data_source")
    if d.get("is_fallback") is None and isinstance(snap, dict):
        d["is_fallback"] = bool(snap.get("is_fallback"))
    else:
        d["is_fallback"] = bool(d.get("is_fallback"))
    return d


@router.get("/knowledge-areas")
def list_knowledge_areas() -> Dict[str, Any]:
    return {"areas": KNOWLEDGE_AREAS}


@router.get("/predictions")
def list_predictions(
    sport: Optional[str] = None,
    limit: Optional[int] = None,
) -> JSONResponse:
    limits = _dashboard_pred_limits()
    db_sport = _db_sport(sport) if sport and sport.upper() != "ALL" else None
    sport_ui = (sport or "").upper()

    with get_db_session() as session:
        if db_sport:
            cap = limit if limit and limit > 0 else limits.get(sport_ui, limits.get(_ui_sport(db_sport), 6))
            rows = _fetch_predictions_for_sport(session, db_sport, cap)
        else:
            rows = []
            for ui_key, db_key in [("MLB", "MLB"), ("NBA", "NBA"), ("FIFA", "SOCCER")]:
                cap = limit if limit and limit > 0 else limits[ui_key]
                rows.extend(_fetch_predictions_for_sport(session, db_key, cap))

    result = [_serialize_prediction_row(r) for r in rows]
    return JSONResponse(content=result, headers={"Cache-Control": "no-store"})


@router.get("/debug/predictions")
def debug_predictions(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    """Return diagnostic counts, latest predictions, and recent pipeline runs."""
    _require_admin(x_admin_key)
    with get_db_session() as session:
        cols = _column_names(engine, "predictions")
        has_prediction_status = "prediction_status" in cols
        has_model_name = "model_name" in cols
        has_data_source = "data_source" in cols
        has_is_fallback = "is_fallback" in cols

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

        select_cols = [
            "prediction_id", "sport", "league", "game_date", "home_team", "away_team",
            "predicted_winner", "confidence_level", "actual_home_score", "actual_away_score",
            "actual_winner", "correct",
            """COALESCE(
                   prediction_status,
                   CASE WHEN actual_winner IS NOT NULL THEN 'FINAL' ELSE 'UPCOMING' END
               ) AS prediction_status""",
            "created_at",
        ]
        if has_model_name:
            select_cols.append("model_name")
        if has_data_source:
            select_cols.append("data_source")
        if has_is_fallback:
            select_cols.append("is_fallback")

        latest = session.execute(
            text(
                f"""
                SELECT {", ".join(select_cols)}
                FROM predictions
                ORDER BY prediction_id DESC
                LIMIT 25
                """
            )
        ).mappings().all()

        pipeline_runs: List[Dict[str, Any]] = []
        try:
            run_tables = sa_inspect(engine).get_table_names()
            if "pipeline_run_log" in run_tables:
                pipeline_runs = [
                    dict(r)
                    for r in session.execute(
                        text(
                            """
                            SELECT run_id, sport, status, error_message,
                                   predictions_count, run_at
                            FROM pipeline_run_log
                            ORDER BY run_id DESC
                            LIMIT 30
                            """
                        )
                    ).mappings().all()
                ]
        except Exception:
            pipeline_runs = []

    latest_predictions = []
    for row in latest:
        d = dict(row)
        d["sport_ui"] = _ui_sport(d.get("sport"))
        if d.get("is_fallback") is not None:
            d["is_fallback"] = bool(d["is_fallback"])
        latest_predictions.append(d)

    return {
        "counts_by_sport": counts_by_sport,
        "latest_predictions": latest_predictions,
        "upcoming_counts": upcoming_counts,
        "final_counts": final_counts,
        "pipeline_runs": pipeline_runs,
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
    d["why_factors"] = _why_factors(snap)
    d["risk_factors"] = _risk_factors(snap)
    d["metrics"] = snap.get("metrics", {}) if isinstance(snap, dict) else {}
    d["starting_pitchers"] = snap.get("starting_pitchers") if isinstance(snap, dict) else None
    d["bullpen"] = snap.get("bullpen") if isinstance(snap, dict) else None
    d["lineups"] = snap.get("lineups") if isinstance(snap, dict) else None
    d["missing_data_warnings"] = snap.get("missing_data_warnings", []) if isinstance(snap, dict) else []
    # Prefer first-class provenance columns; fall back to snapshot for older rows.
    col_source = d.get("data_source")
    d["data_source"] = col_source if col_source is not None else (
        snap.get("data_source") if isinstance(snap, dict) else None
    )
    if d.get("is_fallback") is not None:
        d["is_fallback"] = bool(d.get("is_fallback"))
    else:
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


def _normalize_reviewer_name(name: str) -> str:
    """Case-insensitive identity key: trim + collapse whitespace + lowercase."""
    return " ".join((name or "").strip().split()).lower()


def _normalize_reviewer_email(email: Optional[str]) -> Optional[str]:
    e = (email or "").strip().lower()
    return e or None


def _find_reviewer_by_email(session, email: str) -> Optional[str]:
    row = session.execute(
        text(
            """
            SELECT reviewer_id FROM reviewers
            WHERE email IS NOT NULL AND lower(email) = :email
            ORDER BY created_at ASC
            LIMIT 1
            """
        ),
        {"email": email},
    ).mappings().first()
    return row["reviewer_id"] if row else None


def _find_reviewer_by_normalized_name(session, normalized_name: str) -> Optional[str]:
    """Prefer the oldest row that already has an email (canonical account)."""
    row = session.execute(
        text(
            """
            SELECT reviewer_id FROM reviewers
            WHERE lower(trim(name)) = :n
            ORDER BY
              CASE
                WHEN email IS NOT NULL AND trim(email) != '' THEN 0
                ELSE 1
              END,
              created_at ASC
            LIMIT 1
            """
        ),
        {"n": normalized_name},
    ).mappings().first()
    return row["reviewer_id"] if row else None


@router.post("/reviewers")
def get_or_create_reviewer(payload: ReviewerRequest) -> Dict[str, Any]:
    name = " ".join((payload.name or "").strip().split())
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    email = _normalize_reviewer_email(payload.email)
    custom_id = (payload.reviewer_id or "").strip() or None
    normalized_name = _normalize_reviewer_name(name)
    first_name, last_name = _split_display_name(name)
    ts = datetime.utcnow().isoformat()

    with get_db_session() as session:
        reviewer_id: Optional[str] = None
        created = False

        # Hierarchy: email (canonical) → normalized name → create (or custom id).
        if email:
            reviewer_id = _find_reviewer_by_email(session, email)

        if reviewer_id is None:
            reviewer_id = _find_reviewer_by_normalized_name(session, normalized_name)

        if reviewer_id is None and custom_id:
            # Invite / deep-link upsert by stable id when no email/name match.
            session.execute(
                text("""
                    INSERT INTO reviewers
                        (reviewer_id, name, email, first_name, last_name, analyst_role, profile_public, created_at)
                    VALUES (:rid, :name, :email, :first_name, :last_name, 'analyst', :profile_public, :ts)
                    ON CONFLICT(reviewer_id) DO UPDATE SET
                        name  = excluded.name,
                        email = COALESCE(excluded.email, reviewers.email),
                        first_name = COALESCE(reviewers.first_name, excluded.first_name),
                        last_name = COALESCE(reviewers.last_name, excluded.last_name)
                """),
                {
                    "rid": custom_id, "name": name, "email": email,
                    "first_name": first_name, "last_name": last_name,
                    "profile_public": False, "ts": ts,
                },
            )
            session.commit()
            reviewer_id = custom_id
            created = False
        elif reviewer_id is None:
            reviewer_id = custom_id or str(uuid.uuid4())
            session.execute(
                text(
                    """
                    INSERT INTO reviewers
                        (reviewer_id, name, email, first_name, last_name, analyst_role, profile_public, created_at)
                    VALUES (:rid, :name, :email, :first_name, :last_name, 'analyst', :profile_public, :ts)
                    """
                ),
                {
                    "rid": reviewer_id, "name": name, "email": email,
                    "first_name": first_name, "last_name": last_name,
                    "profile_public": False, "ts": ts,
                },
            )
            session.commit()
            created = True
        else:
            # Attach email / refresh display name on the canonical row.
            session.execute(
                text(
                    """
                    UPDATE reviewers
                    SET name = COALESCE(NULLIF(name, ''), :name),
                        email = COALESCE(email, :email),
                        first_name = COALESCE(NULLIF(first_name, ''), :first_name),
                        last_name = COALESCE(NULLIF(last_name, ''), :last_name)
                    WHERE reviewer_id = :rid
                    """
                ),
                {
                    "rid": reviewer_id,
                    "name": name,
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                },
            )
            # If canonical row has no email yet, set normalized email.
            if email:
                session.execute(
                    text(
                        """
                        UPDATE reviewers
                        SET email = :email
                        WHERE reviewer_id = :rid
                          AND (email IS NULL OR trim(email) = '')
                        """
                    ),
                    {"rid": reviewer_id, "email": email},
                )
            session.commit()
            created = False

        profile = _resolve_reviewer(session, reviewer_id) or {}
        stats = _reviewer_stats(session, reviewer_id)
        history = _reviewer_history(session, reviewer_id)
        preferences = _load_reviewer_preferences(session, reviewer_id)
        custom_sections = _load_custom_sections(session, reviewer_id)
        display_name = profile.get("name") or name

    return {
        "reviewer_id": reviewer_id,
        "name": display_name,
        "created": created,
        "first_name": profile.get("first_name", first_name),
        "last_name": profile.get("last_name", last_name),
        "display_name": profile.get("display_name", name),
        "bio": profile.get("bio"),
        "analyst_role": profile.get("analyst_role", "analyst"),
        "profile_public": profile.get("profile_public", False),
        "onboarding_completed_at": profile.get("onboarding_completed_at"),
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
    return {**resolved, **stats}


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
        # Prefer explicit flag; if pick clearly differs from AI, treat as disagreement.
        pick_l = (payload.reviewer_pick or "").strip().lower()
        ai_l = (pred["predicted_winner"] or "").strip().lower()
        if agree_with_model and pick_l and ai_l and pick_l != ai_l:
            agree_with_model = False
        factors_json = json.dumps(payload.missing_factors)
        session.execute(
            text("""
                INSERT INTO prediction_reviews
                    (review_id, prediction_id, reviewer_id, reviewer_pick,
                     reviewer_confidence, would_bet, agree_with_model,
                     missing_factors, pregame_notes, primary_decision_variable, created_at)
                VALUES
                    (:rid, :pid, :rvid, :pick,
                     :conf, :bet, :agree,
                     :factors, :notes, :primary_var, :ts)
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
                "primary_var": payload.primary_decision_variable,
                "ts": datetime.utcnow().isoformat(),
            },
        )
        session.commit()

    reasoning = compose_analyst_reasoning(payload.missing_factors, payload.pregame_notes)
    return {
        "review_id": review_id,
        "status": "saved",
        "analyst_disagreed": not agree_with_model,
        "analyst_reasoning": reasoning,
    }


@router.post("/review-outcomes")
def submit_postgame_outcome(payload: PostgameOutcomeRequest) -> Dict[str, Any]:
    with get_db_session() as session:
        # Load pregame review
        review = session.execute(
            text(
                "SELECT review_id, prediction_id, reviewer_id, reviewer_pick, "
                "agree_with_model, missing_factors, pregame_notes "
                "FROM prediction_reviews WHERE review_id = :rid"
            ),
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

        reasoning = compose_analyst_reasoning(
            review.get("missing_factors"),
            review.get("pregame_notes"),
        )
        challenge = evaluate_challenge(
            agree_with_model=bool(review["agree_with_model"]),
            reviewer_pick=review["reviewer_pick"],
            predicted_winner=pred["predicted_winner"],
            actual_winner=pred["actual_winner"],
            model_correct_flag=pred["correct"],
            analyst_reasoning=reasoning,
        )
        model_correct = challenge["ai_was_correct"]
        reviewer_correct = challenge["analyst_was_correct"]
        reviewer_beat = challenge["successful_analyst_override"]

        # Successful override: collect what the model missed (reasoning data).
        followup_reason = (payload.followup_reason or "").strip() or None
        if reviewer_beat and not followup_reason:
            raise HTTPException(
                status_code=400,
                detail=OVERRIDE_FOLLOWUP_PROMPT,
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

        # Prefer final_result column when present (added for challenge tracking).
        outcome_cols = _column_names(engine, "review_outcomes")
        if "final_result" in outcome_cols:
            session.execute(
                text("""
                    INSERT INTO review_outcomes
                        (review_id, model_correct, reviewer_correct,
                         reviewer_beat_model, final_result, followup_missing_factors,
                         followup_reason, structured_explanation, factor_tags,
                         should_be_feature, importance, resolved_at)
                    VALUES
                        (:rid, :mc, :rc, :beat, :final_result, :factors, :reason,
                         :structured_explanation, :factor_tags, :should_be_feature,
                         :importance, :ts)
                """),
                {
                    "rid": payload.review_id,
                    "mc": model_correct,
                    "rc": reviewer_correct,
                    "beat": reviewer_beat,
                    "final_result": challenge["final_result"],
                    "factors": json.dumps(payload.followup_missing_factors),
                    "reason": followup_reason,
                    "structured_explanation": structured_explanation,
                    "factor_tags": factor_tags_json,
                    "should_be_feature": should_be_feature,
                    "importance": importance,
                    "ts": datetime.utcnow().isoformat(),
                },
            )
        else:
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
                    "reason": followup_reason,
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
        "analyst_disagreed": challenge["analyst_disagreed"],
        "analyst_reasoning": challenge["analyst_reasoning"],
        "final_result": challenge["final_result"],
        "analyst_was_correct": challenge["analyst_was_correct"],
        "ai_was_correct": challenge["ai_was_correct"],
        "successful_analyst_override": challenge["successful_analyst_override"],
        "override_followup_prompt": challenge["override_followup_prompt"],
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


@router.get("/analysts")
def list_public_analysts() -> List[Dict[str, Any]]:
    eng = engine
    public_true = sql_bool_true("profile_public", eng)
    with get_db_session() as session:
        rows = session.execute(
            text(
                f"""
                SELECT reviewer_id, name, first_name, last_name, bio, analyst_role,
                       profile_public, onboarding_completed_at
                FROM reviewers
                WHERE {public_true}
                ORDER BY name ASC
                """
            )
        ).mappings().all()
    return [_profile_from_row(dict(r)) for r in rows]


@router.get("/decision-variables")
def list_decision_variables() -> Dict[str, Any]:
    return {"variables": PRIMARY_DECISION_VARIABLES}


@router.get("/analysts/{reviewer_id}/profile")
def public_analyst_profile(reviewer_id: str) -> Dict[str, Any]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Analyst not found")
        if not resolved.get("profile_public"):
            raise HTTPException(status_code=404, detail="Profile is not public")
        rid = resolved["reviewer_id"]
        stats = _reviewer_stats(session, rid)
        eng = session.get_bind()
        published_true = sql_bool_true("cs.published", eng)
        case_rows = session.execute(
            text(
                f"""
                SELECT cs.case_id, cs.review_id, cs.prediction_id, cs.ai_missed,
                       cs.decision_factors, cs.missing_variables, cs.data_sources,
                       cs.confidence_rating, cs.created_at,
                       p.home_team, p.away_team, p.game_date, p.sport
                FROM analyst_case_studies cs
                JOIN predictions p ON p.prediction_id = cs.prediction_id
                WHERE cs.reviewer_id = :rid AND {published_true}
                ORDER BY cs.created_at DESC
                LIMIT 20
                """
            ),
            {"rid": rid},
        ).mappings().all()
    public = {k: v for k, v in resolved.items() if k != "onboarding_completed_at"}
    cases = []
    for row in case_rows:
        d = dict(row)
        d["matchup"] = f"{d['away_team']} @ {d['home_team']}"
        cases.append(d)
    return {**public, "stats": stats, "case_studies": cases}


@router.get("/onboarding/questions")
def list_onboarding_questions() -> List[Dict[str, Any]]:
    eng = engine
    active_true = sql_bool_true("active", eng)
    with get_db_session() as session:
        rows = session.execute(
            text(
                f"""
                SELECT question_id, context, title, body_markdown, prompts_json,
                       knowledge_area, sort_order
                FROM analyst_questions
                WHERE context = 'onboarding' AND {active_true}
                ORDER BY sort_order ASC, created_at ASC
                """
            )
        ).mappings().all()
    result = []
    for row in rows:
        result.append(_question_row_to_dict(dict(row)))
    return result


@router.get("/onboarding/status")
def onboarding_status(reviewer_id: str) -> Dict[str, Any]:
    eng = engine
    active_true = sql_bool_true("active", eng)
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        rid = resolved["reviewer_id"]
        if resolved.get("onboarding_completed_at"):
            return {
                "reviewer_id": rid,
                "completed": True,
                "unanswered_count": 0,
                "total_questions": 0,
            }
        total = session.execute(
            text(
                f"""
                SELECT COUNT(*) FROM analyst_questions
                WHERE context = 'onboarding' AND {active_true}
                """
            )
        ).scalar() or 0
        answered = session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analyst_answers aa
                JOIN analyst_questions aq ON aq.question_id = aa.question_id
                WHERE aa.reviewer_id = :rid
                  AND aq.context = 'onboarding'
                """
            ),
            {"rid": rid},
        ).scalar() or 0
    return {
        "reviewer_id": rid,
        "completed": total > 0 and answered >= total,
        "unanswered_count": max(0, total - answered),
        "total_questions": total,
    }


@router.post("/onboarding/answers")
def submit_onboarding_answers(payload: OnboardingAnswersRequest) -> Dict[str, Any]:
    ts = datetime.utcnow().isoformat()
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, payload.reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        rid = resolved["reviewer_id"]

        for item in payload.answers:
            question = session.execute(
                text(
                    """
                    SELECT question_id, knowledge_area FROM analyst_questions
                    WHERE question_id = :qid AND context = 'onboarding'
                    """
                ),
                {"qid": item.question_id},
            ).mappings().first()
            if not question:
                raise HTTPException(status_code=404, detail=f"Question not found: {item.question_id}")

            area = question["knowledge_area"]
            existing = session.execute(
                text(
                    """
                    SELECT answer_id FROM analyst_answers
                    WHERE question_id = :qid AND reviewer_id = :rid
                    """
                ),
                {"qid": item.question_id, "rid": rid},
            ).first()
            if existing:
                session.execute(
                    text(
                        """
                        UPDATE analyst_answers
                        SET answer = :answer, knowledge_area = :area, created_at = :ts
                        WHERE question_id = :qid AND reviewer_id = :rid
                        """
                    ),
                    {"answer": item.answer.strip(), "area": area, "ts": ts, "qid": item.question_id, "rid": rid},
                )
            else:
                session.execute(
                    text(
                        """
                        INSERT INTO analyst_answers
                            (answer_id, question_id, reviewer_id, answer, knowledge_area, created_at)
                        VALUES (:aid, :qid, :rid, :answer, :area, :ts)
                        """
                    ),
                    {
                        "aid": str(uuid.uuid4()),
                        "qid": item.question_id,
                        "rid": rid,
                        "answer": item.answer.strip(),
                        "area": area,
                        "ts": ts,
                    },
                )

        active_true = sql_bool_true("active", session.get_bind())
        total = session.execute(
            text(
                f"""
                SELECT COUNT(*) FROM analyst_questions
                WHERE context = 'onboarding' AND {active_true}
                """
            )
        ).scalar() or 0
        answered = session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analyst_answers aa
                JOIN analyst_questions aq ON aq.question_id = aa.question_id
                WHERE aa.reviewer_id = :rid AND aq.context = 'onboarding'
                """
            ),
            {"rid": rid},
        ).scalar() or 0

        completed = total > 0 and answered >= total
        if completed:
            session.execute(
                text(
                    """
                    UPDATE reviewers
                    SET onboarding_completed_at = :ts
                    WHERE reviewer_id = :rid
                    """
                ),
                {"ts": ts, "rid": rid},
            )
        session.commit()

    return {
        "reviewer_id": rid,
        "status": "saved",
        "completed": completed,
        "answered_count": answered,
        "total_questions": total,
    }


@router.get("/research/current")
def current_research_question(reviewer_id: Optional[str] = None) -> Dict[str, Any]:
    eng = engine
    active_true = sql_bool_true("active", eng)
    featured_true = sql_bool_true("featured", eng)
    with get_db_session() as session:
        row = session.execute(
            text(
                f"""
                SELECT question_id, title, body_markdown, prompts_json, knowledge_area, sort_order
                FROM analyst_questions
                WHERE context = 'research' AND {active_true} AND {featured_true}
                ORDER BY sort_order ASC, created_at DESC
                LIMIT 1
                """
            )
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="No featured research question")
        question = _question_row_to_dict(dict(row))
        if reviewer_id:
            ans = session.execute(
                text(
                    """
                    SELECT answer, knowledge_area, created_at
                    FROM analyst_answers
                    WHERE question_id = :qid AND reviewer_id = :rid
                    """
                ),
                {"qid": question["question_id"], "rid": reviewer_id},
            ).mappings().first()
            question["existing_answer"] = dict(ans) if ans else None
    return question


@router.get("/research/questions")
def list_research_questions() -> List[Dict[str, Any]]:
    eng = engine
    active_true = sql_bool_true("active", eng)
    with get_db_session() as session:
        rows = session.execute(
            text(
                f"""
                SELECT question_id, title, body_markdown, prompts_json, knowledge_area,
                       sort_order, featured
                FROM analyst_questions
                WHERE context = 'research' AND {active_true}
                ORDER BY featured DESC, sort_order ASC, created_at DESC
                """
            )
        ).mappings().all()
    return [_question_row_to_dict(dict(r)) for r in rows]


@router.get("/research/answers")
def list_research_answers(
    knowledge_area: Optional[str] = None,
    question_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses = ["aq.context = 'research'"]
    params: Dict[str, Any] = {}
    if knowledge_area:
        clauses.append("aa.knowledge_area = :area")
        params["area"] = knowledge_area
    if question_id:
        clauses.append("aa.question_id = :qid")
        params["qid"] = question_id
    where = " AND ".join(clauses)
    with get_db_session() as session:
        rows = session.execute(
            text(
                f"""
                SELECT aa.answer_id, aa.question_id, aa.reviewer_id, aa.answer,
                       aa.knowledge_area, aa.created_at,
                       aq.title AS question_title, r.name AS reviewer_name
                FROM analyst_answers aa
                JOIN analyst_questions aq ON aq.question_id = aa.question_id
                JOIN reviewers r ON r.reviewer_id = aa.reviewer_id
                WHERE {where}
                ORDER BY aa.created_at DESC
                LIMIT 100
                """
            ),
            params,
        ).mappings().all()
    return [dict(r) for r in rows]


@router.post("/research/answers")
def submit_research_answers(payload: ResearchAnswersRequest) -> Dict[str, Any]:
    ts = datetime.utcnow().isoformat()
    saved = 0
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, payload.reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        rid = resolved["reviewer_id"]
        for item in payload.answers:
            question = session.execute(
                text(
                    """
                    SELECT question_id, knowledge_area FROM analyst_questions
                    WHERE question_id = :qid AND context = 'research'
                    """
                ),
                {"qid": item.question_id},
            ).mappings().first()
            if not question:
                raise HTTPException(status_code=404, detail=f"Question not found: {item.question_id}")
            area = item.knowledge_area or question["knowledge_area"]
            existing = session.execute(
                text(
                    "SELECT answer_id FROM analyst_answers WHERE question_id = :qid AND reviewer_id = :rid"
                ),
                {"qid": item.question_id, "rid": rid},
            ).first()
            if existing:
                session.execute(
                    text(
                        """
                        UPDATE analyst_answers
                        SET answer = :answer, knowledge_area = :area, created_at = :ts
                        WHERE question_id = :qid AND reviewer_id = :rid
                        """
                    ),
                    {"answer": item.answer.strip(), "area": area, "ts": ts, "qid": item.question_id, "rid": rid},
                )
            else:
                session.execute(
                    text(
                        """
                        INSERT INTO analyst_answers
                            (answer_id, question_id, reviewer_id, answer, knowledge_area, created_at)
                        VALUES (:aid, :qid, :rid, :answer, :area, :ts)
                        """
                    ),
                    {
                        "aid": str(uuid.uuid4()),
                        "qid": item.question_id,
                        "rid": rid,
                        "answer": item.answer.strip(),
                        "area": area,
                        "ts": ts,
                    },
                )
            saved += 1
        session.commit()
    return {"reviewer_id": rid, "status": "saved", "saved_count": saved}


@router.post("/admin/questions")
def admin_create_question(
    payload: AdminQuestionRequest,
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    _require_admin(x_admin_key)
    qid = (payload.question_id or "").strip() or f"research-{uuid.uuid4().hex[:12]}"
    ts = datetime.utcnow().isoformat()
    with get_db_session() as session:
        if payload.featured:
            session.execute(
                text(
                    """
                    UPDATE analyst_questions
                    SET featured = :featured
                    WHERE context = 'research'
                    """
                ),
                {"featured": False},
            )
        session.execute(
            text(
                """
                INSERT INTO analyst_questions
                    (question_id, context, title, body_markdown, prompts_json,
                     knowledge_area, sort_order, active, featured, created_at)
                VALUES
                    (:qid, 'research', :title, :body, :prompts, :area, :sort_order, :active, :featured, :ts)
                ON CONFLICT(question_id) DO UPDATE SET
                    title = excluded.title,
                    body_markdown = excluded.body_markdown,
                    prompts_json = excluded.prompts_json,
                    knowledge_area = excluded.knowledge_area,
                    active = excluded.active,
                    featured = excluded.featured
                """
            ),
            {
                "qid": qid,
                "title": payload.title,
                "body": payload.body_markdown,
                "prompts": json.dumps(payload.prompts),
                "area": payload.knowledge_area,
                "sort_order": 0,
                "active": payload.active,
                "featured": payload.featured,
                "ts": ts,
            },
        )
        session.commit()
    return {"question_id": qid, "status": "saved"}


@router.get("/case-studies/pending")
def pending_case_studies(reviewer_id: str) -> List[Dict[str, Any]]:
    eng = engine
    beat_true = sql_bool_true("ro.reviewer_beat_model", eng)
    with get_db_session() as session:
        rows = session.execute(
            text(
                f"""
                SELECT pr.review_id, pr.prediction_id, pr.reviewer_pick,
                       p.home_team, p.away_team, p.game_date, p.sport,
                       p.predicted_winner, p.actual_winner
                FROM prediction_reviews pr
                JOIN review_outcomes ro ON ro.review_id = pr.review_id
                JOIN predictions p ON p.prediction_id = pr.prediction_id
                LEFT JOIN analyst_case_studies cs ON cs.review_id = pr.review_id
                WHERE pr.reviewer_id = :rid
                  AND {beat_true}
                  AND cs.case_id IS NULL
                ORDER BY ro.resolved_at DESC
                """
            ),
            {"rid": reviewer_id},
        ).mappings().all()
    result = []
    for row in rows:
        d = dict(row)
        d["matchup"] = f"{d['away_team']} @ {d['home_team']}"
        d["sport_ui"] = _ui_sport(d["sport"])
        result.append(d)
    return result


@router.post("/case-studies")
def submit_case_study(payload: CaseStudyRequest) -> Dict[str, Any]:
    ts = datetime.utcnow().isoformat()
    with get_db_session() as session:
        review = session.execute(
            text(
                """
                SELECT pr.review_id, pr.prediction_id, pr.reviewer_id
                FROM prediction_reviews pr
                JOIN review_outcomes ro ON ro.review_id = pr.review_id
                WHERE pr.review_id = :rid AND pr.reviewer_id = :rvid
                """
            ),
            {"rid": payload.review_id, "rvid": payload.reviewer_id},
        ).mappings().first()
        if not review:
            raise HTTPException(status_code=404, detail="Beat-AI review not found")
        eng = session.get_bind()
        beat_true = sql_bool_true("reviewer_beat_model", eng)
        beat = session.execute(
            text(f"SELECT reviewer_beat_model FROM review_outcomes WHERE review_id = :rid AND {beat_true}"),
            {"rid": payload.review_id},
        ).mappings().first()
        if not beat:
            raise HTTPException(status_code=400, detail="Case study only required when analyst beat the model")

        existing = session.execute(
            text("SELECT case_id FROM analyst_case_studies WHERE review_id = :rid"),
            {"rid": payload.review_id},
        ).mappings().first()
        case_id = str(uuid.uuid4())
        if existing:
            session.execute(
                text(
                    """
                    UPDATE analyst_case_studies
                    SET ai_missed = :ai_missed, decision_factors = :decision_factors,
                        missing_variables = :missing_variables, data_sources = :data_sources,
                        confidence_rating = :confidence_rating, created_at = :ts
                    WHERE review_id = :rid
                    """
                ),
                {
                    "ai_missed": payload.ai_missed,
                    "decision_factors": payload.decision_factors,
                    "missing_variables": payload.missing_variables,
                    "data_sources": payload.data_sources,
                    "confidence_rating": payload.confidence_rating,
                    "ts": ts,
                    "rid": payload.review_id,
                },
            )
            case_id = existing["case_id"]
        else:
            session.execute(
                text(
                    """
                    INSERT INTO analyst_case_studies
                        (case_id, review_id, reviewer_id, prediction_id,
                         ai_missed, decision_factors, missing_variables, data_sources,
                         confidence_rating, published, created_at)
                    VALUES
                        (:cid, :rid, :rvid, :pid, :ai_missed, :decision_factors,
                         :missing_variables, :data_sources, :conf, :published, :ts)
                    """
                ),
                {
                    "cid": case_id,
                    "rid": payload.review_id,
                    "rvid": payload.reviewer_id,
                    "pid": review["prediction_id"],
                    "ai_missed": payload.ai_missed,
                    "decision_factors": payload.decision_factors,
                    "missing_variables": payload.missing_variables,
                    "data_sources": payload.data_sources,
                    "conf": payload.confidence_rating,
                    "published": True,
                    "ts": ts,
                },
            )
        session.commit()
    return {"case_id": case_id, "status": "saved"}


@router.get("/comments")
def list_comments(target_type: str, target_id: str) -> List[Dict[str, Any]]:
    if target_type not in ("case_study", "research_question"):
        raise HTTPException(status_code=400, detail="Invalid target_type")
    with get_db_session() as session:
        rows = session.execute(
            text(
                """
                SELECT c.comment_id, c.reviewer_id, c.target_type, c.target_id,
                       c.body, c.created_at, r.name, r.first_name, r.last_name
                FROM analyst_comments c
                JOIN reviewers r ON r.reviewer_id = c.reviewer_id
                WHERE c.target_type = :tt AND c.target_id = :tid
                ORDER BY c.created_at ASC
                """
            ),
            {"tt": target_type, "tid": target_id},
        ).mappings().all()
    return [dict(r) for r in rows]


@router.post("/comments")
def post_comment(payload: CommentRequest) -> Dict[str, Any]:
    with get_db_session() as session:
        resolved = _resolve_reviewer(session, payload.reviewer_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Reviewer not found")
        comment_id = str(uuid.uuid4())
        session.execute(
            text(
                """
                INSERT INTO analyst_comments
                    (comment_id, reviewer_id, target_type, target_id, body, created_at)
                VALUES (:cid, :rid, :tt, :tid, :body, :ts)
                """
            ),
            {
                "cid": comment_id,
                "rid": resolved["reviewer_id"],
                "tt": payload.target_type,
                "tid": payload.target_id,
                "body": payload.body.strip(),
                "ts": datetime.utcnow().isoformat(),
            },
        )
        session.commit()
    return {"comment_id": comment_id, "status": "saved"}
