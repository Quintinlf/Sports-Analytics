from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlencode
sys.path.insert(0, str(Path(__file__).parent.parent))

print("[DEBUG] __name__ =", __name__)

from datetime import date
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy import text

from backend.db import get_db_session, engine, require_engine
from backend.models import Base, AnalystFeedback, FeatureSuggestion
from backend.routes.feedback import router as feedback_router, init_platform
from backend.schemas import (
    FeedbackStatusResponse,
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
    PredictionSummary,
    PredictionsResponse,
)

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "feedback"


ANALYSTS = ["lamar", "anderson", "luis", "alex"]
ANALYST_ROLES = {
    "lamar": "sports_logic",
    "anderson": "sports_logic",
    "luis": "sports_logic",
    "alex": "betting_logic",
}


FORM_SCHEMA: Dict[str, Any] = {
    "version": "2.0",
    "sections": [
        {
            "id": "review_summary",
            "title": "Postgame Summary",
            "fields": [
                {
                    "id": "review_stage",
                    "type": "single_select",
                    "label": "Review stage",
                    "options": ["post_game", "pre_game"],
                },
                {
                    "id": "outcome_correct",
                    "type": "single_select",
                    "label": "Was the winner correct?",
                    "options": ["correct_winner", "wrong_winner", "unknown"],
                },
                {
                    "id": "spread_accuracy",
                    "type": "single_select",
                    "label": "Spread accuracy",
                    "options": [
                        "within_1",
                        "within_2",
                        "within_3",
                        "off_by_3_plus",
                        "not_applicable",
                    ],
                },
                {
                    "id": "confidence_alignment",
                    "type": "single_select",
                    "label": "Confidence alignment",
                    "options": ["aligned", "overconfident", "underconfident"],
                },
                {
                    "id": "variance_assessment",
                    "type": "single_select",
                    "label": "Variance assessment",
                    "options": ["low_variance", "moderate_variance", "high_variance"],
                },
                {
                    "id": "surprise_level",
                    "type": "single_select",
                    "label": "Surprise level",
                    "options": ["not_surprised", "mild_surprise", "big_surprise"],
                },
            ],
        },
        {
            "id": "baseball_logic",
            "title": "Baseball Logic",
            "fields": [
                {
                    "id": "baseball_miss_type",
                    "type": "multi_select",
                    "label": "Baseball signals the model missed",
                    "options": [
                        "no_baseball_issue",
                        "starter_misread",
                        "bullpen_usage",
                        "lineup_change",
                        "defense_shift",
                        "park_factor",
                        "travel_rest",
                        "weather_impact",
                        "injury_news",
                    ],
                },
                {
                    "id": "primary_baseball_driver",
                    "type": "single_select",
                    "label": "Primary baseball driver",
                    "options": [
                        "none",
                        "starter",
                        "bullpen",
                        "lineup",
                        "defense",
                        "park",
                        "travel_rest",
                        "weather",
                        "injury",
                    ],
                },
                {
                    "id": "baseball_confidence",
                    "type": "single_select",
                    "label": "Baseball logic vs model",
                    "options": [
                        "strong_support",
                        "mild_support",
                        "neutral",
                        "mild_conflict",
                        "strong_conflict",
                    ],
                },
                {
                    "id": "baseball_notes",
                    "type": "textarea",
                    "label": "Short notes (optional)",
                    "max_chars": 250,
                },
            ],
        },
        {
            "id": "betting_logic",
            "title": "Betting Logic",
            "fields": [
                {
                    "id": "market_signal",
                    "type": "multi_select",
                    "label": "Market signals",
                    "options": [
                        "no_market_signal",
                        "steam_against",
                        "steam_with",
                        "late_move",
                        "sharp_action",
                    ],
                },
                {
                    "id": "closing_line_value",
                    "type": "single_select",
                    "label": "Closing line value",
                    "options": [
                        "favored_model",
                        "favored_market",
                        "neutral",
                        "not_available",
                    ],
                },
                {
                    "id": "price_sensitivity",
                    "type": "single_select",
                    "label": "Price sensitivity",
                    "options": ["low", "medium", "high"],
                },
                {
                    "id": "betting_notes",
                    "type": "textarea",
                    "label": "Short notes (optional)",
                    "max_chars": 250,
                },
            ],
        },
        {
            "id": "feature_targets",
            "title": "Feature Targets",
            "fields": [
                {
                    "id": "missing_feature_flags",
                    "type": "multi_select",
                    "label": "Feature gaps",
                    "options": [
                        "none_missing",
                        "bullpen_usage",
                        "pitcher_fatigue",
                        "lineup_confirmation",
                        "platoon_splits",
                        "park_factors",
                        "weather_inputs",
                        "umpire_tendencies",
                        "travel_schedule",
                        "defensive_metrics",
                        "market_metrics",
                    ],
                },
                {
                    "id": "feature_priority",
                    "type": "single_select",
                    "label": "Feature priority",
                    "options": ["must_have", "high", "medium", "low"],
                },
                {
                    "id": "feature_target_detail",
                    "type": "textarea",
                    "label": "Short detail (optional)",
                    "max_chars": 250,
                },
            ],
        },
        {
            "id": "meta_scoring",
            "title": "Scoring",
            "fields": [
                {
                    "id": "model_grade",
                    "type": "single_select",
                    "label": "Overall model grade",
                    "options": ["5", "4", "3", "2", "1"],
                },
                {
                    "id": "confidence_grade",
                    "type": "single_select",
                    "label": "Confidence grade",
                    "options": ["5", "4", "3", "2", "1"],
                },
                {
                    "id": "variance_grade",
                    "type": "single_select",
                    "label": "Variance handling grade",
                    "options": ["5", "4", "3", "2", "1"],
                },
                {
                    "id": "review_time",
                    "type": "single_select",
                    "label": "Time spent",
                    "options": ["under_2_min", "under_5_min", "over_5_min"],
                },
            ],
        },
    ],
}


app = FastAPI(title="Sports Analytics Analyst Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the new feedback platform router
app.include_router(feedback_router)


@app.on_event("startup")
def _init_db() -> None:
    require_engine()
    Base.metadata.create_all(bind=engine)
    # Seed predictions + create new review tables
    init_platform(engine)


# ── Feedback platform frontend ──────────────────────────────────────────────

@app.get("/feedback", response_class=FileResponse, include_in_schema=False)
@app.get("/feedback/", response_class=FileResponse, include_in_schema=False)
def feedback_page() -> FileResponse:
    # #region agent log: /feedback route
    try:
        print(f"[ROUTE] /feedback: attempting to serve {_FRONTEND_DIR / 'index.html'}")
        result = FileResponse(_FRONTEND_DIR / "index.html")
        print(f"[ROUTE] /feedback: FileResponse created successfully")
        return result
    except Exception as e:
        import traceback
        print(f"[ROUTE] /feedback: EXCEPTION")
        traceback.print_exc()
        raise
    # #endregion


@app.get("/feedback/styles.css", include_in_schema=False)
def feedback_styles() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/feedback/script.js", include_in_schema=False)
def feedback_script() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "script.js", media_type="application/javascript")


@app.get("/feedback/preview", response_class=HTMLResponse, include_in_schema=False)
def feedback_preview(
    reviewer_id: str = Query(default="quintin"),
    sport: str = Query(default=""),
) -> HTMLResponse:
    # #region agent log: /feedback/preview route
    try:
        print(f"[ROUTE] /feedback/preview: reviewer_id={reviewer_id!r}, sport={sport!r}")
        params: Dict[str, str] = {}
        if reviewer_id:
            params["reviewer_id"] = reviewer_id
        if sport:
            params["sport"] = sport
        query = urlencode(params)
        iframe_src = "/feedback" + (f"?{query}" if query else "")
        print(f"[ROUTE] /feedback/preview: iframe_src={iframe_src!r}")

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Feedback Preview</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; overflow:hidden; background:#0b0f14; }}
    iframe {{ width:100%; height:100%; border:none; display:block; }}
  </style>
</head>
<body>
  <iframe src="{iframe_src}"></iframe>
</body>
</html>"""
        print(f"[ROUTE] /feedback/preview: HTMLResponse created, length={len(html)}")
        return HTMLResponse(html)
    except Exception as e:
        import traceback
        print(f"[ROUTE] /feedback/preview: EXCEPTION")
        traceback.print_exc()
        raise
    # #endregion


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/feedback", status_code=307)


@app.get("/api/v1/forms/feedback-template/{prediction_id}")
def get_feedback_template(prediction_id: str) -> Dict[str, Any]:
    return {"prediction_id": prediction_id, "schema": FORM_SCHEMA}


@app.get("/api/v1/predictions/today", response_model=PredictionsResponse)
def get_todays_predictions() -> PredictionsResponse:
    today = date.today().isoformat()
    sql = text(
        """
        SELECT prediction_id, game_id, game_date, home_team, away_team,
               predicted_spread, win_probability, confidence_level
        FROM predictions
        WHERE game_date = :today
        ORDER BY game_date, prediction_id
        """
    )

    with get_db_session() as session:
        rows = session.execute(sql, {"today": today}).mappings().all()

    games = [
        PredictionSummary(
            prediction_id=str(row.get("prediction_id")),
            game_id=row.get("game_id"),
            game_date=row.get("game_date"),
            home_team=row.get("home_team"),
            away_team=row.get("away_team"),
            predicted_spread=row.get("predicted_spread"),
            win_probability=row.get("win_probability"),
            confidence_level=row.get("confidence_level"),
        )
        for row in rows
    ]
    return PredictionsResponse(games=games)


@app.post("/api/v1/feedback/submit", response_model=FeedbackSubmitResponse)
def submit_feedback(payload: FeedbackSubmitRequest) -> FeedbackSubmitResponse:
    analyst_id = payload.analyst_id.lower().strip()
    if analyst_id not in ANALYSTS:
        raise HTTPException(status_code=400, detail="Unknown analyst_id")

    expected_role = ANALYST_ROLES.get(analyst_id)
    if payload.analyst_role != expected_role:
        raise HTTPException(status_code=400, detail="Analyst role mismatch")

    form_responses = payload.form_responses or {}
    review_section = form_responses.get("review_summary", {})
    feature_section = form_responses.get("feature_targets", {})

    has_confidence_concern = review_section.get("confidence_alignment") in [
        "overconfident",
        "underconfident",
    ]
    has_disagreement = review_section.get("spread_accuracy") in [
        "off_by_3_plus",
    ]
    missing_list = feature_section.get("missing_feature_flags", []) or []
    missing_list = [item for item in missing_list if item != "none_missing"]
    missing_features_count = len(missing_list)

    with get_db_session() as session:
        feedback = AnalystFeedback(
            prediction_id=payload.prediction_id,
            analyst_id=analyst_id,
            analyst_role=payload.analyst_role,
            game_id=payload.game_id,
            form_version=payload.form_version,
            form_responses=form_responses,
            has_confidence_concern=has_confidence_concern,
            has_disagreement=has_disagreement,
            missing_features_count=missing_features_count,
        )
        session.add(feedback)
        session.commit()
        session.refresh(feedback)

        feature_description = feature_section.get("feature_target_detail")
        if feature_description:
            suggestion = FeatureSuggestion(
                feedback_id=feedback.feedback_id,
                analyst_id=analyst_id,
                feature_category=_first_or_none(missing_list),
                feature_name=_extract_feature_name(feature_description),
                description=feature_description,
                calculation_approach=None,
                estimated_impact=feature_section.get("feature_priority"),
            )
            session.add(suggestion)
            session.commit()

    return FeedbackSubmitResponse(feedback_id=str(feedback.feedback_id), status="success")


@app.get("/api/v1/feedback/{prediction_id}/status", response_model=FeedbackStatusResponse)
def get_feedback_status(prediction_id: str) -> FeedbackStatusResponse:
    with get_db_session() as session:
        rows = session.execute(
            text(
                """
                SELECT DISTINCT analyst_id
                FROM analyst_feedback
                WHERE prediction_id = :prediction_id
                """
            ),
            {"prediction_id": prediction_id},
        ).mappings().all()

    submitted_by = sorted({row["analyst_id"] for row in rows})
    pending_from = [a for a in ANALYSTS if a not in submitted_by]

    return FeedbackStatusResponse(
        prediction_id=prediction_id,
        submitted_by=submitted_by,
        pending_from=pending_from,
    )


def _first_or_none(values: Any) -> str | None:
    if not values:
        return None
    if isinstance(values, list):
        return values[0] if values else None
    return str(values)


def _extract_feature_name(text_value: Any) -> str | None:
    if not text_value:
        return None
    text = str(text_value).strip()
    if not text:
        return None
    return text.split()[0][:100]


if __name__ == "__main__":
    import webbrowser
    import uvicorn

    # #region agent log: startup verification
    print("[STARTUP] FastAPI app initialized")
    print("[STARTUP] Registered routes:")
    for r in app.routes:
        if hasattr(r, 'path'):
            print(f"  {r.path}")
    print(f"[STARTUP] _FRONTEND_DIR: {_FRONTEND_DIR}")
    print(f"[STARTUP] _FRONTEND_DIR exists: {_FRONTEND_DIR.exists()}")
    if _FRONTEND_DIR.exists():
        index_html = _FRONTEND_DIR / "index.html"
        print(f"[STARTUP] index.html exists: {index_html.exists()}")
    # #endregion

    dashboard_url = "http://localhost:8000/feedback"
    preview_url = "http://localhost:8000/feedback/preview?reviewer_id=quintin"
    print("[STARTUP] Dashboard:")
    print(dashboard_url)
    print()
    print("[STARTUP] Preview:")
    print(preview_url)
    print()
    print("[STARTUP] Browser opened; starting uvicorn...")
    print("[DEBUG] About to call uvicorn.run")
    opened = webbrowser.open(preview_url, new=2)
    if not opened:
        # On Windows, webbrowser.open can fail when no default association exists.
        print("[STARTUP] WARNING: webbrowser.open could not launch automatically.")
        print("[STARTUP] Fallback: paste the Preview URL into your browser manually.")
        if os.name == "nt":
            try:
                os.startfile(preview_url)  # type: ignore[attr-defined]
                print("[STARTUP] Fallback os.startfile() launch attempted.")
            except Exception:
                pass
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
