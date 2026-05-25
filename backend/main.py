from __future__ import annotations

from datetime import date
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from backend.db import get_db_session, engine, require_engine
from backend.models import Base, AnalystFeedback, FeatureSuggestion
from backend.schemas import (
    FeedbackStatusResponse,
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
    PredictionSummary,
    PredictionsResponse,
)


ANALYSTS = ["lamar", "anderson", "luis", "alex"]
ANALYST_ROLES = {
    "lamar": "sports_logic",
    "anderson": "sports_logic",
    "luis": "sports_logic",
    "alex": "betting_logic",
}


FORM_SCHEMA: Dict[str, Any] = {
    "version": "1.0",
    "sections": [
        {
            "id": "confidence_calibration",
            "title": "Confidence & Disagreement",
            "fields": [
                {
                    "id": "confidence_assessment",
                    "type": "single_select",
                    "label": "Is this prediction confidence appropriate?",
                    "options": [
                        "agree_high",
                        "agree_medium",
                        "agree_low",
                        "should_be_higher",
                        "should_be_lower",
                    ],
                },
                {
                    "id": "confidence_reasoning",
                    "type": "textarea",
                    "label": "Why (if you disagreed)?",
                    "max_chars": 500,
                },
                {
                    "id": "spread_disagreement",
                    "type": "single_select",
                    "label": "Do you agree with the predicted spread?",
                    "options": [
                        "completely_agree",
                        "mostly_agree",
                        "disagree_1_to_2_points",
                        "disagree_3_plus_points",
                        "no_opinion",
                    ],
                },
                {
                    "id": "spread_disagreement_reason",
                    "type": "textarea",
                    "label": "If disagreed, explain the reasoning:",
                    "max_chars": 500,
                },
            ],
        },
        {
            "id": "baseball_logic",
            "title": "Baseball Logic Insights",
            "fields": [
                {
                    "id": "pitcher_concerns",
                    "type": "multi_select",
                    "label": "Any concerns about starting pitchers today?",
                    "options": [
                        "no_concerns",
                        "starter_fatigue",
                        "starter_injury_risk",
                        "bullpen_weakness",
                        "bullpen_strength",
                    ],
                },
                {
                    "id": "pitcher_details",
                    "type": "textarea",
                    "label": "Details (closer on IL, arm slot change, etc.):",
                    "max_chars": 300,
                },
                {
                    "id": "lineup_impact",
                    "type": "single_select",
                    "label": "Lineup changes impact?",
                    "options": [
                        "no_changes",
                        "key_player_out",
                        "key_player_back",
                        "backup_heavy_lineup",
                    ],
                },
                {
                    "id": "lineup_details",
                    "type": "textarea",
                    "label": "Which player(s)? Impact severity (-3 to +3):",
                    "max_chars": 300,
                },
                {
                    "id": "rest_advantage",
                    "type": "single_select",
                    "label": "Rest advantage?",
                    "options": [
                        "no_advantage",
                        "home_rested_1plus_days",
                        "away_rested_1plus_days",
                    ],
                },
                {
                    "id": "weather_impact",
                    "type": "multi_select",
                    "label": "Weather factors affecting prediction?",
                    "options": [
                        "no_weather_factor",
                        "cold_slows_flyballs",
                        "hot_increases_distance",
                        "wind_in",
                        "wind_out",
                        "wind_left",
                        "wind_right",
                        "humidity_high",
                        "rain_delay_risk",
                    ],
                },
                {
                    "id": "weather_magnitude",
                    "type": "slider",
                    "label": "Estimated accuracy impact from weather (percentage points):",
                    "min": -5,
                    "max": 5,
                    "step": 1,
                },
            ],
        },
        {
            "id": "betting_logic",
            "title": "Betting Market Observations",
            "fields": [
                {
                    "id": "market_line_behavior",
                    "type": "single_select",
                    "label": "Market line movement?",
                    "options": [
                        "no_movement",
                        "moved_toward_prediction",
                        "moved_against_prediction",
                        "sharp_money_detected",
                    ],
                },
                {
                    "id": "market_line_details",
                    "type": "textarea",
                    "label": "Line movement details (if applicable):",
                    "max_chars": 300,
                },
                {
                    "id": "model_vs_market",
                    "type": "single_select",
                    "label": "Model confidence vs. market price alignment?",
                    "options": [
                        "well_aligned",
                        "model_underconfident",
                        "model_overconfident",
                        "arbitrage_opportunity",
                    ],
                },
                {
                    "id": "calibration_magnitude",
                    "type": "single_select",
                    "label": "If misaligned, magnitude:",
                    "options": [
                        "minor_1_to_2_pct",
                        "moderate_3_to_5_pct",
                        "significant_6_plus_pct",
                    ],
                },
                {
                    "id": "calibration_commentary",
                    "type": "textarea",
                    "label": "Why the divergence exists:",
                    "max_chars": 300,
                },
            ],
        },
        {
            "id": "missing_features",
            "title": "Missing Features",
            "fields": [
                {
                    "id": "missing_data",
                    "type": "multi_select",
                    "label": "Missing data in current model?",
                    "options": [
                        "no_missing_features",
                        "pitcher_metric",
                        "position_player_metric",
                        "weather_field_metric",
                        "market_metric",
                        "other",
                    ],
                },
                {
                    "id": "feature_description",
                    "type": "textarea",
                    "label": "Describe the feature (what + how calculated):",
                    "max_chars": 500,
                },
                {
                    "id": "feature_impact_estimate",
                    "type": "single_select",
                    "label": "Estimated impact on model accuracy:",
                    "options": [
                        "game_changer",
                        "moderate_improvement",
                        "minor_edge",
                        "speculative",
                    ],
                },
            ],
        },
        {
            "id": "general_feedback",
            "title": "General Feedback",
            "fields": [
                {
                    "id": "general_comments",
                    "type": "textarea",
                    "label": "Any other comments about today's predictions?",
                    "max_chars": 1000,
                },
                {
                    "id": "pattern_observation",
                    "type": "textarea",
                    "label": "Did you spot any patterns in model failures?",
                    "max_chars": 500,
                },
            ],
        },
    ],
}


app = FastAPI(title="MLB Analyst Feedback")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _init_db() -> None:
    require_engine()
    Base.metadata.create_all(bind=engine)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open("backend/static/feedback_form.html", "r", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


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
    confidence_section = form_responses.get("confidence_calibration", {})
    missing_section = form_responses.get("missing_features", {})
    disagreement = confidence_section.get("spread_disagreement")

    has_confidence_concern = confidence_section.get("confidence_assessment") in [
        "should_be_higher",
        "should_be_lower",
    ]
    has_disagreement = disagreement in [
        "disagree_1_to_2_points",
        "disagree_3_plus_points",
    ]
    missing_list = missing_section.get("missing_data", []) or []
    missing_list = [item for item in missing_list if item != "no_missing_features"]
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

        feature_description = missing_section.get("feature_description")
        if feature_description:
            suggestion = FeatureSuggestion(
                feedback_id=feedback.feedback_id,
                analyst_id=analyst_id,
                feature_category=_first_or_none(missing_list),
                feature_name=_extract_feature_name(feature_description),
                description=feature_description,
                calculation_approach=missing_section.get("calculation_approach"),
                estimated_impact=missing_section.get("feature_impact_estimate"),
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
