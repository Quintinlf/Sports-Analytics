from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FeedbackSubmitRequest(BaseModel):
    prediction_id: str = Field(..., description="UUID for the prediction row")
    game_id: str = Field(..., description="Game identifier")
    analyst_id: str = Field(..., description="lamar | anderson | luis | alex")
    analyst_role: str = Field(..., description="sports_logic | betting_logic")
    form_version: str = Field(default="1.0")
    form_responses: Dict[str, Any]


class FeedbackSubmitResponse(BaseModel):
    feedback_id: str
    status: str


class PredictionSummary(BaseModel):
    prediction_id: str
    game_id: Optional[str]
    game_date: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    predicted_spread: Optional[float]
    win_probability: Optional[float]
    confidence_level: Optional[str]


class PredictionsResponse(BaseModel):
    games: list[PredictionSummary]


class FeedbackStatusResponse(BaseModel):
    prediction_id: str
    submitted_by: list[str]
    pending_from: list[str]
