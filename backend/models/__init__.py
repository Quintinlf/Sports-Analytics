from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text, UniqueConstraint, false, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base

__all__ = [
    "Base",
    "AnalystAnswer",
    "AnalystCaseStudy",
    "AnalystComment",
    "AnalystFeedback",
    "AnalystQuestion",
    "FeatureSuggestion",
    "PredictionReview",
    "ReviewOutcome",
    "Reviewer",
    "ReviewerCustomSection",
    "ReviewerPreference",
]


class AnalystFeedback(Base):
    __tablename__ = "analyst_feedback"

    feedback_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    prediction_id: Mapped[str] = mapped_column(String(64), nullable=False)
    analyst_id: Mapped[str] = mapped_column(String(20), nullable=False)
    analyst_role: Mapped[str] = mapped_column(String(30), nullable=False)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    form_version: Mapped[str] = mapped_column(String(10), default="1.0")
    form_responses: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    has_confidence_concern: Mapped[bool] = mapped_column(Boolean, default=False)
    has_disagreement: Mapped[bool] = mapped_column(Boolean, default=False)
    missing_features_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Analyst review platform — additive tables
# ---------------------------------------------------------------------------

class Reviewer(Base):
    __tablename__ = "reviewers"

    reviewer_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyst_role: Mapped[str] = mapped_column(String(30), nullable=False, default="analyst", server_default="analyst")
    profile_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=false())
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalystQuestion(Base):
    __tablename__ = "analyst_questions"

    question_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context: Mapped[str] = mapped_column(String(30), nullable=False, default="onboarding")
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    prompts_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    knowledge_area: Mapped[str | None] = mapped_column(String(80), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    featured: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalystAnswer(Base):
    __tablename__ = "analyst_answers"
    __table_args__ = (UniqueConstraint("question_id", "reviewer_id", name="uq_answer_per_reviewer"),)

    answer_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    question_id: Mapped[str] = mapped_column(String(64), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(100), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    knowledge_area: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalystCaseStudy(Base):
    __tablename__ = "analyst_case_studies"
    __table_args__ = (UniqueConstraint("review_id", name="uq_case_per_review"),)

    case_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    review_id: Mapped[str] = mapped_column(String(36), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(100), nullable=False)
    prediction_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ai_missed: Mapped[str] = mapped_column(Text, nullable=False)
    decision_factors: Mapped[str] = mapped_column(Text, nullable=False)
    missing_variables: Mapped[str] = mapped_column(Text, nullable=False)
    data_sources: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    published: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalystComment(Base):
    __tablename__ = "analyst_comments"

    comment_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reviewer_id: Mapped[str] = mapped_column(String(100), nullable=False)
    target_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PredictionReview(Base):
    """Pregame human review of one AI prediction."""

    __tablename__ = "prediction_reviews"
    __table_args__ = (UniqueConstraint("prediction_id", "reviewer_id", name="uq_review_per_reviewer"),)

    review_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(36), nullable=False)
    reviewer_pick: Mapped[str] = mapped_column(String(100), nullable=False)
    reviewer_confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    would_bet: Mapped[str] = mapped_column(String(20), nullable=False, default="no")
    agree_with_model: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    missing_factors: Mapped[str | None] = mapped_column(Text, nullable=True)
    pregame_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    primary_decision_variable: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewOutcome(Base):
    """Postgame reflection linked to a pregame review.

    Challenge tracking (successful analyst override) uses:
    - model_correct → ai_was_correct
    - reviewer_correct → analyst_was_correct
    - reviewer_beat_model → successful_analyst_override
    - final_result → settled actual winner
    Pregame disagreement lives on PredictionReview.agree_with_model /
    pregame_notes (analyst_disagreed / analyst_reasoning).
    """

    __tablename__ = "review_outcomes"

    review_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reviewer_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reviewer_beat_model: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    final_result: Mapped[str | None] = mapped_column(String(100), nullable=True)
    followup_missing_factors: Mapped[str | None] = mapped_column(Text, nullable=True)
    followup_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    structured_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    factor_tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    should_be_feature: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    importance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewerPreference(Base):
    __tablename__ = "reviewer_preferences"

    reviewer_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    favorite_sports: Mapped[str | None] = mapped_column(Text, nullable=True)
    emails_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    wants_betting_section: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    wants_explanations: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    wants_postgame_reviews: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    email_frequency: Mapped[str] = mapped_column(String(20), nullable=False, default="weekly")
    email_days: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewerCustomSection(Base):
    __tablename__ = "reviewer_custom_sections"

    section_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reviewer_id: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FeatureSuggestion(Base):
    __tablename__ = "feature_suggestions"

    suggestion_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    feedback_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    analyst_id: Mapped[str] = mapped_column(String(20), nullable=False)
    feature_category: Mapped[str] = mapped_column(String(50), nullable=True)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    calculation_approach: Mapped[str] = mapped_column(Text, nullable=True)
    estimated_impact: Mapped[str] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
