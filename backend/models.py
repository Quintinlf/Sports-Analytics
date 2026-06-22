from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewOutcome(Base):
    """Postgame reflection linked to a pregame review."""

    __tablename__ = "review_outcomes"

    review_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reviewer_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reviewer_beat_model: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
