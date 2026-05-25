from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
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
    form_responses: Mapped[dict] = mapped_column(JSONB, nullable=False)

    has_confidence_concern: Mapped[bool] = mapped_column(Boolean, default=False)
    has_disagreement: Mapped[bool] = mapped_column(Boolean, default=False)
    missing_features_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
