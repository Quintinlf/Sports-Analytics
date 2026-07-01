"""Copy this file when adding a new table. Do not import in production."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Index, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base
from backend.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class Game(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Example sports analytics table — rename and adapt before use."""

    __tablename__ = "games"
    __table_args__ = (
        Index("ix_games_sport_game_date", "sport", "game_date"),
    )

    sport: Mapped[str] = mapped_column(String(20), nullable=False)
    league: Mapped[str | None] = mapped_column(String(50), nullable=True)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
