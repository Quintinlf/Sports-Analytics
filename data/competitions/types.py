"""Shared types for competition-aware schedule discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class PredictPolicy(str, Enum):
    FULL_MODEL = "full_model"
    SCHEDULE_ONLY = "schedule_only"


class NbaSeasonPhase(str, Enum):
    REGULAR = "regular"
    PLAYOFFS = "playoffs"
    OFFSEASON = "offseason"


@dataclass(frozen=True)
class Competition:
    id: str
    sport: str
    display_name: str
    provider: str
    provider_league_id: Optional[str]
    priority: int
    predict_policy: PredictPolicy


@dataclass
class UnifiedFixture:
    sport: str
    competition_id: str
    league: str
    provider_game_id: str
    game_date: str
    home_team: str
    away_team: str
    predict_policy: PredictPolicy = PredictPolicy.FULL_MODEL
    season_phase: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_fifa_dict(self) -> Dict[str, Any]:
        """Shape expected by FIFALivePredictionService.build_prediction_rows."""
        return {
            "id": self.provider_game_id,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "utc_date": self.game_date,
            "competition_id": self.competition_id,
            "predict_policy": self.predict_policy.value,
        }

    def to_nba_dict(self) -> Dict[str, Any]:
        """Shape expected by NBALivePredictionService.build_prediction_rows."""
        return {
            "game_id": self.provider_game_id,
            "GAME_ID": self.provider_game_id,
            "game_date": self.game_date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "season_phase": self.season_phase,
            **self.meta,
        }
