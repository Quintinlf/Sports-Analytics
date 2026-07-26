"""Competition catalog and schedule providers for year-round discovery."""
from data.competitions.catalog import (
    REQUIRED_SOCCER_LEAGUE_IDS,
    all_competitions,
    competitions_for_sport,
    get_competition,
)
from data.competitions.resolver import resolve_active, soccer_league_ids
from data.competitions.types import Competition, NbaSeasonPhase, PredictPolicy, UnifiedFixture

__all__ = [
    "Competition",
    "NbaSeasonPhase",
    "PredictPolicy",
    "UnifiedFixture",
    "REQUIRED_SOCCER_LEAGUE_IDS",
    "all_competitions",
    "competitions_for_sport",
    "get_competition",
    "resolve_active",
    "soccer_league_ids",
]
