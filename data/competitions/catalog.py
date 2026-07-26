"""Declarative competition catalog for schedule discovery."""
from __future__ import annotations

from typing import Dict, List, Optional

from data.competitions.types import Competition, PredictPolicy

# TheSportsDB league IDs (verified against public league pages / settings map).
SOCCER_COMPETITIONS: List[Competition] = [
    Competition(
        id="ucl",
        sport="SOCCER",
        display_name="UEFA Champions League",
        provider="thesportsdb",
        provider_league_id="4480",
        priority=10,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
    Competition(
        id="epl",
        sport="SOCCER",
        display_name="Premier League",
        provider="thesportsdb",
        provider_league_id="4328",
        priority=20,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
    Competition(
        id="laliga",
        sport="SOCCER",
        display_name="La Liga",
        provider="thesportsdb",
        provider_league_id="4335",
        priority=30,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
    Competition(
        id="bundesliga",
        sport="SOCCER",
        display_name="Bundesliga",
        provider="thesportsdb",
        provider_league_id="4331",
        priority=40,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
    Competition(
        id="serie_a",
        sport="SOCCER",
        display_name="Serie A",
        provider="thesportsdb",
        provider_league_id="4332",
        priority=50,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
    Competition(
        id="ligue_1",
        sport="SOCCER",
        display_name="Ligue 1",
        provider="thesportsdb",
        provider_league_id="4334",
        priority=60,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
    Competition(
        id="world_cup",
        sport="SOCCER",
        display_name="FIFA World Cup",
        provider="thesportsdb",
        provider_league_id="4429",
        priority=70,
        predict_policy=PredictPolicy.FULL_MODEL,
    ),
    Competition(
        id="euros",
        sport="SOCCER",
        display_name="UEFA European Championships",
        provider="thesportsdb",
        provider_league_id="4502",
        priority=80,
        predict_policy=PredictPolicy.FULL_MODEL,
    ),
]

NBA_COMPETITIONS: List[Competition] = [
    Competition(
        id="nba_regular",
        sport="NBA",
        display_name="NBA Regular Season",
        provider="nba_api",
        provider_league_id="00",
        priority=10,
        predict_policy=PredictPolicy.FULL_MODEL,
    ),
    Competition(
        id="nba_playoffs",
        sport="NBA",
        display_name="NBA Playoffs",
        provider="nba_api",
        provider_league_id="00",
        priority=20,
        predict_policy=PredictPolicy.FULL_MODEL,
    ),
    Competition(
        id="nba_offseason",
        sport="NBA",
        display_name="NBA Offseason",
        provider="nba_api",
        provider_league_id="00",
        priority=90,
        predict_policy=PredictPolicy.SCHEDULE_ONLY,
    ),
]

_ALL: List[Competition] = [*SOCCER_COMPETITIONS, *NBA_COMPETITIONS]
_BY_ID: Dict[str, Competition] = {c.id: c for c in _ALL}


def all_competitions() -> List[Competition]:
    return list(_ALL)


def competitions_for_sport(sport: str) -> List[Competition]:
    key = (sport or "").upper()
    if key == "FIFA":
        key = "SOCCER"
    return [c for c in _ALL if c.sport == key]


def get_competition(competition_id: str) -> Optional[Competition]:
    return _BY_ID.get(competition_id)


REQUIRED_SOCCER_LEAGUE_IDS = frozenset(
    {"4328", "4335", "4331", "4332", "4334", "4480", "4429", "4502"}
)
