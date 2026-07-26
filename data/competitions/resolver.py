"""Resolve which competitions are active for discovery."""
from __future__ import annotations

from typing import List, Optional

from data.competitions.catalog import competitions_for_sport
from data.competitions.types import Competition, NbaSeasonPhase, PredictPolicy


def resolve_active(
    sport: str,
    *,
    season_phase: Optional[NbaSeasonPhase] = None,
) -> List[Competition]:
    """Return catalog competitions that should be queried for *sport*.

    Soccer: all catalogued leagues (club + international).
    NBA: regular / playoffs / offseason based on *season_phase*.
    """
    comps = competitions_for_sport(sport)
    key = (sport or "").upper()
    if key in ("SOCCER", "FIFA"):
        return sorted(comps, key=lambda c: c.priority)

    if key == "NBA":
        if season_phase == NbaSeasonPhase.PLAYOFFS:
            return [c for c in comps if c.id == "nba_playoffs"]
        if season_phase == NbaSeasonPhase.OFFSEASON:
            return [c for c in comps if c.id == "nba_offseason"]
        # Default / regular season
        return [c for c in comps if c.id == "nba_regular"]

    return sorted(comps, key=lambda c: c.priority)


def soccer_league_ids(*, policy: Optional[PredictPolicy] = None) -> List[str]:
    """Provider league IDs for soccer discovery, optionally filtered by policy."""
    comps = resolve_active("SOCCER")
    out: List[str] = []
    for c in comps:
        if not c.provider_league_id:
            continue
        if policy is not None and c.predict_policy != policy:
            continue
        out.append(c.provider_league_id)
    return out
