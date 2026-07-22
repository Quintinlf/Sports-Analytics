"""MLB game context enrichment for prediction snapshots."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False

try:
    import pybaseball as pyb  # noqa: F401
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False


def _pitcher_from_schedule(game: Dict[str, Any], side: str) -> Dict[str, Any]:
    keys = [
        f"{side}_probable_pitcher",
        f"{side}_pitcher",
        f"{side}Pitcher",
    ]
    name = None
    for key in keys:
        if game.get(key):
            name = str(game[key])
            break
    return {"name": name or "TBD", "era": None, "whip": None, "k9": None, "last5": None}


def _try_pybaseball_stats(pitcher_name: str) -> Dict[str, Any]:
    if not PYBASEBALL_AVAILABLE or not pitcher_name or pitcher_name == "TBD":
        return {}
    try:
        from pybaseball import pitching_stats

        season = pitching_stats(2025, qual=1)
        if season is None or season.empty:
            return {}
        row = season[season["Name"].str.contains(pitcher_name.split()[-1], case=False, na=False)]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "era": round(float(r.get("ERA", 0)), 2),
            "whip": round(float(r.get("WHIP", 0)), 2),
            "k9": round(float(r.get("SO9", 0)), 1),
            "last5": None,
        }
    except Exception as exc:
        logger.debug("pybaseball lookup failed for %s: %s", pitcher_name, exc)
        return {}


def _try_statsapi_pitcher_stats(pitcher_name: str) -> Dict[str, Any]:
    if not STATSAPI_AVAILABLE or not pitcher_name or pitcher_name == "TBD":
        return {}
    try:
        players = statsapi.lookup_player(pitcher_name)
        if not players:
            return {}
        pid = players[0].get("id")
        if not pid:
            return {}
        data = statsapi.player_stat_data(pid, group="pitching", type="season")
        stats = (data.get("stats") or [{}])[0].get("stats", {})
        era = stats.get("era")
        whip = stats.get("whip")
        if era is None and whip is None:
            return {}
        return {
            "era": round(float(era), 2) if era is not None else None,
            "whip": round(float(whip), 2) if whip is not None else None,
            "k9": round(float(stats.get("strikeoutsPer9Inn", 0) or 0), 1),
            "last5": None,
        }
    except Exception as exc:
        logger.debug("statsapi pitcher stats failed for %s: %s", pitcher_name, exc)
        return {}


def enrich_pitcher(pitcher: Dict[str, Any]) -> Dict[str, Any]:
    name = pitcher.get("name") or "TBD"
    stats = _try_statsapi_pitcher_stats(name)
    if not stats:
        stats = _try_pybaseball_stats(name)
    pitcher.update({k: v for k, v in stats.items() if v is not None})
    return pitcher


def build_mlb_context(game: Dict[str, Any]) -> Dict[str, Any]:
    """Build MLB context block for feature_snapshot."""
    warnings: List[str] = []
    home_pitcher = enrich_pitcher(_pitcher_from_schedule(game, "home"))
    away_pitcher = enrich_pitcher(_pitcher_from_schedule(game, "away"))

    if home_pitcher.get("name") == "TBD" or away_pitcher.get("name") == "TBD":
        warnings.append("probable_starter_unconfirmed")
    if home_pitcher.get("era") is None and away_pitcher.get("era") is None:
        warnings.append("pitcher_stats_unavailable")
    warnings.append("bullpen_workload_unavailable")

    explanations = []
    if home_pitcher.get("era") is not None:
        explanations.append({
            "label": "Home Starter ERA",
            "weight": 0.28,
            "value": str(home_pitcher["era"]),
        })
    if away_pitcher.get("era") is not None:
        explanations.append({
            "label": "Away Starter ERA",
            "weight": 0.26,
            "value": str(away_pitcher["era"]),
        })
    if not explanations:
        explanations = []

    return {
        "starting_pitchers": {"home": home_pitcher, "away": away_pitcher},
        "bullpen": {
            "home_ip_last_3d": None,
            "away_ip_last_3d": None,
            "fatigue_flag": None,
        },
        "lineups": {"confirmed": False, "notes": "Lineup data not yet ingested"},
        "missing_data_warnings": warnings,
        "explanations": explanations,
        "metrics": {
            "home_starter_era": home_pitcher.get("era"),
            "away_starter_era": away_pitcher.get("era"),
            "home_starter_whip": home_pitcher.get("whip"),
            "away_starter_whip": away_pitcher.get("whip"),
        },
    }
