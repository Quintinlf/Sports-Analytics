"""
NBA Live Feature Builder

Computes the same features used in training (data.feature_engineering +
src.evaluation.vectorized_features) for a single upcoming matchup, using the
league's recent game log. This keeps inference-time features consistent with
what the ensemble model was trained on.

Mirrors data/mlb_live_features.py's shape/contract for the other sport.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data.nba_loader import get_all_nba_teams, get_team_latest_stats
from data.feature_engineering import calculate_rolling_stats
from src.evaluation.vectorized_features import vectorize_high_signal_features

logger = logging.getLogger(__name__)

try:
    from nba_api.stats.endpoints import leaguegamefinder
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

MIN_GAMES_FOR_LIVE_FEATURES = 5
ROLLING_WINDOW = 10


def _resolve_team_id(team_name: str) -> Optional[int]:
    """Resolve a team's full name to its nba_api numeric TEAM_ID."""
    names = get_all_nba_teams()["names"]  # {id: full_name}
    team_name_lower = str(team_name).strip().lower()
    for team_id, full_name in names.items():
        if full_name.lower() == team_name_lower:
            return team_id
    return None


def _season_strings(as_of: datetime) -> List[str]:
    """Return [prior_season, current_season] NBA season strings covering as_of.

    NBA seasons run roughly October -> June, labeled by their start year
    (e.g. "2025-26"). Games before September are treated as still within the
    season that started the previous calendar year.
    """
    year = as_of.year
    if as_of.month >= 9:
        current = f"{year}-{str(year + 1)[-2:]}"
        prior = f"{year - 1}-{str(year)[-2:]}"
    else:
        current = f"{year - 1}-{str(year)[-2:]}"
        prior = f"{year - 2}-{str(year - 1)[-2:]}"
    return [prior, current]


def _fetch_recent_league_games(as_of_date: str) -> pd.DataFrame:
    """Fetch league-wide game logs strictly before as_of_date (leakage-safe)."""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()

    as_of = pd.to_datetime(as_of_date)
    seasons = _season_strings(as_of.to_pydatetime())

    frames = []
    for season in seasons:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
                timeout=30,
            )
            df = finder.get_data_frames()[0]
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("Failed to fetch NBA season %s for live features: %s", season, exc)

    if not frames:
        return pd.DataFrame()

    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games[games["GAME_DATE"] < as_of]
    return games.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def build_nba_live_features(
    home_team: str,
    away_team: str,
    as_of_date: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> Optional[Tuple[pd.DataFrame, int, int]]:
    """
    Build a single-row feature DataFrame for an upcoming NBA matchup, using
    the same rolling + high-signal features the model was trained on.

    Parameters
    ----------
    feature_cols : list, optional
        Ordered column names expected by the model (e.g. an EnsemblePredictor
        component's feature_names). When omitted, columns are derived from
        whatever is available.

    Returns
    -------
    (features_df, home_team_id, away_team_id) or None
        None if either team can't be resolved, league data can't be fetched,
        or either team has fewer than MIN_GAMES_FOR_LIVE_FEATURES completed
        games in the lookback window (e.g. season just started) — callers
        should skip the game in that case rather than guess.
    """
    as_of_date = as_of_date or datetime.utcnow().strftime("%Y-%m-%d")
    as_of_ts = pd.to_datetime(as_of_date)

    home_id = _resolve_team_id(home_team)
    away_id = _resolve_team_id(away_team)
    if home_id is None or away_id is None:
        logger.warning("Could not resolve NBA team id for %r / %r", home_team, away_team)
        return None

    games = _fetch_recent_league_games(as_of_date)
    if games.empty:
        return None

    home_games = games[games["TEAM_ID"] == home_id]
    away_games = games[games["TEAM_ID"] == away_id]
    if len(home_games) < MIN_GAMES_FOR_LIVE_FEATURES or len(away_games) < MIN_GAMES_FOR_LIVE_FEATURES:
        return None

    rolling = calculate_rolling_stats(games, window=ROLLING_WINDOW)
    home_stats = get_team_latest_stats(rolling, home_id)
    away_stats = get_team_latest_stats(rolling, away_id)
    if home_stats is None or away_stats is None:
        return None

    # Recompute rest/back-to-back relative to the upcoming game's date rather
    # than the most recent played game's own (already historical) rest value.
    last_home_date = home_games["GAME_DATE"].max()
    last_away_date = away_games["GAME_DATE"].max()
    home_stats["REST_DAYS"] = float((as_of_ts - last_home_date).days)
    away_stats["REST_DAYS"] = float((as_of_ts - last_away_date).days)
    home_stats["IS_BACK_TO_BACK"] = 1.0 if home_stats["REST_DAYS"] <= 1 else 0.0
    away_stats["IS_BACK_TO_BACK"] = 1.0 if away_stats["REST_DAYS"] <= 1 else 0.0

    matchup_df = pd.DataFrame([{
        "GAME_ID": f"LIVE_{home_id}_{away_id}_{as_of_date}",
        "GAME_DATE": as_of_ts,
        "HOME_TEAM": home_id,
        "AWAY_TEAM": away_id,
    }])
    derived = vectorize_high_signal_features(matchup_df, games, verbose=False)
    derived_row: Dict[str, Any] = derived.iloc[0].to_dict()

    if feature_cols is None:
        feature_cols = (
            [f"HOME_{k}" for k in home_stats.keys()]
            + [f"AWAY_{k}" for k in away_stats.keys()]
            + list(derived_row.keys())
        )

    features: Dict[str, Any] = {}
    for col in feature_cols:
        if col in derived_row:
            features[col] = derived_row[col]
        elif col.startswith("HOME_"):
            features[col] = home_stats.get(col.replace("HOME_", "", 1), 0.0)
        elif col.startswith("AWAY_"):
            features[col] = away_stats.get(col.replace("AWAY_", "", 1), 0.0)
        else:
            features[col] = 0.0

    return pd.DataFrame([features]), home_id, away_id
