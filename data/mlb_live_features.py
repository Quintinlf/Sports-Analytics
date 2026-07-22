"""
MLB Live Feature Builder

Computes the same rolling features used in training (data.mlb_feature_engineering)
for a single upcoming matchup, using each team's recent game log from statsapi.
This keeps inference-time features consistent with what the model was trained on.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from data.mlb_feature_engineering import calculate_mlb_rolling_stats

logger = logging.getLogger(__name__)

try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False

MIN_GAMES_FOR_LIVE_FEATURES = 5
LOOKBACK_DAYS = 45
LOOKBACK_GAMES = 15


def _fetch_team_recent_games(team_name: str, as_of_date: str) -> pd.DataFrame:
    """Fetch a team's recent completed games strictly before as_of_date."""
    if not STATSAPI_AVAILABLE:
        return pd.DataFrame()

    as_of = pd.to_datetime(as_of_date)
    start = (as_of - timedelta(days=LOOKBACK_DAYS)).strftime('%m/%d/%Y')
    end = (as_of - timedelta(days=1)).strftime('%m/%d/%Y')

    try:
        team_lookup = statsapi.lookup_team(team_name)
        if not team_lookup:
            logger.warning("Could not resolve MLB team id for %r", team_name)
            return pd.DataFrame()
        team_id = team_lookup[0]['id']
        games = statsapi.schedule(start_date=start, end_date=end, team=team_id)
    except Exception as exc:
        logger.warning("Failed to fetch recent games for %s: %s", team_name, exc)
        return pd.DataFrame()

    rows = []
    for game in games:
        status = str(game.get('status') or '')
        if status not in ('Final', 'Completed Early'):
            continue
        home_name = game.get('home_name')
        away_name = game.get('away_name')
        home_score, away_score = game.get('home_score'), game.get('away_score')
        if home_score is None or away_score is None:
            continue
        try:
            home_score, away_score = int(home_score), int(away_score)
        except (TypeError, ValueError):
            continue

        is_home = home_name == team_name
        opp_name = away_name if is_home else home_name
        team_runs = home_score if is_home else away_score
        opp_runs = away_score if is_home else home_score

        rows.append({
            'GAME_ID': game.get('game_id'),
            'GAME_DATE': game.get('game_date'),
            'TEAM_NAME': team_name,
            'OPP_NAME': opp_name,
            'IS_HOME': 1 if is_home else 0,
            'R': team_runs,
            'RA': opp_runs,
            'WL': 'W' if team_runs > opp_runs else 'L',
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').tail(LOOKBACK_GAMES).reset_index(drop=True)
    return df


def _latest_rolling_snapshot(team_name: str, as_of_date: str) -> Optional[Dict[str, Any]]:
    """Return the most recent rolling-stat snapshot for a team, or None if insufficient history."""
    games = _fetch_team_recent_games(team_name, as_of_date)
    if len(games) < MIN_GAMES_FOR_LIVE_FEATURES:
        return None

    with_stats = calculate_mlb_rolling_stats(games, window=10)
    latest = with_stats.iloc[-1]
    return {
        col: latest[col]
        for col in with_stats.columns
        if '_ROLL' in col or col in ('WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10')
    }


def build_mlb_live_features(
    home_team: str,
    away_team: str,
    as_of_date: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Build a single-row feature DataFrame for an upcoming MLB matchup, using the
    same rolling stats the model was trained on.

    Parameters
    ----------
    feature_cols : list, optional
        Ordered column names expected by the model (e.g. LGBMWinPredictor.feature_names).
        When omitted, columns are derived from whatever rolling stats are available.

    Returns
    -------
    pd.DataFrame or None
        None if either team has fewer than MIN_GAMES_FOR_LIVE_FEATURES completed
        games in the lookback window (e.g. very early season) — callers should
        fall back to a non-model prediction in that case.
    """
    as_of_date = as_of_date or datetime.utcnow().strftime('%Y-%m-%d')

    home_stats = _latest_rolling_snapshot(home_team, as_of_date)
    away_stats = _latest_rolling_snapshot(away_team, as_of_date)
    if home_stats is None or away_stats is None:
        return None

    if feature_cols is None:
        feature_cols = (
            [f'HOME_{k}' for k in home_stats.keys()]
            + [f'AWAY_{k}' for k in away_stats.keys()]
        )

    features = {}
    for col in feature_cols:
        if col.startswith('HOME_'):
            features[col] = home_stats.get(col.replace('HOME_', '', 1), 0.0)
        elif col.startswith('AWAY_'):
            features[col] = away_stats.get(col.replace('AWAY_', '', 1), 0.0)
        else:
            features[col] = 0.0
    return pd.DataFrame([features])
