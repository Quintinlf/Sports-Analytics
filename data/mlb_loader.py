"""
MLB Data Loading Module

Authoritative source for fetching MLB historical game results and team
metadata via MLB-StatsAPI. Mirrors data/nba_loader.py's shape so downstream
feature-engineering code follows the same pattern used for NBA.

statsapi (not pybaseball) is used for historical results so that team names
stay consistent with data/mlb_predictions_service.py, which already uses
statsapi full team names for live games. pybaseball remains reserved for
pitcher-level enrichment in data/mlb_context.py.
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

warnings.filterwarnings('ignore')

try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False

MLB_SPORT_ID = 1


def get_all_mlb_teams() -> dict:
    """Return dict with MLB team ids, names, and abbreviations (statsapi is authoritative)."""
    if not STATSAPI_AVAILABLE:
        raise ImportError("MLB-StatsAPI required: pip install MLB-StatsAPI")

    teams = statsapi.get('teams', {'sportId': MLB_SPORT_ID}).get('teams', [])
    return {
        'teams': teams,
        'ids': [t['id'] for t in teams],
        'names': {t['id']: t['name'] for t in teams},
        'abbreviations': {t['id']: t.get('abbreviation', t['name']) for t in teams},
    }


def _season_date_range(season: str) -> Tuple[str, str]:
    """Return (start_date, end_date) in MM/DD/YYYY covering a full MLB season."""
    year = int(season)
    return f'03/01/{year}', f'11/15/{year}'


def fetch_mlb_games(
    seasons: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch completed MLB game results for one or more seasons via statsapi.

    Parameters
    ----------
    seasons : list, optional
        Season years as strings, e.g. ['2023', '2024']. Defaults to the last two years.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per team per game (two rows per GAME_ID), columns:
        GAME_ID, GAME_DATE, TEAM_NAME, OPP_NAME, IS_HOME, R, RA, WL.
    """
    if not STATSAPI_AVAILABLE:
        raise ImportError("MLB-StatsAPI required: pip install MLB-StatsAPI")

    if seasons is None:
        current_year = datetime.utcnow().year
        seasons = [str(current_year - 1), str(current_year)]

    rows = []
    for season in seasons:
        if verbose:
            print(f"  Fetching MLB {season}...")
        start_date, end_date = _season_date_range(season)
        try:
            games = statsapi.schedule(start_date=start_date, end_date=end_date, sportId=MLB_SPORT_ID)
        except Exception as exc:
            if verbose:
                print(f"    ERROR: {exc}")
            continue

        season_rows = 0
        for game in games:
            # Skip non-regular-season games (spring training, postseason) when the
            # field is available; if absent, fall back to including everything.
            game_type = game.get('game_type')
            if game_type and game_type != 'R':
                continue

            status = str(game.get('status') or '')
            if status not in ('Final', 'Completed Early'):
                continue

            home_name = game.get('home_name')
            away_name = game.get('away_name')
            home_score = game.get('home_score')
            away_score = game.get('away_score')
            game_date = game.get('game_date')
            game_id = game.get('game_id')

            if not home_name or not away_name or home_score is None or away_score is None:
                continue

            try:
                home_score = int(home_score)
                away_score = int(away_score)
            except (TypeError, ValueError):
                continue

            rows.append({
                'GAME_ID': game_id,
                'GAME_DATE': game_date,
                'TEAM_NAME': home_name,
                'OPP_NAME': away_name,
                'IS_HOME': 1,
                'R': home_score,
                'RA': away_score,
                'WL': 'W' if home_score > away_score else 'L',
            })
            rows.append({
                'GAME_ID': game_id,
                'GAME_DATE': game_date,
                'TEAM_NAME': away_name,
                'OPP_NAME': home_name,
                'IS_HOME': 0,
                'R': away_score,
                'RA': home_score,
                'WL': 'W' if away_score > home_score else 'L',
            })
            season_rows += 1

        if verbose:
            print(f"    -> {season_rows} completed games")
        time.sleep(0.5)  # rate limiting

    if not rows:
        raise ValueError("No MLB game data fetched from any season.")

    combined = pd.DataFrame(rows)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    combined = combined.sort_values(['TEAM_NAME', 'GAME_DATE']).reset_index(drop=True)

    if verbose:
        print(f"  Total: {len(combined)} team-game rows, "
              f"{combined['GAME_DATE'].min().date()} -> {combined['GAME_DATE'].max().date()}")

    return combined
