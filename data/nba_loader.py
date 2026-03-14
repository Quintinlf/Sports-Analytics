"""
NBA Data Loading Module

Authoritative source for fetching NBA game data and upcoming schedules via nba_api.
All other modules should import from here rather than duplicating these functions.
"""

import time
import warnings

import numpy as np
import pandas as pd
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

warnings.filterwarnings('ignore')


def get_all_nba_teams() -> dict:
    """Return dict with NBA team ids, names, and abbreviations."""
    nba_teams = teams.get_teams()
    return {
        'teams': nba_teams,
        'ids': [t['id'] for t in nba_teams],
        'names': {t['id']: t['full_name'] for t in nba_teams},
        'abbreviations': {t['id']: t['abbreviation'] for t in nba_teams},
    }


def fetch_nba_games(
    seasons: list = None,
    season_type: str = 'Regular Season',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch NBA game data for one or more seasons from the NBA stats API.

    Parameters
    ----------
    seasons : list, optional
        Season strings, e.g. ['2023-24', '2024-25']. Defaults to latest two seasons.
    season_type : str
        'Regular Season', 'Playoffs', or 'All Star'.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined game log with GAME_DATE converted to datetime.
    """
    if seasons is None:
        seasons = ['2023-24', '2024-25']

    all_games = []
    for season in seasons:
        if verbose:
            print(f"  Fetching {season}...")
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable='00',
            )
            games = finder.get_data_frames()[0]
            if verbose:
                print(f"    -> {len(games)} records")
            all_games.append(games)
            time.sleep(0.5)  # rate limiting
        except Exception as exc:
            if verbose:
                print(f"    ERROR: {exc}")

    if not all_games:
        raise ValueError("No game data fetched from any season.")

    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])

    if verbose:
        print(f"  Total: {len(combined)} records, "
              f"{combined['GAME_DATE'].min().date()} -> {combined['GAME_DATE'].max().date()}")

    return combined


def fetch_with_retry(season: str, max_retries: int = 3) -> pd.DataFrame | None:
    """Fetch one season with exponential-backoff retry; returns None on failure."""
    for attempt in range(max_retries):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                league_id_nullable='00',
                timeout=60,
            )
            return finder.get_data_frames()[0]
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_upcoming_games(days_ahead: int = 7, verbose: bool = True) -> list:
    """
    Fetch today's NBA schedule from the live scoreboard endpoint.

    Returns
    -------
    list of dict
        Each dict has: game_id, game_date, home_team, away_team,
        home_team_id, away_team_id, game_status, game_status_text.
    """
    try:
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        upcoming = []

        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            for game in games_data['scoreboard']['games']:
                upcoming.append({
                    'game_id': game.get('gameId'),
                    'game_date': game.get('gameTimeUTC'),
                    'home_team': game.get('homeTeam', {}).get('teamName'),
                    'away_team': game.get('awayTeam', {}).get('teamName'),
                    'home_team_id': game.get('homeTeam', {}).get('teamId'),
                    'away_team_id': game.get('awayTeam', {}).get('teamId'),
                    'game_status': game.get('gameStatus'),
                    'game_status_text': game.get('gameStatusText'),
                })

        if verbose:
            print(f"  Found {len(upcoming)} upcoming games")
        return upcoming
    except Exception as exc:
        if verbose:
            print(f"  Error fetching upcoming games: {exc}")
        return []


def get_team_latest_stats(games_df: pd.DataFrame, team_id: int) -> dict | None:
    """Return most recent rolling-stat snapshot for *team_id* from *games_df*."""
    team_games = games_df[games_df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    if len(team_games) == 0:
        return None
    latest = team_games.iloc[-1]
    return {
        col: latest[col]
        for col in games_df.columns
        if '_ROLL' in col or col in ('WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10')
    }


def prepare_prediction_features(
    home_stats: dict,
    away_stats: dict,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for an upcoming game.

    Parameters
    ----------
    home_stats : dict  Rolling stats for the home team (keys without prefix).
    away_stats : dict  Rolling stats for the away team.
    feature_cols : list  Ordered list of column names expected by the model.

    Returns
    -------
    pd.DataFrame  One row, columns = feature_cols.
    """
    features = {}
    for col in feature_cols:
        if col.startswith('HOME_'):
            features[col] = home_stats.get(col.replace('HOME_', '', 1), 0)
        elif col.startswith('AWAY_'):
            features[col] = away_stats.get(col.replace('AWAY_', '', 1), 0)
        else:
            features[col] = 0
    return pd.DataFrame([features])
