"""data.extended_data_loader

Extended dataset utilities built on top of the authoritative modules:
- data.nba_loader
- data.feature_engineering

Responsibilities:
- Fetch multi-season historical game logs
- Compute leakage-safe rolling stats
- Optionally cache games to SQLite
- Build a matchup-level training dataset for model training
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.database.database_handler import SportsAnalyticsDB
from data.feature_engineering import prepare_training_data as _prepare_training_data
from data.nba_loader import fetch_nba_games, get_all_nba_teams


def fetch_comprehensive_nba_data(
    seasons: Optional[List[str]] = None,
    season_type: str = 'Regular Season',
    use_cache: bool = True,
    db_path: str = 'sports_analytics.db',
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch multi-season game logs and compute rolling stats.

    Notes
    -----
    - Rolling stats are computed with leakage prevention inside
      data.feature_engineering.calculate_rolling_stats.
    - Cache reads are intentionally conservative: this function will always
      fetch fresh data (nba_api) and then write it to the DB if enabled.
      This keeps behavior predictable until a robust cache hydrate is added.
    """
    from data.feature_engineering import calculate_rolling_stats

    if seasons is None:
        seasons = ['2022-23', '2023-24', '2024-25']

    if verbose:
        print('=' * 70)
        print(f'COMPREHENSIVE DATA FETCH: {len(seasons)} seasons')
        print('=' * 70)
        print('Seasons:', ', '.join(seasons))
        print('Season type:', season_type)
        print('Cache:', 'enabled' if use_cache else 'disabled')

    games_df = fetch_nba_games(seasons=seasons, season_type=season_type, verbose=verbose)
    games_with_stats = calculate_rolling_stats(games_df, window=5)

    if use_cache:
        cache_games_to_db(games_with_stats, db_path=db_path, verbose=verbose)

    return games_with_stats


def cache_games_to_db(
    games_df: pd.DataFrame,
    db_path: str,
    verbose: bool = False,
) -> int:
    """Cache unique games to the DB (one row per GAME_ID)."""
    if games_df.empty:
        return 0

    cached_count = 0
    with SportsAnalyticsDB(db_path) as db:
        for game_id in games_df['GAME_ID'].unique():
            game_rows = games_df[games_df['GAME_ID'] == game_id]
            if len(game_rows) < 2:
                continue

            home_mask = game_rows['MATCHUP'].astype(str).str.contains('vs\.', na=False)
            if int(home_mask.sum()) == 1:
                home_row = game_rows[home_mask].iloc[0]
                away_row = game_rows[~home_mask].iloc[0]
            else:
                home_row = game_rows.iloc[0]
                away_row = game_rows.iloc[1]

            def _to_json_safe(obj):
                """Convert pandas Series or dict to JSON-serializable format."""
                if isinstance(obj, pd.Series):
                    obj = obj.to_dict()
                if isinstance(obj, dict):
                    return {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in obj.items()}
                return obj
            
            game_data = {
                'game_id': str(game_id),
                'game_date': str(pd.to_datetime(home_row['GAME_DATE']).date()),
                'season': str(home_row.get('SEASON_ID', '') or ''),
                'home_team': str(home_row.get('TEAM_NAME', home_row.get('TEAM_ID'))),
                'away_team': str(away_row.get('TEAM_NAME', away_row.get('TEAM_ID'))),
                'home_team_id': int(home_row.get('TEAM_ID')),
                'away_team_id': int(away_row.get('TEAM_ID')),
                'home_score': int(home_row.get('PTS')) if pd.notna(home_row.get('PTS')) else None,
                'away_score': int(away_row.get('PTS')) if pd.notna(away_row.get('PTS')) else None,
                'game_status': 'Final',
                'stats': {'home': _to_json_safe(home_row), 'away': _to_json_safe(away_row)},
            }

            try:
                db.cache_game(game_data)
                cached_count += 1
            except Exception as exc:
                if verbose:
                    print(f'Warning: could not cache game {game_id}: {exc}')

    if verbose:
        print(f'Cached {cached_count} unique games')
    return cached_count


def prepare_training_data(
    games_with_stats: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Compatibility wrapper for training data prep."""
    matchup_df, y, feature_cols = _prepare_training_data(games_with_stats, verbose=verbose)
    return matchup_df, y, feature_cols


def get_extended_training_dataset(
    db_path: str = 'sports_analytics.db',
    verbose: bool = True,
) -> Dict[str, object]:
    """Return a ready-to-train dataset bundle.

    Returns
    -------
    dict with keys:
      - games_df
      - matchup_df
      - X
      - y
      - feature_names
      - team_data
    """
    if verbose:
        print('\n' + '=' * 70)
        print('LOADING EXTENDED TRAINING DATASET')
        print('=' * 70)

    games_df = fetch_comprehensive_nba_data(
        seasons=['2022-23', '2023-24', '2024-25'],
        season_type='Regular Season',
        use_cache=True,
        db_path=db_path,
        verbose=verbose,
    )

    matchup_df, y, feature_names = prepare_training_data(games_df, verbose=verbose)
    X = matchup_df[feature_names].values
    team_info = get_all_nba_teams()

    if verbose:
        print('=' * 70)
        print('DATASET READY')
        print(f'Samples: {len(X)}')
        print(f'Features: {len(feature_names)}')
        print(f'Teams: {len(team_info.get("ids", []))}')
        print('=' * 70 + '\n')

    return {
        'games_df': games_df,
        'matchup_df': matchup_df,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'team_data': team_info,
    }


def refresh_recent_data(
    existing_df: pd.DataFrame,
    days_back: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """Refresh the last N days of games and recompute rolling stats."""
    from data.feature_engineering import calculate_rolling_stats

    if existing_df.empty:
        return existing_df

    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered_df = existing_df[pd.to_datetime(existing_df['GAME_DATE']) < cutoff_date].copy()

    current_season = '2024-25'
    recent_games = fetch_nba_games(seasons=[current_season], verbose=verbose)
    recent_games = recent_games[pd.to_datetime(recent_games['GAME_DATE']) >= cutoff_date]

    updated_df = pd.concat([filtered_df, recent_games], ignore_index=True)
    updated_df = updated_df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    updated_df = calculate_rolling_stats(updated_df, window=5)

    if verbose:
        print(f'Added {len(recent_games)} recent rows')

    return updated_df
