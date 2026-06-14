"""
Feature Engineering Module

Authoritative source for all feature construction:
- Rolling team statistics (leakage-protected)
- Matchup feature creation (home vs. away rows)
- Player-level rolling aggregates (stubs for integration_pipeline.py)
- Training-data preparation helper
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from data.sport_config import get_form_windows, get_feature_set

NBA_DEFAULT_FORM_WINDOW = get_form_windows("NBA")[0]
NBA_BOX_SCORE_STATS = get_feature_set("NBA", category="box_score_stats")


# ---------------------------------------------------------------------------
# Team-level features
# ---------------------------------------------------------------------------

def calculate_rolling_stats(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """
    Add rolling-average, streak, rest-day, and win-rate columns to game logs.

    All rolling windows use shift(1) so the current game is never included
    (strict chronological / leakage-safe).

    Columns added
    -------------
    {COL}_ROLL      : rolling mean of box-score stats (see sport_config NBA)
    WIN_STREAK      : consecutive wins (+) / losses (-), based on prior games
    REST_DAYS       : calendar days since last game for this team
    IS_BACK_TO_BACK : 1 if REST_DAYS == 1 else 0
    WIN_RATE_10     : rolling 10-game win percentage (prior games only)
    """
    if window is None:
        window = NBA_DEFAULT_FORM_WINDOW
    df = df.copy().sort_values(['TEAM_ID', 'GAME_DATE'])

    # Basic rolling averages (shift to exclude current game)
    for col in NBA_BOX_SCORE_STATS:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    # Win streak (positive = consecutive wins, negative = consecutive losses)
    def _streak(wl_series: pd.Series) -> pd.Series:
        result, cur = [], 0
        for wl in wl_series:
            if wl == 'W':
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            result.append(cur)
        return pd.Series(result, index=wl_series.index)

    # Shift by 1 so the current game's outcome is not visible
    df['WIN_STREAK'] = (
        df.groupby('TEAM_ID')['WL']
        .transform(_streak)
        .shift(1)
        .fillna(0)
    )

    # Rest days
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(2)
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)

    # 10-game win rate (excluding current game)
    df['WIN_RATE_10'] = df.groupby('TEAM_ID')['WL'].transform(
        lambda x: (x == 'W').shift(1).rolling(window=10, min_periods=1).mean()
    )

    return df


def create_matchup_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from per-team rows to one-row-per-game with HOME_* / AWAY_* columns.

    The home team is the one whose MATCHUP contains 'vs.' and the away team
    has '@' in MATCHUP. Falls back to row order when the pattern is absent.

    Returns
    -------
    pd.DataFrame with columns:
        GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM,
        HOME_TEAM_NAME, AWAY_TEAM_NAME,
        HOME_<stat>_ROLL … AWAY_<stat>_ROLL, WIN_STREAK, REST_DAYS …,
        HOME_PTS, AWAY_PTS, POINT_DIFF, HOME_WIN
    """
    from data.nba_loader import get_all_nba_teams
    team_names = get_all_nba_teams()['names']

    rolling_cols = [
        c for c in games_df.columns
        if '_ROLL' in c or c in ('WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10')
    ]

    matchups = []
    for game_id, grp in games_df.groupby('GAME_ID'):
        if len(grp) != 2:
            continue

        home_mask = grp['MATCHUP'].str.contains('vs.', na=False)
        if home_mask.sum() == 1:
            home = grp[home_mask].iloc[0]
            away = grp[~home_mask].iloc[0]
        else:
            home, away = grp.iloc[0], grp.iloc[1]

        row = {
            'GAME_ID': game_id,
            'GAME_DATE': home['GAME_DATE'],
            'HOME_TEAM': home['TEAM_ID'],
            'AWAY_TEAM': away['TEAM_ID'],
            'HOME_TEAM_NAME': team_names.get(home['TEAM_ID'], 'Unknown'),
            'AWAY_TEAM_NAME': team_names.get(away['TEAM_ID'], 'Unknown'),
        }
        for col in rolling_cols:
            row[f'HOME_{col}'] = home[col]
            row[f'AWAY_{col}'] = away[col]

        row['HOME_PTS'] = home['PTS']
        row['AWAY_PTS'] = away['PTS']
        row['POINT_DIFF'] = home['PTS'] - away['PTS']
        row['HOME_WIN'] = 1 if home['WL'] == 'W' else 0
        matchups.append(row)

    return pd.DataFrame(matchups)


def prepare_training_data(
    games_with_stats: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build the matchup DataFrame and return (matchup_df, y_point_diff, feature_cols).

    feature_cols excludes identity/name columns and outcome columns so that
    the resulting X matrix can be fed directly to any model.
    """
    if verbose:
        print("  Building matchup features...")

    matchup_df = create_matchup_features(games_with_stats)
    matchup_df = matchup_df.dropna()

    if verbose:
        print(f"    {len(matchup_df)} training rows, "
              f"{matchup_df['GAME_DATE'].min()} -> {matchup_df['GAME_DATE'].max()}")

    _exclude = {
        'HOME_TEAM', 'AWAY_TEAM', 'HOME_TEAM_NAME', 'AWAY_TEAM_NAME',
        'HOME_PTS', 'AWAY_PTS', 'HOME_WIN',
    }
    feature_cols = [
        c for c in matchup_df.columns
        if (c.startswith('HOME_') or c.startswith('AWAY_')) and c not in _exclude
    ]

    y = matchup_df['POINT_DIFF'].values
    if verbose:
        print(f"    {len(feature_cols)} feature columns")

    return matchup_df, y, feature_cols


# ---------------------------------------------------------------------------
# Player-level stubs (required by integration_pipeline.py)
# ---------------------------------------------------------------------------

def calculate_player_rolling_stats(
    player_logs: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add per-player rolling averages.

    Rolls PTS, REB, AST, FG_PCT, MIN (shift(1) for leakage safety).
    """
    df = player_logs.copy().sort_values(['PLAYER_ID', 'GAME_DATE'])
    for col in ('PTS', 'REB', 'AST', 'FG_PCT', 'MIN'):
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
    return df


def aggregate_player_stats_by_team(
    player_logs_with_rolling: pd.DataFrame,
    game_date=None,
) -> dict:
    """
    Aggregate player rolling stats to the team level.

    Returns a dict mapping team_id -> {feature_name: value}.
    Useful for building the 36 player-level features in EnhancedFeaturePipeline.
    """
    required = [c for c in player_logs_with_rolling.columns if c.endswith('_ROLL')]
    if not required:
        return {}

    if game_date is not None:
        df = player_logs_with_rolling[
            player_logs_with_rolling['GAME_DATE'] < game_date
        ]
    else:
        df = player_logs_with_rolling

    team_stats = {}
    for team_id, grp in df.groupby('TEAM_ID'):
        latest = grp.sort_values('GAME_DATE').groupby('PLAYER_ID').last()
        team_stats[team_id] = {
            **{f'TEAM_AVG_{col}': latest[col].mean() for col in required},
            **{f'TEAM_TOP_{col}': latest[col].max() for col in required},
        }
    return team_stats
