"""
MLB Feature Engineering Module

Mirrors data/feature_engineering.py for baseball: leakage-safe rolling
run-differential stats and matchup-level (home vs away) feature construction.

Output schema matches what machine_learning.lightgbm_models.prepare_features_and_target
already expects (POINT_DIFF, HOME_WIN, HOME_*/AWAY_* rolling columns), so that
module needs zero changes to support MLB.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

MLB_DEFAULT_FORM_WINDOW = 10
MLB_BOX_SCORE_STATS = ['R', 'RA']


# ---------------------------------------------------------------------------
# Team-level features
# ---------------------------------------------------------------------------

def calculate_mlb_rolling_stats(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """
    Add rolling-average, streak, rest-day, and win-rate columns to MLB team-game logs.

    All rolling windows use shift(1) so the current game is never included
    (strict chronological / leakage-safe). Adapted from
    data.feature_engineering.calculate_rolling_stats for runs scored/allowed
    instead of box-score stats.

    Columns added
    -------------
    R_ROLL, RA_ROLL  : rolling mean runs for / against (prior games only)
    WIN_STREAK       : consecutive wins (+) / losses (-), based on prior games
    REST_DAYS        : calendar days since last game for this team
    IS_BACK_TO_BACK  : 1 if REST_DAYS == 1 else 0
    WIN_RATE_10      : rolling 10-game win percentage (prior games only)
    """
    if window is None:
        window = MLB_DEFAULT_FORM_WINDOW
    df = df.copy().sort_values(['TEAM_NAME', 'GAME_DATE'])

    for col in MLB_BOX_SCORE_STATS:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_NAME')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    def _streak(wl_series: pd.Series) -> pd.Series:
        result, cur = [], 0
        for wl in wl_series:
            if wl == 'W':
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            result.append(cur)
        return pd.Series(result, index=wl_series.index)

    df['WIN_STREAK'] = (
        df.groupby('TEAM_NAME')['WL']
        .transform(_streak)
        .shift(1)
        .fillna(0)
    )

    df['REST_DAYS'] = df.groupby('TEAM_NAME')['GAME_DATE'].diff().dt.days.fillna(2)
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)

    df['WIN_RATE_10'] = df.groupby('TEAM_NAME')['WL'].transform(
        lambda x: (x == 'W').shift(1).rolling(window=10, min_periods=1).mean()
    )

    return df


def create_mlb_matchup_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from per-team rows to one-row-per-game with HOME_* / AWAY_* columns.

    Uses the IS_HOME flag set directly by data.mlb_loader.fetch_mlb_games
    (no matchup-string parsing needed, unlike the NBA equivalent).

    Returns
    -------
    pd.DataFrame with columns:
        GAME_ID, GAME_DATE, HOME_TEAM_NAME, AWAY_TEAM_NAME,
        HOME_<stat>_ROLL ... AWAY_<stat>_ROLL, WIN_STREAK, REST_DAYS ...,
        HOME_R, AWAY_R, POINT_DIFF, HOME_WIN
    """
    rolling_cols = [
        c for c in games_df.columns
        if '_ROLL' in c or c in ('WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10')
    ]

    matchups = []
    for game_id, grp in games_df.groupby('GAME_ID'):
        if len(grp) != 2:
            continue

        home_mask = grp['IS_HOME'] == 1
        if home_mask.sum() != 1:
            continue
        home = grp[home_mask].iloc[0]
        away = grp[~home_mask].iloc[0]

        row = {
            'GAME_ID': game_id,
            'GAME_DATE': home['GAME_DATE'],
            'HOME_TEAM_NAME': home['TEAM_NAME'],
            'AWAY_TEAM_NAME': away['TEAM_NAME'],
        }
        for col in rolling_cols:
            row[f'HOME_{col}'] = home[col]
            row[f'AWAY_{col}'] = away[col]

        row['HOME_R'] = home['R']
        row['AWAY_R'] = away['R']
        row['POINT_DIFF'] = home['R'] - away['R']
        row['HOME_WIN'] = 1 if home['WL'] == 'W' else 0
        matchups.append(row)

    return pd.DataFrame(matchups)


def prepare_mlb_training_data(
    games_with_stats: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build the MLB matchup DataFrame and return (matchup_df, y_point_diff, feature_cols).

    Mirrors data.feature_engineering.prepare_training_data.
    """
    if verbose:
        print("  Building MLB matchup features...")

    matchup_df = create_mlb_matchup_features(games_with_stats)
    matchup_df = matchup_df.dropna()

    if verbose:
        print(f"    {len(matchup_df)} training rows, "
              f"{matchup_df['GAME_DATE'].min()} -> {matchup_df['GAME_DATE'].max()}")

    _exclude = {
        'HOME_TEAM_NAME', 'AWAY_TEAM_NAME', 'HOME_R', 'AWAY_R', 'HOME_WIN',
    }
    feature_cols = [
        c for c in matchup_df.columns
        if (c.startswith('HOME_') or c.startswith('AWAY_')) and c not in _exclude
    ]

    y = matchup_df['POINT_DIFF'].values
    if verbose:
        print(f"    {len(feature_cols)} feature columns")

    return matchup_df, y, feature_cols
