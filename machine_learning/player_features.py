"""
Player Feature Engineering - Rolling Stats & Team Aggregation

Computes rolling statistics per player and aggregates to team level
with minutes-weighting. Maintains chronological integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_player_rolling_stats(
    df_logs: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """
    Calculate rolling 5-game statistics per player.
    
    Parameters:
    -----------
    df_logs : pd.DataFrame
        Player game logs (from fetch_player_logs_for_team)
    window : int
        Rolling window size (default: 5 games)
    
    Returns:
    --------
    pd.DataFrame
        Same as input with new rolling stat columns appended
    """
    
    if len(df_logs) == 0:
        return df_logs
    
    df = df_logs.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    
    # Ensure numeric columns
    stat_cols = ['PTS', 'REB', 'AST', 'MIN', 'FGA', 'FG_PCT', 'STL', 'BLK', 'TOV']
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Rolling averages (grouped by player)
    for col in stat_cols:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        else:
            df[f'{col}_ROLL'] = 0.0
    
    # Efficiency metric: Points per FGA
    df['FGA'] = df['FGA'].fillna(0)
    df['PTS'] = df['PTS'].fillna(0)
    df['PTS_PER_FGA'] = df['PTS'] / (df['FGA'] + 1)  # Avoid division by zero
    
    df['PTS_PER_FGA_ROLL'] = df.groupby('PLAYER_ID')['PTS_PER_FGA'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # Rotation stability: Std dev of minutes
    df['MIN_STD_ROLL'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.rolling(window=window, min_periods=2).std().fillna(0)
    )
    
    # Fill any remaining NaN
    df = df.fillna(0)
    
    return df


def aggregate_player_stats_by_team(
    df_logs_rolled: pd.DataFrame,
    team_id: int,
    weight_by_minutes: bool = True
) -> Dict[str, float]:
    """
    Aggregate rolling player stats to team level (18 features per team).
    
    CHRONOLOGICALLY SAFE: Uses most recent game date only.
    
    Parameters:
    -----------
    df_logs_rolled : pd.DataFrame
        Player logs with rolling stats (from calculate_player_rolling_stats)
    team_id : int
        Team ID for filtering
    weight_by_minutes : bool
        Weight aggregates by minutes played (default: True)
    
    Returns:
    --------
    Dict[str, float]
        18 features with keys:
        - PLAYER_PTS_ROLL_WEIGHTED, PLAYER_REB_ROLL_WEIGHTED, etc. (8 features)
        - PLAYER_TOP_SCORER_PPG, PLAYER_TOP_REBOUNDER_RPG, PLAYER_TOP_PLAYMAKER_APG (3)
        - PLAYER_TOP_SCORER_SHARE (1)
        - PLAYER_BENCH_SCORING_PCT (1)
        - PLAYER_ACTIVE_ROTATION_SIZE, PLAYER_ROTATION_STABILITY (2)
        - PLAYER_KEY_PLAYER_MISSING, PLAYER_MINUTES_DROP_40PCT (2)
        - PLAYER_SCORING_CONCENTRATION, PLAYER_DEFENSIVE_CONTRIBUTORS (2)
    """
    
    features = {}
    
    # Default feature names (for zero-filling)
    default_features = [
        'PLAYER_PTS_ROLL_WEIGHTED', 'PLAYER_REB_ROLL_WEIGHTED', 
        'PLAYER_AST_ROLL_WEIGHTED', 'PLAYER_FGA_ROLL_WEIGHTED',
        'PLAYER_FG_PCT_ROLL_WEIGHTED', 'PLAYER_PTS_PER_FGA_ROLL_WEIGHTED',
        'PLAYER_TOP_SCORER_PPG', 'PLAYER_TOP_REBOUNDER_RPG',
        'PLAYER_TOP_PLAYMAKER_APG', 'PLAYER_TOP_SCORER_SHARE',
        'PLAYER_BENCH_SCORING_PCT', 'PLAYER_ACTIVE_ROTATION_SIZE',
        'PLAYER_ROTATION_STABILITY', 'PLAYER_KEY_PLAYER_MISSING',
        'PLAYER_MINUTES_DROP_40PCT', 'PLAYER_SCORING_CONCENTRATION',
        'PLAYER_DEFENSIVE_CONTRIBUTORS', 'PLAYER_BENCH_SCORER_COUNT'
    ]
    
    if len(df_logs_rolled) == 0:
        # Return zeros if no data
        for key in default_features:
            features[key] = 0.0
        return features
    
    # Filter to team's games
    df_team = df_logs_rolled[df_logs_rolled['TEAM_ID'] == team_id].copy()
    
    if len(df_team) == 0:
        for key in default_features:
            features[key] = 0.0
        return features
    
    # Use most recent game only
    latest_date = df_team['GAME_DATE'].max()
    df_recent = df_team[df_team['GAME_DATE'] == latest_date].copy()
    
    if len(df_recent) == 0:
        for key in default_features:
            features[key] = 0.0
        return features
    
    # Minutes weighting
    if weight_by_minutes and 'MIN_ROLL' in df_recent.columns:
        min_total = df_recent['MIN_ROLL'].sum()
        if min_total > 0:
            weights = df_recent['MIN_ROLL'] / min_total
        else:
            weights = np.ones(len(df_recent)) / len(df_recent)
    else:
        weights = np.ones(len(df_recent)) / len(df_recent)
    
    # === FEATURE 1-6: Minutes-Weighted Aggregates ===
    weighted_cols = {
        'PLAYER_PTS_ROLL_WEIGHTED': 'PTS_ROLL',
        'PLAYER_REB_ROLL_WEIGHTED': 'REB_ROLL',
        'PLAYER_AST_ROLL_WEIGHTED': 'AST_ROLL',
        'PLAYER_FGA_ROLL_WEIGHTED': 'FGA_ROLL',
        'PLAYER_FG_PCT_ROLL_WEIGHTED': 'FG_PCT_ROLL',
        'PLAYER_PTS_PER_FGA_ROLL_WEIGHTED': 'PTS_PER_FGA_ROLL',
    }
    
    for feat_name, col_name in weighted_cols.items():
        if col_name in df_recent.columns:
            features[feat_name] = float((df_recent[col_name] * weights).sum())
        else:
            features[feat_name] = 0.0
    
    # === FEATURE 7-9: Top Performers ===
    features['PLAYER_TOP_SCORER_PPG'] = float(df_recent['PTS_ROLL'].max()) if 'PTS_ROLL' in df_recent.columns else 0.0
    features['PLAYER_TOP_REBOUNDER_RPG'] = float(df_recent['REB_ROLL'].max()) if 'REB_ROLL' in df_recent.columns else 0.0
    features['PLAYER_TOP_PLAYMAKER_APG'] = float(df_recent['AST_ROLL'].max()) if 'AST_ROLL' in df_recent.columns else 0.0
    
    # === FEATURE 10: Star Player Dependency ===
    if 'PTS_ROLL' in df_recent.columns:
        total_pts = df_recent['PTS_ROLL'].sum()
        features['PLAYER_TOP_SCORER_SHARE'] = float(df_recent['PTS_ROLL'].max() / total_pts) if total_pts > 0 else 0.0
    else:
        features['PLAYER_TOP_SCORER_SHARE'] = 0.0
    
    # === FEATURE 11: Bench Scoring ===
    if 'MIN_ROLL' in df_recent.columns and 'PTS_ROLL' in df_recent.columns and len(df_recent) >= 5:
        df_sorted = df_recent.sort_values('MIN_ROLL', ascending=False)
        bench = df_sorted.iloc[5:]
        total_pts = df_sorted['PTS_ROLL'].sum()
        features['PLAYER_BENCH_SCORING_PCT'] = float(bench['PTS_ROLL'].sum() / total_pts) if total_pts > 0 else 0.0
    else:
        features['PLAYER_BENCH_SCORING_PCT'] = 0.0
    
    # === FEATURE 12: Active Rotation Size ===
    features['PLAYER_ACTIVE_ROTATION_SIZE'] = float((df_recent['MIN_ROLL'] > 5).sum()) if 'MIN_ROLL' in df_recent.columns else 0.0
    
    # === FEATURE 13: Rotation Stability ===
    features['PLAYER_ROTATION_STABILITY'] = float(df_recent['MIN_STD_ROLL'].mean()) if 'MIN_STD_ROLL' in df_recent.columns else 0.0
    
    # === FEATURE 14: Key Player Missing Detection ===
    if 'MIN_ROLL' in df_recent.columns and 'MIN' in df_recent.columns:
        top_3 = df_recent.nlargest(3, 'MIN_ROLL')['PLAYER_ID'].tolist()
        missing_count = 0
        for pid in top_3:
            player_data = df_recent[df_recent['PLAYER_ID'] == pid]
            if len(player_data) > 0 and player_data['MIN'].iloc[-1] < 10:
                missing_count += 1
        features['PLAYER_KEY_PLAYER_MISSING'] = float(missing_count)
    else:
        features['PLAYER_KEY_PLAYER_MISSING'] = 0.0
    
    # === FEATURE 15: Injury Proxy (Minutes Drop >40%) ===
    if 'MIN' in df_recent.columns and 'MIN_ROLL' in df_recent.columns:
        drops = 0
        for _, p in df_recent.iterrows():
            if p['MIN_ROLL'] > 15:
                pct_drop = (p['MIN_ROLL'] - p['MIN']) / (p['MIN_ROLL'] + 1)
                if pct_drop > 0.40:
                    drops += 1
        features['PLAYER_MINUTES_DROP_40PCT'] = float(drops)
    else:
        features['PLAYER_MINUTES_DROP_40PCT'] = 0.0
    
    # === FEATURE 16: Scoring Concentration ===
    if 'PTS_ROLL' in df_recent.columns and len(df_recent) > 1:
        pts_nz = df_recent['PTS_ROLL'].values[df_recent['PTS_ROLL'].values > 0]
        if len(pts_nz) > 1:
            features['PLAYER_SCORING_CONCENTRATION'] = float((pts_nz.max() - pts_nz.mean()) / (pts_nz.max() + 1))
        else:
            features['PLAYER_SCORING_CONCENTRATION'] = 0.0
    else:
        features['PLAYER_SCORING_CONCENTRATION'] = 0.0
    
    # === FEATURE 17: Defensive Contributors ===
    if 'STL_ROLL' in df_recent.columns and 'BLK_ROLL' in df_recent.columns:
        features['PLAYER_DEFENSIVE_CONTRIBUTORS'] = float(((df_recent['STL_ROLL'] > 1.0) | (df_recent['BLK_ROLL'] > 0.5)).sum())
    else:
        features['PLAYER_DEFENSIVE_CONTRIBUTORS'] = 0.0
    
    # === FEATURE 18: Bench Scorer Count ===
    if 'MIN_ROLL' in df_recent.columns:
        features['PLAYER_BENCH_SCORER_COUNT'] = float(((df_recent['MIN_ROLL'] > 2) & (df_recent['MIN_ROLL'] <= 15)).sum())
    else:
        features['PLAYER_BENCH_SCORER_COUNT'] = 0.0
    
    # Fill missing with 0
    for key in default_features:
        if key not in features:
            features[key] = 0.0
    
    return features
