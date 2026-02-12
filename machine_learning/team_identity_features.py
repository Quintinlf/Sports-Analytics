"""
Team Identity and Opponent-Adjusted Statistics Feature Engineering

This module provides functions for:
1. Encoding team identities as numerical indices
2. Computing opponent-adjusted statistics normalized vs league average
3. Regularizing feature importance to prevent overfitting to streak patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def add_team_identity_encoding(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode team identities as numerical indices for machine learning features.
    
    Creates consistent numerical team IDs (0 to N-1) for both home and away teams,
    allowing models to learn team-specific strengths beyond rolling statistics.
    
    Parameters
    ----------
    matchup_df : pd.DataFrame
        Matchup dataframe with HOME_TEAM and AWAY_TEAM columns containing
        team names or identifiers.
    
    Returns
    -------
    pd.DataFrame
        Input dataframe with added columns:
        - HOME_TEAM_ID : int (0 to N-1 where N is number of unique teams)
        - AWAY_TEAM_ID : int (0 to N-1)
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'HOME_TEAM': [1610612738, 1610612738, 1610612739],
    ...     'AWAY_TEAM': [1610612739, 1610612740, 1610612738]
    ... })
    >>> result = add_team_identity_encoding(df)
    >>> 'HOME_TEAM_ID' in result.columns and 'AWAY_TEAM_ID' in result.columns
    True
    """
    df = matchup_df.copy()
    
    # Get all unique teams from both home and away columns
    all_teams = pd.concat([
        df['HOME_TEAM'] if 'HOME_TEAM' in df.columns else pd.Series([]),
        df['AWAY_TEAM'] if 'AWAY_TEAM' in df.columns else pd.Series([])
    ]).unique()
    
    all_teams = sorted(all_teams)
    
    # Create mapping from team identifier to numerical ID (0 to N-1)
    team_to_id = {team: idx for idx, team in enumerate(all_teams)}
    
    # Apply mapping to create ID columns
    if 'HOME_TEAM' in df.columns:
        df['HOME_TEAM_ID'] = df['HOME_TEAM'].map(team_to_id).fillna(-1).astype(int)
    
    if 'AWAY_TEAM' in df.columns:
        df['AWAY_TEAM_ID'] = df['AWAY_TEAM'].map(team_to_id).fillna(-1).astype(int)
    
    return df


def add_opponent_adjusted_stats(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute opponent-adjusted statistics normalized against league averages.
    
    Normalizes team statistics against league averages by computing
    (team_stat / league_avg - 1), producing values where 0 indicates
    league average performance. This helps models understand relative
    team strength rather than absolute values.
    
    Parameters
    ----------
    matchup_df : pd.DataFrame
        Matchup dataframe with rolling statistics columns. Expected patterns:
        - HOME_*_ROLL, AWAY_*_ROLL for basic stats
        - HOME_*_ROLL, AWAY_*_ROLL for advanced stats
    
    Returns
    -------
    pd.DataFrame
        Input dataframe with added opponent-adjusted columns with _ADJ suffix.
        For each stat pair (HOME_X_ROLL, AWAY_X_ROLL), creates:
        - HOME_X_ADJ : float (normalized deviation from league average)
        - AWAY_X_ADJ : float (normalized deviation from league average)
    
    Notes
    -----
    Adjusted stat formula: (team_stat / league_avg - 1)
    - Value of 0.0 means team is at league average
    - Value of +0.1 means team is 10% above league average
    - Value of -0.1 means team is 10% below league average
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'HOME_PTS_ROLL': [110.0, 105.0, 100.0],
    ...     'AWAY_PTS_ROLL': [100.0, 108.0, 95.0]
    ... })
    >>> result = add_opponent_adjusted_stats(df)
    >>> 'HOME_PTS_ADJ' in result.columns
    True
    """
    df = matchup_df.copy()
    
    # Define stat patterns to adjust (rolling averages)
    stat_bases = [
        'PTS', 'FG', 'FG3', 'FT', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'TS_PCT', 'EFG_PCT', 'AST_TO_RATIO', 'POSS_APPROX', 'OFF_RTG_APPROX', 'FT_RATE',
        'PLUS_MINUS'
    ]
    
    for stat_base in stat_bases:
        home_col = f'HOME_{stat_base}_ROLL'
        away_col = f'AWAY_{stat_base}_ROLL'
        
        # Only process if both columns exist
        if home_col in df.columns and away_col in df.columns:
            # Calculate league average (mean across all home and away values)
            home_values = df[home_col].replace([np.inf, -np.inf], np.nan).fillna(0)
            away_values = df[away_col].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            league_avg = (home_values.mean() + away_values.mean()) / 2
            
            # Avoid division by zero
            if league_avg > 0:
                # Create adjusted columns: (team_stat / league_avg - 1)
                df[f'HOME_{stat_base}_ADJ'] = (home_values / league_avg - 1).replace([np.inf, -np.inf], 0).fillna(0)
                df[f'AWAY_{stat_base}_ADJ'] = (away_values / league_avg - 1).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def get_team_id_mapping(matchup_df: pd.DataFrame) -> Dict:
    """
    Extract team identifier to ID mapping from a matchup dataframe.
    
    Useful for retrieving the team encoding scheme after it has been applied,
    allowing external code to map team names/IDs to their numerical indices.
    
    Parameters
    ----------
    matchup_df : pd.DataFrame
        Matchup dataframe with HOME_TEAM, AWAY_TEAM, HOME_TEAM_ID, AWAY_TEAM_ID columns.
    
    Returns
    -------
    Dict
        Dictionary mapping team identifiers to numerical IDs.
        Returns empty dict if required columns are missing.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'HOME_TEAM': [1610612738, 1610612739],
    ...     'HOME_TEAM_ID': [0, 1],
    ...     'AWAY_TEAM': [1610612739, 1610612738],
    ...     'AWAY_TEAM_ID': [1, 0]
    ... })
    >>> mapping = get_team_id_mapping(df)
    >>> len(mapping) == 2
    True
    """
    required_cols = ['HOME_TEAM', 'AWAY_TEAM', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    
    if not all(col in matchup_df.columns for col in required_cols):
        return {}
    
    # Build mapping from both home and away pairs
    mapping = {}
    
    for _, row in matchup_df.iterrows():
        mapping[row['HOME_TEAM']] = row['HOME_TEAM_ID']
        mapping[row['AWAY_TEAM']] = row['AWAY_TEAM_ID']
    
    return mapping


def regularize_win_streak_weight(
    feature_importance: Dict[str, float],
    max_ratio: float = 2.0
) -> Dict[str, float]:
    """
    Regularize feature importance by capping WIN_STREAK weight.
    
    Prevents over-reliance on win streak patterns by ensuring WIN_STREAK
    importance does not exceed a multiple of the next highest feature.
    This helps prevent the model from becoming a "streak chaser" that
    ignores fundamental team statistics.
    
    Parameters
    ----------
    feature_importance : Dict[str, float]
        Dictionary mapping feature names to their importance values.
    max_ratio : float, optional
        Maximum allowed ratio of WIN_STREAK to second-highest feature.
        Default is 2.0 (WIN_STREAK can be at most 2x the next feature).
    
    Returns
    -------
    Dict[str, float]
        Modified feature importance dictionary with WIN_STREAK capped
        if it exceeded the max_ratio threshold.
    
    Examples
    --------
    >>> importance = {
    ...     'HOME_WIN_STREAK': 1000.0,
    ...     'HOME_PTS_ADJ': 200.0,
    ...     'AWAY_PTS_ADJ': 150.0
    ... }
    >>> result = regularize_win_streak_weight(importance, max_ratio=2.0)
    >>> result['HOME_WIN_STREAK'] <= result['HOME_PTS_ADJ'] * 2
    True
    """
    result = feature_importance.copy()
    
    # Find all WIN_STREAK related features
    streak_features = [k for k in result.keys() if 'WIN_STREAK' in k.upper()]
    
    if not streak_features:
        # No WIN_STREAK features to regularize
        return result
    
    # Get all non-streak features
    other_features = {k: v for k, v in result.items() if 'WIN_STREAK' not in k.upper()}
    
    if not other_features:
        # No other features to compare against
        return result
    
    # Find maximum importance among non-streak features
    max_other_importance = max(other_features.values())
    max_allowed_streak = max_other_importance * max_ratio
    
    # Cap each WIN_STREAK feature if it exceeds the limit
    for streak_feature in streak_features:
        if result[streak_feature] > max_allowed_streak:
            result[streak_feature] = max_allowed_streak
    
    return result
