"""
Advanced Feature Engineering for NBA Predictions

Computes advanced basketball metrics from box score data:
- True Shooting % (TS%)
- Effective Field Goal % (EFG%)
- Assist-to-Turnover Ratio
- Approximate Possessions & Offensive Rating
- Free Throw Rate
- PLUS_MINUS rolling average

Also fetches season-level advanced stats from NBA API:
- OFF_RATING, DEF_RATING, NET_RATING, PACE
"""

import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


def calculate_advanced_rolling_stats(games_df, window=5):
    """
    Compute advanced metrics from box score data and add rolling averages.
    
    Call this AFTER calculate_rolling_stats() from data_loader.py.
    Adds rolling versions of: TS_PCT, EFG_PCT, AST_TO_RATIO, 
    POSS_APPROX, OFF_RTG_APPROX, FT_RATE, PLUS_MINUS
    
    Args:
        games_df: DataFrame from calculate_rolling_stats()
        window: Rolling window size (default 5 games)
    
    Returns:
        DataFrame with additional advanced rolling features
    """
    df = games_df.copy()
    
    # --- Compute per-game advanced metrics ---
    
    # True Shooting Percentage
    if 'FGA' in df.columns and 'FTA' in df.columns:
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']) + 0.001)
    
    # Effective Field Goal Percentage
    if 'FGM' in df.columns and 'FG3M' in df.columns and 'FGA' in df.columns:
        df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / (df['FGA'] + 0.001)
    
    # Assist-to-Turnover Ratio
    if 'AST' in df.columns and 'TOV' in df.columns:
        df['AST_TO_RATIO'] = df['AST'] / (df['TOV'] + 1)
    
    # Approximate Possessions (Dean Oliver formula, simplified)
    if all(c in df.columns for c in ['FGA', 'FTA', 'OREB', 'TOV']):
        df['POSS_APPROX'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
    
    # Approximate Offensive Rating (points per 100 possessions)
    if 'POSS_APPROX' in df.columns:
        df['OFF_RTG_APPROX'] = df['PTS'] / (df['POSS_APPROX'] + 0.001) * 100
    
    # Free Throw Rate
    if 'FTA' in df.columns and 'FGA' in df.columns:
        df['FT_RATE'] = df['FTA'] / (df['FGA'] + 0.001)

    # --- Four Factors: Turnover Percentage (turnovers per possession opportunity) ---
    # Lower is better — the "unforced errors" metric
    if all(c in df.columns for c in ['TOV', 'FGA', 'FTA']):
        df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'] + 0.001)

    # --- Four Factors: Offensive Rebound Percentage (approx, without opponent totals) ---
    # Higher is better — second-chance opportunity creation
    if all(c in df.columns for c in ['OREB', 'DREB']):
        df['OREB_PCT_APPROX'] = df['OREB'] / (df['OREB'] + df['DREB'] + 0.001)

    # --- Net Rating per 100 Possessions (decision quality "north star") ---
    # PLUS_MINUS / possessions * 100 → how many points better per 100 possessions
    if all(c in df.columns for c in ['PLUS_MINUS', 'POSS_APPROX']):
        df['NET_RTG_APPROX'] = df['PLUS_MINUS'] / (df['POSS_APPROX'] + 0.001) * 100
    elif 'PLUS_MINUS' in df.columns:
        df['NET_RTG_APPROX'] = df['PLUS_MINUS']  # fallback: raw plus/minus

    # --- Pythagorean Win Probability (expected wins from scoring efficiency) ---
    # Formula: PF^1.67 / (PF^1.67 + PA^1.67)  — if record > Pyth → over-performing
    # PA (points allowed) = PTS - PLUS_MINUS
    if all(c in df.columns for c in ['PTS', 'PLUS_MINUS']):
        _pf = df['PTS'].clip(lower=1)
        _pa = (df['PTS'] - df['PLUS_MINUS']).clip(lower=1)
        df['PYT_WIN_PCT'] = _pf ** 1.67 / (_pf ** 1.67 + _pa ** 1.67)

    # --- Roll the advanced metrics ---
    advanced_cols = [
        'TS_PCT', 'EFG_PCT', 'AST_TO_RATIO',
        'POSS_APPROX', 'OFF_RTG_APPROX', 'FT_RATE',
        # Four Factors additions
        'TOV_PCT', 'OREB_PCT_APPROX',
        # Decision-quality metrics
        'NET_RTG_APPROX', 'PYT_WIN_PCT',
    ]

    for col in advanced_cols:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    # PLUS_MINUS rolling (shifted to prevent leakage)
    if 'PLUS_MINUS' in df.columns:
        df['PLUS_MINUS_ROLL'] = df.groupby('TEAM_ID')['PLUS_MINUS'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    n_new = len(advanced_cols) + 1
    print(f"   ✅ Added {n_new} advanced rolling features (incl. Four Factors + Net Rating + Pythagorean Wins)")
    return df


def fetch_season_advanced_stats(seasons=None):
    """
    Fetch team-level advanced stats from NBA API (OFF_RATING, DEF_RATING, etc.)
    
    These are season aggregates merged by TEAM_ID into matchup features.
    Falls back gracefully if API call fails.
    
    Args:
        seasons: List of season strings, e.g. ['2023-24', '2024-25']
    
    Returns:
        DataFrame with TEAM_ID + advanced stats, or None on failure
    """
    if seasons is None:
        seasons = ['2024-25']
    
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
    except ImportError:
        print("   ⚠️  nba_api not installed, skipping advanced stats")
        return None
    
    all_stats = []
    
    for season in seasons:
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            
            # Keep key columns
            keep_cols = ['TEAM_ID']
            for col in ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'PIE',
                        'TS_PCT', 'AST_PCT', 'AST_TO', 'OREB_PCT', 'DREB_PCT',
                        'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT']:
                if col in stats.columns:
                    keep_cols.append(col)
            
            season_stats = stats[keep_cols].copy()
            season_stats['SEASON'] = season
            all_stats.append(season_stats)
            
            print(f"   ✅ Advanced stats for {season}: {len(season_stats)} teams")
            time.sleep(0.6)  # Rate limiting
            
        except Exception as e:
            print(f"   ⚠️  Could not fetch advanced stats for {season}: {e}")
            continue
    
    if not all_stats:
        return None
    
    return pd.concat(all_stats, ignore_index=True)


def merge_advanced_stats_to_matchups(matchup_df, advanced_df):
    """
    Merge season-level advanced stats into matchup features.
    
    Adds HOME_ADV_OFF_RATING, HOME_ADV_DEF_RATING, etc. and AWAY_ equivalents.
    
    Args:
        matchup_df: Matchup DataFrame from create_matchup_features()
        advanced_df: DataFrame from fetch_season_advanced_stats()
    
    Returns:
        Enhanced matchup DataFrame
    """
    if advanced_df is None:
        return matchup_df
    
    df = matchup_df.copy()
    
    # Map game dates to seasons
    def date_to_season(date):
        if date.month >= 10:
            return f"{date.year}-{str(date.year + 1)[2:]}"
        else:
            return f"{date.year - 1}-{str(date.year)[2:]}"
    
    df['_SEASON'] = df['GAME_DATE'].apply(date_to_season)
    
    # Get stat columns (everything except TEAM_ID and SEASON)
    stat_cols = [c for c in advanced_df.columns if c not in ['TEAM_ID', 'SEASON']]
    
    # Merge for home team
    home_rename = {'TEAM_ID': 'HOME_TEAM', 'SEASON': '_SEASON'}
    home_rename.update({c: f'HOME_ADV_{c}' for c in stat_cols})
    home_merge = advanced_df.rename(columns=home_rename)
    df = df.merge(home_merge, on=['HOME_TEAM', '_SEASON'], how='left')
    
    # Merge for away team
    away_rename = {'TEAM_ID': 'AWAY_TEAM', 'SEASON': '_SEASON'}
    away_rename.update({c: f'AWAY_ADV_{c}' for c in stat_cols})
    away_merge = advanced_df.rename(columns=away_rename)
    df = df.merge(away_merge, on=['AWAY_TEAM', '_SEASON'], how='left')
    
    df = df.drop(columns=['_SEASON'], errors='ignore')
    
    n_adv_cols = len([c for c in df.columns if '_ADV_' in c])
    print(f"   ✅ Merged {n_adv_cols} season-level advanced stat columns")
    
    return df


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
    
    # Rolling averages (grouped by player, shifted to prevent leakage)
    for col in stat_cols:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        else:
            df[f'{col}_ROLL'] = 0.0
    
    # Efficiency metric: Points per FGA
    df['FGA'] = df['FGA'].fillna(0)
    df['PTS'] = df['PTS'].fillna(0)
    df['PTS_PER_FGA'] = df['PTS'] / (df['FGA'] + 1)  # Avoid division by zero
    
    df['PTS_PER_FGA_ROLL'] = df.groupby('PLAYER_ID')['PTS_PER_FGA'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    # Rotation stability: Std dev of minutes (shifted to prevent leakage)
    df['MIN_STD_ROLL'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=2).std().fillna(0)
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
        'PLUS_MINUS',
        # Four Factors
        'TOV_PCT', 'OREB_PCT_APPROX',
        # Decision-quality metrics
        'NET_RTG_APPROX', 'PYT_WIN_PCT',
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


"""
REBUILD SCRIPT: Fresh dataset + retrained models with FIXED features

This script:
1. Clears all caches
2. Rebuilds dataset from scratch (2023-24 + 2024-25)
3. Retrains ML models with clean features (no leakage)

Run this once, then report the results.
"""

import os
import sys
import shutil
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
from loaders.data_loader import fetch_nba_games, calculate_rolling_stats
from learners.model_trainer import GPPredictor

print("\n" + "="*80)
print("🧹 STEP 1: CLEARING CACHES")
print("="*80)

# Clear Python cache
cache_dirs = ['.cache', '__pycache__', 'machine_learning/__pycache__']
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"   ✅ Removed {cache_dir}")

# Clear database
if os.path.exists('sports_analytics.db'):
    os.remove('sports_analytics.db')
    print(f"   ✅ Removed sports_analytics.db")

print("✅ Caches cleared\n")

print("="*80)
print("🔄 STEP 2: REBUILDING DATASET (2023-24 + 2024-25)")
print("="*80)

try:
    games_df = fetch_nba_games(
        seasons=['2023-24', '2024-25'], 
        season_type='Regular Season', 
        verbose=True
    )
    print(f"\n✅ Fetched {len(games_df)} total game records")
    
    print("\n📊 Calculating rolling stats with FIXED features (shift applied)...")
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Dataset built: {len(games_with_stats)} games")
    print(f"   Date range: {games_with_stats['GAME_DATE'].min()} → {games_with_stats['GAME_DATE'].max()}\n")
    
except Exception as e:
    print(f"❌ Error rebuilding dataset: {e}")
    sys.exit(1)

print("="*80)
print("📊 STEP 3: CHRONOLOGICAL TRAIN/TEST SPLIT")
print("="*80)

# Sort by date and split
games_sorted = games_with_stats.sort_values('GAME_DATE').reset_index(drop=True)
split_idx = int(len(games_sorted) * 0.80)

train_df = games_sorted.iloc[:split_idx].copy()
test_df = games_sorted.iloc[split_idx:].copy()

print(f"\n📚 Training set:")
print(f"   Games: {len(train_df)}")
print(f"   Dates: {train_df['GAME_DATE'].min()} → {train_df['GAME_DATE'].max()}")

print(f"\n🧪 Test set:")
print(f"   Games: {len(test_df)}")
print(f"   Dates: {test_df['GAME_DATE'].min()} → {test_df['GAME_DATE'].max()}")

print("\n" + "="*80)
print("🧠 STEP 4: RETRAINING MODELS FROM SCRATCH")
print("="*80)

try:
    print("\n⏳ Training Gaussian Process model...")
    print("   (This may take a few minutes...)")
    
    gp_model = GPPredictor(kernel_type='matern', length_scale=1.0, noise_level=0.1)
    gp_model.fit(train_df, test_df)
    
    print("✅ Gaussian Process model trained")
    
except Exception as e:
    print(f"❌ Error training model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✨ REBUILD COMPLETE")
print("="*80)

print(f"""
Summary:
  ✅ Caches cleared
  ✅ Dataset rebuilt
  ✅ Features recalculated with shift(1) (NO LEAKAGE)
  ✅ Models retrained from scratch

Next: Run backtest and compare new accuracy to 94.3%
Expected: ~55-65% accuracy (matching your live 60%)
""")