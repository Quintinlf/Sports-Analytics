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
    
    # --- Roll the advanced metrics ---
    advanced_cols = [
        'TS_PCT', 'EFG_PCT', 'AST_TO_RATIO',
        'POSS_APPROX', 'OFF_RTG_APPROX', 'FT_RATE'
    ]
    
    for col in advanced_cols:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # PLUS_MINUS rolling (proxy for Net Rating)
    if 'PLUS_MINUS' in df.columns:
        df['PLUS_MINUS_ROLL'] = df.groupby('TEAM_ID')['PLUS_MINUS'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    print(f"   ✅ Added {len(advanced_cols) + 1} advanced rolling features")
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
