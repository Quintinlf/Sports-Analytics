"""
Extended Data Loader with 3-Season Support and Database Caching
Builds on data_loader.py with enhanced capabilities for iterative training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Tuple
import sys
import os

# Import base data loader functions
sys.path.append(os.path.dirname(__file__))
from data_loader import (
    fetch_nba_games, 
    calculate_rolling_stats, 
    create_matchup_features,
    get_all_nba_teams
)
from database_handler import SportsAnalyticsDB


def fetch_comprehensive_nba_data(
    seasons: List[str] = ['2022-23', '2023-24', '2024-25'],
    season_type: str = 'Regular Season',
    use_cache: bool = True,
    db_path: str = "sports_analytics.db",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch comprehensive 3-season NBA data with database caching
    
    Parameters:
    - seasons: List of seasons to fetch (default: 3 most recent)
    - season_type: 'Regular Season', 'Playoffs', etc.
    - use_cache: Whether to use/update database cache
    - db_path: Path to SQLite database
    - verbose: Print progress messages
    
    Returns:
    - DataFrame with all game data including rolling stats
    """
    
    if verbose:
        print("=" * 70)
        print(f"📊 COMPREHENSIVE DATA FETCH: {len(seasons)} Seasons")
        print("=" * 70)
        print(f"Seasons: {', '.join(seasons)}")
        print(f"Season Type: {season_type}")
        print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")
        print()
    
    # Check cache first if enabled
    if use_cache:
        db = SportsAnalyticsDB(db_path)
        cached_games = []
        
        for season in seasons:
            cached = db.get_cached_games(season=season)
            if cached and verbose:
                print(f"💾 Found {len(cached)} cached games for {season}")
            cached_games.extend(cached)
        
        db.close()
        
        # If we have substantial cached data, use it
        if cached_games and len(cached_games) > 1000:
            if verbose:
                print(f"\n✅ Using {len(cached_games)} cached games")
            
            # Convert cached data to DataFrame format
            # Note: This is a simplified version - expand as needed
            # For now, we'll still fetch fresh data
            pass
    
    # Fetch fresh data from NBA API
    if verbose:
        print("🌐 Fetching fresh data from NBA API...\n")
    
    try:
        games_df = fetch_nba_games(
            seasons=seasons,
            season_type=season_type,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n📈 Calculating rolling statistics...")
        
        # Calculate rolling stats
        games_with_stats = calculate_rolling_stats(games_df, window=5)
        
        # Cache to database if enabled
        if use_cache:
            if verbose:
                print(f"💾 Caching games to database...")
            cache_games_to_db(games_with_stats, db_path, verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"✅ COMPREHENSIVE DATA READY: {len(games_with_stats)} game records")
            print(f"📅 Date Range: {games_with_stats['GAME_DATE'].min().date()} to {games_with_stats['GAME_DATE'].max().date()}")
            unique_teams = games_with_stats['TEAM_ID'].nunique()
            print(f"🏀 Teams: {unique_teams}")
            print("=" * 70 + "\n")
        
        return games_with_stats
        
    except Exception as e:
        print(f"❌ Error fetching comprehensive data: {e}")
        raise


def cache_games_to_db(games_df: pd.DataFrame, db_path: str, verbose: bool = False) -> int:
    """
    Cache game data to database for future use
    
    Parameters:
    - games_df: DataFrame with game data
    - db_path: Path to SQLite database
    - verbose: Print progress
    
    Returns:
    - Number of games cached
    """
    db = SportsAnalyticsDB(db_path)
    cached_count = 0
    
    # Group by game to avoid duplicates (each game appears twice, once per team)
    for game_id in games_df['GAME_ID'].unique():
        game_rows = games_df[games_df['GAME_ID'] == game_id]
        
        if len(game_rows) >= 2:
            # Determine home/away
            # Matchup format: "TEAM @ TEAM" or "TEAM vs. TEAM"
            home_row = game_rows[game_rows['MATCHUP'].str.contains('vs.')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains('vs.')]) > 0 else game_rows.iloc[0]
            away_row = game_rows[game_rows['MATCHUP'].str.contains('@')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains('@')]) > 0 else game_rows.iloc[1]
            
            game_data = {
                'game_id': str(game_id),
                'game_date': str(home_row['GAME_DATE'].date()),
                'season': home_row.get('SEASON_ID', ''),
                'home_team': home_row['TEAM_NAME'] if 'TEAM_NAME' in home_row else str(home_row['TEAM_ID']),
                'away_team': away_row['TEAM_NAME'] if 'TEAM_NAME' in away_row else str(away_row['TEAM_ID']),
                'home_team_id': int(home_row['TEAM_ID']),
                'away_team_id': int(away_row['TEAM_ID']),
                'home_score': int(home_row['PTS']) if 'PTS' in home_row and pd.notna(home_row['PTS']) else None,
                'away_score': int(away_row['PTS']) if 'PTS' in away_row and pd.notna(away_row['PTS']) else None,
                'game_status': 'Final',
                'stats': {
                    'home': home_row.to_dict(),
                    'away': away_row.to_dict()
                }
            }
            
            try:
                db.cache_game(game_data)
                cached_count += 1
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not cache game {game_id}: {e}")
    
    db.close()
    
    if verbose:
        print(f"   ✅ Cached {cached_count} unique games")
    
    return cached_count


def prepare_training_data(
    games_with_stats: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare training data for models from games with rolling stats
    
    Parameters:
    - games_with_stats: DataFrame with calculated rolling stats
    - verbose: Print information
    
    Returns:
    - Tuple of (matchup_df, targets, feature_names)
    """
    
    if verbose:
        print("🔧 Preparing training data...")
    
    # Create matchup features
    matchup_df = create_matchup_features(games_with_stats)
    
    # Remove rows with NaN in target or features
    matchup_df = matchup_df.dropna()
    
    if verbose:
        print(f"   ✅ Created {len(matchup_df)} training samples")
    
    # Extract features and target
    target = matchup_df['POINT_DIFF'].values
    
    # Feature columns (all HOME_* and AWAY_* columns except identifiers)
    feature_cols = [col for col in matchup_df.columns 
                   if (col.startswith('HOME_') or col.startswith('AWAY_'))
                   and col not in ['HOME_TEAM_ID', 'AWAY_TEAM_ID']]
    
    features = matchup_df[feature_cols]
    
    if verbose:
        print(f"   📊 Features: {len(feature_cols)} columns")
        print(f"   🎯 Target: POINT_DIFF (home points - away points)")
    
    return matchup_df, target, feature_cols


def get_extended_training_dataset(
    db_path: str = "sports_analytics.db",
    verbose: bool = True
) -> Dict[str, any]:
    """
    Get complete extended training dataset ready for all models
    
    Returns dictionary with:
    - games_df: Raw game data
    - matchup_df: Matchup features
    - X: Feature matrix
    - y: Target vector
    - feature_names: List of feature column names
    - team_data: Dictionary of team information
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("🚀 LOADING EXTENDED TRAINING DATASET")
        print("=" * 70 + "\n")
    
    # Fetch comprehensive 3-season data
    games_df = fetch_comprehensive_nba_data(
        seasons=['2022-23', '2023-24', '2024-25'],
        use_cache=True,
        db_path=db_path,
        verbose=verbose
    )
    
    # Prepare training data
    matchup_df, y, feature_names = prepare_training_data(games_df, verbose=verbose)
    X = matchup_df[feature_names].values
    
    # Get team data
    team_info = get_all_nba_teams()
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ DATASET READY FOR TRAINING")
        print("=" * 70)
        print(f"📊 Training Samples: {len(X)}")
        print(f"📈 Features: {len(feature_names)}")
        print(f"🏀 Teams: {len(team_info['ids'])}")
        print("=" * 70 + "\n")
    
    return {
        'games_df': games_df,
        'matchup_df': matchup_df,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'team_data': team_info
    }


def refresh_recent_data(
    existing_df: pd.DataFrame,
    days_back: int = 7,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Refresh with most recent games (useful during active season)
    
    Parameters:
    - existing_df: Existing games DataFrame
    - days_back: How many days back to re-fetch
    - verbose: Print progress
    
    Returns:
    - Updated DataFrame with latest data
    """
    
    if verbose:
        print(f"🔄 Refreshing data from last {days_back} days...")
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    # Remove old recent data
    filtered_df = existing_df[existing_df['GAME_DATE'] < cutoff_date].copy()
    
    # Fetch fresh recent data
    current_season = '2024-25'  # Update based on current year
    recent_games = fetch_nba_games(
        seasons=[current_season],
        verbose=verbose
    )
    
    # Filter to recent only
    recent_games = recent_games[recent_games['GAME_DATE'] >= cutoff_date]
    
    # Combine
    updated_df = pd.concat([filtered_df, recent_games], ignore_index=True)
    updated_df = updated_df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    
    # Recalculate rolling stats
    updated_df = calculate_rolling_stats(updated_df)
    
    if verbose:
        print(f"   ✅ Added {len(recent_games)} recent game records")
    
    return updated_df
