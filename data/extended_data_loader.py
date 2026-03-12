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
from experimental.loaders.data_loader import (
    fetch_nba_games, 
    calculate_rolling_stats, 
    create_matchup_features,
    get_all_nba_teams
)
from database.database_handler import SportsAnalyticsDB


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
    
    # Feature columns (all HOME_* and AWAY_* columns except identifiers and names)
    feature_cols = [col for col in matchup_df.columns 
                   if (col.startswith('HOME_') or col.startswith('AWAY_'))
                   and col not in ['HOME_TEAM_ID', 'AWAY_TEAM_ID', 
                                   'HOME_TEAM_NAME', 'AWAY_TEAM_NAME',
                                   'HOME_TEAM', 'AWAY_TEAM',
                                   'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]
    
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


"""
NBA Data Loading and Preprocessing Module

Handles:
- Fetching historical NBA game data via nba_api
- Calculating rolling statistics and advanced features
- Creating matchup datasets for model training
- Fetching upcoming games for predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from nba_api.live.nba.endpoints import scoreboard
import time
import warnings

warnings.filterwarnings('ignore')


def get_all_nba_teams():
    """Get all NBA teams with IDs and names"""
    nba_teams = teams.get_teams()
    team_ids = [team['id'] for team in nba_teams]
    team_names = {team['id']: team['full_name'] for team in nba_teams}
    team_abbreviations = {team['id']: team['abbreviation'] for team in nba_teams}
    
    return {
        'teams': nba_teams,
        'ids': team_ids,
        'names': team_names,
        'abbreviations': team_abbreviations
    }


def fetch_nba_games(seasons=['2023-24', '2024-25'], season_type='Regular Season', verbose=True):
    """
    Fetch NBA game data from multiple seasons
    
    Parameters:
    - seasons: List of season strings (e.g., ['2023-24', '2024-25'])
    - season_type: 'Regular Season', 'Playoffs', or 'All Star'
    - verbose: Print progress messages
    
    Returns:
    - DataFrame with all game data
    """
    all_games = []
    
    for season in seasons:
        if verbose:
            print(f"📥 Fetching {season} season...")
        
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable='00'
            )
            games = gamefinder.get_data_frames()[0]
            
            if verbose:
                print(f"   ✅ Got {len(games)} game records from {season}")
            
            all_games.append(games)
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Error fetching {season}: {e}")
            continue
    
    if not all_games:
        raise ValueError("No game data fetched!")
    
    # Combine all seasons
    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    
    if verbose:
        print(f"\n✅ Total: {len(combined)} game records")
        print(f"📅 Date range: {combined['GAME_DATE'].min()} to {combined['GAME_DATE'].max()}")
    
    return combined


def calculate_rolling_stats(df, window=5):
    """
    Calculate rolling averages and advanced features
    
    Features created:
    - Rolling averages (5-game window): PTS, FG_PCT, FG3_PCT, REB, AST, STL, BLK, TOV
    - WIN_STREAK: Consecutive wins/losses
    - REST_DAYS: Days since last game
    - IS_BACK_TO_BACK: Playing consecutive days
    - WIN_RATE_10: Rolling 10-game win percentage
    
    Parameters:
    - df: Game DataFrame
    - window: Rolling window size (default: 5)
    
    Returns:
    - DataFrame with rolling stats
    """
    df = df.copy()
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    # Basic rolling stats (shifted to exclude current game - prevent leakage)
    rolling_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
    
    # Win Streak - Consecutive wins (positive) or losses (negative)
    def calculate_streak(wl_series):
        streak = []
        current_streak = 0
        for wl in wl_series:
            if wl == 'W':
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
            streak.append(current_streak)
        return pd.Series(streak, index=wl_series.index)
    
    # Shift WIN_STREAK by 1 to use only prior games (prevent leakage)
    df['WIN_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(calculate_streak).shift(1).fillna(0)
    
    # Rest Days - Days between games
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(2)
    
    # Back-to-Back Indicator
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)
    
    # Team Momentum - Rolling win rate (last 10 games, shifted to exclude current game)
    df['WIN_RATE_10'] = df.groupby('TEAM_ID')['WL'].transform(
        lambda x: (x == 'W').shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    return df


def create_matchup_features(games_df):
    """
    Create matchup dataset where each row is a game with both teams' stats
    
    Parameters:
    - games_df: DataFrame with rolling stats
    
    Returns:
    - DataFrame with matchup features (home vs away)
    """
    team_data = get_all_nba_teams()
    team_names = team_data['names']
    
    matchups = []
    
    # Group by GAME_ID to get both teams
    for game_id, game_group in games_df.groupby('GAME_ID'):
        if len(game_group) == 2:
            # Sort to identify home/away (home team usually listed first in MATCHUP)
            game_group = game_group.sort_values('MATCHUP', ascending=False)
            
            home_team = game_group.iloc[0]
            away_team = game_group.iloc[1]
            
            matchup = {
                'GAME_ID': game_id,
                'GAME_DATE': home_team['GAME_DATE'],
                'HOME_TEAM': home_team['TEAM_ID'],
                'AWAY_TEAM': away_team['TEAM_ID'],
                'HOME_TEAM_NAME': team_names.get(home_team['TEAM_ID'], 'Unknown'),
                'AWAY_TEAM_NAME': team_names.get(away_team['TEAM_ID'], 'Unknown'),
            }
            
            # Add rolling features for both teams
            for prefix, team_data in [('HOME', home_team), ('AWAY', away_team)]:
                for col in games_df.columns:
                    if '_ROLL' in col or col in ['WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10']:
                        matchup[f'{prefix}_{col}'] = team_data[col]
            
            # Target variables
            matchup['HOME_PTS'] = home_team['PTS']
            matchup['AWAY_PTS'] = away_team['PTS']
            matchup['POINT_DIFF'] = home_team['PTS'] - away_team['PTS']
            matchup['HOME_WIN'] = 1 if home_team['WL'] == 'W' else 0
            
            matchups.append(matchup)
    
    return pd.DataFrame(matchups)


def fetch_upcoming_games(days_ahead=7, verbose=True):
    """
    Fetch upcoming NBA games from live scoreboard
    
    Parameters:
    - days_ahead: Number of days to look ahead (default: 7)
    - verbose: Print progress messages
    
    Returns:
    - List of upcoming games with team info
    """
    try:
        # Get today's scoreboard
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        upcoming = []
        
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            games = games_data['scoreboard']['games']
            
            for game in games:
                game_info = {
                    'game_id': game.get('gameId'),
                    'game_date': game.get('gameTimeUTC'),
                    'home_team': game.get('homeTeam', {}).get('teamName'),
                    'away_team': game.get('awayTeam', {}).get('teamName'),
                    'home_team_id': game.get('homeTeam', {}).get('teamId'),
                    'away_team_id': game.get('awayTeam', {}).get('teamId'),
                    'game_status': game.get('gameStatus'),
                    'game_status_text': game.get('gameStatusText')
                }
                upcoming.append(game_info)
        
        if verbose:
            print(f"✅ Found {len(upcoming)} upcoming games")
        
        return upcoming
        
    except Exception as e:
        if verbose:
            print(f"❌ Error fetching upcoming games: {e}")
        return []


def get_team_latest_stats(games_df, team_id):
    """
    Get the most recent rolling stats for a team
    
    Parameters:
    - games_df: DataFrame with rolling stats
    - team_id: NBA team ID
    
    Returns:
    - Dictionary of latest stats
    """
    team_games = games_df[games_df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return None
    
    latest = team_games.iloc[-1]
    
    stats = {}
    for col in games_df.columns:
        if '_ROLL' in col or col in ['WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10']:
            stats[col] = latest[col]
    
    return stats


def prepare_prediction_features(home_stats, away_stats, feature_cols):
    """
    Prepare features for a single game prediction
    
    Parameters:
    - home_stats: Home team stats dictionary
    - away_stats: Away team stats dictionary
    - feature_cols: List of expected feature column names
    
    Returns:
    - Feature array ready for model prediction
    """
    features = {}
    
    # Match the training feature format
    for col in feature_cols:
        if col.startswith('HOME_'):
            stat_name = col.replace('HOME_', '')
            features[col] = home_stats.get(stat_name, 0)
        elif col.startswith('AWAY_'):
            stat_name = col.replace('AWAY_', '')
            features[col] = away_stats.get(stat_name, 0)
    
    return pd.DataFrame([features])


if __name__ == "__main__":
    # Test the module
    print("🏀 Testing NBA Data Loader...")
    
    # Get teams
    team_data = get_all_nba_teams()
    print(f"✅ Found {len(team_data['ids'])} teams")
    
    # Fetch recent games
    games = fetch_nba_games(seasons=['2024-25'], verbose=True)
    print(f"✅ Fetched {len(games)} games")
    
    # Calculate rolling stats
    games_with_stats = calculate_rolling_stats(games)
    print(f"✅ Calculated rolling stats")
    
    # Create matchups
    matchups = create_matchup_features(games_with_stats)
    print(f"✅ Created {len(matchups)} matchups")
    
    print("\n🎉 Data loader module working correctly!")

"""Run NBA predictions for today's games using nba_api and a Bayesian Ridge model.

This script fetches recent seasons' game logs, engineers rolling features,
trains a BayesianRidge model on historical games (excluding today), and
predicts point differential and win probability for games occurring today.

Usage: python run_basketball_today.py
"""
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import warnings

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')


def fetch_with_retry(season, max_retries=3):
    for attempt in range(max_retries):
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                league_id_nullable='00',
                timeout=60,
            )
            games = gamefinder.get_data_frames()[0]
            return games
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


def calculate_rolling_stats(df, window=5):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    for col in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

    def calculate_streak(wl_series):
        streak = []
        current_streak = 0
        for wl in wl_series:
            if wl == 'W':
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
            streak.append(current_streak)
        return pd.Series(streak, index=wl_series.index)

    if 'WL' in df.columns:
        df['WIN_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(calculate_streak)

    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(2)
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)

    if 'WL' in df.columns:
        df['WIN_RATE_10'] = df.groupby('TEAM_ID')['WL'].transform(
            lambda x: (x == 'W').rolling(window=10, min_periods=1).mean()
        )

    return df


def create_matchup_features(games_df, team_names):
    matchups = []
    for game_id, game_group in games_df.groupby('GAME_ID'):
        if len(game_group) == 2:
            # ensure a consistent order: home first
            game_group = game_group.sort_values('MATCHUP', ascending=False)
            home = game_group.iloc[0]
            away = game_group.iloc[1]

            def get_feat(row, prefix):
                return {
                    f'{prefix}_PTS_ROLL': row.get('PTS_ROLL', 0),
                    f'{prefix}_FG_PCT_ROLL': row.get('FG_PCT_ROLL', 0),
                    f'{prefix}_FG3_PCT_ROLL': row.get('FG3_PCT_ROLL', 0),
                    f'{prefix}_REB_ROLL': row.get('REB_ROLL', 0),
                    f'{prefix}_AST_ROLL': row.get('AST_ROLL', 0),
                    f'{prefix}_STL_ROLL': row.get('STL_ROLL', 0),
                    f'{prefix}_BLK_ROLL': row.get('BLK_ROLL', 0),
                    f'{prefix}_TOV_ROLL': row.get('TOV_ROLL', 0),
                    f'{prefix}_WIN_STREAK': row.get('WIN_STREAK', 0),
                    f'{prefix}_REST_DAYS': row.get('REST_DAYS', 2),
                    f'{prefix}_IS_BACK_TO_BACK': row.get('IS_BACK_TO_BACK', 0),
                    f'{prefix}_WIN_RATE_10': row.get('WIN_RATE_10', 0),
                }

            matchup = {
                'GAME_ID': game_id,
                'GAME_DATE': home['GAME_DATE'],
                'HOME_TEAM': home['TEAM_ID'],
                'AWAY_TEAM': away['TEAM_ID'],
                'HOME_TEAM_NAME': team_names.get(home['TEAM_ID'], 'Unknown'),
                'AWAY_TEAM_NAME': team_names.get(away['TEAM_ID'], 'Unknown'),
            }
            matchup.update(get_feat(home, 'HOME'))
            matchup.update(get_feat(away, 'AWAY'))
            # targets if available
            if 'PTS' in home and 'PTS' in away:
                matchup['HOME_PTS'] = home['PTS']
                matchup['AWAY_PTS'] = away['PTS']
                matchup['POINT_DIFF'] = home['PTS'] - away['PTS']
                matchup['HOME_WIN'] = 1 if home.get('WL') == 'W' else 0

            matchups.append(matchup)

    return pd.DataFrame(matchups)


def main():
    print('🏀 Starting NBA today runner')

    nba_teams = teams.get_teams()
    team_ids = [t['id'] for t in nba_teams]
    team_names = {t['id']: t['full_name'] for t in nba_teams}

    seasons = ['2023-24', '2024-25']
    frames = []
    for s in seasons:
        print(f'📥 Fetching {s}...')
        df = fetch_with_retry(s)
        if df is not None:
            frames.append(df)

    if not frames:
        print('⚠️ NBA API unavailable; exiting')
        return

    games = pd.concat(frames, ignore_index=True)
    games = games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    games = calculate_rolling_stats(games, window=5)

    matchup_df = create_matchup_features(games, team_names)

    # Ensure GAME_DATE is datetime.date for filtering
    matchup_df['GAME_DATE'] = pd.to_datetime(matchup_df['GAME_DATE']).dt.date
    today = datetime.utcnow().date()

    todays = matchup_df[matchup_df['GAME_DATE'] == today]
    if todays.empty:
        # If no games matched UTC date, also try local date
        local_today = datetime.now().date()
        todays = matchup_df[matchup_df['GAME_DATE'] == local_today]

    if todays.empty:
        print('ℹ️ No games found for today in fetched data.')
        print('Available date range: ', matchup_df['GAME_DATE'].min(), 'to', matchup_df['GAME_DATE'].max())
        return

    print(f'🔎 Found {len(todays)} games for today')

    # Prepare training set (exclude today's games)
    training = matchup_df[~matchup_df['GAME_DATE'].isin(todays['GAME_DATE'])].dropna(subset=['POINT_DIFF'])
    feature_cols = [c for c in matchup_df.columns if c.endswith('_ROLL') or 'WIN_STREAK' in c or 'REST_DAYS' in c or 'IS_BACK_TO_BACK' in c or 'WIN_RATE_10' in c]

    X_train = training[feature_cols].fillna(0)
    y_train = training['POINT_DIFF']

    print('📚 Training model on historical games:', X_train.shape)
    model = BayesianRidge()
    model.fit(X_train, y_train)

    X_today = todays[feature_cols].fillna(0)
    mu, std = model.predict(X_today, return_std=True)

    from scipy.stats import norm

    probs = 1 - norm.cdf(0, loc=mu, scale=std)

    results = todays[['GAME_ID', 'HOME_TEAM_NAME', 'AWAY_TEAM_NAME']].copy()
    results['PRED_POINT_DIFF'] = mu
    results['PRED_STD'] = std
    results['HOME_WIN_PROB'] = probs

    pd.set_option('display.float_format', '{:.3f}'.format)
    print('\n🏁 Predictions for today:')
    print(results.sort_values('HOME_WIN_PROB', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()

"""
Results Fetcher Module

Handles:
- Live score retrieval from NBA API
- Manual result entry option
- Auto-matching results to predictions
"""

import json
import time
from datetime import datetime, timedelta
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamelog
import warnings

warnings.filterwarnings('ignore')


def fetch_live_scores(target_date=None):
    """
    Fetch live/completed game scores from NBA API
    
    Parameters:
    - target_date: Date string 'YYYY-MM-DD' (default: today)
    
    Returns:
    - List of dicts with game results
    """
    try:
        # Get scoreboard
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        results = []
        
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            games = games_data['scoreboard']['games']
            
            for game in games:
                # Only include finished games
                status = game.get('gameStatus', 0)
                if status == 3:  # Game finished
                    home_team = game.get('homeTeam', {})
                    away_team = game.get('awayTeam', {})
                    
                    result = {
                        'game_id': game.get('gameId'),
                        'game_date': game.get('gameTimeUTC', '')[:10],
                        'home_team': home_team.get('teamName'),
                        'away_team': away_team.get('teamName'),
                        'home_score': home_team.get('score', 0),
                        'away_score': away_team.get('score', 0),
                        'spread': home_team.get('score', 0) - away_team.get('score', 0),
                        'fetched_at': datetime.now().isoformat()
                    }
                    results.append(result)
        
        print(f"✅ Fetched {len(results)} completed games")
        return results
        
    except Exception as e:
        print(f"❌ Error fetching live scores: {e}")
        return []


def manual_result_entry(game_id, home_team, away_team, home_score, away_score, notes=''):
    """
    Manually enter a game result
    
    Parameters:
    - game_id: Game identifier
    - home_team: Home team name
    - away_team: Away team name
    - home_score: Home team final score
    - away_score: Away team final score
    - notes: Optional notes about special circumstances
    
    Returns:
    - Dict with result info
    """
    result = {
        'game_id': game_id,
        'game_date': datetime.now().date().isoformat(),
        'home_team': home_team,
        'away_team': away_team,
        'home_score': int(home_score),
        'away_score': int(away_score),
        'spread': int(home_score) - int(away_score),
        'entry_method': 'manual',
        'notes': notes,
        'entered_at': datetime.now().isoformat()
    }
    
    print(f"📝 Manual result entered: {home_team} {home_score} - {away_team} {away_score}")
    return result


def match_results_to_predictions(results, validator):
    """
    Match fetched results to logged predictions
    
    Parameters:
    - results: List of result dicts from fetch_live_scores()
    - validator: PredictionValidator instance
    
    Returns:
    - Number of matches found and logged
    """
    matches_found = 0
    
    for result in results:
        # Find matching predictions
        for i, pred in enumerate(validator.predictions):
            if pred['actual_spread'] is not None:
                continue  # Already has result
            
            # Match by team names and date
            home_match = pred['home_team'] in result['home_team'] or result['home_team'] in pred['home_team']
            away_match = pred['away_team'] in result['away_team'] or result['away_team'] in pred['away_team']
            
            if home_match and away_match:
                # Found a match!
                validator.log_result(i, result['home_score'], result['away_score'])
                matches_found += 1
                print(f"   ✓ Matched: {result['home_team']} vs {result['away_team']}")
                break
    
    if matches_found == 0:
        print("   No new matches found")
    
    return matches_found


if __name__ == "__main__":
    print("🏀 Testing Results Fetcher...")
    
    # Test live score fetching
    results = fetch_live_scores()
    print(f"✅ Fetched {len(results)} results")
    
    # Test manual entry
    manual = manual_result_entry(
        game_id='test_001',
        home_team='Lakers',
        away_team='Warriors',
        home_score=115,
        away_score=108,
        notes='Test game'
    )
    print(f"✅ Manual entry created")
    
    print("\n🎉 Results fetcher module working correctly!")

"""
Game Data Parser for CSV/Text Input
Parses game schedules and results from various formats
"""

import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional
import io


def parse_game_csv_text(csv_text: str) -> pd.DataFrame:
    """
    Parse game data from CSV text format
    
    Expected format (comma-separated):
    Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
    
    Parameters:
    - csv_text: Raw CSV text content
    
    Returns:
    - DataFrame with parsed game data
    """
    
    # Parse CSV
    df = pd.read_csv(io.StringIO(csv_text))
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Rename columns for consistency
    column_mapping = {
        'Visitor/Neutral': 'away_team',
        'Home/Neutral': 'home_team',
        'Date': 'date',
        'Start (ET)': 'start_time'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Get PTS columns (there are two: away PTS and home PTS)
    pts_columns = [col for col in df.columns if col == 'PTS']
    if len(pts_columns) >= 2:
        df['away_score'] = df[pts_columns[0]]
        df['home_score'] = df[pts_columns[1]]
    else:
        # Create empty score columns if PTS columns not found
        df['away_score'] = None
        df['home_score'] = None
    
    # Parse date
    df['game_date'] = pd.to_datetime(df['date'] + ' 2026', format='%a %b %d %Y', errors='coerce')
    df['game_date_str'] = df['game_date'].dt.strftime('%Y-%m-%d')
    
    # Determine if game has been played (has scores)
    df['has_result'] = pd.notna(df['away_score']) & pd.notna(df['home_score'])
    
    # Convert scores to int where available
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    
    # Calculate actual spread for completed games
    df['actual_spread'] = df.apply(
        lambda row: row['home_score'] - row['away_score'] 
        if pd.notna(row['home_score']) and pd.notna(row['away_score']) 
        else None,
        axis=1
    )
    
    # Determine actual winner
    df['actual_winner'] = df.apply(
        lambda row: row['home_team'] if pd.notna(row['actual_spread']) and row['actual_spread'] > 0 
        else row['away_team'] if pd.notna(row['actual_spread']) and row['actual_spread'] < 0 
        else 'TIE' if pd.notna(row['actual_spread']) and row['actual_spread'] == 0
        else None,
        axis=1
    )
    
    # Select and order relevant columns
    output_columns = [
        'game_date', 'game_date_str', 'start_time',
        'away_team', 'home_team',
        'away_score', 'home_score', 'actual_spread', 'actual_winner',
        'has_result', 'Arena'
    ]
    
    available_columns = [col for col in output_columns if col in df.columns]
    df = df[available_columns]
    
    # Sort by date
    df = df.sort_values('game_date').reset_index(drop=True)
    
    return df


def separate_completed_and_upcoming(games_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Separate games into completed and upcoming
    
    Parameters:
    - games_df: DataFrame with all games
    
    Returns:
    - Dictionary with 'completed' and 'upcoming' DataFrames
    """
    
    completed = games_df[games_df['has_result'] == True].copy()
    upcoming = games_df[games_df['has_result'] == False].copy()
    
    return {
        'completed': completed,
        'upcoming': upcoming
    }


def games_to_prediction_format(games_df: pd.DataFrame) -> List[Dict]:
    """
    Convert games DataFrame to list of dictionaries for prediction
    
    Parameters:
    - games_df: DataFrame with game data
    
    Returns:
    - List of game dictionaries
    """
    
    games_list = []
    
    for idx, row in games_df.iterrows():
        game = {
            'game_id': f"game_{row['game_date_str']}_{idx}",
            'game_date': row['game_date_str'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'start_time': row.get('start_time'),
            'venue': row.get('Arena')
        }
        
        # Add actual results if available
        if row.get('has_result'):
            game['actual_home_score'] = int(row['home_score'])
            game['actual_away_score'] = int(row['away_score'])
            game['actual_spread'] = float(row['actual_spread'])
            game['actual_winner'] = row['actual_winner']
        
        games_list.append(game)
    
    return games_list


def parse_game_data_from_text(text: str, verbose: bool = True) -> Dict:
    """
    Main function to parse game data from raw text
    
    Parameters:
    - text: Raw CSV text with game data
    - verbose: Print parsing information
    
    Returns:
    - Dictionary with parsed data
    """
    
    if verbose:
        print("=" * 70)
        print("📋 PARSING GAME DATA")
        print("=" * 70 + "\n")
    
    # Parse CSV
    games_df = parse_game_csv_text(text)
    
    if verbose:
        print(f"✅ Parsed {len(games_df)} games")
        print(f"📅 Date range: {games_df['game_date'].min().date()} to {games_df['game_date'].max().date()}")
    
    # Separate completed and upcoming
    separated = separate_completed_and_upcoming(games_df)
    completed_games = separated['completed']
    upcoming_games = separated['upcoming']
    
    if verbose:
        print(f"✅ Completed games: {len(completed_games)}")
        print(f"🔮 Upcoming games: {len(upcoming_games)}")
    
    # Convert to prediction format
    completed_list = games_to_prediction_format(completed_games)
    upcoming_list = games_to_prediction_format(upcoming_games)
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 PARSING COMPLETE")
        print("=" * 70 + "\n")
    
    return {
        'all_games': games_df,
        'completed_games': completed_games,
        'upcoming_games': upcoming_games,
        'completed_list': completed_list,
        'upcoming_list': upcoming_list,
        'total_games': len(games_df),
        'completed_count': len(completed_games),
        'upcoming_count': len(upcoming_games)
    }


# Sample usage
if __name__ == "__main__":
    # Test with sample data
    sample_text = """Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
Sun Feb 1 2026,3:30p,Milwaukee Bucks,79,Boston Celtics,107,Box Score,,19156,2:09,TD Garden,
Sun Feb 8 2026,12:30p,New York Knicks,,Boston Celtics,,,,,,TD Garden,
"""
    
    result = parse_game_data_from_text(sample_text)
    print(f"Parsed {result['total_games']} games")
    print(f"Completed: {result['completed_count']}, Upcoming: {result['upcoming_count']}")


"""
H2H Analysis and Post-Prediction Adjustment

Contains two cooperating classes:

  H2HTrendsAnalyzer
      Queries historical head-to-head games between two teams and returns
      matchup-specific stats (win rate, recency-weighted rate, avg spread).
      CRITICAL: Maintains chronological safety via strict date filtering.

  H2HPostPredictor
      Applies H2H trends as a post-hoc probability adjustment AFTER model
      prediction. This ensures H2H does not interfere with model training.

      Architecture:
          Model Prediction -> H2H Adjustment -> Final Output

      This is NOT a model retrain, inner-loop adjustment, or feature step.
      This IS rule-based calibration / probabilistic blending /
      context-aware post-processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

class H2HTrendsAnalyzer:
    """
    Analyzes head-to-head matchup history between two teams.
    
    This provides context-aware signal orthogonal to rolling team stats:
    - Rolling stats: "How good are they generally?"
    - H2H: "How do they perform against THIS opponent?"
    
    CHRONOLOGICAL SAFETY:
    - Accepts explicit prediction_date parameter
    - Uses STRICT < comparison (not <=) to prevent data leakage
    - Safe for both live predictions and historical backtests
    """
    
    def __init__(self, games_df: Optional[pd.DataFrame] = None, cache_size: int = 256):
        """
        Initialize H2H analyzer.
        
        Parameters:
        -----------
        games_df : pd.DataFrame, optional
            Historical games DataFrame with columns:
            - HOME_TEAM_NAME, AWAY_TEAM_NAME
            - GAME_DATE (datetime or parseable string)
            - HOME_WIN (1 if home won, 0 otherwise)
            - POINT_DIFF (home_pts - away_pts)
            
            If None, analyzer will return neutral stats (used for testing).
            
        cache_size : int
            In-memory cache size to prevent redundant queries.
            Default 256 covers ~30 teams * 30 teams / 4 (typical games per day).
        """
        self.games_df = games_df
        self.cache = {}
        self.cache_size = cache_size
        
        # Ensure GAME_DATE is datetime if df provided
        if self.games_df is not None and 'GAME_DATE' in self.games_df.columns:
            self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
    
    def get_h2h_stats(
        self, 
        team_a_name: str, 
        team_b_name: str, 
        prediction_date: str,
        window: int = 10
    ) -> Dict:
        """
        Get head-to-head statistics for matchup.
        
        Parameters:
        -----------
        team_a_name : str
            First team (typically away team for consistent interpretation)
        team_b_name : str
            Second team (typically home team)
        prediction_date : str or datetime
            CRITICAL: Only uses games with GAME_DATE < prediction_date.
            Must be explicit - never defaults to "today".
        window : int
            Number of recent H2H games to analyze (default: 10)
        
        Returns:
        --------
        dict with keys:
            - h2h_games_count: Number of H2H games found
            - h2h_win_rate: P(team_a wins) based on historical H2H
            - h2h_win_rate_recent: Recency-weighted win rate (last 3 games = 2x)
            - h2h_avg_spread: Average point differential (team_a perspective)
            - h2h_last_game_date: Date of most recent H2H game
            - insufficient_data: True if < 2 games (insufficient for inference)
        
        Example:
        --------
        >>> analyzer = H2HTrendsAnalyzer(games_df)
        >>> stats = analyzer.get_h2h_stats(
        ...     "Atlanta Hawks", 
        ...     "Philadelphia 76ers", 
        ...     "2026-02-19"
        ... )
        >>> print(f"Hawks win {stats['h2h_win_rate']:.1%} of H2H games")
        """
        
        # Cache key (sorted so A@B and B@A hit same cache entry)
        cache_key = (
            tuple(sorted([team_a_name, team_b_name])), 
            str(prediction_date),
            window
        )
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Convert prediction_date to datetime
        pred_date = pd.to_datetime(prediction_date)
        
        # If no historical data, return neutral
        if self.games_df is None or len(self.games_df) == 0:
            return self._neutral_h2h()
        
        # Filter to games between these two teams, BEFORE prediction date
        h2h_mask = (
            (
                ((self.games_df['HOME_TEAM_NAME'] == team_a_name) & 
                 (self.games_df['AWAY_TEAM_NAME'] == team_b_name)) |
                ((self.games_df['HOME_TEAM_NAME'] == team_b_name) & 
                 (self.games_df['AWAY_TEAM_NAME'] == team_a_name))
            ) &
            (self.games_df['GAME_DATE'] < pred_date)  # STRICT < (no leakage)
        )
        
        h2h_games = self.games_df[h2h_mask].copy()
        
        # Insufficient data if < 2 games
        if len(h2h_games) < 2:
            result = self._neutral_h2h(insufficient=True)
            self._cache_result(cache_key, result)
            return result
        
        # Sort by date descending (most recent first)
        h2h_games = h2h_games.sort_values('GAME_DATE', ascending=False)
        
        # Take last N games
        last_n = h2h_games.head(window)
        
        # Calculate team_a win statistics
        team_a_wins = 0
        spreads = []
        
        for _, game in last_n.iterrows():
            if game['HOME_TEAM_NAME'] == team_a_name:
                # team_a was home
                team_a_won = (game['HOME_WIN'] == 1)
                spread = game['POINT_DIFF']  # Positive = team_a won by this much
            else:
                # team_a was away
                team_a_won = (game['HOME_WIN'] == 0)
                spread = -game['POINT_DIFF']  # Flip sign for team_a perspective
            
            if team_a_won:
                team_a_wins += 1
            spreads.append(spread)
        
        # Overall win rate
        win_rate = team_a_wins / len(last_n)
        
        # Recency-weighted win rate (last 3 games count 2x)
        recent_games = last_n.head(3)
        if len(recent_games) > 0:
            recent_wins = sum(
                1 for _, game in recent_games.iterrows()
                if (
                    (game['HOME_TEAM_NAME'] == team_a_name and game['HOME_WIN'] == 1) or
                    (game['AWAY_TEAM_NAME'] == team_a_name and game['HOME_WIN'] == 0)
                )
            )
            # Weighted average: recent games 2x weight, older games 1x
            older_games_count = len(last_n) - len(recent_games)
            older_wins = team_a_wins - recent_wins
            
            if older_games_count > 0:
                recency_win_rate = (2 * recent_wins + older_wins) / (2 * len(recent_games) + older_games_count)
            else:
                recency_win_rate = recent_wins / len(recent_games)
        else:
            recency_win_rate = win_rate
        
        # Average spread
        avg_spread = np.mean(spreads) if spreads else 0.0
        
        # Build result
        result = {
            'h2h_games_count': len(last_n),
            'h2h_win_rate': float(win_rate),
            'h2h_win_rate_recent': float(recency_win_rate),
            'h2h_avg_spread': float(avg_spread),
            'h2h_last_game_date': str(last_n.iloc[0]['GAME_DATE'].date()) if len(last_n) > 0 else None,
            'insufficient_data': False
        }
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _cache_result(self, key: Tuple, result: Dict):
        """Store result in cache with size limit."""
        if len(self.cache) >= self.cache_size:
            # Simple eviction: remove oldest entry (first added)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = result
    
    @staticmethod
    def _neutral_h2h(insufficient: bool = False) -> Dict:
        """
        Return neutral H2H stats when no data available.
        
        Neutral = 50/50 probabilities, zero spread.
        This ensures H2H doesn't affect prediction when insufficient data.
        """
        return {
            'h2h_games_count': 0,
            'h2h_win_rate': 0.5,
            'h2h_win_rate_recent': 0.5,
            'h2h_avg_spread': 0.0,
            'h2h_last_game_date': None,
            'insufficient_data': True
        }
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache = {}
    
    def get_cache_stats(self) -> Dict:
        """Return cache statistics for monitoring."""
        return {
            'cache_size': len(self.cache),
            'cache_capacity': self.cache_size,
            'hit_rate': None  # Would need hit/miss tracking to compute
        }


class H2HPostPredictor:
    """
    Post-prediction adjustment using H2H trends.
    
    Takes model output (GP, LightGBM, Ensemble) and blends with H2H signal
    using weighted probabilistic combination.
    
    Key Design Decisions:
    1. Applied AFTER retraining loop completes (doesn't trigger retraining)
    2. Uses probabilistic blending (not heuristic confidence scaling)
    3. Preserves original model prediction in output (transparency)
    4. Adds h2h_agreement flag for disagreement detection
    
    Mathematical Approach:
        P_final = w_model * P_model + w_h2h * P_h2h
        
    Where:
        - P_model: Model's predicted probability
        - P_h2h: H2H win rate (from historical matchups)
        - w_model: Model weight (default: 0.85)
        - w_h2h: H2H weight (default: 0.15)
    """
    
    def __init__(
        self, 
        h2h_analyzer, 
        model_weight: float = 0.85, 
        h2h_weight: float = 0.15,
        h2h_min_games: int = 2
    ):
        """
        Initialize H2H post-predictor.
        
        Parameters:
        -----------
        h2h_analyzer : H2HTrendsAnalyzer
            Analyzer instance for querying H2H stats
        model_weight : float
            Weight for model probability (default: 0.85)
            Conservative default - trust model more than H2H
        h2h_weight : float
            Weight for H2H probability (default: 0.15)
            Start small, increase if validation shows improvement
        h2h_min_games : int
            Minimum H2H games required for adjustment (default: 2)
            If fewer games, falls back to model-only prediction
        
        Notes:
        ------
        - Weights must sum to 1.0
        - Start conservative (0.85/0.15) and tune based on validation
        - Can disable H2H by setting h2h_weight=0
        """
        if not np.isclose(model_weight + h2h_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {model_weight + h2h_weight}")
        
        self.h2h_analyzer = h2h_analyzer
        self.model_weight = model_weight
        self.h2h_weight = h2h_weight
        self.h2h_min_games = h2h_min_games
    
    def compute_four_factors_edge(
        self,
        home_team: str,
        away_team: str,
        matchup_features: dict
    ) -> dict:
        """
        Compare Dean Oliver's Four Factors between home and away team.

        Factors (by impact, per Oliver):
          1. eFG%       – shot selection quality / efficiency  (higher better)
          2. TOV%       – turnover rate / unforced errors       (lower  better)
          3. OREB%      – offensive rebound aggression          (higher better)
          4. FT Rate    – free throw drawing / paint attacks    (higher better)

        Also includes:
          - Net Rating differential (points per 100 possessions)
          - Pythagorean Win% alignment with model prediction

        Returns dict with edge counts per team + net_rating + pythagorean fields.
        """
        factors = [
            ('eFG%',    'EFG_PCT',        True),   # higher is better
            ('TOV%',    'TOV_PCT',        False),  # lower is better
            ('OREB%',   'OREB_PCT_APPROX', True),  # higher is better
            ('FT Rate', 'FT_RATE',        True),   # higher is better
        ]

        home_adv, away_adv = [], []
        for label, key, higher_better in factors:
            h = matchup_features.get(f'HOME_{key}_ROLL')
            a = matchup_features.get(f'AWAY_{key}_ROLL')
            if h is None or a is None:
                continue
            if higher_better:
                (home_adv if h > a else away_adv).append(label)
            else:
                (home_adv if h < a else away_adv).append(label)

        # Net Rating differential
        h_net = matchup_features.get('HOME_NET_RTG_APPROX_ROLL')
        a_net = matchup_features.get('AWAY_NET_RTG_APPROX_ROLL')
        net_rating_edge = None
        if h_net is not None and a_net is not None:
            diff = float(h_net) - float(a_net)
            net_rating_edge = {
                'home': round(float(h_net), 1),
                'away': round(float(a_net), 1),
                'diff': round(diff, 1),
                'favors': home_team if diff > 0 else away_team,
            }

        # Pythagorean Win% alignment
        h_pyt = matchup_features.get('HOME_PYT_WIN_PCT_ROLL')
        a_pyt = matchup_features.get('AWAY_PYT_WIN_PCT_ROLL')
        pythagorean_edge = None
        if h_pyt is not None and a_pyt is not None:
            pythagorean_edge = {
                'home': round(float(h_pyt), 3),
                'away': round(float(a_pyt), 3),
                'favors': home_team if float(h_pyt) > float(a_pyt) else away_team,
            }

        n_home = len(home_adv)
        n_away = len(away_adv)
        return {
            'four_factors_home_count': n_home,
            'four_factors_away_count': n_away,
            'four_factors_home_advantages': home_adv,
            'four_factors_away_advantages': away_adv,
            'four_factors_edge': (
                home_team if n_home > n_away
                else (away_team if n_away > n_home else 'EVEN')
            ),
            'net_rating_edge': net_rating_edge,
            'pythagorean_edge': pythagorean_edge,
        }

    def adjust_prediction(
        self,
        model_prediction: Dict,
        prediction_date: str,
        away_team: Optional[str] = None,
        home_team: Optional[str] = None,
        matchup_features: Optional[dict] = None,  # Four Factors / Net Rtg / Pythagorean context
    ) -> Dict:
        """
        Apply H2H adjustment to model prediction.
        
        Parameters:
        -----------
        model_prediction : dict
            Output from predictor.py or iterative_predictor.py
            Required keys:
                - 'home_team', 'away_team' (or pass explicitly)
                - 'win_probability' (P(home wins))
                - 'confidence_score' (0-1 scale)
                - 'predicted_spread'
                
        prediction_date : str or datetime
            Date for H2H query (YYYY-MM-DD format)
            CRITICAL: Must pass explicit date for chronological safety
            
        away_team : str, optional
            Away team name (if not in model_prediction)
        home_team : str, optional
            Home team name (if not in model_prediction)
        
        Returns:
        --------
        dict with original prediction plus:
            - h2h_games_count: Number of H2H games found
            - h2h_win_rate: H2H probability used for blending
            - h2h_avg_spread: Average H2H spread
            - h2h_insufficient_data: Boolean flag
            - h2h_weight_used: Weight applied to H2H signal
            - model_weight_used: Weight applied to model signal
            - h2h_agreement: True if H2H and model agree on winner
            - adjusted_win_probability: Final blended probability
            - adjusted_confidence_score: Recalculated confidence (0-1)
            - adjusted_confidence_level: 'HIGH', 'MEDIUM', or 'LOW'
            - original_win_probability: Copy of model's original prob
            - original_confidence_score: Copy of model's original confidence
        
        Example:
        --------
        >>> adjuster = H2HPostPredictor(h2h_analyzer)
        >>> model_pred = {
        ...     'home_team': 'Philadelphia 76ers',
        ...     'away_team': 'Atlanta Hawks',
        ...     'win_probability': 0.72,  # Model says 72% for 76ers
        ...     'confidence_score': 0.65
        ... }
        >>> final = adjuster.adjust_prediction(model_pred, '2026-02-19')
        >>> print(f"Adjusted: {final['adjusted_win_probability']:.1%}")
        >>> print(f"H2H agrees: {final['h2h_agreement']}")
        """
        
        # Extract team names
        home = home_team or model_prediction.get('home_team')
        away = away_team or model_prediction.get('away_team')
        
        if not home or not away:
            raise ValueError("Must provide home_team and away_team")
        
        # Extract model probability (P(home wins))
        model_home_prob = model_prediction.get('win_probability', 0.5)
        model_confidence = model_prediction.get('confidence_score', 0.0)
        
        # Query H2H stats (away team perspective for consistency)
        h2h_stats = self.h2h_analyzer.get_h2h_stats(
            away, home, prediction_date
        )
        
        # Check if sufficient H2H data
        if h2h_stats['insufficient_data'] or h2h_stats['h2h_games_count'] < self.h2h_min_games:
            # Insufficient data - return model prediction unchanged
            return self._return_unadjusted(model_prediction, h2h_stats, reason='insufficient_data')
        
        # Get H2H probability
        # h2h_win_rate is P(away wins) from analyzer
        h2h_away_prob = h2h_stats['h2h_win_rate']
        h2h_home_prob = 1.0 - h2h_away_prob
        
        # Probabilistic blend
        adjusted_home_prob = (
            self.model_weight * model_home_prob +
            self.h2h_weight * h2h_home_prob
        )
        
        # Clip to valid probability range
        adjusted_home_prob = np.clip(adjusted_home_prob, 0.01, 0.99)
        
        # Determine winners for agreement check
        model_winner = 'home' if model_home_prob > 0.5 else 'away'
        h2h_winner = 'home' if h2h_home_prob > 0.5 else 'away'
        h2h_agrees = (model_winner == h2h_winner)
        
        # Recalculate confidence from adjusted probability
        # Confidence = how far from 0.5 (certainty measure)
        adjusted_certainty = abs(adjusted_home_prob - 0.5) * 2.0  # Scale to [0, 1]
        
        # Map to confidence levels (matching predictor.py thresholds)
        if adjusted_certainty >= 0.50:
            adjusted_confidence_level = 'HIGH'
        elif adjusted_certainty >= 0.30:
            adjusted_confidence_level = 'MEDIUM'
        else:
            adjusted_confidence_level = 'LOW'
        
        # Decision-quality context (Four Factors / Net Rating / Pythagorean Wins)
        decision_quality = {}
        if matchup_features is not None:
            decision_quality = self.compute_four_factors_edge(
                home, away, matchup_features
            )

        # Build enriched output
        return {
            **model_prediction,  # Preserve all original fields
            **decision_quality,  # Four Factors + Net Rating + Pythagorean fields
            
            # Original values (for comparison/validation)
            'original_win_probability': float(model_home_prob),
            'original_confidence_score': float(model_confidence),
            'original_confidence_level': model_prediction.get('confidence', 'UNKNOWN'),
            
            # H2H metadata
            'h2h_games_count': h2h_stats['h2h_games_count'],
            'h2h_win_rate': float(h2h_stats['h2h_win_rate']),
            'h2h_win_rate_recent': float(h2h_stats['h2h_win_rate_recent']),
            'h2h_avg_spread': float(h2h_stats['h2h_avg_spread']),
            'h2h_last_game_date': h2h_stats['h2h_last_game_date'],
            'h2h_insufficient_data': False,
            
            # Blending parameters
            'h2h_weight_used': float(self.h2h_weight),
            'model_weight_used': float(self.model_weight),
            'h2h_agreement': bool(h2h_agrees),
            
            # Adjusted outputs (FINAL PREDICTIONS)
            'adjusted_win_probability': float(adjusted_home_prob),
            'adjusted_confidence_score': float(adjusted_certainty),
            'adjusted_confidence_level': adjusted_confidence_level,
            
            # Update main prediction fields (so downstream code uses adjusted values)
            'win_probability': float(adjusted_home_prob),
            'confidence_score': float(adjusted_certainty),
            'confidence': adjusted_confidence_level,
        }
    
    def _return_unadjusted(
        self, 
        model_prediction: Dict, 
        h2h_stats: Dict, 
        reason: str
    ) -> Dict:
        """
        Return prediction unchanged when H2H adjustment not applicable.
        
        Still adds H2H metadata fields (set to neutral/None) for
        consistent output schema.
        """
        model_prob = model_prediction.get('win_probability', 0.5)
        model_confidence = model_prediction.get('confidence_score', 0.0)
        
        return {
            **model_prediction,
            
            # H2H metadata (neutral/insufficient)
            'h2h_games_count': h2h_stats.get('h2h_games_count', 0),
            'h2h_win_rate': 0.5,  # Neutral
            'h2h_win_rate_recent': 0.5,
            'h2h_avg_spread': 0.0,
            'h2h_last_game_date': None,
            'h2h_insufficient_data': True,
            
            # No blending occurred
            'h2h_weight_used': 0.0,
            'model_weight_used': 1.0,
            'h2h_agreement': None,  # Can't determine without data
            
            # "Adjusted" values same as original
            'adjusted_win_probability': float(model_prob),
            'adjusted_confidence_score': float(model_confidence),
            'adjusted_confidence_level': model_prediction.get('confidence', 'UNKNOWN'),

            # Original values
            'original_win_probability': float(model_prob),
            'original_confidence_score': float(model_confidence),
            'original_confidence_level': model_prediction.get('confidence', 'UNKNOWN'),

            # Metadata
            'adjustment_skipped_reason': reason,

            # Decision-quality context (empty when matchup_features not passed)
            'four_factors_home_count': None,
            'four_factors_away_count': None,
            'four_factors_home_advantages': [],
            'four_factors_away_advantages': [],
            'four_factors_edge': None,
            'net_rating_edge': None,
            'pythagorean_edge': None,
        }
    
    def set_weights(self, model_weight: float, h2h_weight: float):
        """
        Update blending weights dynamically.
        
        Useful for:
        - A/B testing different weight configurations
        - Adaptive weighting based on recent performance
        - Disabling H2H (set h2h_weight=0)
        """
        if not np.isclose(model_weight + h2h_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {model_weight + h2h_weight}")
        
        self.model_weight = model_weight
        self.h2h_weight = h2h_weight
