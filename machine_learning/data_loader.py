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
    
    # Basic rolling stats
    rolling_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
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
    
    df['WIN_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(calculate_streak)
    
    # Rest Days - Days between games
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(2)
    
    # Back-to-Back Indicator
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)
    
    # Team Momentum - Rolling win rate (last 10 games)
    df['WIN_RATE_10'] = df.groupby('TEAM_ID')['WL'].transform(
        lambda x: (x == 'W').rolling(window=10, min_periods=1).mean()
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
