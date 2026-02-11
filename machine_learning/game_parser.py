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
        lambda row: row['home_team'] if row['actual_spread'] > 0 
        else row['away_team'] if row['actual_spread'] < 0 
        else 'TIE'
        if pd.notna(row['actual_spread'])
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
