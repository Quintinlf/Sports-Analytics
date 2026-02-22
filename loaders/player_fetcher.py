"""
Player Game Log Fetcher - Chronologically Safe Data Retrieval

Fetches player game logs from nba_api with strict leakage prevention.
All data is filtered to games BEFORE prediction date.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from nba_api.stats.endpoints import playergamelog, commonteamroster
from typing import Optional, Dict


# NBA Team ID Mapping (Normalized 0-29 to Raw IDs)
TEAM_ID_MAP = {
    0: 1610612737,   # Hawks
    1: 1610612738,   # Celtics
    2: 1610612751,   # Nets
    3: 1610612766,   # Hornets
    4: 1610612741,   # Bulls
    5: 1610612739,   # Cavaliers
    6: 1610612742,   # Mavericks
    7: 1610612743,   # Nuggets
    8: 1610612765,   # Pistons
    9: 1610612744,   # Warriors
    10: 1610612745,  # Rockets
    11: 1610612754,  # Pacers
    12: 1610612746,  # Clippers
    13: 1610612747,  # Lakers
    14: 1610612763,  # Grizzlies
    15: 1610612748,  # Heat
    16: 1610612749,  # Bucks
    17: 1610612750,  # Timberwolves
    18: 1610612740,  # Pelicans
    19: 1610612752,  # Knicks
    20: 1610612760,  # Thunder
    21: 1610612753,  # Magic
    22: 1610612755,  # 76ers
    23: 1610612756,  # Suns
    24: 1610612757,  # Trail Blazers
    25: 1610612758,  # Kings
    26: 1610612759,  # Spurs
    27: 1610612761,  # Raptors
    28: 1610612762,  # Jazz
    29: 1610612764,  # Wizards
}


def normalize_team_id(team_id: int) -> int:
    """Convert team_id from normalized (0-29) or raw format to raw format."""
    if team_id < 100:
        # Assume normalized ID
        return TEAM_ID_MAP.get(team_id, team_id)
    else:
        # Already raw NBA ID
        return team_id


def convert_minutes_to_float(min_str: str) -> float:
    """Convert MM:SS or float string to float minutes."""
    if pd.isna(min_str):
        return 0.0
    
    try:
        if isinstance(min_str, str):
            if ':' in min_str:
                parts = min_str.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes + seconds / 60.0
            else:
                return float(min_str)
        else:
            return float(min_str)
    except (ValueError, AttributeError):
        return 0.0


def fetch_player_logs_for_team(
    team_id: int,
    season: str = '2024-25',
    before_date: Optional[datetime] = None,
    verbose: bool = False,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch all player game logs for a team's roster.
    
    CHRONOLOGICALLY SAFE: Only returns games with GAME_DATE < before_date.
    
    Parameters:
    -----------
    team_id : int
        Team ID (normalized 0-29 or raw 1610612738 format)
    season : str
        NBA season (e.g., '2024-25')
    before_date : datetime, optional
        CRITICAL: Filter to games before this date. Prevents leakage.
    verbose : bool
        Print debug info
    max_retries : int
        Retry attempts on rate limiting
    
    Returns:
    --------
    pd.DataFrame
        Columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_DATE, MIN, PTS, REB, AST,
                 FGA, FG_PCT, FG3M, FTM, STL, BLK, TOV, PLUS_MINUS
        Empty DataFrame on failure (does not crash)
    """
    
    # Convert team_id to raw format
    raw_team_id = normalize_team_id(team_id)
    if verbose:
        print(f"   Fetching logs for team_id={raw_team_id} (converted from {team_id})")
    
    try:
        # STEP 1: Get team roster
        for attempt in range(max_retries):
            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=raw_team_id,
                    season=season
                )
                df_roster = roster.get_data_frames()[0]
                player_ids = df_roster['PLAYER_ID'].tolist()
                
                if verbose:
                    print(f"   ✅ Retrieved {len(player_ids)} players on roster")
                
                time.sleep(0.6)  # Rate limiting
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    if verbose:
                        print(f"   ❌ Error fetching roster: {e}")
                    return pd.DataFrame()
        
        if len(player_ids) == 0:
            return pd.DataFrame()
        
        # STEP 2: Fetch game logs for each player
        all_logs = []
        
        for player_id in player_ids:
            try:
                logs = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season'
                )
                df_logs = logs.get_data_frames()[0]
                
                if len(df_logs) > 0:
                    # Ensure GAME_DATE is datetime
                    df_logs['GAME_DATE'] = pd.to_datetime(df_logs['GAME_DATE'])
                    
                    # CHRONOLOGICAL SAFETY: Filter to games BEFORE prediction date
                    if before_date is not None:
                        before_date_dt = pd.to_datetime(before_date)
                        df_logs = df_logs[df_logs['GAME_DATE'] < before_date_dt]
                    
                    if len(df_logs) > 0:
                        df_logs['PLAYER_ID'] = player_id
                        all_logs.append(df_logs)
                
                time.sleep(0.6)  # Rate limiting
                
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2.0)  # Aggressive backoff on rate limit
                    continue
                else:
                    if verbose:
                        print(f"   ⚠️  Player {player_id} error: {e}")
                    continue
        
        if not all_logs:
            return pd.DataFrame()
        
        # STEP 3: Combine and clean
        combined = pd.concat(all_logs, ignore_index=True)
        combined = combined.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Convert MIN to float
        combined['MIN'] = combined['MIN'].apply(convert_minutes_to_float)
        
        # Select critical columns
        critical_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'GAME_DATE', 'MIN',
                        'PTS', 'REB', 'AST', 'FGA', 'FG_PCT', 'FG3M', 'FTM',
                        'STL', 'BLK', 'TOV', 'PLUS_MINUS']
        
        available_cols = [col for col in critical_cols if col in combined.columns]
        combined = combined[available_cols]
        
        if verbose:
            print(f"   ✅ Retrieved {len(combined)} games for {combined['PLAYER_ID'].nunique()} players")
        
        return combined
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Critical error: {e}")
        return pd.DataFrame()


def get_all_team_ids() -> Dict[int, str]:
    """Return mapping of normalized IDs (0-29) to team names."""
    return {
        0: 'Hawks', 1: 'Celtics', 2: 'Nets', 3: 'Hornets', 4: 'Bulls',
        5: 'Cavaliers', 6: 'Mavericks', 7: 'Nuggets', 8: 'Pistons', 9: 'Warriors',
        10: 'Rockets', 11: 'Pacers', 12: 'Clippers', 13: 'Lakers', 14: 'Grizzlies',
        15: 'Heat', 16: 'Bucks', 17: 'Timberwolves', 18: 'Pelicans', 19: 'Knicks',
        20: 'Thunder', 21: 'Magic', 22: '76ers', 23: 'Suns', 24: 'Trail Blazers',
        25: 'Kings', 26: 'Spurs', 27: 'Raptors', 28: 'Jazz', 29: 'Wizards',
    }
