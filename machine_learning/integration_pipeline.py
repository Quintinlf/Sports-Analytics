"""
Enhanced Feature Integration Pipeline

Combines 94 team-level features + 36 player-level features = 130 total.
Maintains strict chronological integrity throughout.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, Tuple, Dict, Optional

from machine_learning.player_fetcher import fetch_player_logs_for_team, normalize_team_id
from feature_selection.player_features import calculate_player_rolling_stats, aggregate_player_stats_by_team
from machine_learning.player_cache import PlayerDataCache


class EnhancedFeaturePipeline:
    """Production pipeline for enhanced 130-feature game feature building."""
    
    def __init__(self, player_data_source: Union[str, PlayerDataCache] = 'api'):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        player_data_source : str or PlayerDataCache
            'api' to fetch from nba_api directly
            PlayerDataCache instance to use cached data
        """
        self.data_source = player_data_source
        self.team_feature_count = 94
        self.player_feature_count = 36
        self.total_feature_count = 130
        
        # Player feature names (18 per team)
        self.player_feature_names = [
            'PLAYER_PTS_ROLL_WEIGHTED',
            'PLAYER_REB_ROLL_WEIGHTED',
            'PLAYER_AST_ROLL_WEIGHTED',
            'PLAYER_FGA_ROLL_WEIGHTED',
            'PLAYER_FG_PCT_ROLL_WEIGHTED',
            'PLAYER_PTS_PER_FGA_ROLL_WEIGHTED',
            'PLAYER_TOP_SCORER_PPG',
            'PLAYER_TOP_REBOUNDER_RPG',
            'PLAYER_TOP_PLAYMAKER_APG',
            'PLAYER_TOP_SCORER_SHARE',
            'PLAYER_BENCH_SCORING_PCT',
            'PLAYER_ACTIVE_ROTATION_SIZE',
            'PLAYER_ROTATION_STABILITY',
            'PLAYER_KEY_PLAYER_MISSING',
            'PLAYER_MINUTES_DROP_40PCT',
            'PLAYER_SCORING_CONCENTRATION',
            'PLAYER_DEFENSIVE_CONTRIBUTORS',
            'PLAYER_BENCH_SCORER_COUNT',
        ]
    
    def _get_player_data(
        self,
        team_id: int,
        before_date: datetime,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Fetch player data from API or cache."""
        
        if isinstance(self.data_source, PlayerDataCache):
            # Use cache
            logs = self.data_source.get_cached_logs(
                team_id=normalize_team_id(team_id),
                before_date=before_date,
                verbose=verbose
            )
        else:
            # Use API
            logs = fetch_player_logs_for_team(
                team_id=team_id,
                season='2024-25',
                before_date=before_date,
                verbose=verbose
            )
        
        return logs
    
    def build_player_features(
        self,
        game_date: datetime,
        team_id: int,
        team_type: str = 'HOME'
    ) -> Dict[str, float]:
        """
        Build 18 player features for one team.
        
        CHRONOLOGICALLY SAFE: Only uses data before game_date.
        
        Parameters:
        -----------
        game_date : datetime
            Game date (prediction date)
        team_id : int
            Team ID
        team_type : str
            'HOME' or 'AWAY' (for feature naming)
        
        Returns:
        --------
        Dict[str, float]
            18 features with keys like 'HOME_PLAYER_PTS_ROLL_WEIGHTED'
        """
        
        features = {}
        
        try:
            # Fetch player logs (STRICTLY BEFORE game_date)
            logs = self._get_player_data(team_id, before_date=game_date, verbose=False)
            
            if len(logs) == 0:
                # No data available, fill with zeros
                for feat_name in self.player_feature_names:
                    features[f'{team_type}_{feat_name}'] = 0.0
                return features
            
            # Calculate rolling stats
            logs_rolled = calculate_player_rolling_stats(logs, window=5)
            
            # Aggregate to team level
            team_features = aggregate_player_stats_by_team(
                logs_rolled,
                team_id=normalize_team_id(team_id),
                weight_by_minutes=True
            )
            
            # Add team type prefix
            for key, val in team_features.items():
                features[f'{team_type}_{key}'] = val
            
        except Exception as e:
            # On error, fill with zeros
            for feat_name in self.player_feature_names:
                features[f'{team_type}_{feat_name}'] = 0.0
        
        return features
    
    def build_enhanced_features(
        self,
        game_date: datetime,
        home_team_id: int,
        away_team_id: int,
        matchup_df: pd.DataFrame,
        team_feature_cols: list,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float], list]:
        """
        Build complete 130-feature vector for a game.
        
        CHRONOLOGICALLY SAFE: All player data filtered to before game_date.
        
        Parameters:
        -----------
        game_date : datetime
            Game date (prediction target)
        home_team_id : int
            Home team ID
        away_team_id : int
            Away team ID
        matchup_df : pd.DataFrame
            Pre-computed team features (e.g., matchup_df_sorted from notebook)
        team_feature_cols : list
            List of 94 team feature column names
        verbose : bool
            Print debug info
        
        Returns:
        --------
        Tuple of:
            - feature_vector: np.array (130,) with all features
            - feature_dict: Dict with feature names and values
            - feature_names: List of 130 feature names in order
        """
        
        feature_dict = {}
        
        # === STEP 1: Extract 94 team features ===
        if verbose:
            print(f"Extracting team features...")
        
        # Find matching game in matchup_df
        game_mask = (
            (matchup_df['GAME_DATE'] == game_date) &
            (matchup_df['HOME_TEAM_ID'] == normalize_team_id(home_team_id)) &
            (matchup_df['AWAY_TEAM_ID'] == normalize_team_id(away_team_id))
        )
        
        matching_games = matchup_df[game_mask]
        
        if len(matching_games) > 0:
            game_row = matching_games.iloc[0]
            for col in team_feature_cols:
                if col in game_row.index:
                    val = game_row[col]
                    feature_dict[col] = float(val) if not pd.isna(val) else 0.0
                else:
                    feature_dict[col] = 0.0
        else:
            # No matching game found, fill team features with zeros
            for col in team_feature_cols:
                feature_dict[col] = 0.0
            if verbose:
                print(f"⚠️  No matching game in matchup_df")
        
        # === STEP 2: Build 18 HOME player features ===
        if verbose:
            print(f"Building HOME player features...")
        
        home_features = self.build_player_features(
            game_date=game_date,
            team_id=home_team_id,
            team_type='HOME'
        )
        feature_dict.update(home_features)
        
        # === STEP 3: Build 18 AWAY player features ===
        if verbose:
            print(f"Building AWAY player features...")
        
        away_features = self.build_player_features(
            game_date=game_date,
            team_id=away_team_id,
            team_type='AWAY'
        )
        feature_dict.update(away_features)
        
        # === STEP 4: Build ordered feature vector (130,) ===
        feature_names = team_feature_cols.copy()
        
        for team_type in ['HOME', 'AWAY']:
            for feat_name in self.player_feature_names:
                feature_names.append(f'{team_type}_{feat_name}')
        
        feature_vector = np.array(
            [feature_dict.get(col, 0.0) for col in feature_names],
            dtype=np.float32
        )
        
        # === STEP 5: Validation ===
        if verbose:
            print(f"Validating features...")
            print(f"  Vector size: {len(feature_vector)} (expected 130)")
            print(f"  NaN count: {np.isnan(feature_vector).sum()}")
            print(f"  Inf count: {np.isinf(feature_vector).sum()}")
        
        # Handle NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct size
        if len(feature_vector) != 130:
            if verbose:
                print(f"⚠️  Feature vector size mismatch: {len(feature_vector)} (expected 130)")
            # Pad or trim
            if len(feature_vector) < 130:
                feature_vector = np.pad(feature_vector, (0, 130 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:130]
        
        return feature_vector, feature_dict, feature_names


def build_enhanced_game_features(
    game_date: datetime,
    home_team_id: int,
    away_team_id: int,
    matchup_df: pd.DataFrame,
    team_feature_cols: list,
    player_data_source: Union[str, PlayerDataCache] = 'api',
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, float], list]:
    """
    Convenience function: Build enhanced 130-feature vector for a single game.
    
    CHRONOLOGICALLY SAFE: All player data is strictly before game_date.
    
    Parameters:
    -----------
    game_date : datetime
        Game date
    home_team_id : int
        Home team ID
    away_team_id : int
        Away team ID
    matchup_df : pd.DataFrame
        Pre-computed team features
    team_feature_cols : list
        List of 94 team feature column names
    player_data_source : str or PlayerDataCache
        'api' or PlayerDataCache instance
    verbose : bool
        Print debug info
    
    Returns:
    --------
    Tuple of (feature_vector, feature_dict, feature_names)
    """
    
    pipeline = EnhancedFeaturePipeline(player_data_source=player_data_source)
    
    return pipeline.build_enhanced_features(
        game_date=game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        matchup_df=matchup_df,
        team_feature_cols=team_feature_cols,
        verbose=verbose
    )
