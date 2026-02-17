"""
H2H Trends Module - Context-Aware Matchup Analysis

Queries historical head-to-head games between two teams to provide
matchup-specific context that rolling stats cannot capture.

CRITICAL: Maintains chronological safety via strict date filtering.
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
