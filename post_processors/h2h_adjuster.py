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
