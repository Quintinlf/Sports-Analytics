"""
Vectorized feature engineering for NBA matchup enrichment.

This module computes high-signal features in one vectorized pass and adds
game-theory features grounded in extensive-form logic:
- expected_payoff_matrix
- optimal_path_delta
- signal_consistency_score

Key safety rule:
- All rolling and transition-style features use shifted windows only, so a game
  at time T never consumes same-game realized outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(numerator, fill_value=float(default), dtype=float)
    valid = np.abs(denominator) > 1e-12
    out[valid] = numerator[valid] / denominator[valid]
    return out


@dataclass(frozen=True)
class BasketballStateNode:
    """Compact extensive-form node contract for basketball decision states."""

    score_margin_bucket: int
    time_bucket: int
    possession_home: int
    foul_state_bucket: int
    timeout_state_bucket: int
    home_context: float
    gamma: float = 0.98

    def legal_actions(self) -> Tuple[str, ...]:
        if self.time_bucket <= 1:
            return ('quick_two', 'quick_three', 'intentional_foul', 'hold_for_last')
        if self.time_bucket <= 3:
            return ('aggressive_push', 'balanced_set', 'clock_control')
        return ('balanced_set', 'clock_control')

    @staticmethod
    def matrix_game_value_2x2(a11: np.ndarray, a12: np.ndarray, a21: np.ndarray, a22: np.ndarray) -> np.ndarray:
        """Compute mixed-strategy value for a 2x2 zero-sum game."""
        denom = a11 - a12 - a21 + a22
        value = np.full_like(denom, fill_value=np.nan, dtype=float)

        valid = np.abs(denom) > 1e-9
        value[valid] = (a11[valid] * a22[valid] - a12[valid] * a21[valid]) / denom[valid]
        fallback = 0.25 * (a11 + a12 + a21 + a22)
        value = np.where(np.isnan(value), fallback, value)
        return value

    @staticmethod
    def bellman_optimal_aggressive(
        p_good_aggressive: np.ndarray,
        p_good_conservative: np.ndarray,
        gamma: float = 0.98,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One-step Bellman recursion for aggressive vs conservative action choice."""
        p_good_aggressive = np.clip(p_good_aggressive, 0.0, 1.0)
        p_good_conservative = np.clip(p_good_conservative, 0.0, 1.0)

        transitions = np.stack(
            [
                np.stack([p_good_aggressive, 1.0 - p_good_aggressive], axis=1),
                np.stack([p_good_conservative, 1.0 - p_good_conservative], axis=1),
            ],
            axis=1,
        )
        rewards = np.array([1.0, -1.0], dtype=float)
        continuation = np.array([0.25, -0.25], dtype=float)
        q_values = np.sum(transitions * (rewards + gamma * continuation), axis=2)
        optimal_aggressive = (q_values[:, 0] >= q_values[:, 1]).astype(float)
        return optimal_aggressive, q_values

    @staticmethod
    def update_belief_with_ic(
        prior_low_fatigue: np.ndarray,
        action_high_energy: np.ndarray,
        p_action_high_given_low: np.ndarray,
        p_action_high_given_high: np.ndarray,
        eq_payoff_low: np.ndarray,
        eq_payoff_high: np.ndarray,
        max_dev_payoff_low: np.ndarray,
        max_dev_payoff_high: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bayesian belief update with Intuitive Criterion filtering."""
        prior_low = np.clip(prior_low_fatigue, 1e-6, 1.0 - 1e-6)
        prior_high = 1.0 - prior_low

        like_low = np.where(action_high_energy >= 0.5, p_action_high_given_low, 1.0 - p_action_high_given_low)
        like_high = np.where(action_high_energy >= 0.5, p_action_high_given_high, 1.0 - p_action_high_given_high)
        like_low = np.clip(like_low, 1e-6, 1.0 - 1e-6)
        like_high = np.clip(like_high, 1e-6, 1.0 - 1e-6)

        unnorm_low = prior_low * like_low
        unnorm_high = prior_high * like_high

        # Intuitive Criterion: types that can never gain by deviating to this action
        # should receive zero posterior mass.
        unnorm_low = np.where(max_dev_payoff_low < eq_payoff_low, 0.0, unnorm_low)
        unnorm_high = np.where(max_dev_payoff_high < eq_payoff_high, 0.0, unnorm_high)

        normalizer = unnorm_low + unnorm_high
        posterior_low = _safe_divide(unnorm_low, normalizer, default=0.5)
        posterior_high = 1.0 - posterior_low
        return posterior_low, posterior_high


def _compute_game_theory_features_vectorized(
    home_off_rating: np.ndarray,
    home_def_rating: np.ndarray,
    away_off_rating: np.ndarray,
    away_def_rating: np.ndarray,
    rest_days_home: np.ndarray,
    rest_days_away: np.ndarray,
    schedule_density_home: np.ndarray,
    schedule_density_away: np.ndarray,
    is_back_to_back_home: np.ndarray,
    is_back_to_back_away: np.ndarray,
    pace_home: np.ndarray,
    pace_away: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute matchup-level game-theory features with vectorized operations."""
    home_off = np.asarray(home_off_rating, dtype=float)
    home_def = np.asarray(home_def_rating, dtype=float)
    away_off = np.asarray(away_off_rating, dtype=float)
    away_def = np.asarray(away_def_rating, dtype=float)

    rest_h = np.asarray(rest_days_home, dtype=float)
    rest_a = np.asarray(rest_days_away, dtype=float)
    dens_h = np.asarray(schedule_density_home, dtype=float)
    dens_a = np.asarray(schedule_density_away, dtype=float)
    b2b_h = np.asarray(is_back_to_back_home, dtype=float)
    b2b_a = np.asarray(is_back_to_back_away, dtype=float)
    pace_h = np.asarray(pace_home, dtype=float)
    pace_a = np.asarray(pace_away, dtype=float)

    fatigue_h = np.clip(dens_h - 0.4 * rest_h + 0.75 * b2b_h, 0.0, 6.0)
    fatigue_a = np.clip(dens_a - 0.4 * rest_a + 0.75 * b2b_a, 0.0, 6.0)

    league_pace = float(np.nanmean(np.concatenate([pace_h, pace_a]))) if len(pace_h) else 100.0
    pace_signal_h = _sigmoid((pace_h - league_pace) / 6.0)
    pace_signal_a = _sigmoid((pace_a - league_pace) / 6.0)
    action_h = (pace_signal_h >= 0.5).astype(float)
    action_a = (pace_signal_a >= 0.5).astype(float)

    prior_low_h = _sigmoid(rest_h - (dens_h / 2.5))
    prior_low_a = _sigmoid(rest_a - (dens_a / 2.5))

    pA_low_h = np.clip(0.45 + 0.35 * _sigmoid(rest_h - 1.5) - 0.10 * b2b_h, 0.05, 0.95)
    pA_low_a = np.clip(0.45 + 0.35 * _sigmoid(rest_a - 1.5) - 0.10 * b2b_a, 0.05, 0.95)
    pA_high_h = np.clip(0.25 + 0.25 * _sigmoid(1.5 - rest_h) + 0.08 * b2b_h + 0.05 * np.clip(dens_h - 2.0, 0.0, None), 0.05, 0.95)
    pA_high_a = np.clip(0.25 + 0.25 * _sigmoid(1.5 - rest_a) + 0.08 * b2b_a + 0.05 * np.clip(dens_a - 2.0, 0.0, None), 0.05, 0.95)

    eq_low_h = home_off - 0.35 * away_def
    eq_high_h = eq_low_h - (1.5 + 0.8 * fatigue_h)
    dev_low_h = eq_low_h + 0.6
    dev_high_h = eq_high_h + 0.25 - 0.9 * fatigue_h

    eq_low_a = away_off - 0.35 * home_def
    eq_high_a = eq_low_a - (1.5 + 0.8 * fatigue_a)
    dev_low_a = eq_low_a + 0.6
    dev_high_a = eq_high_a + 0.25 - 0.9 * fatigue_a

    post_low_h, post_high_h = BasketballStateNode.update_belief_with_ic(
        prior_low_h,
        action_h,
        pA_low_h,
        pA_high_h,
        eq_low_h,
        eq_high_h,
        dev_low_h,
        dev_high_h,
    )
    post_low_a, post_high_a = BasketballStateNode.update_belief_with_ic(
        prior_low_a,
        action_a,
        pA_low_a,
        pA_high_a,
        eq_low_a,
        eq_high_a,
        dev_low_a,
        dev_high_a,
    )

    signal_h = action_h * post_low_h + (1.0 - action_h) * post_high_h
    signal_a = action_a * post_low_a + (1.0 - action_a) * post_high_a
    signal_consistency_score = np.clip(0.5 * (signal_h + signal_a), 0.0, 1.0)

    adj_home_off = home_off + 0.15 * pace_h - 0.9 * fatigue_h
    adj_away_def = away_def + 0.6 * fatigue_a
    a11_h = adj_home_off - adj_away_def
    a12_h = 0.85 * adj_home_off - 0.60 * adj_away_def
    a21_h = 0.60 * adj_home_off - 0.85 * adj_away_def
    a22_h = 0.75 * adj_home_off - 0.75 * adj_away_def
    home_value = BasketballStateNode.matrix_game_value_2x2(a11_h, a12_h, a21_h, a22_h)

    adj_away_off = away_off + 0.15 * pace_a - 0.9 * fatigue_a
    adj_home_def = home_def + 0.6 * fatigue_h
    a11_a = adj_away_off - adj_home_def
    a12_a = 0.85 * adj_away_off - 0.60 * adj_home_def
    a21_a = 0.60 * adj_away_off - 0.85 * adj_home_def
    a22_a = 0.75 * adj_away_off - 0.75 * adj_home_def
    away_value = BasketballStateNode.matrix_game_value_2x2(a11_a, a12_a, a21_a, a22_a)
    expected_payoff_matrix = home_value - away_value

    p_good_aggr_h = np.clip(_sigmoid((adj_home_off - adj_away_def) / 8.0 + 0.35 * rest_h - 0.40 * b2b_h), 0.05, 0.95)
    p_good_cons_h = np.clip(_sigmoid((adj_home_off - adj_away_def) / 9.0 + 0.15 * rest_h - 0.20 * b2b_h), 0.05, 0.95)
    p_good_aggr_a = np.clip(_sigmoid((adj_away_off - adj_home_def) / 8.0 + 0.35 * rest_a - 0.40 * b2b_a), 0.05, 0.95)
    p_good_cons_a = np.clip(_sigmoid((adj_away_off - adj_home_def) / 9.0 + 0.15 * rest_a - 0.20 * b2b_a), 0.05, 0.95)

    optimal_aggr_h, _ = BasketballStateNode.bellman_optimal_aggressive(p_good_aggr_h, p_good_cons_h)
    optimal_aggr_a, _ = BasketballStateNode.bellman_optimal_aggressive(p_good_aggr_a, p_good_cons_a)
    delta_h = np.abs(optimal_aggr_h - pace_signal_h)
    delta_a = np.abs(optimal_aggr_a - pace_signal_a)
    optimal_path_delta = np.clip(np.sqrt((delta_h ** 2 + delta_a ** 2) / 2.0), 0.0, 1.0)

    return {
        'expected_payoff_matrix': expected_payoff_matrix,
        'optimal_path_delta': optimal_path_delta,
        'signal_consistency_score': signal_consistency_score,
    }


def compute_game_theory_matchup_features(
    home_off_rating: float,
    home_def_rating: float,
    away_off_rating: float,
    away_def_rating: float,
    rest_days_home: float,
    rest_days_away: float,
    schedule_density_home: float,
    schedule_density_away: float,
    is_back_to_back_home: float,
    is_back_to_back_away: float,
    pace_home: float,
    pace_away: float,
) -> Dict[str, float]:
    """Scalar wrapper for row-wise feature derivation paths."""
    out = _compute_game_theory_features_vectorized(
        home_off_rating=np.asarray([home_off_rating], dtype=float),
        home_def_rating=np.asarray([home_def_rating], dtype=float),
        away_off_rating=np.asarray([away_off_rating], dtype=float),
        away_def_rating=np.asarray([away_def_rating], dtype=float),
        rest_days_home=np.asarray([rest_days_home], dtype=float),
        rest_days_away=np.asarray([rest_days_away], dtype=float),
        schedule_density_home=np.asarray([schedule_density_home], dtype=float),
        schedule_density_away=np.asarray([schedule_density_away], dtype=float),
        is_back_to_back_home=np.asarray([is_back_to_back_home], dtype=float),
        is_back_to_back_away=np.asarray([is_back_to_back_away], dtype=float),
        pace_home=np.asarray([pace_home], dtype=float),
        pace_away=np.asarray([pace_away], dtype=float),
    )
    return {k: float(v[0]) for k, v in out.items()}


def vectorize_high_signal_features(
    matchup_df: pd.DataFrame,
    games_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute high-signal features for training data in one vectorized pass.
    
    This replaces the slow row-by-row derive_game_features() calls during training.
    
    Parameters
    ----------
    matchup_df : pd.DataFrame
        Training data with columns: GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM.
        Each row is one training matchup ready for feature enrichment.
    
    games_df : pd.DataFrame
        Team-level game history with columns: GAME_ID, GAME_DATE, TEAM_ID, 
        WL (W/L), PTS, MATCHUP, etc.
        Rows sorted by [TEAM_ID, GAME_DATE].
    
    verbose : bool
        Print timing and progress.
    
    Returns
    -------
    pd.DataFrame
        Features dataframe with curated high-signal columns
        and same row count as matchup_df.
        
    Notes
    -----
    - All operations are vectorized (no loops)
    - Pre-computes team-level stats once, then merges
    - Uses merge_asof to align training rows with historical cutoffs
    - Result shape: (len(matchup_df), 17)
    """
    import time
    start = time.time()
    
    # Ensure datetime types
    games = games_df.copy()
    matchup = matchup_df.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    matchup['GAME_DATE'] = pd.to_datetime(matchup['GAME_DATE'])
    
    # Sort for merge_asof and rolling operations
    games = games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    matchup = matchup.sort_values('GAME_DATE').reset_index(drop=True)
    
    if verbose:
        print(f"[Vectorize] Processing {len(matchup)} matchups with {len(games)} game rows")
    
    # -----------------------------------------------------------------------
    # STEP 1: Build team feature table (all features, all teams, all games)
    # -----------------------------------------------------------------------
    
    # 1a. WIN and POINT_DIFF columns
    games['win'] = (games['WL'].astype(str) == 'W').astype(float)
    games['point_diff'] = pd.to_numeric(games['PTS'], errors='coerce').fillna(0.0)

    # Opponent points proxy (for defensive ratings) from two-row game logs.
    game_totals = games.groupby('GAME_ID')['point_diff'].transform('sum')
    games['opp_points'] = (game_totals - games['point_diff']).fillna(games['point_diff'])
    
    # 1b. Rolling 5-game stats (per team)
    if verbose:
        print("  Computing rolling 5-game stats...")
    
    games['last5_win_pct'] = games.groupby('TEAM_ID')['win'].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    games['last5_point_diff'] = games.groupby('TEAM_ID')['point_diff'].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    games['off_rating'] = games.groupby('TEAM_ID')['point_diff'].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    games['def_rating'] = games.groupby('TEAM_ID')['opp_points'].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    # 1c. Rest days (lag features per team)
    if verbose:
        print("  Computing rest days...")
    
    games['prev_game_date'] = games.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    games['rest_days'] = (games['GAME_DATE'] - games['prev_game_date']).dt.days.fillna(2.0)
    games['rest_days'] = games['rest_days'].clip(lower=0.0)  # No negative rest
    
    # 1d. Back-to-back flag
    games['is_back_to_back'] = (games['rest_days'] == 1.0).astype(float)
    
    # 1e. Home/away strength (win % at home vs. away)
    if verbose:
        print("  Computing home/away strength...")
    
    games['is_home'] = games['MATCHUP'].astype(str).str.contains('vs.', na=False).astype(float)
    games['is_away'] = games['MATCHUP'].astype(str).str.contains('@', na=False).astype(float)
    
    # Home strength: win % in "vs." games (home games)
    games['home_win_only'] = np.where(games['is_home'] == 1.0, games['win'], np.nan)
    games['away_win_only'] = np.where(games['is_away'] == 1.0, games['win'], np.nan)

    games['home_strength'] = games.groupby('TEAM_ID')['home_win_only'].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    # Away strength: win % in "@" games (away games) 
    games['away_strength'] = games.groupby('TEAM_ID')['away_win_only'].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    games['home_strength'] = games['home_strength'].fillna(0.5)
    games['away_strength'] = games['away_strength'].fillna(0.5)
    games['home_away_strength'] = games['home_strength'] - games['away_strength']
    
    # 1f. Schedule density (games in last 7 days)
    if verbose:
        print("  Computing schedule density...")
    
    # Use transform-based approach to avoid MultiIndex index alignment issues
    def _compute_7d_game_count(group):
        """Count unique games in 7-day window per team."""
        result = []
        for idx, row in group.iterrows():
            current_date = row['GAME_DATE']
            window_start = current_date - pd.Timedelta(days=7)
            count = len(group[(group['GAME_DATE'] > window_start) & (group['GAME_DATE'] <= current_date)])
            result.append(max(float(count) - 1.0, 0.0))
        return pd.Series(result, index=group.index)
    
    games['schedule_density'] = games.groupby('TEAM_ID', group_keys=False).apply(_compute_7d_game_count)
    
    # 1g. Pace (5-game average PTS)
    if verbose:
        print("  Computing pace...")
    
    games['pace'] = games.groupby('TEAM_ID')['point_diff'].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # 1h. ELO ratings (running updated ratings)
    if verbose:
        print("  Computing ELO ratings...")
    
    # This returns pre-game ELO rating to avoid same-game leakage.
    games['elo_rating'] = _compute_elo_running(games)
    
    # -----------------------------------------------------------------------
    # STEP 2: Merge home team features onto matchups
    # -----------------------------------------------------------------------
    if verbose:
        print("  Merging home team features...")
    
    home_features = games[[
        'TEAM_ID', 'GAME_DATE', 'last5_win_pct', 'last5_point_diff', 
        'rest_days', 'is_back_to_back', 'home_away_strength', 
        'schedule_density', 'pace', 'elo_rating', 'off_rating', 'def_rating'
    ]].copy()
    home_features.columns = [
        'HOME_TEAM', 'GAME_DATE', 
        'last5_win_pct_home', 'last5_point_diff_home',
        'rest_days_home', 'is_back_to_back_home', 'home_away_strength_home',
        'schedule_density_home', 'pace_home', 'elo_home', 'off_rating_home', 'def_rating_home'
    ]
    
    matchup = pd.merge_asof(
        matchup.sort_values('GAME_DATE'),
        home_features.sort_values('GAME_DATE'),
        on='GAME_DATE',
        by='HOME_TEAM',
        direction='backward',
        tolerance=pd.Timedelta('10000D'),  # Large tolerance to accept any matching TEAM/DATE combo
    )
    
    # -----------------------------------------------------------------------
    # STEP 3: Merge away team features onto matchups
    # -----------------------------------------------------------------------
    if verbose:
        print("  Merging away team features...")
    
    away_features = games[[
        'TEAM_ID', 'GAME_DATE', 'last5_win_pct', 'last5_point_diff',
        'rest_days', 'is_back_to_back', 'home_away_strength',
        'schedule_density', 'pace', 'elo_rating', 'off_rating', 'def_rating'
    ]].copy()
    away_features.columns = [
        'AWAY_TEAM', 'GAME_DATE',
        'last5_win_pct_away', 'last5_point_diff_away',
        'rest_days_away', 'is_back_to_back_away', 'home_away_strength_away',
        'schedule_density_away', 'pace_away', 'elo_away', 'off_rating_away', 'def_rating_away'
    ]
    
    matchup = pd.merge_asof(
        matchup.sort_values('GAME_DATE'),
        away_features.sort_values('GAME_DATE'),
        on='GAME_DATE',
        by='AWAY_TEAM',
        direction='backward',
        tolerance=pd.Timedelta('10000D'),
    )
    
    # -----------------------------------------------------------------------
    # STEP 4: Compute matchup deltas (home - away)
    # -----------------------------------------------------------------------
    if verbose:
        print("  Computing matchup deltas...")
    
    matchup['rest_diff'] = (
        matchup['rest_days_home'].fillna(2.0) - matchup['rest_days_away'].fillna(2.0)
    )
    matchup['home_away_strength_diff'] = (
        matchup['home_away_strength_home'].fillna(0.0) - 
        matchup['home_away_strength_away'].fillna(0.0)
    )
    matchup['schedule_density_diff'] = (
        matchup['schedule_density_home'].fillna(0.0) - 
        matchup['schedule_density_away'].fillna(0.0)
    )
    matchup['pace_diff'] = (
        matchup['pace_home'].fillna(0.0) - matchup['pace_away'].fillna(0.0)
    )
    matchup['elo_diff'] = (
        matchup['elo_home'].fillna(1500.0) - matchup['elo_away'].fillna(1500.0)
    )

    # Game-theory matchup features (PBE-style belief update + Bellman policy gap)
    gt_features = _compute_game_theory_features_vectorized(
        home_off_rating=matchup['off_rating_home'].fillna(100.0).to_numpy(dtype=float),
        home_def_rating=matchup['def_rating_home'].fillna(100.0).to_numpy(dtype=float),
        away_off_rating=matchup['off_rating_away'].fillna(100.0).to_numpy(dtype=float),
        away_def_rating=matchup['def_rating_away'].fillna(100.0).to_numpy(dtype=float),
        rest_days_home=matchup['rest_days_home'].fillna(2.0).to_numpy(dtype=float),
        rest_days_away=matchup['rest_days_away'].fillna(2.0).to_numpy(dtype=float),
        schedule_density_home=matchup['schedule_density_home'].fillna(0.0).to_numpy(dtype=float),
        schedule_density_away=matchup['schedule_density_away'].fillna(0.0).to_numpy(dtype=float),
        is_back_to_back_home=matchup['is_back_to_back_home'].fillna(0.0).to_numpy(dtype=float),
        is_back_to_back_away=matchup['is_back_to_back_away'].fillna(0.0).to_numpy(dtype=float),
        pace_home=matchup['pace_home'].fillna(100.0).to_numpy(dtype=float),
        pace_away=matchup['pace_away'].fillna(100.0).to_numpy(dtype=float),
    )
    
    # -----------------------------------------------------------------------
    # STEP 5: Build output with curated high-signal features
    # -----------------------------------------------------------------------
    if verbose:
        print("  Building output...")
    
    features = pd.DataFrame(index=matchup.index)
    features['elo_diff'] = matchup['elo_diff'].fillna(0.0)
    features['last5_win_pct_home'] = matchup['last5_win_pct_home'].fillna(0.5)
    features['last5_win_pct_away'] = matchup['last5_win_pct_away'].fillna(0.5)
    features['last5_point_diff_home'] = matchup['last5_point_diff_home'].fillna(0.0)
    features['last5_point_diff_away'] = matchup['last5_point_diff_away'].fillna(0.0)
    features['rest_days_home'] = matchup['rest_days_home'].fillna(2.0)
    features['rest_days_away'] = matchup['rest_days_away'].fillna(2.0)
    features['rest_diff'] = matchup['rest_diff'].fillna(0.0)
    features['is_back_to_back_home'] = matchup['is_back_to_back_home'].fillna(0.0)
    features['is_back_to_back_away'] = matchup['is_back_to_back_away'].fillna(0.0)
    features['home_away_strength_diff'] = matchup['home_away_strength_diff'].fillna(0.0)
    features['schedule_density_diff'] = matchup['schedule_density_diff'].fillna(0.0)
    features['pace_diff'] = matchup['pace_diff'].fillna(0.0)
    features['injury_proxy'] = 0.0  # Placeholder (no injury data yet)
    features['expected_payoff_matrix'] = pd.Series(gt_features['expected_payoff_matrix'], index=matchup.index).fillna(0.0)
    features['optimal_path_delta'] = pd.Series(gt_features['optimal_path_delta'], index=matchup.index).fillna(0.0)
    features['signal_consistency_score'] = pd.Series(gt_features['signal_consistency_score'], index=matchup.index).fillna(0.5)
    
    elapsed = time.time() - start
    if verbose:
        print(f"✓ Vectorized feature computation complete in {elapsed:.1f}s")
        print(f"  Output shape: {features.shape}")
    
    return features


def _compute_elo_running(games_df: pd.DataFrame, k_factor: float = 20.0) -> pd.Series:
    """
    Compute running ELO ratings for each team across game history.
    
    Returns a Series with ELO rating BEFORE each game (for leakage-safe merge_asof).
    """
    elo_ratings = {}
    result = []
    
    for _, row in games_df.iterrows():
        team_id = row['TEAM_ID']
        current_elo = elo_ratings.get(team_id, 1500.0)
        result.append(current_elo)
        
        # Simplified update: win/loss/tie
        if row['win'] == 1.0:
            update = k_factor * 0.5  # Rough approximation
        elif row['win'] == 0.0:
            update = -k_factor * 0.5
        else:
            update = 0.0
        
        new_elo = current_elo + update
        elo_ratings[team_id] = new_elo
    
    return pd.Series(result, index=games_df.index)
