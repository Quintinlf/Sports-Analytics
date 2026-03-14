"""
Shot analysis helpers (experimental).
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class ShotAnalyzer:
    """Analyze shot quality and zone-level efficiency."""

    @staticmethod
    def calculate_shot_quality(shot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add expected-points estimate for each shot row.

        Expected columns:
        - SHOT_TYPE in {'2PT','3PT'}
        - FG_PCT in [0, 1]
        """
        df = shot_data.copy()
        df['Expected_Points'] = np.where(
            df['SHOT_TYPE'].astype(str).str.upper().str.contains('3'),
            pd.to_numeric(df['FG_PCT'], errors='coerce').fillna(0.0) * 3.0,
            pd.to_numeric(df['FG_PCT'], errors='coerce').fillna(0.0) * 2.0,
        )
        return df

    @staticmethod
    def analyze_shot_zones(shots: pd.DataFrame, zones: List[str]) -> Dict[str, Dict]:
        """Aggregate makes, attempts, FG%, and expected points by zone."""
        stats: Dict[str, Dict] = {}
        for zone in zones:
            zone_df = shots[shots['ZONE'] == zone]
            if len(zone_df) == 0:
                continue

            makes = pd.to_numeric(zone_df.get('SHOT_MADE', 0), errors='coerce').fillna(0).sum()
            attempts = len(zone_df)
            fg_pct = float(makes / attempts) if attempts > 0 else 0.0

            exp_pts_col = (
                pd.to_numeric(zone_df['Expected_Points'], errors='coerce').fillna(0.0)
                if 'Expected_Points' in zone_df.columns
                else pd.Series(np.zeros(attempts), index=zone_df.index)
            )

            stats[zone] = {
                'attempts': int(attempts),
                'makes': int(makes),
                'fg_pct': fg_pct,
                'expected_points': float(exp_pts_col.mean()),
            }
        return stats

    @staticmethod
    def rolling_bayesian_average(
        series: pd.Series,
        window: int = 5,
        prior_weight: float = 2.0,
    ) -> pd.Series:
        """Rolling Bayesian average with a global prior mean."""
        s = pd.to_numeric(series, errors='coerce')
        overall_mean = float(s.mean()) if len(s) else 0.0
        rolling_sum = s.rolling(window=window, min_periods=1).sum()
        rolling_count = s.rolling(window=window, min_periods=1).count()
        return (rolling_sum + prior_weight * overall_mean) / (rolling_count + prior_weight)
