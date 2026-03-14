"""
Adaptive learner utilities for post-prediction error feedback.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


class AdaptiveLearner:
    """
    Lightweight adaptive learner that proposes team-level adjustments and
    EPAA-weight refinements from matched prediction results.
    """

    def __init__(self, mcmc_model=None):
        self.mcmc_model = mcmc_model
        self.learning_history: List[Dict] = []

    def calculate_team_error_adjustments(
        self,
        matches: List[Dict],
        team_data: Dict = None,
    ) -> Dict[str, float]:
        """Estimate per-team bias adjustments from historical spread errors."""
        team_errors: Dict[str, List[float]] = {}

        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            error = float(match['predicted_spread']) - float(match['actual_spread'])

            team_errors.setdefault(home_team, []).append(error)
            team_errors.setdefault(away_team, []).append(-error)

        adjustments: Dict[str, float] = {}
        for team, errors in team_errors.items():
            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors))
            n = len(errors)

            if n > 3 and std_error > 0:
                t_stat = mean_error / (std_error / np.sqrt(n))
                p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), n - 1))
                adjustments[team] = -mean_error * 0.1 if p_value < 0.10 else 0.0
            else:
                adjustments[team] = 0.0

        return adjustments

    def propose_mcmc_refinement(
        self,
        matches: List[Dict],
        current_epaa_weight: float = 0.5,
    ) -> Dict:
        """Propose EPAA weight update based on error profile."""
        if not matches:
            return {
                'current_epaa_weight': current_epaa_weight,
                'proposed_epaa_weight': current_epaa_weight,
                'weight_change': 0.0,
                'reasoning': ['No matches available.'],
                'metrics': {'mae': None, 'win_accuracy': None, 'uncertainty_calibrated': None},
            }

        df = pd.DataFrame(matches)
        mae = float(df['spread_error'].mean())
        win_accuracy = float(df['correct_winner'].mean())

        proposed = current_epaa_weight
        reasoning: List[str] = []

        if win_accuracy < 0.55:
            proposed = max(0.2, current_epaa_weight - 0.1)
            reasoning.append('Low win accuracy suggests reducing EPAA weight.')
        elif win_accuracy > 0.70 and mae > 10:
            proposed = min(0.8, current_epaa_weight + 0.05)
            reasoning.append('Direction signal is good; increase EPAA weight slightly.')
        elif mae < 8:
            proposed = min(0.7, current_epaa_weight + 0.02)
            reasoning.append('Strong error profile; keep/increase EPAA contribution.')

        high_unc = df[df['uncertainty'] > df['uncertainty'].median()]
        low_unc = df[df['uncertainty'] <= df['uncertainty'].median()]
        uncertainty_calibrated = (
            float(high_unc['spread_error'].mean()) > float(low_unc['spread_error'].mean())
            if len(high_unc) > 0 and len(low_unc) > 0
            else False
        )

        if not uncertainty_calibrated:
            reasoning.append('Uncertainty calibration appears weak; review GP kernel settings.')

        out = {
            'current_epaa_weight': current_epaa_weight,
            'proposed_epaa_weight': proposed,
            'weight_change': proposed - current_epaa_weight,
            'reasoning': reasoning,
            'metrics': {
                'mae': mae,
                'win_accuracy': win_accuracy,
                'uncertainty_calibrated': uncertainty_calibrated,
            },
        }
        self.learning_history.append({'timestamp': datetime.now().isoformat(), 'summary': out})
        return out

    def generate_learning_report(
        self,
        matches: List[Dict],
        team_data: Dict = None,
        current_epaa_weight: float = 0.5,
    ) -> str:
        """Create a markdown report with actionable learning recommendations."""
        if not matches:
            return '# Adaptive Learning Report\n\nNo matches available.\n'

        df = pd.DataFrame(matches)
        mae = float(df['spread_error'].mean())
        rmse = float(np.sqrt((df['spread_error'] ** 2).mean()))
        win_acc = float(df['correct_winner'].mean())
        bias = float((df['predicted_spread'] - df['actual_spread']).mean())

        adjustments = self.calculate_team_error_adjustments(matches, team_data or {})
        refinement = self.propose_mcmc_refinement(matches, current_epaa_weight)

        lines = [
            '# Adaptive Learning Report',
            '',
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f'Games analyzed: {len(df)}',
            '',
            '## Performance',
            f'- Win accuracy: {win_acc:.1%}',
            f'- MAE: {mae:.2f}',
            f'- RMSE: {rmse:.2f}',
            f'- Mean bias: {bias:+.2f}',
            '',
            '## EPAA Refinement',
            f"- Current weight: {refinement['current_epaa_weight']:.2f}",
            f"- Proposed weight: {refinement['proposed_epaa_weight']:.2f}",
            f"- Change: {refinement['weight_change']:+.2f}",
        ]

        if refinement['reasoning']:
            lines.append('- Rationale:')
            for reason in refinement['reasoning']:
                lines.append(f'  - {reason}')

        significant = {k: v for k, v in adjustments.items() if abs(v) > 0.5}
        lines += ['', '## Team Adjustments']
        if significant:
            for team, adj in sorted(significant.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]:
                lines.append(f'- {team}: {adj:+.2f}')
        else:
            lines.append('- No statistically significant team-level adjustment identified.')

        return '\n'.join(lines) + '\n'
