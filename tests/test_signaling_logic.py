import unittest

import numpy as np

from src.evaluation.vectorized_features import (
    BasketballStateNode,
    compute_game_theory_matchup_features,
)


class SignalingLogicTests(unittest.TestCase):
    def test_intuitive_criterion_zeros_weak_type_mass(self):
        prior_low = np.array([0.20], dtype=float)
        action_high = np.array([1.0], dtype=float)
        p_high_given_low = np.array([0.80], dtype=float)
        p_high_given_high = np.array([0.70], dtype=float)

        eq_low = np.array([2.0], dtype=float)
        eq_high = np.array([1.5], dtype=float)
        max_dev_low = np.array([2.3], dtype=float)
        max_dev_high = np.array([1.0], dtype=float)  # m < x for weak/high-fatigue type

        post_low, post_high = BasketballStateNode.update_belief_with_ic(
            prior_low_fatigue=prior_low,
            action_high_energy=action_high,
            p_action_high_given_low=p_high_given_low,
            p_action_high_given_high=p_high_given_high,
            eq_payoff_low=eq_low,
            eq_payoff_high=eq_high,
            max_dev_payoff_low=max_dev_low,
            max_dev_payoff_high=max_dev_high,
        )

        self.assertGreater(post_low[0], 0.99)
        self.assertLess(post_high[0], 0.01)

    def test_weak_type_high_energy_signal_scores_as_less_consistent(self):
        # Home side is configured as weak/fatigued but shows high-energy pace.
        features = compute_game_theory_matchup_features(
            home_off_rating=103.0,
            home_def_rating=112.0,
            away_off_rating=110.0,
            away_def_rating=106.0,
            rest_days_home=0.0,
            rest_days_away=2.0,
            schedule_density_home=5.0,
            schedule_density_away=2.0,
            is_back_to_back_home=1.0,
            is_back_to_back_away=0.0,
            pace_home=118.0,
            pace_away=99.0,
        )

        score = float(features['signal_consistency_score'])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertLessEqual(score, 0.50)


if __name__ == '__main__':
    unittest.main()
