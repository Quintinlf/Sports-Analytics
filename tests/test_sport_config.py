"""Tests for centralized sport configuration."""
from __future__ import annotations

import unittest

from data.sport_config import (
    SPORTS,
    build_outcome_options,
    binary_home_win_probabilities,
    get_config,
    get_feature_set,
    get_form_windows,
    get_model_type,
    get_outcome_types,
    get_feedback_categories,
    get_email_labels,
    format_matchup,
    outcome_display_label,
)


class TestSportConfig(unittest.TestCase):
    def test_all_sports_defined(self) -> None:
        self.assertEqual(set(SPORTS), {"NBA", "MLB", "SOCCER"})

    def test_outcome_types(self) -> None:
        self.assertEqual(get_outcome_types("NBA"), ["HOME_WIN", "AWAY_WIN"])
        self.assertEqual(get_outcome_types("MLB"), ["HOME_WIN", "AWAY_WIN"])
        self.assertEqual(get_outcome_types("SOCCER"), ["HOME_WIN", "DRAW", "AWAY_WIN"])

    def test_form_windows(self) -> None:
        for sport in SPORTS:
            windows = get_form_windows(sport)
            self.assertIn(5, windows)
            self.assertIn(10, windows)

    def test_feature_sets_include_elo(self) -> None:
        for sport in SPORTS:
            elo_feats = get_feature_set(sport, category="elo")
            self.assertIn("elo_diff", elo_feats)

    def test_model_types_are_strings(self) -> None:
        self.assertEqual(get_model_type("NBA"), "ensemble")
        self.assertEqual(get_model_type("MLB"), "lightgbm")
        self.assertEqual(get_model_type("SOCCER"), "xgboost_multiclass")

    def test_email_labels(self) -> None:
        nba = get_email_labels("NBA")
        self.assertIn("report_title", nba)
        self.assertIn("🏀", nba["icon"])

    def test_feedback_categories(self) -> None:
        mlb = get_feedback_categories("MLB")
        self.assertIn("pitcher", mlb)
        self.assertIn("bullpen", mlb)

    def test_binary_outcome_options(self) -> None:
        probs = binary_home_win_probabilities(0.62)
        options = build_outcome_options("MLB", "NYY", "BOS", probs)
        self.assertEqual(len(options), 2)
        self.assertEqual(options[0]["option_name"], "NYY Win")
        self.assertAlmostEqual(options[0]["probability"], 0.62)
        self.assertEqual(options[0]["rank"], 1)

    def test_soccer_three_way_options(self) -> None:
        probs = {"HOME_WIN": 0.52, "DRAW": 0.28, "AWAY_WIN": 0.20}
        options = build_outcome_options("SOCCER", "Sweden", "Tunisia", probs)
        self.assertEqual(len(options), 3)
        self.assertEqual(options[0]["option_name"], "Sweden Win")
        self.assertEqual(outcome_display_label("DRAW", "Sweden", "Tunisia"), "Draw")

    def test_matchup_format(self) -> None:
        self.assertEqual(format_matchup("NBA", "Knicks", "Spurs"), "Spurs @ Knicks")
        self.assertEqual(format_matchup("MLB", "NYY", "BOS"), "BOS at NYY")
        self.assertEqual(format_matchup("SOCCER", "Sweden", "Tunisia"), "Sweden vs Tunisia")

    def test_unknown_sport_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_config("NHL")


if __name__ == "__main__":
    unittest.main()
