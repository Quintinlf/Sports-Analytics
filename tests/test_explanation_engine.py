"""Tests for the cross-sport prediction explanation layer."""
from __future__ import annotations

import json
import unittest

import pandas as pd

from data.explanation_engine import (
    build_risk_factors,
    build_snapshot,
    explain_fifa_prediction,
    explain_mlb_prediction,
    explain_nba_prediction,
    factors_from_home_away_pairs,
    factors_from_squad_profiles,
    _MLB_PAIR_LABELS,
)


class TestExplanationEngine(unittest.TestCase):
    def test_mlb_factors_favor_predicted_winner(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "HOME_R_ROLL": 5.5,
                    "AWAY_R_ROLL": 3.2,
                    "HOME_RA_ROLL": 3.0,
                    "AWAY_RA_ROLL": 5.1,
                    "HOME_WIN_RATE_10": 0.7,
                    "AWAY_WIN_RATE_10": 0.4,
                    "HOME_REST_DAYS": 2.0,
                    "AWAY_REST_DAYS": 1.0,
                    "HOME_WIN_STREAK": 3.0,
                    "AWAY_WIN_STREAK": 0.0,
                    "HOME_IS_BACK_TO_BACK": 0.0,
                    "AWAY_IS_BACK_TO_BACK": 1.0,
                }
            ]
        )
        result = explain_mlb_prediction(
            features=features,
            predicted_winner="Padres",
            home_team="Padres",
            away_team="Dodgers",
            win_probability=0.62,
            confidence_level="MEDIUM",
            missing_data_warnings=["pitcher_stats_unavailable"],
        )
        self.assertGreaterEqual(len(result["why_factors"]), 3)
        self.assertLessEqual(len(result["why_factors"]), 5)
        labels = {f["label"] for f in result["why_factors"]}
        self.assertTrue(
            labels & {
                "Recent scoring (runs / game)",
                "Recent run prevention",
                "Last-10 win rate",
                "Win streak",
                "Rest days",
                "Avoiding a back-to-back",
            }
        )
        for f in result["why_factors"]:
            self.assertIn("Padres", f["detail"])
            self.assertGreater(f["strength"], 0)
        risk_codes = {r["code"] for r in result["risk_factors"]}
        self.assertIn("pitcher_stats_unavailable", risk_codes)
        snap = build_snapshot(
            sport="MLB",
            data_source="mlb_statsapi",
            is_fallback=False,
            confidence_score=0.62,
            explanations=result["explanations"],
            metrics={"model_win_probability": 0.62},
            why_factors=result["why_factors"],
            risk_factors=result["risk_factors"],
        )
        self.assertEqual(snap["schema_version"], 2)
        self.assertEqual(len(snap["why_factors"]), len(result["why_factors"]))
        # JSON-serializable for feature_snapshot column
        json.dumps(snap)

    def test_mlb_away_winner_uses_away_advantage(self) -> None:
        features = {
            "HOME_R_ROLL": 2.0,
            "AWAY_R_ROLL": 6.0,
            "HOME_WIN_RATE_10": 0.3,
            "AWAY_WIN_RATE_10": 0.8,
        }
        factors = factors_from_home_away_pairs(
            features,
            predicted_winner="Yankees",
            home_team="Red Sox",
            away_team="Yankees",
            pair_labels=_MLB_PAIR_LABELS,
            top_n=5,
        )
        self.assertTrue(factors)
        self.assertTrue(all(f["side"] == "away" for f in factors))

    def test_nba_factors_include_elo_and_form(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "HOME_PTS_ROLL": 118.0,
                    "AWAY_PTS_ROLL": 105.0,
                    "HOME_FG_PCT_ROLL": 0.49,
                    "AWAY_FG_PCT_ROLL": 0.44,
                    "HOME_WIN_RATE_10": 0.7,
                    "AWAY_WIN_RATE_10": 0.4,
                    "HOME_REST_DAYS": 2.0,
                    "AWAY_REST_DAYS": 1.0,
                    "elo_diff": 85.0,
                    "rest_diff": 1.0,
                    "injury_proxy": 0.0,
                }
            ]
        )
        result = explain_nba_prediction(
            features=features,
            predicted_winner="Celtics",
            home_team="Celtics",
            away_team="Knicks",
            win_probability=0.71,
            confidence_level="HIGH",
        )
        self.assertGreaterEqual(len(result["why_factors"]), 3)
        labels = " ".join(f["label"] for f in result["why_factors"]).lower()
        self.assertTrue(
            "elo" in labels or "scoring" in labels or "win rate" in labels or "home court" in labels
        )
        self.assertEqual(result["risk_factors"], [])

    def test_nba_close_matchup_risk(self) -> None:
        risks = build_risk_factors(
            win_probability=0.52,
            confidence_level="LOW",
        )
        codes = {r["code"] for r in risks}
        self.assertIn("close_matchup", codes)
        self.assertIn("low_confidence", codes)

    def test_fifa_squad_metric_factors(self) -> None:
        home = {
            "xg_per90": 1.8,
            "gls": 12.0,
            "poss": 58.0,
            "ga": 4.0,
            "pass_pct": 86.0,
        }
        away = {
            "xg_per90": 1.1,
            "gls": 7.0,
            "poss": 45.0,
            "ga": 9.0,
            "pass_pct": 79.0,
        }
        result = explain_fifa_prediction(
            home_metrics=home,
            away_metrics=away,
            predicted_winner="France",
            home_team="France",
            away_team="Belgium",
            win_probability=0.58,
            confidence_level="MEDIUM",
            outcome_probabilities={"HOME_WIN": 0.58, "DRAW": 0.30, "AWAY_WIN": 0.12},
        )
        self.assertGreaterEqual(len(result["why_factors"]), 3)
        self.assertLessEqual(len(result["why_factors"]), 5)
        for f in result["why_factors"]:
            self.assertIn("France", f["detail"])
        risk_codes = {r["code"] for r in result["risk_factors"]}
        self.assertIn("draw_likely", risk_codes)

    def test_fifa_factors_from_squad_profiles_dedupe_labels(self) -> None:
        home = {"gls": 10.0, "goals_per90": 1.5, "xg": 1.4}
        away = {"gls": 5.0, "goals_per90": 0.8, "xg": 0.9}
        factors = factors_from_squad_profiles(
            home,
            away,
            predicted_winner="Brazil",
            home_team="Brazil",
            away_team="Chile",
            top_n=5,
        )
        labels = [f["label"] for f in factors]
        self.assertEqual(len(labels), len(set(labels)))

    def test_snapshot_explanations_auto_fill_from_why(self) -> None:
        snap = build_snapshot(
            sport="NBA",
            data_source="nba_api",
            is_fallback=False,
            confidence_score=0.66,
            explanations=[],
            metrics={},
            why_factors=[
                {
                    "label": "Elo rating edge",
                    "detail": "Celtics: elo_diff = 40",
                    "side": "home",
                    "strength": 1.0,
                    "source_feature": "elo_diff",
                }
            ],
            risk_factors=[],
        )
        self.assertEqual(len(snap["explanations"]), 1)
        self.assertEqual(snap["explanations"][0]["label"], "Elo rating edge")


if __name__ == "__main__":
    unittest.main()
