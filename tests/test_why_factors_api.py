"""API serialization of why_factors from stored snapshots (incl. legacy rows)."""
from __future__ import annotations

import unittest

from backend.routes.feedback import _explanations, _risk_factors, _why_factors


class TestWhyFactorsApiSerialization(unittest.TestCase):
    def test_schema_v2_passes_through(self) -> None:
        snap = {
            "schema_version": 2,
            "why_factors": [
                {
                    "label": "Elo rating edge",
                    "detail": "Celtics +85 Elo advantage",
                    "strength": 0.8,
                    "side": "home",
                }
            ],
            "risk_factors": [
                {"label": "Close matchup", "detail": "Win probability difference is small"}
            ],
            "explanations": [],
        }
        why = _why_factors(snap)
        self.assertEqual(len(why), 1)
        self.assertEqual(why[0]["label"], "Elo rating edge")
        self.assertEqual(len(_risk_factors(snap)), 1)

    def test_legacy_v1_synthesizes_why_from_explanations(self) -> None:
        snap = {
            "schema_version": 1,
            "explanations": [
                {"label": "Home Starter ERA", "weight": 0.28, "value": "2.65"},
                {"label": "Away Starter ERA", "weight": 0.26, "value": "4.0"},
            ],
        }
        why = _why_factors(snap)
        self.assertEqual(len(why), 2)
        self.assertEqual(why[0]["detail"], "2.65")
        self.assertEqual(why[0]["strength"], 0.28)

    def test_legacy_explanations_without_value_still_renderable(self) -> None:
        """NBA/FIFA demo rows often omit value — must not become empty why_factors."""
        snap = {
            "schema_version": 1,
            "explanations": [
                {"label": "Home advantage", "weight": 0.63},
                {"label": "Possession", "weight": 0.55},
            ],
        }
        expl = _explanations(snap)
        self.assertTrue(all(e.get("value") for e in expl))
        why = _why_factors(snap)
        self.assertEqual(len(why), 2)
        self.assertEqual(why[0]["detail"], "Home advantage")
        self.assertEqual(why[1]["detail"], "Possession")


if __name__ == "__main__":
    unittest.main()
