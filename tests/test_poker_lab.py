"""Shuffle & Fourier lab API: exact where it claims to be, stored where it must be."""
from __future__ import annotations

import random
import unittest

from fastapi.testclient import TestClient

from backend.main import app
from backend.routes.poker_lab import LAB_DATA, cut_distribution
from poker.fourier import total_variation, walk_distribution
from poker.shuffle import PROCEDURES, cut
from poker.shuffle_analysis import rising_sequences


class TestPokerLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_seven_riffles_is_the_first_below_one_half(self) -> None:
        data = self.client.get("/api/poker/lab/riffles").json()
        self.assertEqual(data["riffles_needed_half"], 7)
        table = {row["riffles"]: row["tv_distance"] for row in data["table"]}
        self.assertAlmostEqual(table[7], 0.334, places=3)
        self.assertAlmostEqual(table[10], 0.043, places=3)

    def test_exact_cut_distribution_matches_the_cut_function(self) -> None:
        exact = cut_distribution()
        self.assertEqual(exact[0], 0.0, "cut() never cuts at 0")
        self.assertAlmostEqual(sum(exact), 1.0, places=12)
        rng = random.Random(11)
        deck = list(range(52))
        counts = [0] * 52
        trials = 60_000
        for _ in range(trials):
            counts[cut(deck, rng)[0]] += 1
        sampled = [c / trials for c in counts]
        self.assertLess(total_variation(exact, sampled), 0.02)

    def test_cut_walk_mixes_rotations_but_never_the_deck(self) -> None:
        data = self.client.get("/api/poker/lab/cuts", params={"sd": 5.2, "max_cuts": 25}).json()
        walk = {row["cuts"]: row for row in data["walk"]}
        self.assertLess(walk[25]["tv_on_rotations"], walk[1]["tv_on_rotations"])
        for row in data["walk"]:
            self.assertLessEqual(row["tv_on_rotations"], row["upper_bound"] + 1e-12)
            self.assertEqual(row["tv_from_shuffled"], 1.0)
        self.assertAlmostEqual(data["coefficients"][0]["modulus"], 1.0)
        self.assertAlmostEqual(data["orthogonality_check"]["same"], 52.0)
        self.assertLess(data["orthogonality_check"]["different"], 1e-9)

    def test_inverse_transform_reproduces_the_walk(self) -> None:
        """What lab.js does for the k slider: invert P_hat(m)^k."""
        import cmath

        data = self.client.get("/api/poker/lab/cuts").json()
        coeffs = [complex(c["re"], c["im"]) for c in data["coefficients"]]
        k = 5
        inverse = [
            sum(coeffs[m] ** k * cmath.exp(-2j * cmath.pi * j * m / 52) for m in range(52)).real / 52
            for j in range(52)
        ]
        direct = walk_distribution(data["distribution"], k)
        self.assertLess(max(abs(a - b) for a, b in zip(inverse, direct)), 1e-12)

    def test_shuffle_frames_are_reproducible_and_counted(self) -> None:
        params = {"procedure": "casino standard", "seed": 42}
        first = self.client.get("/api/poker/lab/shuffle", params=params).json()
        again = self.client.get("/api/poker/lab/shuffle", params=params).json()
        self.assertEqual(first, again)
        self.assertEqual([f["step"] for f in first["frames"]],
                         ["new deck", "riffle", "riffle", "strip", "riffle", "cut"])
        for frame in first["frames"]:
            self.assertEqual(sorted(frame["arrangement"]), list(range(52)))
            self.assertEqual(frame["rising_sequences"], rising_sequences(frame["arrangement"]))
        self.assertLessEqual(first["frames"][1]["rising_sequences"], 2)

    def test_unknown_procedure_is_404(self) -> None:
        self.assertEqual(self.client.get("/api/poker/lab/shuffle",
                                         params={"procedure": "shuffle harder"}).status_code, 404)

    def test_every_procedure_has_a_stored_report(self) -> None:
        self.assertTrue(LAB_DATA.exists(), "run scripts/build_shuffle_lab.py")
        data = self.client.get("/api/poker/lab/procedures").json()
        self.assertEqual({p["key"] for p in data["procedures"]}, set(PROCEDURES))
        for p in data["procedures"]:
            self.assertIsNotNone(p["report"], p["key"])
        by_key = {p["key"]: p["report"] for p in data["procedures"]}
        self.assertEqual(by_key["casino standard"]["significant_fraction"], 1.0)
        self.assertAlmostEqual(by_key["7x riffle"]["exact_tv"], 0.334, places=3)

    def test_lab_and_poker_pages_are_served(self) -> None:
        page = self.client.get("/poker/lab")
        self.assertEqual(page.status_code, 200)
        self.assertIn("/poker/lab.js", page.text)
        self.assertEqual(self.client.get("/poker/lab.js").status_code, 200)
        self.assertIn('href="/poker/lab"', self.client.get("/poker").text)


if __name__ == "__main__":
    unittest.main()
