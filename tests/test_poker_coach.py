"""The coach: bots play like their settings, the range model follows the bot, and advice is sane."""
from __future__ import annotations

import unittest
from random import Random

from fastapi.testclient import TestClient

from backend.main import app
from poker.cards import parse_cards
from poker.coach import OPPONENT_LESSONS, opponent_stats, simulate
from poker.evaluator import evaluate
from poker.opponents import PROFILES
from poker.session import LearningMode, PokerSession


class BotsPlayTheirSettings(unittest.TestCase):
    def test_share_of_hands_played_matches_vpip(self):
        for key, profile in PROFILES.items():
            stats = opponent_stats(key)
            self.assertAlmostEqual(stats["plays_preflop"], profile.vpip, delta=0.08, msg=key)
            self.assertAlmostEqual(stats["raises_preflop"], profile.pfr, delta=0.08, msg=key)

    def test_the_calling_station_calls_and_the_rock_folds(self):
        self.assertLess(opponent_stats("calling_station")["folds_to_flop_bet"], 0.15)
        self.assertGreater(opponent_stats("tight_passive")["folds_to_flop_bet"], 0.6)
        self.assertGreater(opponent_stats("maniac")["bluff_share_of_bets"], 0.5)

    def test_every_profile_has_a_lesson(self):
        self.assertEqual(set(OPPONENT_LESSONS), set(PROFILES))


class Simulation(unittest.TestCase):
    def test_aces_crush_seven_deuce_and_the_river_is_exact(self):
        hero = parse_cards("As Ah")
        board = parse_cards("Kd 9c 4s 3h Jc")
        villain = tuple(parse_cards("7d 2c"))
        sim = simulate(hero, board, {villain: 1.0}, {villain: 1.0}, Random(1))
        self.assertTrue(sim.exact)
        self.assertEqual(sim.equity, 1.0)
        flop = simulate(hero, board[:3], {villain: 1.0}, {villain: 0.0}, Random(1), trials=400)
        self.assertGreater(flop.equity, 0.85)
        self.assertEqual(flop.fold_chance, 1.0)


class CoachAtTheTable(unittest.TestCase):
    def _play_until(self, session, done, limit=60):
        for _ in range(limit):
            data = session.to_dict()
            if data["hand"]["is_complete"]:
                if done(data):
                    return data
                session.start_hand()
                continue
            coach = session.coach()
            if done(data):
                return data
            action = coach["best"]
            amount = coach["test_bet"] + session.hero.committed_street if action in ("bet", "raise") else 0
            session.hero_action(action, amount)
        self.fail("never reached the wanted state")

    def test_coach_is_the_default_and_explains_every_choice(self):
        session = PokerSession(opponent_profile="balanced", seed=3)
        session.start_hand()
        self.assertEqual(session.learning_mode, LearningMode.COACH)
        data = session.to_dict()
        coach = data["coach"]
        self.assertTrue(coach["lines"] and coach["advice"] and coach["opponent_tip"])
        legal = data["hand"]["legal_actions"]
        for action, allowed in (("check", legal["can_check"]), ("call", legal["can_call"])):
            self.assertEqual(action in coach["verdicts"], allowed, action)
        self.assertIn(coach["best"], coach["verdicts"])
        self.assertGreater(coach["trials"], 0)

    def test_folding_when_checking_is_free_is_always_a_mistake(self):
        session = PokerSession(opponent_profile="calling_station", seed=8)
        session.start_hand()
        data = self._play_until(session, lambda d: d["coach"] and "check" in d["coach"]["verdicts"])
        self.assertEqual(data["coach"]["verdicts"]["fold"]["grade"], "mistake")

    def test_finished_hands_get_a_recap_of_each_choice(self):
        session = PokerSession(opponent_profile="maniac", seed=4)
        session.start_hand()
        data = self._play_until(session, lambda d: d["hand"]["is_complete"] and d["recap"]["decisions"])
        recap = data["recap"]
        self.assertTrue(recap["lines"] and recap["takeaway"])
        self.assertTrue(all(d["grade"] in ("good", "ok", "mistake") for d in recap["decisions"]))

    def test_other_modes_skip_the_coach(self):
        session = PokerSession(opponent_profile="balanced", seed=3, learning_mode=LearningMode.BEGINNER)
        session.start_hand()
        self.assertIsNone(session.to_dict()["coach"])


class LearnApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_opponent_lessons_carry_measured_numbers(self):
        rows = self.client.get("/api/poker/learn/opponents").json()["opponents"]
        self.assertEqual({r["key"] for r in rows}, set(PROFILES))
        for r in rows:
            self.assertTrue(r["plan"] and r["leak"])
            self.assertIn("folds_to_flop_bet", r["measured"])

    def test_showdown_quiz_names_the_real_winner(self):
        for seed in range(20):
            q = self.client.get(f"/api/poker/learn/showdown?seed={seed}").json()
            board = parse_cards(q["board"])
            a, b = (evaluate(parse_cards(h["cards"]) + board) for h in q["hands"])
            self.assertEqual(q["winner"], 0 if a > b else 1 if b > a else None)

    def test_learn_page_is_served(self):
        self.assertEqual(self.client.get("/poker/learn").status_code, 200)
        self.assertEqual(self.client.get("/poker/learn.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
