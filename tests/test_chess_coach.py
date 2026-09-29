"""Chess coach: reviews explain real mistakes, principles fire on real habits, games persist."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.main import app
from backend.routes.feedback import init_platform
from chess_coach import build_profile, identify_opening, review_game
from chess_coach.principles import BY_KEY, PRINCIPLES
from scripts.db_utils import create_database_engine, ensure_default_reviewers

# 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7#  (evaluations white-centric, one per position)
SCHOLAR = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
SCHOLAR_EVALS = [20, 30, 25, -10, 40, 30, 1000, 1000]
SCHOLAR_BEST = ["e2e4", "e7e5", "b1c3", "g7g6", "f1c4", "d8e7", "h5f7"]


def _quiet_game(n_moves: int) -> tuple[list[str], list[int], list[str]]:
    """Knights shuffling out and back: legal, long, and breaks development and castling."""
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = (cycle * (n_moves // 4 + 1))[:n_moves]
    return moves, [0] * (n_moves + 1), moves[:]


class Review(unittest.TestCase):
    def test_mate_in_one_blunder_is_named_as_mate(self):
        r = review_game(SCHOLAR, SCHOLAR_EVALS, SCHOLAR_BEST, "black", "1-0")
        self.assertEqual(r["outcome"], "loss")
        self.assertEqual(r["counts"]["blunder"], 1)
        moment = r["key_moments"][0]
        self.assertEqual(moment["san"], "Nf6")
        self.assertIn("checkmate", moment["headline"])
        self.assertIn("Qxf7#", moment["headline"])
        self.assertEqual(r["lessons"][0]["title"], BY_KEY["their_threat"].title)
        self.assertIn("3...Nf6", r["lessons"][0]["evidence"])

    def test_early_queen_is_flagged_for_the_side_that_did_it(self):
        white = review_game(SCHOLAR, SCHOLAR_EVALS, SCHOLAR_BEST, "white", "1-0")
        self.assertEqual(white["outcome"], "win")
        self.assertIn("early_queen", white["broken"])
        black = review_game(SCHOLAR, SCHOLAR_EVALS, SCHOLAR_BEST, "black", "1-0")
        self.assertNotIn("early_queen", black["broken"])

    def test_development_and_castling_principles_fire_on_a_passive_game(self):
        moves, evals, best = _quiet_game(26)
        r = review_game(moves, evals, best, "white", "1/2-1/2")
        self.assertIn("develop", r["broken"])
        self.assertIn("castle", r["broken"])
        self.assertIn("one_move_each", r["broken"])
        self.assertIn("center", r["broken"])

    def test_opening_names_come_from_the_lichess_list(self):
        self.assertEqual(identify_opening("e2e4 e7e5 g1f3 b8c6 f1b5".split())["name"], "Ruy Lopez")
        self.assertEqual(identify_opening(SCHOLAR)["eco"], "C20")
        self.assertIsNone(identify_opening([]))

    def test_book_moves_and_engine_choices_lose_nothing(self):
        # A shallow browser search wobbles by tens of centipawns; 1.e4 e5 2.Bc4 must not be "mistakes".
        moves = ["e2e4", "e7e5", "f1c4", "c7c6", "b1c3", "a7a6"]
        evals = [98, 71, 79, -97, 2, -36, 14]            # white's point of view
        best = [None, None, None, None, "d1e2", "g8f6"]
        rows = review_game(moves, evals, best, "white", "*")["moves"]
        self.assertEqual([r["quality"] for r in rows[:4]], ["book"] * 4)
        self.assertEqual([r["cp_loss"] for r in rows[:4]], [0] * 4)
        self.assertEqual(rows[4]["cp_loss"], 38)       # out of book, not the engine's move: counted
        self.assertEqual(rows[5]["cp_loss"], 50)
        same = review_game(moves[:5], evals[:6], best[:4] + ["b1c3"], "white", "*")["moves"]
        self.assertEqual(same[4]["cp_loss"], 0)          # the engine's own choice: counted as no loss

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            review_game(SCHOLAR, SCHOLAR_EVALS[:-1], SCHOLAR_BEST, "black", "1-0")
        with self.assertRaises(ValueError):
            review_game(["e2e5"], [0, 0], [None], "white", "*")

    def test_engineless_games_still_get_opening_lessons(self):
        moves, _, _ = _quiet_game(26)
        r = review_game(moves, [None] * 27, [None] * 26, "white", "*")
        self.assertIsNone(r["acpl"]["all"])
        self.assertIn("develop", r["broken"])

    def test_every_principle_has_teaching_text(self):
        phases = {p.phase for p in PRINCIPLES}
        self.assertEqual(phases, {"opening", "middlegame", "endgame"})
        for p in PRINCIPLES:
            self.assertTrue(p.title and p.why and p.practice, p.key)


class Profile(unittest.TestCase):
    def test_profile_counts_what_the_reviews_stored(self):
        loss = review_game(SCHOLAR, SCHOLAR_EVALS, SCHOLAR_BEST, "black", "1-0")
        win = review_game(SCHOLAR, SCHOLAR_EVALS, SCHOLAR_BEST, "white", "1-0")
        p = build_profile([loss, win, win])
        self.assertEqual(p["record"], {"win": 2, "draw": 0, "loss": 1})
        self.assertEqual(p["by_color"]["white"]["score"], 1.0)
        self.assertTrue(any("queen" in r["title"].lower() for r in p["recurring"]))
        self.assertTrue(p["summary"])

    def test_style_needs_enough_moves_and_is_measured_against_opponents(self):
        short = build_profile([review_game(SCHOLAR, SCHOLAR_EVALS, SCHOLAR_BEST, "white", "1-0")])
        self.assertFalse(any("capture" in s for s in short["summary"]))
        moves, evals, best = _quiet_game(26)
        quiet = review_game(moves, evals, best, "white", "1/2-1/2")
        p = build_profile([quiet] * 4)                     # 52 of your moves, none forcing
        self.assertEqual(p["habits"]["forcing_share"], 0.0)
        self.assertTrue(any("about the same as they do" in s for s in p["summary"]), p["summary"])

    def test_empty_profile_invites_a_first_game(self):
        self.assertEqual(build_profile([])["games"], 0)


class Api(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_database_engine(f"sqlite:///{os.path.join(self._tmp.name, 'c.db')}")
        self._env = patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "false"})
        self._env.start()
        init_platform(self.engine)
        ensure_default_reviewers(self.engine)
        self._patch = patch("backend.routes.chess.engine", self.engine)
        self._patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self._patch.stop()
        self._env.stop()
        self.engine.dispose()
        self._tmp.cleanup()

    def _post(self, color="black", result="1-0", opponent="stockfish-3"):
        return self.client.post("/api/chess/games", json={
            "moves": SCHOLAR, "evals": SCHOLAR_EVALS, "best_moves": SCHOLAR_BEST,
            "player_color": color, "result": result, "opponent": opponent})

    def test_record_against_each_opponent_style(self):
        self._post(opponent="rusher-2")
        self._post(color="white", opponent="rusher-5")
        self._post(opponent="stockfish-3")
        rows = {r["opponent"]: r for r in self.client.get("/api/chess/profile").json()["by_opponent"]}
        self.assertEqual(rows["rusher"], {"opponent": "rusher", "games": 2, "win": 1, "draw": 0, "loss": 1})
        self.assertEqual(rows["stockfish"]["games"], 1)

    def test_coach_and_lesson_scripts_are_served(self):
        for path in ("/chess/coach.js", "/chess/lessons.js"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_games_are_saved_counted_and_reopened(self):
        first = self._post()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["games_played"], 1)
        self.assertEqual(self._post(color="white").json()["games_played"], 2)
        listing = self.client.get("/api/chess/games").json()
        self.assertEqual(listing["total"], 2)
        self.assertEqual({g["outcome"] for g in listing["games"]}, {"win", "loss"})
        one = self.client.get(f"/api/chess/games/{first.json()['game_id']}").json()
        self.assertEqual(one["moves"], SCHOLAR)
        self.assertEqual(one["review"]["opening"]["eco"], "C20")

    def test_profile_and_principles_endpoints(self):
        self._post()
        self._post(color="white")
        prof = self.client.get("/api/chess/profile").json()
        self.assertEqual(prof["games_played"], 2)
        self.assertEqual(prof["record"]["win"], 1)
        tips = self.client.get("/api/chess/principles").json()
        self.assertTrue(tips["opening"] and tips["middlegame"] and tips["endgame"])

    def test_old_reviews_are_redone_when_the_coach_changes(self):
        game_id = self._post().json()["game_id"]
        with self.engine.begin() as conn:
            stale = json.loads(conn.execute(text("SELECT review FROM chess_games WHERE game_id = :g"),
                                            {"g": game_id}).scalar())
            self.assertEqual(stale["engine"]["evals"], SCHOLAR_EVALS)
            stale.update(coach_version=0, lessons=[], acpl={**stale["acpl"], "all": 999.0})
            conn.execute(text("UPDATE chess_games SET review = :r, acpl = 999 WHERE game_id = :g"),
                         {"r": json.dumps(stale), "g": game_id})
        one = self.client.get(f"/api/chess/games/{game_id}").json()
        self.assertEqual(one["evals"], SCHOLAR_EVALS)
        self.assertNotIn("engine", one["review"])
        self.assertTrue(one["review"]["lessons"])
        with self.engine.connect() as conn:
            acpl = conn.execute(text("SELECT acpl FROM chess_games WHERE game_id = :g"), {"g": game_id}).scalar()
        self.assertNotEqual(acpl, 999)

    def test_illegal_or_short_games_are_rejected(self):
        bad = self.client.post("/api/chess/games", json={
            "moves": ["e2e5"], "evals": [0, 0], "best_moves": [None], "player_color": "white", "result": "*"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.client.get("/api/chess/games/nope").status_code, 404)

    def test_pages_are_served(self):
        for path in ("/chess", "/chess/chess.js", "/chess/chess.css", "/workspace"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        self.assertEqual(self.client.get("/chess/secret.py").status_code, 404)


if __name__ == "__main__":
    unittest.main()
