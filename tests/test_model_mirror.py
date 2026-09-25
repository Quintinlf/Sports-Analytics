"""You vs the AI: the owner's picks default to the model's."""
from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.model_mirror import AGREED, AUTO, LATE, OVERRODE, Game, scoreboard, sign_test_p
from backend.routes.feedback import init_platform
from scripts.db_utils import (
    DEFAULT_REVIEWER_ID,
    create_database_engine,
    ensure_default_reviewers,
    insert_prediction,
)

START = datetime(2026, 9, 20, 23, 0, tzinfo=timezone.utc)


def game(pid, model="LAL", actual="LAL", manual=None, picked=None, start=START, sport="NBA"):
    return Game(prediction_id=pid, sport=sport, matchup="BOS @ LAL", game_date="2026-09-20",
                model_pick=model, actual=actual, manual_pick=manual, picked_at=picked, start=start)


class TestScoring(unittest.TestCase):
    def test_untouched_games_are_the_ai_against_itself(self) -> None:
        board = scoreboard([game(1), game(2, actual="BOS"), game(3)])
        self.assertEqual((board["you_correct"], board["ai_correct"], board["lead"]), (2, 2, 0))
        self.assertEqual(board["picks"]["auto"], 3)
        self.assertIn("AI against itself", board["verdict"])
        self.assertIsNone(board["overrides"]["p_value"])

    def test_only_overrides_move_the_score(self) -> None:
        early = START - timedelta(hours=2)
        games = [
            game(1),                                              # auto, both right
            game(2, manual="BOS", actual="BOS", picked=early),    # override won
            game(3, manual="BOS", actual="BOS", picked=early),    # override won
            game(4, manual="BOS", actual="LAL", picked=early),    # override lost
            game(5, manual="LAL", actual="BOS", picked=early),    # agreed, both wrong
        ]
        board = scoreboard(games)
        self.assertEqual(board["you_correct"] - board["ai_correct"],
                         board["overrides"]["won"] - board["overrides"]["lost"])
        self.assertEqual(board["overrides"], {"won": 2, "lost": 1, "neither_right": 0,
                                              "p_value": 1.0})
        self.assertEqual(board["picks"], {"auto": 1, "agreed": 1, "overrode": 3,
                                          "late_not_scored": 0})
        self.assertEqual(board["lead"], 1)
        self.assertEqual([g["prediction_id"] for g in board["recent_overrides"]], [4, 3, 2])

    def test_pick_after_tip_off_falls_back_to_the_model(self) -> None:
        late = game(1, manual="BOS", actual="BOS", picked=START + timedelta(minutes=5))
        self.assertEqual(late.source, LATE)
        self.assertEqual(late.pick, "LAL")
        board = scoreboard([late])
        self.assertEqual((board["you_correct"], board["ai_correct"]), (0, 0))
        self.assertEqual(board["picks"]["late_not_scored"], 1)

    def test_unknown_start_is_scored_but_flagged(self) -> None:
        g = game(1, manual="BOS", actual="BOS", picked=START, start=None)
        self.assertEqual(g.source, OVERRODE)
        self.assertEqual(scoreboard([g])["unverified_timing"], 1)

    def test_date_only_start_is_not_a_tip_off_time(self) -> None:
        row = {"prediction_id": 1, "predicted_winner": "LAL", "actual_winner": None,
               "home_team": "LAL", "away_team": "BOS", "start_time_utc": "2026-09-20",
               "reviewer_pick": "BOS", "picked_at": "2026-09-20T18:00:00"}
        self.assertIsNone(Game.from_row(row).start)
        self.assertEqual(Game.from_row(row).source, OVERRODE)

    def test_matching_is_case_and_space_insensitive(self) -> None:
        g = game(1, model="Lakers", actual=" lakers ", manual="LAKERS", picked=START - timedelta(hours=1))
        self.assertEqual(g.source, AGREED)
        self.assertTrue(g.you_correct and g.ai_correct)

    def test_sign_test_is_exact(self) -> None:
        self.assertEqual(sign_test_p(1, 0), 1.0)
        self.assertAlmostEqual(sign_test_p(10, 0), 2 / 1024)
        self.assertAlmostEqual(sign_test_p(8, 2), 2 * (1 + 10 + 45) / 1024)
        self.assertEqual(sign_test_p(3, 7), sign_test_p(7, 3))

    def test_upcoming_lists_auto_and_manual_picks(self) -> None:
        stale = Game(prediction_id=3, sport="NBA", matchup="BOS @ LAL", game_date="2026-09-01",
                     model_pick="LAL", actual=None)
        board = scoreboard([game(1, actual=None), stale,
                            game(2, actual=None, manual="BOS", picked=START - timedelta(days=1))],
                           today="2026-09-20")
        self.assertEqual([(g["prediction_id"], g["source"]) for g in board["upcoming"]],
                         [(1, AUTO), (2, OVERRODE)])
        self.assertEqual(board["settled_games"], 0)


class TestScoreboardApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = create_database_engine(f"sqlite:///{os.path.join(self._tmpdir.name, 'm.db')}")
        self._env = patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "false"})
        self._env.start()
        init_platform(self.engine)
        ensure_default_reviewers(self.engine)
        Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        @contextmanager
        def _session():
            db = Session()
            try:
                yield db
            finally:
                db.close()

        self._patches = [
            patch("backend.routes.feedback.get_db_session", _session),
            patch("backend.routes.feedback.engine", self.engine),
            patch("backend.routes.mirror.get_db_session", _session),
            patch("backend.routes.mirror.engine", self.engine),
        ]
        for p in self._patches:
            p.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._env.stop()
        self.engine.dispose()
        self._tmpdir.cleanup()

    def _prediction(self, gid: str, start: datetime, predicted: str = "NYY") -> int:
        return insert_prediction(self.engine, {
            "sport": "MLB", "league": "MLB", "provider_game_id": gid,
            "game_date": start.date().isoformat(), "start_time_utc": start.isoformat(),
            "home_team": "NYY", "away_team": "BOS", "predicted_winner": predicted,
            "win_probability": 0.6, "confidence_level": "MEDIUM", "model_name": "test",
            "prediction_status": "UPCOMING",
        })

    def _settle(self, pid: int, winner: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE predictions SET actual_winner = :w, prediction_status = 'SETTLED' "
                              "WHERE prediction_id = :pid"), {"w": winner, "pid": pid})

    def test_owner_override_beats_the_model_and_nothing_is_written(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(days=1)
        auto_pid = self._prediction("g-auto", future)
        override_pid = self._prediction("g-over", future)
        response = self.client.post("/api/feedback/prediction-reviews", json={
            "prediction_id": override_pid, "reviewer_id": DEFAULT_REVIEWER_ID,
            "reviewer_pick": "BOS", "reviewer_confidence": 4, "agree_with_model": False,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self._settle(auto_pid, "NYY")
        self._settle(override_pid, "BOS")

        board = self.client.get("/api/mirror/scoreboard").json()
        self.assertTrue(board["owner"]["found"])
        self.assertEqual(board["owner"]["reviewer_id"], DEFAULT_REVIEWER_ID)
        self.assertEqual((board["you_correct"], board["ai_correct"]), (2, 1))
        self.assertEqual(board["picks"]["auto"], 1)
        self.assertEqual(board["overrides"]["won"], 1)
        self.assertEqual(board["unverified_timing"], 0)

        with self.engine.connect() as conn:
            reviews = conn.execute(text("SELECT COUNT(*) FROM prediction_reviews")).scalar()
        self.assertEqual(reviews, 1, "auto picks are resolved, never inserted")

    def test_unknown_owner_is_the_ai_against_itself(self) -> None:
        pid = self._prediction("g1", datetime.now(timezone.utc) + timedelta(days=1))
        self._settle(pid, "BOS")
        board = self.client.get("/api/mirror/scoreboard", params={"reviewer": "nobody"}).json()
        self.assertFalse(board["owner"]["found"])
        self.assertEqual((board["you_correct"], board["ai_correct"], board["picks"]["auto"]), (0, 0, 1))

    def test_same_name_accounts_resolve_to_the_canonical_one(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO reviewers (reviewer_id, name, created_at) "
                              "VALUES ('dup-1', 'QUINTIN', '2000-01-01T00:00:00')"))
        owner = self.client.get("/api/mirror/scoreboard", params={"reviewer": "Quintin"}).json()["owner"]
        self.assertEqual(owner["reviewer_id"], DEFAULT_REVIEWER_ID, "the row with an email wins")
        self.assertEqual(owner["same_name_accounts"], 1)

    def test_home_page_is_the_hub(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/poker?play=1", response.text)
        self.assertIn("You vs the AI", response.text)
        self.assertEqual(self.client.get("/home/home.js").status_code, 200)
        self.assertEqual(self.client.get("/home/secret.env").status_code, 404)


if __name__ == "__main__":
    unittest.main()
