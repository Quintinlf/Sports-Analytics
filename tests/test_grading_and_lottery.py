"""Every pick graded after the fact; Powerball tickets committed and graded."""
from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from backend.lottery_picks import (
    EASTERN, POWERBALL, grade, model_ticket, next_draw, prize_for, submit_pick, validate_ticket,
)
from backend.main import app
from backend.routes.feedback import init_platform
from lottery.history import Draw
from lottery.popularity import PopularityModel
from scripts.db_utils import (
    DEFAULT_REVIEWER_ID, create_database_engine, ensure_default_reviewers, insert_prediction,
)


class _Db(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_database_engine(f"sqlite:///{os.path.join(self._tmp.name, 'g.db')}")
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

        self._patches = [patch(f"backend.routes.{m}.{name}", value)
                         for m in ("feedback", "mirror", "lottery")
                         for name, value in (("get_db_session", _session), ("engine", self.engine))
                         if not (m == "lottery" and name == "get_db_session")]
        for p in self._patches:
            p.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._env.stop()
        self.engine.dispose()
        self._tmp.cleanup()


class TestGrading(_Db):
    def _game(self, gid: str, start: datetime, winner: str) -> int:
        pid = insert_prediction(self.engine, {
            "sport": "MLB", "league": "MLB", "provider_game_id": gid,
            "game_date": start.date().isoformat(), "start_time_utc": start.isoformat(),
            "home_team": "NYY", "away_team": "BOS", "predicted_winner": "NYY",
            "win_probability": 0.6, "confidence_level": "MEDIUM", "model_name": "test",
            "prediction_status": "UPCOMING"})
        self._winners[pid] = winner
        return pid

    def _review(self, pid: int, reviewer: str, pick: str) -> None:
        r = self.client.post("/api/feedback/prediction-reviews", json={
            "prediction_id": pid, "reviewer_id": reviewer, "reviewer_pick": pick,
            "reviewer_confidence": 3, "agree_with_model": pick == "NYY"})
        self.assertEqual(r.status_code, 200, r.text)

    def _settle_all(self) -> None:
        with self.engine.begin() as conn:
            for pid, w in self._winners.items():
                conn.execute(text("UPDATE predictions SET actual_winner = :w, prediction_status = 'SETTLED' "
                                  "WHERE prediction_id = :p"), {"w": w, "p": pid})

    def test_everyone_is_graded_and_the_owner_defaults_to_the_model(self) -> None:
        self._winners = {}
        future = datetime.now(timezone.utc) + timedelta(days=1)
        a = self._game("a", future, "NYY")          # owner auto: both right
        b = self._game("b", future, "BOS")          # owner overrides to BOS: wins
        c = self._game("c", future, "BOS")          # only timothy picks: BOS, right
        self._game("d", future + timedelta(days=5), "NYY")  # never settled here
        self._review(b, DEFAULT_REVIEWER_ID, "BOS")
        self._review(c, "timothy", "BOS")
        self._settle_all()
        with self.engine.begin() as conn:  # d stays unsettled
            conn.execute(text("UPDATE predictions SET actual_winner = NULL, prediction_status = 'UPCOMING' "
                              "WHERE provider_game_id = 'd'"))

        board = self.client.get("/api/mirror/leaderboard").json()
        self.assertEqual(board["grading"]["inserted"], 4)  # owner a, b, c and timothy c
        rows = {r["reviewer_id"]: r for r in board["leaderboard"]}
        owner = rows[DEFAULT_REVIEWER_ID]
        self.assertTrue(owner["is_owner"])
        self.assertEqual((owner["picks"], owner["correct"], owner["model_correct"]), (3, 2, 1))
        self.assertEqual((owner["auto"], owner["overrides"], owner["won"]), (2, 1, 1))
        timothy = rows["timothy"]
        self.assertEqual((timothy["picks"], timothy["correct"], timothy["model_correct"]), (1, 1, 0))
        self.assertEqual((timothy["won"], timothy["lost"]), (1, 0))
        with self.engine.connect() as conn:
            stored = conn.execute(text("SELECT COUNT(*) FROM pick_grades")).scalar()
            reviews = conn.execute(text("SELECT COUNT(*) FROM prediction_reviews")).scalar()
        self.assertEqual(stored, 4)
        self.assertEqual(reviews, 2, "default picks are graded, never inserted as reviews")
        again = self.client.get("/api/mirror/leaderboard").json()["grading"]
        self.assertEqual((again["inserted"], again["updated"]), (0, 0))

    def test_a_corrected_result_regrades(self) -> None:
        self._winners = {}
        pid = self._game("x", datetime.now(timezone.utc) + timedelta(days=1), "NYY")
        self._settle_all()
        self.client.get("/api/mirror/leaderboard")
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE predictions SET actual_winner = 'BOS' WHERE prediction_id = :p"), {"p": pid})
        again = self.client.get("/api/mirror/leaderboard").json()
        self.assertEqual(again["grading"]["updated"], 1)
        owner = next(r for r in again["leaderboard"] if r["is_owner"])
        self.assertEqual(owner["correct"], 0)


class TestPowerballRules(unittest.TestCase):
    def test_draw_schedule(self) -> None:
        mon_evening = datetime(2026, 9, 21, 20, 0, tzinfo=EASTERN)
        self.assertEqual(next_draw(POWERBALL, mon_evening), date(2026, 9, 21))
        after_draw = datetime(2026, 9, 21, 23, 30, tzinfo=EASTERN)
        self.assertEqual(next_draw(POWERBALL, after_draw), date(2026, 9, 23))
        friday = datetime(2026, 9, 25, 9, 0, tzinfo=EASTERN)
        self.assertEqual(next_draw(POWERBALL, friday), date(2026, 9, 26))

    def test_model_ticket_is_seeded_valid_and_uncrowded(self) -> None:
        day = date(2026, 9, 26)
        first, again = model_ticket(POWERBALL, day), model_ticket(POWERBALL, day)
        self.assertEqual(first, again)
        self.assertNotEqual(first["whites"], model_ticket(POWERBALL, date(2026, 9, 28))["whites"])
        validate_ticket(POWERBALL, day, first["whites"], first["special"])
        birthdays = PopularityModel(POWERBALL.matrix).crowd_score((3, 7, 12, 21, 28))
        self.assertLess(first["crowd_score"], 1.0)
        self.assertLess(first["crowd_score"], birthdays)

    def test_prize_tiers(self) -> None:
        day = date(2026, 9, 26)
        self.assertEqual(prize_for(POWERBALL, day, 5, True), (None, True))
        self.assertEqual(prize_for(POWERBALL, day, 3, True), (100, False))
        self.assertEqual(prize_for(POWERBALL, day, 0, True), (4, False))
        self.assertEqual(prize_for(POWERBALL, day, 2, False), (0, False))

    def test_bad_tickets_are_refused(self) -> None:
        day = date(2026, 9, 26)
        for whites, special in [((1, 1, 2, 3, 4), 5), ((1, 2, 3, 4, 70), 5), ((1, 2, 3, 4, 5), 27), ((1, 2, 3, 4), 5)]:
            with self.assertRaises(ValueError):
                validate_ticket(POWERBALL, day, whites, special)


class TestPowerballApi(_Db):
    def test_model_ticket_committed_owner_defaults_then_graded(self) -> None:
        with patch("lottery.sources.fetch_draws", return_value=[]):
            first = self.client.get("/api/lottery/powerball").json()
        self.assertIsNotNone(first["model_next"])
        self.assertEqual(first["your_next"]["source"], "auto")
        self.assertEqual(first["your_next"]["whites"], first["model_next"]["whites"])

        r = self.client.post("/api/lottery/powerball/picks", json={"whites": [5, 17, 33, 48, 60], "special": 9})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["reviewer_id"], DEFAULT_REVIEWER_ID)
        bad = self.client.post("/api/lottery/powerball/picks", json={"whites": [5, 5, 33, 48, 60], "special": 9})
        self.assertEqual(bad.status_code, 400)

        # the draw happens: whites 5 17 33 + two misses, Powerball 9
        draw_day = date.fromisoformat(first["next_draw"])
        drawn = Draw("powerball", draw_day, (5, 17, 33, 61, 62), 9, POWERBALL.era_index(draw_day))
        with self.engine.begin() as conn:
            self.assertEqual(grade(conn, POWERBALL, [drawn]), 2)
            mine = conn.execute(text("SELECT matched_white, matched_special, prize FROM lottery_picks "
                                     "WHERE owner = :o"), {"o": DEFAULT_REVIEWER_ID}).first()
        self.assertEqual(tuple(mine), (3, True, 100))

    def test_source_outage_does_not_break_the_page(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO lottery_picks (pick_id, game_key, draw_date, owner, source, whites, "
                              "special, created_at) VALUES ('p1', 'powerball', :d, 'model', 'model', "
                              "'01 02 03 04 05', 6, CURRENT_TIMESTAMP)"), {"d": (next_draw(POWERBALL) - timedelta(days=7)).isoformat()})
        with patch("lottery.sources.fetch_draws", side_effect=RuntimeError("portal down")):
            page = self.client.get("/api/lottery/powerball").json()
        self.assertEqual(page["status"]["fetch_error"], "portal down")


if __name__ == "__main__":
    unittest.main()
