"""Fixes for the Copilot review of PR #2: results survive re-syncs, settlement grades
every scored game, dates stay on their day, stand-in start times are not trusted."""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from sqlalchemy import text

from backend.model_mirror import Game, OVERRODE
from data.prediction_time import enrich_prediction_times
from scripts.db_utils import create_database_engine, insert_prediction
from scripts.settle_live_predictions import settle

BASE = {"sport": "MLB", "league": "MLB", "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "win_probability": 0.6, "confidence_level": "MEDIUM",
        "model_name": "test"}


class _Db(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_database_engine(f"sqlite:///{os.path.join(self._tmp.name, 'r.db')}")
        self._env = patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "false"})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self.engine.dispose()
        self._tmp.cleanup()

    def row(self, pid: int) -> dict:
        with self.engine.connect() as conn:
            return dict(conn.execute(text("SELECT * FROM predictions WHERE prediction_id = :p"), {"p": pid}).mappings().first())


class TestResyncKeepsResults(_Db):
    def test_schedule_resync_does_not_erase_a_settled_game(self) -> None:
        pid = insert_prediction(self.engine, {**BASE, "provider_game_id": "g1", "game_date": "2026-09-20",
                                              "prediction_status": "SETTLED", "actual_home_score": 5,
                                              "actual_away_score": 3, "actual_winner": "NYY", "correct": 1})
        again = insert_prediction(self.engine, {**BASE, "provider_game_id": "g1", "game_date": "2026-09-20",
                                                "predicted_winner": "BOS", "prediction_status": "UPCOMING",
                                                "actual_home_score": None, "actual_away_score": None,
                                                "actual_winner": None})
        self.assertEqual(pid, again)
        r = self.row(pid)
        self.assertEqual((r["actual_home_score"], r["actual_away_score"], r["actual_winner"]), (5, 3, "NYY"))
        self.assertEqual(r["prediction_status"], "SETTLED")
        self.assertEqual(r["predicted_winner"], "NYY", "the recorded pick is history too")

    def test_unsettled_game_still_updates(self) -> None:
        pid = insert_prediction(self.engine, {**BASE, "provider_game_id": "g2", "game_date": "2026-09-28"})
        insert_prediction(self.engine, {**BASE, "provider_game_id": "g2", "game_date": "2026-09-28",
                                        "predicted_winner": "BOS", "win_probability": 0.55})
        self.assertEqual(self.row(pid)["predicted_winner"], "BOS")


class TestSettlement(_Db):
    def _scored(self, gid: str, home: int, away: int, status: str = "UPCOMING") -> int:
        pid = insert_prediction(self.engine, {**BASE, "provider_game_id": gid, "game_date": "2026-09-20"})
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE predictions SET actual_home_score = :h, actual_away_score = :a, "
                              "prediction_status = :s WHERE prediction_id = :p"),
                         {"h": home, "a": away, "s": status, "p": pid})
        return pid

    def test_winner_is_derived_and_graded(self) -> None:
        home_win, away_win, draw = self._scored("h", 4, 2), self._scored("a", 1, 6), self._scored("d", 2, 2)
        legacy = self._scored("f", 7, 1, status="FINAL")
        settle(self.engine, void_after_days=3)
        self.assertEqual((self.row(home_win)["actual_winner"], self.row(home_win)["correct"]), ("NYY", 1))
        self.assertEqual((self.row(away_win)["actual_winner"], self.row(away_win)["correct"]), ("BOS", 0))
        self.assertEqual(self.row(draw)["actual_winner"], "Draw")
        r = self.row(legacy)
        self.assertEqual((r["prediction_status"], r["correct"], r["actual_winner"]), ("SETTLED", 1, "NYY"))

    def test_void_needs_both_scores_missing(self) -> None:
        half = insert_prediction(self.engine, {**BASE, "provider_game_id": "p", "game_date": "2026-01-02"})
        empty = insert_prediction(self.engine, {**BASE, "provider_game_id": "e", "game_date": "2026-01-02"})
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE predictions SET actual_away_score = 3 WHERE prediction_id = :p"), {"p": half})
        settle(self.engine, void_after_days=3)
        self.assertNotEqual(self.row(half)["prediction_status"], "VOID")
        self.assertEqual(self.row(empty)["prediction_status"], "VOID")


class TestDatesAndStandInTimes(unittest.TestCase):
    def test_date_only_time_field_keeps_the_calendar_day(self) -> None:
        row = enrich_prediction_times({"start_time_utc": "2026-08-07", "game_date": ""})
        self.assertEqual(row["game_date_pacific"], "2026-08-07")

    def test_noon_pacific_stand_in_is_not_a_verified_start(self) -> None:
        stand_in = {"prediction_id": 1, "predicted_winner": "NYY", "actual_winner": "BOS", "home_team": "NYY",
                    "away_team": "BOS", "start_time_utc": "2026-09-20T19:00:00Z",   # 12:00 PDT
                    "reviewer_pick": "BOS", "picked_at": "2026-09-20T22:00:00"}      # 3 pm PDT
        g = Game.from_row(stand_in)
        self.assertIsNone(g.start)
        self.assertEqual(g.source, OVERRODE, "a 3 pm pick is not 'late' against an unknown start")
        real = dict(stand_in, start_time_utc="2026-09-21T02:10:00Z")                  # 7:10 pm PDT
        self.assertIsNotNone(Game.from_row(real).start)


if __name__ == "__main__":
    unittest.main()
