"""Tests for unified predictions and prediction_options schema (Phase 1)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime

from scripts.db_utils import (
    create_database_engine,
    ensure_unified_schema,
    get_prediction_options,
    get_predictions_by_date,
    insert_prediction,
    insert_prediction_options,
)


class TestUnifiedPredictionsDbUtils(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "test.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_schema_creation(self) -> None:
        ensure_unified_schema(self.engine)
        ensure_unified_schema(self.engine)  # idempotent

    def test_insert_prediction_and_options(self) -> None:
        ensure_unified_schema(self.engine)
        today = datetime.utcnow().date().isoformat()

        prediction_id = insert_prediction(
            self.engine,
            {
                "sport": "MLB",
                "league": "MLB",
                "game_date": today,
                "home_team": "NYY",
                "away_team": "BOS",
                "predicted_winner": "NYY Win",
                "confidence_level": "MEDIUM",
                "feature_snapshot": {"rest_diff": 1},
            },
        )
        self.assertIsInstance(prediction_id, int)
        self.assertGreater(prediction_id, 0)

        count = insert_prediction_options(
            self.engine,
            prediction_id,
            [
                {"option_name": "NYY Win", "probability": 0.62, "rank": 1},
                {"option_name": "BOS Win", "probability": 0.38, "rank": 2},
            ],
        )
        self.assertEqual(count, 2)

        options = get_prediction_options(self.engine, prediction_id)
        self.assertEqual(len(options), 2)
        self.assertEqual(options[0]["option_name"], "NYY Win")
        self.assertAlmostEqual(options[0]["probability"], 0.62)

    def test_get_predictions_by_date_and_sport(self) -> None:
        ensure_unified_schema(self.engine)
        today = datetime.utcnow().date().isoformat()

        insert_prediction(
            self.engine,
            {
                "sport": "MLB",
                "league": "MLB",
                "game_date": today,
                "home_team": "NYY",
                "away_team": "BOS",
                "predicted_winner": "NYY Win",
                "confidence_level": "MEDIUM",
                "feature_snapshot": None,
            },
        )
        insert_prediction(
            self.engine,
            {
                "sport": "NBA",
                "league": "NBA",
                "game_date": today,
                "home_team": "NYK",
                "away_team": "SAS",
                "predicted_winner": "NYK Win",
                "confidence_level": "HIGH",
                "feature_snapshot": None,
            },
        )

        all_preds = get_predictions_by_date(self.engine, today, today)
        self.assertEqual(len(all_preds), 2)

        mlb_preds = get_predictions_by_date(self.engine, today, today, sport="MLB")
        self.assertEqual(len(mlb_preds), 1)
        self.assertEqual(mlb_preds[0]["sport"], "MLB")
        self.assertEqual(mlb_preds[0]["home_team"], "NYY")

        snapshot = json.loads(mlb_preds[0]["feature_snapshot"]) if mlb_preds[0]["feature_snapshot"] else {}
        self.assertEqual(snapshot, {})


class TestUnifiedPredictionsDatabaseHandler(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "test_handler.db")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_handler_unified_methods_on_legacy_schema(self) -> None:
        from data.database.database_handler import SportsAnalyticsDB

        db = SportsAnalyticsDB(self.db_path)
        today = datetime.utcnow().date().isoformat()

        prediction_id = db.insert_unified_prediction(
            {
                "sport": "MLB",
                "league": "MLB",
                "game_date": today,
                "home_team": "LAD",
                "away_team": "SFG",
                "predicted_winner": "LAD Win",
                "confidence_level": "HIGH",
                "feature_snapshot": {"era_diff": 0.5},
                "win_probability": 0.67,
                "predicted_spread": 1.1,
            }
        )
        db.insert_prediction_options(
            prediction_id,
            [
                {"option_name": "LAD Win", "probability": 0.67, "rank": 1},
                {"option_name": "SFG Win", "probability": 0.33, "rank": 2},
            ],
        )

        rows = db.get_unified_predictions_by_date(today, today, sport="MLB")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["predicted_winner"], "LAD Win")

        options = db.get_prediction_options(prediction_id)
        self.assertEqual(len(options), 2)
        db.close()


class TestRunDailyPredictionsDualWrite(unittest.TestCase):
    def test_dual_write_mlb_and_unified(self) -> None:
        from sqlalchemy import text

        from scripts.run_daily_predictions import (
            _build_prediction_options,
            _fetch_mlb_predictions,
            _insert_mlb_predictions,
            _insert_unified_predictions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "daily.db")
            engine = create_database_engine(f"sqlite:///{db_path}")

            from scripts.run_daily_predictions import _ensure_mlb_predictions_table

            try:
                _ensure_mlb_predictions_table(engine)
                rows = _fetch_mlb_predictions()

                mlb_count = _insert_mlb_predictions(engine, rows)
                unified_count = _insert_unified_predictions(engine, rows)

                self.assertEqual(mlb_count, 1)
                self.assertEqual(unified_count, 1)

                with engine.begin() as conn:
                    mlb_rows = conn.execute(text("SELECT COUNT(*) FROM mlb_predictions")).scalar()
                    pred_rows = conn.execute(
                        text("SELECT COUNT(*) FROM predictions WHERE sport = 'MLB'")
                    ).scalar()
                    opt_rows = conn.execute(text("SELECT COUNT(*) FROM prediction_options")).scalar()

                self.assertEqual(mlb_rows, 1)
                self.assertEqual(pred_rows, 1)
                self.assertEqual(opt_rows, 2)

                options = _build_prediction_options(rows[0])
                self.assertEqual(len(options), 2)
                self.assertAlmostEqual(sum(o["probability"] for o in options), 1.0, places=5)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
