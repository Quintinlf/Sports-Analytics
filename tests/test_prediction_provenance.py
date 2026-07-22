"""Provenance columns and admin-gated debug diagnostics."""
from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.routes.feedback import init_platform
from data.prediction_service import ensure_pipeline_run_log, record_pipeline_run
from scripts.db_utils import create_database_engine, insert_prediction


class TestPredictionProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "prov.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        self._env_patcher = patch.dict(
            os.environ,
            {"ENABLE_DEMO_PREDICTIONS": "true", "ADMIN_API_KEY": "test-admin"},
        )
        self._env_patcher.start()
        init_platform(self.engine)
        self._Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        @contextmanager
        def _test_db_session():
            db = self._Session()
            try:
                yield db
            finally:
                db.close()

        self._db_patcher = patch("backend.routes.feedback.get_db_session", _test_db_session)
        self._engine_patcher = patch("backend.routes.feedback.engine", self.engine)
        self._db_patcher.start()
        self._engine_patcher.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._engine_patcher.stop()
        self._db_patcher.stop()
        self._env_patcher.stop()
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_insert_persists_data_source_and_is_fallback(self) -> None:
        pid = insert_prediction(
            self.engine,
            {
                "sport": "MLB",
                "league": "MLB",
                "game_date": "2026-07-21",
                "home_team": "NYY",
                "away_team": "BOS",
                "predicted_winner": "NYY",
                "win_probability": 0.61,
                "confidence_level": "MEDIUM",
                "model_name": "MLB-LightGBM-v1",
                "data_source": "mlb_statsapi",
                "is_fallback": False,
                "prediction_status": "UPCOMING",
                "feature_snapshot": '{"data_source":"mlb_statsapi","is_fallback":false}',
                "created_at": "2026-07-21T12:00:00",
            },
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT model_name, data_source, is_fallback FROM predictions "
                    "WHERE prediction_id = :pid"
                ),
                {"pid": pid},
            ).mappings().first()
        self.assertEqual(row["model_name"], "MLB-LightGBM-v1")
        self.assertEqual(row["data_source"], "mlb_statsapi")
        self.assertFalse(bool(row["is_fallback"]))

    def test_list_predictions_includes_provenance(self) -> None:
        insert_prediction(
            self.engine,
            {
                "sport": "NBA",
                "league": "NBA",
                "game_date": "2026-07-21",
                "home_team": "LAL",
                "away_team": "BOS",
                "predicted_winner": "LAL",
                "win_probability": 0.55,
                "confidence_level": "LOW",
                "model_name": "NBA-Ensemble-v1",
                "data_source": "nba_api",
                "is_fallback": False,
                "prediction_status": "UPCOMING",
                "created_at": "2026-07-21T12:00:00",
            },
        )
        res = self.client.get("/api/feedback/predictions?sport=NBA")
        self.assertEqual(res.status_code, 200)
        rows = res.json()
        self.assertTrue(rows)
        self.assertIn("model_name", rows[0])
        self.assertIn("data_source", rows[0])
        self.assertIn("is_fallback", rows[0])
        self.assertIn("created_at", rows[0])

    def test_debug_predictions_requires_admin(self) -> None:
        denied = self.client.get("/api/feedback/debug/predictions")
        self.assertEqual(denied.status_code, 403)

        ensure_pipeline_run_log(self.engine)
        record_pipeline_run(self.engine, "MLB", "ok", predictions_count=3)
        allowed = self.client.get(
            "/api/feedback/debug/predictions",
            headers={"X-Admin-Key": "test-admin"},
        )
        self.assertEqual(allowed.status_code, 200)
        body = allowed.json()
        self.assertIn("counts_by_sport", body)
        self.assertIn("pipeline_runs", body)
        self.assertTrue(any(r.get("sport") == "MLB" for r in body["pipeline_runs"]))


if __name__ == "__main__":
    unittest.main()
