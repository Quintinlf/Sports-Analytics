"""Leader observability API + seed role."""
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
from scripts.db_utils import (
    DEFAULT_REVIEWER_ID,
    create_database_engine,
    ensure_default_reviewers,
    insert_prediction,
)
from data.prediction_time import pacific_today


class TestLeaderApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "leader.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        self._env = patch.dict(
            os.environ,
            {
                "ENABLE_DEMO_PREDICTIONS": "false",
                "ADMIN_API_KEY": "leader-admin-key",
            },
        )
        self._env.start()
        init_platform(self.engine)
        ensure_default_reviewers(self.engine)
        self._Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        @contextmanager
        def _test_db_session():
            db = self._Session()
            try:
                yield db
            finally:
                db.close()

        self._patches = [
            patch("backend.routes.feedback.get_db_session", _test_db_session),
            patch("backend.routes.feedback.engine", self.engine),
            patch("backend.routes.leader.get_db_session", _test_db_session),
            patch("backend.routes.leader.engine", self.engine),
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

    def test_quintin_seeded_as_leader(self) -> None:
        with self.engine.connect() as conn:
            role = conn.execute(
                text("SELECT analyst_role FROM reviewers WHERE reviewer_id = :rid"),
                {"rid": DEFAULT_REVIEWER_ID},
            ).scalar()
            timothy = conn.execute(
                text("SELECT analyst_role FROM reviewers WHERE reviewer_id = 'timothy'")
            ).scalar()
        self.assertEqual(role, "leader")
        self.assertEqual(timothy, "analyst")

    def test_health_requires_leader(self) -> None:
        denied = self.client.get("/api/leader/health")
        self.assertEqual(denied.status_code, 403)

        ok_admin = self.client.get(
            "/api/leader/health",
            headers={"X-Admin-Key": "leader-admin-key"},
        )
        self.assertEqual(ok_admin.status_code, 200)
        body = ok_admin.json()
        self.assertIn("pipelines", body)
        self.assertIn("stale_predictions", body)

        # A reviewer id is public (it appears in URLs), so it is not a credential.
        id_only = self.client.get(
            f"/api/leader/health?reviewer_id={DEFAULT_REVIEWER_ID}"
        )
        self.assertEqual(id_only.status_code, 403)
        wrong_key = self.client.get("/api/leader/health", headers={"X-Admin-Key": "guess"})
        self.assertEqual(wrong_key.status_code, 403)

    def test_reviewers_and_performance(self) -> None:
        today = pacific_today()
        insert_prediction(
            self.engine,
            {
                "sport": "MLB",
                "league": "MLB",
                "provider_game_id": "lead-1",
                "game_date": today,
                "home_team": "NYY",
                "away_team": "BOS",
                "predicted_winner": "NYY",
                "win_probability": 0.62,
                "confidence_level": "HIGH",
                "model_name": "MLB-LightGBM-v1",
                "prediction_status": "SETTLED",
                "actual_home_score": 5,
                "actual_away_score": 3,
                "actual_winner": "NYY",
                "correct": 1,
                "data_source": "test",
                "is_fallback": False,
            },
        )
        headers = {"X-Admin-Key": "leader-admin-key"}
        perf = self.client.get("/api/leader/performance", headers=headers)
        self.assertEqual(perf.status_code, 200)
        overall = perf.json()["overall"]
        self.assertGreaterEqual(overall["settled"], 1)

        rev = self.client.get("/api/leader/reviewers", headers=headers)
        self.assertEqual(rev.status_code, 200)
        names = [r["name"] for r in rev.json()["reviewers"]]
        self.assertIn("Quintin", names)
        self.assertIn("Timothy", names)

    def test_leader_page_served(self) -> None:
        res = self.client.get("/leader")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Leader Observability", res.text)


class TestLeaderEmailSkip(unittest.TestCase):
    def test_load_reviewers_skips_leader_unless_opted_in(self) -> None:
        from scripts.send_weekly_feedback_form import load_reviewers

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = create_database_engine(f"sqlite:///{os.path.join(tmp.name, 'e.db')}")
        init_platform(engine)
        ensure_default_reviewers(engine)

        # Leader with emails_enabled=False (default) should be skipped
        reviewers = load_reviewers(engine, allowlist=None, weekday=0)  # Sunday PT
        ids = {r["reviewer_id"] for r in reviewers}
        self.assertNotIn(DEFAULT_REVIEWER_ID, ids)

        # Opt-in leader should be included when day matches
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE reviewer_preferences
                    SET emails_enabled = 1, email_days = '[0,3]'
                    WHERE reviewer_id = :rid
                    """
                ),
                {"rid": DEFAULT_REVIEWER_ID},
            )
        reviewers2 = load_reviewers(engine, allowlist=None, weekday=0)
        ids2 = {r["reviewer_id"] for r in reviewers2}
        self.assertIn(DEFAULT_REVIEWER_ID, ids2)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
