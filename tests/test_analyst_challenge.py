"""Analyst challenge tracking — successful override + follow-up reasoning."""
from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from backend.analyst_challenge import (
    OVERRIDE_FOLLOWUP_PROMPT,
    compose_analyst_reasoning,
    evaluate_challenge,
)
from backend.main import app
from backend.routes.feedback import init_platform
from scripts.db_utils import create_database_engine


class TestChallengeHelpers(unittest.TestCase):
    def test_compose_reasoning(self) -> None:
        text_out = compose_analyst_reasoning(
            ["bullpen", "injury"],
            "Saw heavy usage last 3 days",
        )
        self.assertIn("Saw heavy usage", text_out or "")
        self.assertIn("bullpen", text_out or "")

    def test_successful_override(self) -> None:
        result = evaluate_challenge(
            agree_with_model=False,
            reviewer_pick="Padres",
            predicted_winner="Dodgers",
            actual_winner="Padres",
            model_correct_flag=0,
            analyst_reasoning="Bullpen edge",
        )
        self.assertTrue(result["analyst_disagreed"])
        self.assertTrue(result["analyst_was_correct"])
        self.assertFalse(result["ai_was_correct"])
        self.assertTrue(result["successful_analyst_override"])
        self.assertEqual(result["final_result"], "Padres")
        self.assertEqual(result["override_followup_prompt"], OVERRIDE_FOLLOWUP_PROMPT)

    def test_agree_or_wrong_is_not_override(self) -> None:
        agree = evaluate_challenge(
            agree_with_model=True,
            reviewer_pick="Dodgers",
            predicted_winner="Dodgers",
            actual_winner="Dodgers",
            model_correct_flag=1,
        )
        self.assertFalse(agree["successful_analyst_override"])

        both_wrong = evaluate_challenge(
            agree_with_model=False,
            reviewer_pick="Giants",
            predicted_winner="Dodgers",
            actual_winner="Padres",
            model_correct_flag=0,
        )
        self.assertFalse(both_wrong["successful_analyst_override"])


class TestChallengeApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "challenge.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        with patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "true"}):
            init_platform(self.engine)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO predictions
                        (sport, league, game_date, home_team, away_team,
                         predicted_winner, confidence_level, prediction_status,
                         actual_home_score, actual_away_score, actual_winner, correct,
                         feature_snapshot, created_at)
                    VALUES
                        ('MLB', 'NL', '2099-06-01', 'Padres', 'Dodgers',
                         'Dodgers', 'HIGH', 'FINAL',
                         5, 2, 'Padres', 0,
                         '{}', '2099-06-01T12:00:00')
                    """
                )
            )
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
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_override_requires_followup_and_persists_final_result(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Override Analyst"})
        self.assertEqual(create.status_code, 200)
        rid = create.json()["reviewer_id"]

        with self.engine.connect() as conn:
            pred = conn.execute(
                text(
                    "SELECT prediction_id, actual_winner FROM predictions "
                    "WHERE actual_winner = 'Padres' LIMIT 1"
                )
            ).mappings().first()
        self.assertIsNotNone(pred)

        pregame = self.client.post(
            "/api/feedback/prediction-reviews",
            json={
                "prediction_id": pred["prediction_id"],
                "reviewer_id": rid,
                "reviewer_pick": "Padres",
                "reviewer_confidence": 4,
                "agree_with_model": False,
                "pregame_notes": "Padres bullpen fresher",
                "missing_factors": ["bullpen"],
            },
        )
        self.assertEqual(pregame.status_code, 200)
        body = pregame.json()
        self.assertTrue(body["analyst_disagreed"])
        self.assertIn("bullpen", body["analyst_reasoning"] or "")
        review_id = body["review_id"]

        missing = self.client.post(
            "/api/feedback/review-outcomes",
            json={"review_id": review_id},
        )
        self.assertEqual(missing.status_code, 400)
        self.assertIn("model missed", missing.json()["detail"].lower())

        ok = self.client.post(
            "/api/feedback/review-outcomes",
            json={
                "review_id": review_id,
                "followup_reason": "Model missed bullpen workload over the last three days.",
                "followup_missing_factors": ["bullpen"],
            },
        )
        self.assertEqual(ok.status_code, 200)
        data = ok.json()
        self.assertTrue(data["successful_analyst_override"])
        self.assertTrue(data["analyst_was_correct"])
        self.assertFalse(data["ai_was_correct"])
        self.assertEqual(data["final_result"], "Padres")
        self.assertEqual(data["override_followup_prompt"], OVERRIDE_FOLLOWUP_PROMPT)

        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT reviewer_beat_model, final_result, followup_reason "
                    "FROM review_outcomes WHERE review_id = :rid"
                ),
                {"rid": review_id},
            ).mappings().first()
        self.assertTrue(row["reviewer_beat_model"])
        self.assertEqual(row["final_result"], "Padres")
        self.assertIn("bullpen", row["followup_reason"].lower())


if __name__ == "__main__":
    unittest.main()
