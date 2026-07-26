"""Phase 1: trusted analyst profiles and onboarding question storage."""
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
    TRUSTED_ANALYSTS,
    _backfill_reviewer_names,
    _ensure_reviewer_profile_columns,
    _split_display_name,
    create_database_engine,
    ensure_default_reviewers,
)


class TestAnalystPhase1(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "phase1.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        # These tests exercise review/case-study flows against *some* existing
        # prediction row, not real model output — opt into the demo seed data
        # (off by default in production) so init_platform() populates one.
        with patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "true"}):
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
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_reviewer_profile_migration_idempotent(self) -> None:
        with self.engine.begin() as conn:
            cols = _ensure_reviewer_profile_columns(conn, self.engine)
            _ensure_reviewer_profile_columns(conn, self.engine)
        for col in ("first_name", "last_name", "bio", "analyst_role", "profile_public", "onboarding_completed_at"):
            self.assertIn(col, cols)

    def test_name_backfill(self) -> None:
        ts = "2026-01-01T00:00:00"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO reviewers (reviewer_id, name, email, first_name, last_name, analyst_role, profile_public, created_at)
                    VALUES ('test-user', 'Jane Doe', NULL, NULL, NULL, 'analyst', :profile_public, :ts)
                    """
                ),
                {"ts": ts, "profile_public": False},
            )
            _backfill_reviewer_names(conn)
            row = conn.execute(
                text("SELECT first_name, last_name FROM reviewers WHERE reviewer_id = 'test-user'")
            ).mappings().first()
        self.assertEqual(row["first_name"], "Jane")
        self.assertEqual(row["last_name"], "Doe")

    def test_split_display_name(self) -> None:
        self.assertEqual(_split_display_name("Lamar"), ("Lamar", ""))
        self.assertEqual(_split_display_name("Jane Doe"), ("Jane", "Doe"))

    def test_trusted_analyst_seeds(self) -> None:
        with patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "true"}):
            ensure_default_reviewers(self.engine)
        with self.engine.connect() as conn:
            for analyst in TRUSTED_ANALYSTS:
                row = conn.execute(
                    text(
                        """
                        SELECT reviewer_id, analyst_role, profile_public, first_name
                        FROM reviewers WHERE reviewer_id = :rid
                        """
                    ),
                    {"rid": analyst["reviewer_id"]},
                ).mappings().first()
                self.assertIsNotNone(row, msg=f"missing seed {analyst['reviewer_id']}")
                self.assertEqual(row["analyst_role"], "trusted_analyst")
                self.assertTrue(bool(row["profile_public"]))
                self.assertEqual(row["first_name"], analyst["first_name"])

    def test_trusted_analysts_not_seeded_without_demo_flag(self) -> None:
        import tempfile
        from scripts.db_utils import create_database_engine

        with tempfile.TemporaryDirectory() as tmp:
            engine = create_database_engine(f"sqlite:///{tmp}/no_demo.db")
            # Create reviewers table via ensure with demo off.
            with patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": ""}, clear=False):
                # Remove the key if present
                os.environ.pop("ENABLE_DEMO_PREDICTIONS", None)
                init_platform(engine)
                ensure_default_reviewers(engine)
            with engine.connect() as conn:
                for analyst in TRUSTED_ANALYSTS:
                    row = conn.execute(
                        text("SELECT reviewer_id FROM reviewers WHERE reviewer_id = :rid"),
                        {"rid": analyst["reviewer_id"]},
                    ).first()
                    self.assertIsNone(row, msg=f"unexpected seed {analyst['reviewer_id']}")
            engine.dispose()

    def test_onboarding_questions_seeded(self) -> None:
        with self.engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM analyst_questions WHERE context = 'onboarding'")
            ).scalar()
        self.assertGreaterEqual(count, 2)

    def test_post_reviewers_returns_profile_fields(self) -> None:
        res = self.client.post("/api/feedback/reviewers", json={"name": "Test Analyst"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("first_name", data)
        self.assertIn("last_name", data)
        self.assertIn("display_name", data)
        self.assertIn("analyst_role", data)
        self.assertEqual(data["first_name"], "Test")
        self.assertEqual(data["last_name"], "Analyst")

    def test_public_analysts_list(self) -> None:
        res = self.client.get("/api/feedback/analysts")
        self.assertEqual(res.status_code, 200)
        ids = {a["reviewer_id"] for a in res.json()}
        self.assertIn("lamar", ids)
        self.assertIn("melissa", ids)
        self.assertIn("alex", ids)

    def test_public_profile_hides_email(self) -> None:
        res = self.client.get("/api/feedback/analysts/lamar/profile")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertNotIn("email", data)
        self.assertEqual(data["reviewer_id"], "lamar")
        self.assertIn("stats", data)

    def test_onboarding_answers_and_completion(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Onboard User"})
        rid = create.json()["reviewer_id"]

        questions = self.client.get("/api/feedback/onboarding/questions").json()
        self.assertTrue(len(questions) >= 2)

        status_before = self.client.get(f"/api/feedback/onboarding/status?reviewer_id={rid}").json()
        self.assertFalse(status_before["completed"])

        answers = [{"question_id": q["question_id"], "answer": f"Answer for {q['title']}"} for q in questions]
        submit = self.client.post(
            "/api/feedback/onboarding/answers",
            json={"reviewer_id": rid, "answers": answers},
        )
        self.assertEqual(submit.status_code, 200)
        self.assertTrue(submit.json()["completed"])

        status_after = self.client.get(f"/api/feedback/onboarding/status?reviewer_id={rid}").json()
        self.assertTrue(status_after["completed"])

        with self.engine.connect() as conn:
            stored = conn.execute(
                text("SELECT COUNT(*) FROM analyst_answers WHERE reviewer_id = :rid"),
                {"rid": rid},
            ).scalar()
        self.assertEqual(stored, len(questions))

    def test_onboarding_questions_count_four(self) -> None:
        with self.engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM analyst_questions WHERE context = 'onboarding'")
            ).scalar()
        self.assertGreaterEqual(count, 4)

    def test_trusted_analyst_preferences_seeded_as_booleans(self) -> None:
        with patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "true"}):
            ensure_default_reviewers(self.engine)
        with self.engine.connect() as conn:
            prefs = conn.execute(
                text(
                    """
                    SELECT emails_enabled, wants_betting_section, wants_explanations,
                           wants_postgame_reviews
                    FROM reviewer_preferences
                    WHERE reviewer_id = 'lamar'
                    """
                )
            ).mappings().first()
            profile = conn.execute(
                text("SELECT profile_public FROM reviewers WHERE reviewer_id = 'lamar'")
            ).mappings().first()
        self.assertIsNotNone(prefs)
        self.assertIsNotNone(profile)
        for col in (
            "emails_enabled",
            "wants_betting_section",
            "wants_explanations",
            "wants_postgame_reviews",
        ):
            self.assertTrue(bool(prefs[col]), msg=f"{col} should be truthy for trusted analyst")
        self.assertTrue(bool(profile["profile_public"]))

    def test_research_featured_question(self) -> None:
        res = self.client.get("/api/feedback/research/current")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["question_id"], "research-nash-equilibrium")
        self.assertEqual(
            data["title"],
            "Can modeling opponents' decisions improve sports prediction?",
        )
        self.assertEqual(data["knowledge_area"], "Game Theory")
        self.assertIn("Nash equilibrium", data["body_markdown"])
        self.assertIn("prompts", data)

    def test_onboarding_sports_expertise_prompt(self) -> None:
        questions = self.client.get("/api/feedback/onboarding/questions").json()
        sports_q = next(q for q in questions if q["question_id"] == "onboard-sports")
        prompt = sports_q["prompts"][0]["prompt"]
        self.assertIn("sports/leagues", prompt.lower())
        self.assertIn("confidently", prompt.lower())

    def test_knowledge_areas_includes_ux(self) -> None:
        res = self.client.get("/api/feedback/knowledge-areas")
        self.assertEqual(res.status_code, 200)
        self.assertIn("UX", res.json()["areas"])

    def test_question_upsert_preserves_answers_and_completion(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Preserve User"})
        rid = create.json()["reviewer_id"]
        questions = self.client.get("/api/feedback/onboarding/questions").json()
        answers = [{"question_id": q["question_id"], "answer": f"Saved: {q['title']}"} for q in questions]
        self.client.post(
            "/api/feedback/onboarding/answers",
            json={"reviewer_id": rid, "answers": answers},
        )
        with self.engine.connect() as conn:
            before_count = conn.execute(
                text("SELECT COUNT(*) FROM analyst_answers WHERE reviewer_id = :rid"),
                {"rid": rid},
            ).scalar()
            completed_before = conn.execute(
                text("SELECT onboarding_completed_at FROM reviewers WHERE reviewer_id = :rid"),
                {"rid": rid},
            ).scalar()
        init_platform(self.engine)
        with self.engine.connect() as conn:
            after_count = conn.execute(
                text("SELECT COUNT(*) FROM analyst_answers WHERE reviewer_id = :rid"),
                {"rid": rid},
            ).scalar()
            completed_after = conn.execute(
                text("SELECT onboarding_completed_at FROM reviewers WHERE reviewer_id = :rid"),
                {"rid": rid},
            ).scalar()
            title = conn.execute(
                text("SELECT title FROM analyst_questions WHERE question_id = 'onboard-sports'")
            ).scalar()
        self.assertEqual(before_count, after_count)
        self.assertEqual(completed_before, completed_after)
        self.assertEqual(title, "Sports expertise")

    def test_prediction_per_sport_limits(self) -> None:
        res = self.client.get("/api/feedback/predictions")
        self.assertEqual(res.status_code, 200)
        preds = res.json()
        by_sport: dict[str, int] = {}
        for p in preds:
            sport = p.get("sport_ui") or p.get("sport")
            by_sport[sport] = by_sport.get(sport, 0) + 1
        for sport, count in by_sport.items():
            self.assertLessEqual(count, 6, msg=f"{sport} exceeded default limit")

    def test_prediction_limit_env_override(self) -> None:
        with patch.dict(os.environ, {"DASHBOARD_PRED_LIMIT_MLB": "2"}):
            from backend.routes import feedback as fb

            limits = fb._dashboard_pred_limits()
            self.assertEqual(limits["MLB"], 2)

    def test_prediction_priority_prefers_high_confidence(self) -> None:
        import json

        from backend.routes.feedback import _fetch_predictions_for_sport

        ts = "2026-07-08T12:00:00"
        low_snap = json.dumps({"explanations": []})
        high_snap = json.dumps({
            "explanations": [{"label": "ERA", "weight": 0.5, "value": "2.10"}],
        })
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO predictions
                        (sport, league, game_date, home_team, away_team, predicted_winner,
                         confidence_level, prediction_status, feature_snapshot, created_at)
                    VALUES
                        ('MLB', 'AL', '2099-01-01', 'Low Team', 'Low Opp', 'Low Team',
                         'LOW', 'UPCOMING', :low_snap, :ts),
                        ('MLB', 'AL', '2099-01-02', 'High Team', 'High Opp', 'High Team',
                         'HIGH', 'UPCOMING', :high_snap, :ts)
                    """
                ),
                {"ts": ts, "low_snap": low_snap, "high_snap": high_snap},
            )
        with self._Session() as session:
            rows = _fetch_predictions_for_sport(session, "MLB", 50)
        teams = [r["home_team"] for r in rows if r["home_team"] in ("High Team", "Low Team")]
        self.assertEqual(len(teams), 2)
        self.assertEqual(teams[0], "High Team")

    def test_research_answers_submit_and_list(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Research User"})
        rid = create.json()["reviewer_id"]
        q = self.client.get("/api/feedback/research/current").json()
        submit = self.client.post(
            "/api/feedback/research/answers",
            json={
                "reviewer_id": rid,
                "answers": [{
                    "question_id": q["question_id"],
                    "answer": "Game theory could model bullpen usage as sequential games.",
                    "knowledge_area": "Game Theory",
                }],
            },
        )
        self.assertEqual(submit.status_code, 200)
        listed = self.client.get("/api/feedback/research/answers?knowledge_area=Game%20Theory")
        self.assertTrue(any(a["reviewer_id"] == rid for a in listed.json()))

    def test_admin_create_question_requires_key(self) -> None:
        res = self.client.post(
            "/api/feedback/admin/questions",
            json={"title": "Test Q", "body_markdown": "Body", "prompts": ["Why?"]},
        )
        self.assertEqual(res.status_code, 403)

    def test_admin_create_question_with_key(self) -> None:
        with patch.dict(os.environ, {"ADMIN_API_KEY": "test-admin-key"}):
            res = self.client.post(
                "/api/feedback/admin/questions",
                headers={"X-Admin-Key": "test-admin-key"},
                json={
                    "title": "Bullpen fatigue study",
                    "body_markdown": "How do you measure bullpen workload?",
                    "prompts": ["3-day IP", "High-leverage outs"],
                    "knowledge_area": "Bullpens",
                    "featured": False,
                },
            )
        self.assertEqual(res.status_code, 200)
        self.assertIn("question_id", res.json())

    def test_decision_variables_endpoint(self) -> None:
        res = self.client.get("/api/feedback/decision-variables")
        self.assertEqual(res.status_code, 200)
        vars_ = res.json()["variables"]
        self.assertIn("starting_pitcher", vars_)
        self.assertIn("bullpen", vars_)

    def test_primary_decision_variable_on_pregame(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Pregame User"})
        rid = create.json()["reviewer_id"]
        with self.engine.connect() as conn:
            pred = conn.execute(
                text(
                    """
                    SELECT prediction_id, predicted_winner, home_team, away_team
                    FROM predictions
                    WHERE prediction_status = 'UPCOMING'
                    LIMIT 1
                    """
                )
            ).mappings().first()
        self.assertIsNotNone(pred)
        wrong_pick = pred["away_team"] if pred["predicted_winner"] == pred["home_team"] else pred["home_team"]
        res = self.client.post(
            "/api/feedback/prediction-reviews",
            json={
                "prediction_id": pred["prediction_id"],
                "reviewer_id": rid,
                "reviewer_pick": wrong_pick,
                "reviewer_confidence": 4,
                "agree_with_model": False,
                "primary_decision_variable": "bullpen",
            },
        )
        self.assertEqual(res.status_code, 200)
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT primary_decision_variable FROM prediction_reviews
                    WHERE reviewer_id = :rid AND prediction_id = :pid
                    """
                ),
                {"rid": rid, "pid": pred["prediction_id"]},
            ).mappings().first()
        self.assertEqual(row["primary_decision_variable"], "bullpen")

    def _beat_ai_review(self, reviewer_id: str) -> str:
        with self.engine.connect() as conn:
            pred = conn.execute(
                text(
                    """
                    SELECT prediction_id, predicted_winner, actual_winner
                    FROM predictions
                    WHERE actual_winner IS NOT NULL AND correct = 0
                    LIMIT 1
                    """
                )
            ).mappings().first()
        self.assertIsNotNone(pred)
        pregame = self.client.post(
            "/api/feedback/prediction-reviews",
            json={
                "prediction_id": pred["prediction_id"],
                "reviewer_id": reviewer_id,
                "reviewer_pick": pred["actual_winner"],
                "reviewer_confidence": 5,
                "agree_with_model": False,
            },
        )
        self.assertEqual(pregame.status_code, 200)
        review_id = pregame.json()["review_id"]
        outcome = self.client.post(
            "/api/feedback/review-outcomes",
            json={
                "review_id": review_id,
                "followup_reason": "Bullpen usage and matchup platoon splits the model ignored.",
            },
        )
        self.assertEqual(outcome.status_code, 200)
        self.assertTrue(outcome.json()["reviewer_beat_model"])
        self.assertTrue(outcome.json()["successful_analyst_override"])
        return review_id

    def test_case_study_flow(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Case Study User"})
        rid = create.json()["reviewer_id"]
        review_id = self._beat_ai_review(rid)

        pending = self.client.get(f"/api/feedback/case-studies/pending?reviewer_id={rid}")
        self.assertEqual(pending.status_code, 200)
        self.assertEqual(len(pending.json()), 1)
        self.assertEqual(pending.json()[0]["review_id"], review_id)

        submit = self.client.post(
            "/api/feedback/case-studies",
            json={
                "review_id": review_id,
                "reviewer_id": rid,
                "ai_missed": "Bullpen fatigue",
                "decision_factors": "3-day workload",
                "missing_variables": "Reliever IP last 3 days",
                "data_sources": "Baseball Savant",
                "confidence_rating": 4,
            },
        )
        self.assertEqual(submit.status_code, 200)

        pending_after = self.client.get(f"/api/feedback/case-studies/pending?reviewer_id={rid}")
        self.assertEqual(len(pending_after.json()), 0)

        profile = self.client.get(f"/api/feedback/analysts/{rid}/profile")
        self.assertEqual(profile.status_code, 404)  # not public by default

    def test_comments_on_research_question(self) -> None:
        create = self.client.post("/api/feedback/reviewers", json={"name": "Comment User"})
        rid = create.json()["reviewer_id"]
        qid = self.client.get("/api/feedback/research/current").json()["question_id"]
        post = self.client.post(
            "/api/feedback/comments",
            json={
                "reviewer_id": rid,
                "target_type": "research_question",
                "target_id": qid,
                "body": "Nash equilibrium applies to late-inning bullpen decisions.",
            },
        )
        self.assertEqual(post.status_code, 200)
        listed = self.client.get(
            f"/api/feedback/comments?target_type=research_question&target_id={qid}"
        )
        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any("Nash equilibrium" in c["body"] for c in listed.json()))

    def test_mlb_context_module(self) -> None:
        from data.mlb_context import build_mlb_context

        ctx = build_mlb_context({
            "home_probable_pitcher": "Test Pitcher",
            "away_probable_pitcher": "Other Pitcher",
        })
        self.assertIn("starting_pitchers", ctx)
        self.assertIn("missing_data_warnings", ctx)
        self.assertEqual(ctx["starting_pitchers"]["home"]["name"], "Test Pitcher")


if __name__ == "__main__":
    unittest.main()
