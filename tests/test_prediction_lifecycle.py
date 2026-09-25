"""Phase 0: identity, Pacific display dates, settlement, upcoming filters."""
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
from backend.routes.feedback import init_platform
from data.prediction_time import (
    PRED_SETTLED,
    PRED_UPCOMING,
    PRED_VOID,
    enrich_prediction_times,
    pacific_today,
)
from scripts.db_utils import (
    _compute_game_signature,
    create_database_engine,
    insert_prediction,
)
from scripts.settle_live_predictions import settle


class TestPredictionTime(unittest.TestCase):
    def test_enrich_uses_utc_start_and_pacific_display(self) -> None:
        # 2026-08-07 05:00 UTC → 2026-08-06 evening Pacific
        row = enrich_prediction_times(
            {
                "sport": "MLB",
                "home_team": "NYY",
                "away_team": "BOS",
                "start_time_utc": "2026-08-07T05:00:00Z",
            }
        )
        self.assertTrue(str(row["start_time_utc"]).startswith("2026-08-07T05:00:00"))
        self.assertEqual(row["game_date_pacific"], "2026-08-06")
        self.assertEqual(row["game_date"], "2026-08-06")

    def test_date_only_game_date_preserved_as_pacific(self) -> None:
        row = enrich_prediction_times(
            {
                "sport": "MLB",
                "home_team": "NYY",
                "away_team": "BOS",
                "game_date": "2026-08-07",
            }
        )
        self.assertEqual(row["game_date"], "2026-08-07")
        self.assertEqual(row["game_date_pacific"], "2026-08-07")
        self.assertTrue(row.get("start_time_utc"))

    def test_signature_prefers_provider_id_not_calendar_date(self) -> None:
        a = _compute_game_signature(
            {
                "sport": "MLB",
                "provider_game_id": "12345",
                "game_date": "2026-08-06",
                "home_team": "NYY",
                "away_team": "BOS",
            }
        )
        b = _compute_game_signature(
            {
                "sport": "MLB",
                "provider_game_id": "12345",
                "game_date": "2026-08-07",  # different display date
                "home_team": "NYY",
                "away_team": "BOS",
            }
        )
        self.assertEqual(a, b)


class TestLifecycleIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "life.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        # Force schema auto-migrate for sqlite
        os.environ.pop("SCHEMA_AUTO_MIGRATE", None)
        self._env = patch.dict(
            os.environ,
            {"ENABLE_DEMO_PREDICTIONS": "false", "ADMIN_API_KEY": "test-admin"},
        )
        self._env.start()
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
        self._env.stop()
        self.engine.dispose()
        self._tmpdir.cleanup()

    def _insert(
        self,
        *,
        sport: str,
        provider: str,
        start_utc: str,
        status: str = PRED_UPCOMING,
        scores=None,
        winner: str = "Home",
    ) -> int:
        home, away = "Home", "Away"
        data = {
            "sport": sport,
            "league": sport,
            "provider_game_id": provider,
            "start_time_utc": start_utc,
            "home_team": home,
            "away_team": away,
            "predicted_winner": winner,
            "win_probability": 0.6,
            "confidence_level": "MEDIUM",
            "model_name": f"{sport}-Test",
            "data_source": "test",
            "is_fallback": False,
            "prediction_status": status,
            "created_at": datetime.utcnow().isoformat(),
        }
        if scores:
            data["actual_home_score"], data["actual_away_score"] = scores
            data["actual_winner"] = home if scores[0] > scores[1] else away
        return insert_prediction(self.engine, data)

    def test_insert_persists_lifecycle_columns(self) -> None:
        pid = self._insert(
            sport="MLB",
            provider="mlb-1",
            start_utc="2026-08-07T02:00:00Z",
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT start_time_utc, game_date_pacific, game_status, "
                    "prediction_status, provider_game_id FROM predictions "
                    "WHERE prediction_id = :pid"
                ),
                {"pid": pid},
            ).mappings().one()
        self.assertEqual(row["provider_game_id"], "mlb-1")
        self.assertEqual(str(row["game_date_pacific"]), "2026-08-06")
        self.assertEqual(row["game_status"], "SCHEDULED")
        self.assertEqual(row["prediction_status"], PRED_UPCOMING)

    def test_upsert_by_provider_ignores_display_date_drift(self) -> None:
        pid1 = self._insert(
            sport="MLB", provider="mlb-dup", start_utc="2026-08-07T02:00:00Z"
        )
        pid2 = self._insert(
            sport="MLB", provider="mlb-dup", start_utc="2026-08-07T05:00:00Z"
        )
        self.assertEqual(pid1, pid2)
        with self.engine.connect() as conn:
            n = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE provider_game_id = 'mlb-dup'")
            ).scalar()
        self.assertEqual(int(n), 1)

    def test_settle_marks_scored_and_voids_stale(self) -> None:
        today = pacific_today()
        # Upcoming today
        self._insert(
            sport="MLB",
            provider="mlb-today",
            start_utc=f"{today}T20:00:00Z",
        )
        # Scored → should SETTLED
        self._insert(
            sport="MLB",
            provider="mlb-scored",
            start_utc=f"{today}T20:00:00Z",
            scores=(5, 3),
        )
        # Stale upcoming → VOID
        stale_day = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        self._insert(
            sport="MLB",
            provider="mlb-stale",
            start_utc=f"{stale_day}T20:00:00Z",
        )

        stats = settle(self.engine, void_after_days=3, dry_run=False)
        self.assertGreaterEqual(stats["settled"], 1)
        self.assertGreaterEqual(stats["voided"], 1)

        with self.engine.connect() as conn:
            scored = conn.execute(
                text(
                    "SELECT prediction_status, game_status FROM predictions "
                    "WHERE provider_game_id = 'mlb-scored'"
                )
            ).mappings().one()
            stale = conn.execute(
                text(
                    "SELECT prediction_status FROM predictions "
                    "WHERE provider_game_id = 'mlb-stale'"
                )
            ).mappings().one()
        self.assertEqual(scored["prediction_status"], PRED_SETTLED)
        self.assertEqual(scored["game_status"], "FINAL")
        self.assertEqual(stale["prediction_status"], PRED_VOID)

    def test_dashboard_excludes_settled_and_stale(self) -> None:
        today = pacific_today()
        self._insert(
            sport="MLB",
            provider="mlb-live",
            start_utc=f"{today}T23:00:00Z",
        )
        self._insert(
            sport="MLB",
            provider="mlb-done",
            start_utc=f"{today}T20:00:00Z",
            scores=(4, 1),
        )
        settle(self.engine, void_after_days=3, dry_run=False)

        resp = self.client.get("/api/feedback/predictions?sport=MLB")
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        ids = {r.get("provider_game_id") for r in rows}
        self.assertIn("mlb-live", ids)
        self.assertNotIn("mlb-done", ids)
        for r in rows:
            self.assertFalse(r.get("settled"))
            self.assertNotIn(str(r.get("prediction_status", "")).upper(), {"SETTLED", "VOID"})


class TestModelArtifacts(unittest.TestCase):
    def test_check_model_artifacts_mlb_fifa_present(self) -> None:
        from scripts.check_model_artifacts import check_all

        # MLB + FIFA are committed; NBA may be present in CI.
        code = check_all(["MLB", "FIFA"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
