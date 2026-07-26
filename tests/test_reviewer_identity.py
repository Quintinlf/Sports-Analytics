"""Reviewer identity: email-canonical + case-insensitive name matching."""
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
from backend.routes.feedback import (
    _normalize_reviewer_email,
    _normalize_reviewer_name,
    init_platform,
)
from scripts.db_utils import create_database_engine, ensure_reviewer_email_unique_index
from scripts.send_weekly_feedback_form import (
    email_already_claimed_for_window,
    ensure_email_send_log,
)


class TestReviewerIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "identity.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        with patch.dict(os.environ, {"ENABLE_DEMO_PREDICTIONS": "false"}):
            init_platform(self.engine)
        ensure_reviewer_email_unique_index(self.engine)
        ensure_email_send_log(self.engine)
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

    def test_normalize_helpers(self) -> None:
        self.assertEqual(_normalize_reviewer_name("  Quintin "), "quintin")
        self.assertEqual(_normalize_reviewer_name("Quintin"), "quintin")
        self.assertEqual(_normalize_reviewer_email(" QuintinLF7@Gmail.com "), "quintinlf7@gmail.com")
        self.assertIsNone(_normalize_reviewer_email("  "))

    def test_quintin_vs_Quintin_same_reviewer(self) -> None:
        a = self.client.post("/api/feedback/reviewers", json={"name": "Quintin"})
        self.assertEqual(a.status_code, 200)
        rid_a = a.json()["reviewer_id"]

        b = self.client.post("/api/feedback/reviewers", json={"name": "quintin"})
        self.assertEqual(b.status_code, 200)
        rid_b = b.json()["reviewer_id"]

        self.assertEqual(rid_a, rid_b)
        with self.engine.connect() as conn:
            n = conn.execute(
                text("SELECT COUNT(*) FROM reviewers WHERE lower(trim(name)) = 'quintin'")
            ).scalar()
        self.assertEqual(n, 1)

    def test_same_email_reuses_existing_reviewer(self) -> None:
        a = self.client.post(
            "/api/feedback/reviewers",
            json={"name": "Quintin Alpha", "email": "analyst@example.com"},
        )
        self.assertEqual(a.status_code, 200)
        rid_a = a.json()["reviewer_id"]

        b = self.client.post(
            "/api/feedback/reviewers",
            json={"name": "Totally Different", "email": "Analyst@Example.com"},
        )
        self.assertEqual(b.status_code, 200)
        self.assertEqual(b.json()["reviewer_id"], rid_a)

        with self.engine.connect() as conn:
            n = conn.execute(
                text(
                    "SELECT COUNT(*) FROM reviewers WHERE lower(email) = 'analyst@example.com'"
                )
            ).scalar()
        self.assertEqual(n, 1)

    def test_unique_email_index_blocks_second_row(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO reviewers (reviewer_id, name, email, created_at)
                    VALUES ('r1', 'One', 'dup@example.com', '2026-01-01')
                    """
                )
            )
        with self.assertRaises(Exception):
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO reviewers (reviewer_id, name, email, created_at)
                        VALUES ('r2', 'Two', 'DUP@example.com', '2026-01-02')
                        """
                    )
                )

    def test_email_window_claim_is_inbox_scoped(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO email_send_log
                        (reviewer_id, email_type, send_date, email, sent_at)
                    VALUES
                        ('r-a', 'weekly_digest', '2026-07-27', 'same@example.com', 't')
                    """
                )
            )
        self.assertTrue(
            email_already_claimed_for_window(
                self.engine, "SAME@example.com", "weekly_digest", "2026-07-27"
            )
        )
        self.assertFalse(
            email_already_claimed_for_window(
                self.engine, "other@example.com", "weekly_digest", "2026-07-27"
            )
        )


if __name__ == "__main__":
    unittest.main()
