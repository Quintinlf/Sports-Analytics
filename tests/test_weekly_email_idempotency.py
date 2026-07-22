"""Weekly digest sends at most once per reviewer per UTC date."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.db_utils import create_database_engine
from scripts.send_weekly_feedback_form import (
    EMAIL_TYPE_WEEKLY_DIGEST,
    already_sent,
    ensure_email_send_log,
    main,
    record_send,
)


class TestWeeklyEmailIdempotency(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "email.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        ensure_email_send_log(self.engine)
        self.send_date = "2026-07-20"

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_record_is_unique_per_reviewer_type_date(self) -> None:
        record_send(
            self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.send_date, "a@example.com"
        )
        record_send(
            self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.send_date, "a@example.com"
        )
        record_send(
            self.engine, "rev-b", EMAIL_TYPE_WEEKLY_DIGEST, self.send_date, "b@example.com"
        )
        self.assertTrue(already_sent(self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.send_date))
        self.assertTrue(already_sent(self.engine, "rev-b", EMAIL_TYPE_WEEKLY_DIGEST, self.send_date))
        with self.engine.begin() as conn:
            from sqlalchemy import text

            count = conn.execute(text("SELECT COUNT(*) FROM email_send_log")).scalar()
        self.assertEqual(count, 2)

    def test_main_skips_second_run_for_same_day(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS reviewers (
                        reviewer_id TEXT PRIMARY KEY,
                        name TEXT,
                        email TEXT,
                        created_at TEXT
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS reviewer_preferences (
                        reviewer_id TEXT PRIMARY KEY,
                        favorite_sports TEXT,
                        emails_enabled INTEGER,
                        email_days TEXT
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO reviewers (reviewer_id, name, email, created_at)
                    VALUES ('rev-a', 'Ada', 'ada@example.com', '2026-01-01')
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO reviewer_preferences (reviewer_id, favorite_sports, emails_enabled, email_days)
                    VALUES ('rev-a', '["MLB"]', 1, '[0,1,2,3,4,5,6]')
                    """
                )
            )

        send_calls: list[str] = []

        def _fake_send(to_email: str, subject: str, html_content: str) -> None:
            send_calls.append(to_email)

        env = {
            "DATABASE_URL": str(self.engine.url),
            "FEEDBACK_BASE_URL": "http://localhost:8000",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_USER": "u",
            "SMTP_PASS": "p",
            "FEEDBACK_EMAIL_FROM": "from@example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("scripts.send_weekly_feedback_form.create_database_engine", return_value=self.engine):
                with patch("scripts.send_weekly_feedback_form.ensure_default_reviewers"):
                    with patch(
                        "scripts.send_weekly_feedback_form.load_featured_research_question",
                        return_value=None,
                    ):
                        with patch(
                            "scripts.send_weekly_feedback_form.load_reviewer_stats",
                            return_value={
                                "agreement_pct": 0,
                                "beat_ai_count": 0,
                                "pending_pregame": 0,
                                "pending_postgame": 0,
                                "pending_case_studies": 0,
                            },
                        ):
                            with patch(
                                "scripts.send_weekly_feedback_form.load_predictions",
                                return_value=[],
                            ):
                                with patch(
                                    "scripts.send_weekly_feedback_form.send_email",
                                    side_effect=_fake_send,
                                ):
                                    main()
                                    main()

        self.assertEqual(send_calls, ["ada@example.com"])


if __name__ == "__main__":
    unittest.main()
