"""Onboarding invite emails are one-shot and leave weekly_digest unchanged."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import text

from scripts.db_utils import create_database_engine
from scripts.send_analyst_invite import (
    EMAIL_TYPE_ANALYST_INVITE,
    INVITE_SEND_KEY,
    INVITE_SUBJECT,
    invite_already_sent,
    load_invite_candidates,
    main as invite_main,
    render_invite_email,
)
from scripts.send_weekly_feedback_form import (
    EMAIL_TYPE_WEEKLY_DIGEST,
    already_sent,
    ensure_email_send_log,
    main as weekly_main,
    record_send,
)


class TestAnalystInviteEmail(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "invite.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        ensure_email_send_log(self.engine)
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
                    VALUES
                      ('rev-a', 'Ada', 'ada@example.com', '2026-01-01'),
                      ('rev-b', 'Bob', 'bob@example.com', '2026-01-01')
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO reviewer_preferences
                      (reviewer_id, favorite_sports, emails_enabled, email_days)
                    VALUES
                      ('rev-a', '["MLB"]', 1, '[0,1,2,3,4,5,6]'),
                      ('rev-b', '["NBA"]', 1, '[0,1,2,3,4,5,6]')
                    """
                )
            )

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_invite_html_has_required_sections(self) -> None:
        html = render_invite_email(
            {"reviewer_id": "rev-a", "name": "Ada"},
            "https://example.com",
        )
        self.assertIn("You're Invited: Become a Sports AI Analyst", html)
        self.assertIn("How It Works", html)
        self.assertIn("Analyst, Not Beta Tester", html)
        self.assertIn("When Humans Beat The AI", html)
        self.assertIn("Your Ideas Matter", html)
        self.assertIn("Research Questions", html)
        self.assertIn("Email Schedule", html)
        self.assertIn("Sunday", html)
        self.assertIn("Wednesday", html)
        self.assertIn("Future Development", html)
        self.assertIn("reviewer_id=rev-a", html)

    def test_invite_does_not_duplicate(self) -> None:
        send_calls: list[tuple[str, str]] = []

        def _fake_send(to_email: str, subject: str, html_content: str) -> None:
            send_calls.append((to_email, subject))

        env = {
            "DATABASE_URL": str(self.engine.url),
            "FEEDBACK_BASE_URL": "http://localhost:8000",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_USER": "u",
            "SMTP_PASS": "p",
            "FEEDBACK_EMAIL_FROM": "from@example.com",
            "ANALYST_INVITE_ALLOWLIST": "ada@example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("scripts.send_analyst_invite.create_database_engine", return_value=self.engine):
                with patch("scripts.send_analyst_invite.ensure_default_reviewers"):
                    with patch("scripts.send_analyst_invite.send_email", side_effect=_fake_send):
                        invite_main()
                        with self.assertRaises(SystemExit) as second:
                            invite_main()
                        self.assertEqual(second.exception.code, 0)

        self.assertEqual(len(send_calls), 1)
        self.assertEqual(send_calls[0][0], "ada@example.com")
        self.assertEqual(send_calls[0][1], INVITE_SUBJECT)
        self.assertTrue(invite_already_sent(self.engine, "rev-a"))
        self.assertFalse(
            already_sent(self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, "2026-07-20")
        )

    def test_invite_allowlist_required(self) -> None:
        env = {
            "DATABASE_URL": str(self.engine.url),
            "FEEDBACK_BASE_URL": "http://localhost:8000",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_USER": "u",
            "SMTP_PASS": "p",
            "FEEDBACK_EMAIL_FROM": "from@example.com",
        }
        # Clear allowlist vars
        with patch.dict(
            os.environ,
            {**env, "ANALYST_INVITE_ALLOWLIST": "", "WEEKLY_EMAIL_ALLOWLIST": ""},
            clear=False,
        ):
            with patch("scripts.send_analyst_invite.create_database_engine", return_value=self.engine):
                with self.assertRaises(SystemExit) as ctx:
                    invite_main()
        self.assertEqual(ctx.exception.code, 1)

    def test_weekly_digest_unchanged_by_invite_log(self) -> None:
        """Recording an invite must not block weekly_digest for the same day."""
        record_send(
            self.engine,
            "rev-a",
            EMAIL_TYPE_ANALYST_INVITE,
            INVITE_SEND_KEY,
            "ada@example.com",
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
            "WEEKLY_EMAIL_ALLOWLIST": "ada@example.com",
            "WEEKLY_EMAIL_FORCE": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "scripts.send_weekly_feedback_form.create_database_engine",
                return_value=self.engine,
            ):
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
                                    weekly_main()
                                    weekly_main()

        self.assertEqual(send_calls, ["ada@example.com"])
        self.assertTrue(invite_already_sent(self.engine, "rev-a"))

    def test_load_candidates_respects_prior_invite(self) -> None:
        record_send(
            self.engine,
            "rev-a",
            EMAIL_TYPE_ANALYST_INVITE,
            INVITE_SEND_KEY,
            "ada@example.com",
        )
        cands = load_invite_candidates(
            self.engine, allowlist=["ada@example.com", "bob@example.com"]
        )
        emails = {c["email"] for c in cands}
        self.assertEqual(emails, {"bob@example.com"})


if __name__ == "__main__":
    unittest.main()
