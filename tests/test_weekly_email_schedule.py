"""Pacific-time digest windows + duplicate claim semantics."""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from sqlalchemy import text

from scripts.db_utils import create_database_engine
from scripts.send_weekly_feedback_form import (
    EMAIL_TYPE_WEEKLY_DIGEST,
    PACIFIC_TZ,
    already_sent,
    ensure_email_send_log,
    is_digest_send_day,
    load_reviewers,
    main,
    pacific_cron_weekday,
    pacific_send_window_key,
    release_send_claim,
    try_claim_send,
)

PT = PACIFIC_TZ


def _pt(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=PT)


class TestPacificScheduleHelpers(unittest.TestCase):
    def test_sunday_midnight_pt_is_send_day(self) -> None:
        # 2026-07-26 is a Sunday
        now = _pt(2026, 7, 26, 0, 0)
        self.assertEqual(pacific_cron_weekday(now), 0)
        self.assertTrue(is_digest_send_day(now))
        self.assertEqual(pacific_send_window_key(now), "2026-07-26")

    def test_wednesday_midnight_pt_is_send_day(self) -> None:
        # 2026-07-22 is a Wednesday
        now = _pt(2026, 7, 22, 0, 5)
        self.assertEqual(pacific_cron_weekday(now), 3)
        self.assertTrue(is_digest_send_day(now))
        self.assertEqual(pacific_send_window_key(now), "2026-07-22")

    def test_saturday_pt_does_not_send(self) -> None:
        # 2026-07-25 is a Saturday — former UTC-midnight-Sunday bug window
        now = _pt(2026, 7, 25, 17, 0)  # Saturday 5pm PT ≈ old UTC Sun 00:00 in PDT
        self.assertEqual(pacific_cron_weekday(now), 6)
        self.assertFalse(is_digest_send_day(now))

    def test_utc_sunday_midnight_maps_to_saturday_pt_in_summer(self) -> None:
        # Explicit conversion check: 2026-07-26 00:00 UTC → 2026-07-25 17:00 PDT
        utc = datetime(2026, 7, 26, 0, 0, tzinfo=ZoneInfo("UTC"))
        pt = utc.astimezone(PT)
        self.assertEqual(pt.weekday(), 5)  # Saturday
        self.assertFalse(is_digest_send_day(pt))

    def test_dst_and_pst_midnight_targets(self) -> None:
        # PDT: midnight PT = 07:00 UTC
        pdt_midnight = _pt(2026, 7, 26, 0, 0)
        self.assertEqual(
            pdt_midnight.astimezone(ZoneInfo("UTC")).strftime("%H:%M"),
            "07:00",
        )
        # PST: midnight PT = 08:00 UTC (2026-01-04 is Sunday in PST)
        pst_midnight = _pt(2026, 1, 4, 0, 0)
        self.assertEqual(
            pst_midnight.astimezone(ZoneInfo("UTC")).strftime("%H:%M"),
            "08:00",
        )
        self.assertTrue(is_digest_send_day(pst_midnight))


class TestClaimIdempotency(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "sched.db")
        self.engine = create_database_engine(f"sqlite:///{db_path}")
        ensure_email_send_log(self.engine)
        self.window = "2026-07-26"

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_second_claim_same_window_fails(self) -> None:
        self.assertTrue(
            try_claim_send(
                self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window, "a@x.com"
            )
        )
        self.assertFalse(
            try_claim_send(
                self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window, "a@x.com"
            )
        )
        self.assertTrue(
            already_sent(self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window)
        )

    def test_release_allows_retry(self) -> None:
        self.assertTrue(
            try_claim_send(
                self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window, "a@x.com"
            )
        )
        release_send_claim(self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window)
        self.assertTrue(
            try_claim_send(
                self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window, "a@x.com"
            )
        )

    def test_invite_key_does_not_block_digest(self) -> None:
        try_claim_send(self.engine, "rev-a", "analyst_invite", "onboarding", "a@x.com")
        self.assertTrue(
            try_claim_send(
                self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, self.window, "a@x.com"
            )
        )


class TestMainPacificGate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "main.db")
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
                    VALUES ('rev-a', 'Ada', 'ada@example.com', '2026-01-01')
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO reviewer_preferences
                      (reviewer_id, favorite_sports, emails_enabled, email_days)
                    VALUES ('rev-a', '["MLB"]', 1, '[0,3]')
                    """
                )
            )

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()

    def _run_main(self, now: datetime, force: str = "") -> list[str]:
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
            "WEEKLY_EMAIL_FORCE": force,
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
                                    try:
                                        main(now=now)
                                    except SystemExit:
                                        pass
        return send_calls

    def test_saturday_main_sends_nothing(self) -> None:
        calls = self._run_main(_pt(2026, 7, 25, 17, 0))
        self.assertEqual(calls, [])

    def test_sunday_sends_once_and_duplicate_skips(self) -> None:
        sunday = _pt(2026, 7, 26, 0, 10)
        calls1 = self._run_main(sunday)
        calls2 = self._run_main(sunday)
        self.assertEqual(calls1, ["ada@example.com"])
        self.assertEqual(calls2, [])
        self.assertTrue(
            already_sent(self.engine, "rev-a", EMAIL_TYPE_WEEKLY_DIGEST, "2026-07-26")
        )

    def test_wednesday_sends(self) -> None:
        calls = self._run_main(_pt(2026, 7, 22, 0, 0))
        self.assertEqual(calls, ["ada@example.com"])
        self.assertEqual(pacific_send_window_key(_pt(2026, 7, 22, 0, 0)), "2026-07-22")

    def test_load_reviewers_uses_pacific_weekday(self) -> None:
        saturday = _pt(2026, 7, 25, 12, 0)
        sunday = _pt(2026, 7, 26, 12, 0)
        self.assertEqual(load_reviewers(self.engine, now=saturday), [])
        loaded = load_reviewers(self.engine, now=sunday)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["email"], "ada@example.com")


if __name__ == "__main__":
    unittest.main()
