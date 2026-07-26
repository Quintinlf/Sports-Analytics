"""Send one-time analyst onboarding / invitation emails.

Separate from the Sunday/Wednesday weekly_digest path. Uses the same
email_send_log table with email_type=analyst_invite and a fixed send_date
key so each reviewer receives at most one invitation ever.

Safe launch (required):
  ANALYST_INVITE_ALLOWLIST=a@example.com,b@example.com
  FEEDBACK_BASE_URL=https://your-service.onrender.com
  SMTP_* / FEEDBACK_EMAIL_FROM
  DATABASE_URL (or SUPABASE_*)

Usage:
  python -m scripts.send_analyst_invite
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.config  # noqa: F401 — load .env

from sqlalchemy import text
from sqlalchemy.engine import Engine

from scripts.db_utils import create_database_engine, ensure_default_reviewers, resolve_database_url
from scripts.send_weekly_feedback_form import (
    already_sent,
    ensure_email_send_log,
    record_send,
    send_email,
)

logger = logging.getLogger("analyst_invite")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

EMAIL_TYPE_ANALYST_INVITE = "analyst_invite"
# Lifetime idempotency key (not a calendar date) — one invite per reviewer.
INVITE_SEND_KEY = "onboarding"

INVITE_SUBJECT = "🏆 You're Invited: Become a Sports AI Analyst 🤖"


def invite_already_sent(engine: Engine, reviewer_id: str) -> bool:
    return already_sent(engine, reviewer_id, EMAIL_TYPE_ANALYST_INVITE, INVITE_SEND_KEY)


def load_invite_candidates(
    engine: Engine,
    allowlist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Reviewers with email who have not yet received an analyst_invite.

    Allowlist is required for safety at the CLI layer (main refuses empty allowlist).
    """
    sql = text(
        """
        SELECT r.reviewer_id, r.name, r.email,
               rp.favorite_sports, rp.emails_enabled
        FROM reviewers r
        LEFT JOIN reviewer_preferences rp ON rp.reviewer_id = r.reviewer_id
        WHERE r.email IS NOT NULL AND TRIM(r.email) != ''
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()

    allow = {e.strip().lower() for e in (allowlist or []) if e.strip()}
    result: List[Dict[str, Any]] = []
    for row in rows:
        if row["emails_enabled"] is not None and not bool(row["emails_enabled"]):
            continue
        email = (row["email"] or "").strip()
        if allow and email.lower() not in allow:
            continue
        if invite_already_sent(engine, row["reviewer_id"]):
            continue
        result.append(
            {
                "reviewer_id": row["reviewer_id"],
                "name": row["name"] or "Analyst",
                "email": email,
            }
        )
    return result


def render_invite_email(reviewer: Dict[str, Any], base_url: str) -> str:
    """Professional HTML invitation — headings, sections, emoji accents."""
    name = reviewer.get("name") or "Analyst"
    cta = f"{base_url.rstrip('/')}/feedback?reviewer_id={reviewer['reviewer_id']}"
    contact = os.getenv("ANALYST_CONTACT_EMAIL", "quintinlf7@gmail.com")

    section = (
        "margin:0 0 22px 0;padding:18px 20px;background:#f8fafc;"
        "border-radius:10px;border:1px solid #e2e8f0;"
    )
    h2 = "margin:0 0 10px 0;font-size:18px;color:#0f172a;"
    p = "margin:0 0 10px 0;font-size:15px;line-height:1.55;color:#334155;"
    li = "margin:0 0 6px 0;font-size:15px;line-height:1.5;color:#334155;"

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /></head>
<body style="margin:0;padding:0;background:#eef2ff;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:680px;margin:28px auto;background:#ffffff;border-radius:14px;
              overflow:hidden;box-shadow:0 8px 28px rgba(15,23,42,0.08);">
    <div style="background:linear-gradient(135deg,#1e3a8a,#312e81);padding:28px 24px;color:#fff;">
      <div style="font-size:13px;letter-spacing:0.08em;opacity:0.85;text-transform:uppercase;">
        Sports Analytics Platform
      </div>
      <h1 style="margin:10px 0 0 0;font-size:26px;line-height:1.25;">
        🏆 You're Invited: Become a Sports AI Analyst 🤖
      </h1>
      <p style="margin:12px 0 0 0;font-size:15px;opacity:0.95;">Hi {name},</p>
    </div>

    <div style="padding:24px;">
      <div style="{section}">
        <h2 style="{h2}">🏆 You're Invited: Become a Sports AI Analyst</h2>
        <p style="{p}">
          You are joining the <strong>first group of analysts</strong> helping improve our
          Sports AI prediction platform.
        </p>
        <p style="{p}">
          The goal is not only predicting winners — it is
          <strong>teaching the AI how humans reason about sports</strong>: context, injuries,
          matchups, and judgment that raw stats alone miss.
        </p>
      </div>

      <div style="{section}">
        <h2 style="{h2}">🤖 How It Works</h2>
        <ul style="margin:0;padding-left:20px;">
          <li style="{li}">The AI analyzes <strong>MLB</strong>, <strong>NBA</strong>, and
            <strong>FIFA / soccer</strong> predictions.</li>
          <li style="{li}">It uses statistics, historical data, team information, and machine learning.</li>
          <li style="{li}">You receive your dashboard digest on
            <strong>Sundays</strong> and <strong>Wednesdays</strong>.</li>
        </ul>
      </div>

      <div style="{section}">
        <h2 style="{h2}">🧠 Your Role: Analyst, Not Beta Tester</h2>
        <p style="{p}">
          Your job is to provide <strong>reasoning</strong>, not just mark predictions right or wrong.
        </p>
        <p style="{p}">Strong feedback includes:</p>
        <ul style="margin:0;padding-left:20px;">
          <li style="{li}">Injuries and availability</li>
          <li style="{li}">Matchup context</li>
          <li style="{li}">Player / team trends</li>
          <li style="{li}">Strategy and game-plan edges</li>
          <li style="{li}">Information the model likely missed</li>
        </ul>
      </div>

      <div style="{section}">
        <h2 style="{h2}">⚔️ When Humans Beat The AI</h2>
        <p style="{p}">
          If you <strong>disagree with the AI and you are correct</strong>, your reasoning is
          especially valuable.
        </p>
        <p style="{p}">
          The system will ask <em>why you saw something the AI missed</em>. The purpose is
          <strong>learning from expert reasoning</strong> — not ranking people.
        </p>
      </div>

      <div style="{section}">
        <h2 style="{h2}">💡 Your Ideas Matter</h2>
        <p style="{p}">Use the free feedback areas on the dashboard for:</p>
        <ul style="margin:0;padding-left:20px;">
          <li style="{li}">Teams you want analyzed</li>
          <li style="{li}">Sports ideas</li>
          <li style="{li}">Statistics the model should consider</li>
          <li style="{li}">Missing logic</li>
          <li style="{li}">Product improvements</li>
        </ul>
      </div>

      <div style="{section}">
        <h2 style="{h2}">📚 Research Questions</h2>
        <p style="{p}">
          You may receive focused research questions / topics designed to collect structured
          reasoning and improve the system over time.
        </p>
      </div>

      <div style="{section}">
        <h2 style="{h2}">📩 Email Schedule</h2>
        <ul style="margin:0;padding-left:20px;">
          <li style="{li}"><strong>Sunday</strong> — analyst digest</li>
          <li style="{li}"><strong>Wednesday</strong> — analyst digest</li>
        </ul>
        <p style="{p}">
          Contact <strong>Quintin</strong> at
          <a href="mailto:{contact}" style="color:#1d4ed8;">{contact}</a> if:
        </p>
        <ul style="margin:0;padding-left:20px;">
          <li style="{li}">Emails are missing</li>
          <li style="{li}">Emails feel excessive</li>
          <li style="{li}">You want specific teams or sports analyzed</li>
        </ul>
      </div>

      <div style="{section}">
        <h2 style="{h2}">🚀 Future Development</h2>
        <p style="{p}">
          Future versions may include deeper analytics and decision tools. Right now the focus is
          building a reliable <strong>human + AI feedback loop</strong>.
        </p>
      </div>

      <div style="text-align:center;margin:28px 0 8px 0;">
        <a href="{cta}"
           style="display:inline-block;background:#1d4ed8;color:#fff;text-decoration:none;
                  font-weight:600;padding:14px 28px;border-radius:10px;font-size:16px;">
          Open your analyst dashboard →
        </a>
      </div>
      <p style="font-size:13px;color:#64748b;text-align:center;margin:12px 0 0 0;">
        Or paste this link: {cta}
      </p>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    try:
        db_url = resolve_database_url(default=None, required=True)
    except RuntimeError:
        logger.error(
            "SUPABASE_DATABASE_URL, SUPERBASE_DATABASE_URL, or DATABASE_URL is required."
        )
        sys.exit(1)

    base_url = (os.getenv("FEEDBACK_BASE_URL") or "").rstrip("/")
    if not base_url:
        logger.error("FEEDBACK_BASE_URL is required.")
        sys.exit(1)

    allowlist_raw = (
        os.getenv("ANALYST_INVITE_ALLOWLIST") or os.getenv("WEEKLY_EMAIL_ALLOWLIST") or ""
    ).strip()
    allowlist = [e.strip() for e in allowlist_raw.split(",") if e.strip()]
    if not allowlist:
        logger.error(
            "ANALYST_INVITE_ALLOWLIST is required (comma-separated emails). "
            "Refusing to invite everyone by default."
        )
        sys.exit(1)

    engine = create_database_engine(db_url)
    ensure_default_reviewers(engine)
    ensure_email_send_log(engine)

    candidates = load_invite_candidates(engine, allowlist=allowlist)
    if not candidates:
        logger.warning(
            "No invite candidates (check allowlist, emails on reviewers, or prior invites)."
        )
        sys.exit(0)

    sent = failed = skipped = 0
    for reviewer in candidates:
        try:
            if invite_already_sent(engine, reviewer["reviewer_id"]):
                skipped += 1
                continue
            html = render_invite_email(reviewer, base_url)
            send_email(reviewer["email"], INVITE_SUBJECT, html)
            record_send(
                engine,
                reviewer["reviewer_id"],
                EMAIL_TYPE_ANALYST_INVITE,
                INVITE_SEND_KEY,
                reviewer["email"],
            )
            sent += 1
            logger.info("Sent analyst invite to %s (%s)", reviewer["name"], reviewer["email"])
        except Exception as exc:
            failed += 1
            logger.error(
                "Failed invite to %s (%s): %s",
                reviewer["name"],
                reviewer["email"],
                exc,
                exc_info=True,
            )

    logger.info(
        "Invite run complete. emails_sent=%s emails_failed=%s emails_skipped=%s",
        sent,
        failed,
        skipped,
    )
    if sent == 0 and failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
