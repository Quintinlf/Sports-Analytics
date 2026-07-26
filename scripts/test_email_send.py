"""Test email sender for the AI Sports Analyst Feedback Platform.

Creates (or updates) a 'quintin' reviewer in the local SQLite database and
sends a styled HTML test email with a clickable dashboard link.

Required environment variables:
  SMTP_USER   — Gmail / SMTP username (e.g. you@gmail.com)
  SMTP_PASS   — App password (Gmail) or SMTP password

Optional environment variables:
  SMTP_HOST         — Default: smtp.gmail.com
  SMTP_PORT         — Default: 587  (STARTTLS)
  TEST_EMAIL        — Recipient address; defaults to SMTP_USER
  SQLITE_DATABASE_URL — Local SQLite path for this script (preferred when
                        DATABASE_URL / SUPERBASE_* point at Postgres)
  DATABASE_URL / SUPERBASE_DATABASE_URL / SUPABASE_DATABASE_URL
                      — Used only when the resolved URL is SQLite
  FEEDBACK_BASE_URL — Base URL of the running platform
                      Default: http://localhost:8000

Usage (PowerShell):
  $env:SMTP_USER = "you@gmail.com"
  $env:SMTP_PASS = "your-app-password"
  $env:TEST_EMAIL = "you@gmail.com"
  python scripts/test_email_send.py

Usage (bash / WSL):
  SMTP_USER="you@gmail.com" SMTP_PASS="..." python scripts/test_email_send.py
"""
from __future__ import annotations

import os
import smtplib
import sqlite3
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Path alignment when invoked as `python scripts/test_email_send.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.config  # noqa: F401 — load .env before database URL resolution

from scripts.db_utils import DEFAULT_SQLITE_URL, resolve_database_url

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REVIEWER_ID   = "quintin"
REVIEWER_NAME = "Quintin"

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
TEST_EMAIL = os.getenv("TEST_EMAIL", SMTP_USER)

# Configurable base URL — supports staging, ngrok tunnels, etc.
FEEDBACK_BASE_URL = os.getenv("FEEDBACK_BASE_URL", "http://localhost:8000").rstrip("/")
DASHBOARD_LINK = f"{FEEDBACK_BASE_URL}/feedback?reviewer_id={REVIEWER_ID}"

# This helper uses sqlite3 against a local file. Prefer SQLITE_DATABASE_URL when
# the canonical resolver points at Postgres (typical local SUPERBASE setup).
_resolved = resolve_database_url(default=DEFAULT_SQLITE_URL)
_db_url = (
    _resolved
    if _resolved.startswith("sqlite")
    else os.getenv("SQLITE_DATABASE_URL", DEFAULT_SQLITE_URL)
)
_db_path_str = _db_url.replace("sqlite:///", "")
DB_PATH = Path(_db_path_str).resolve()

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require(var: str, value: str) -> None:
    if not value:
        print(f"ERROR: Environment variable {var} is required but not set.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Reviewer upsert
# ---------------------------------------------------------------------------

def upsert_reviewer(email: str) -> None:
    """Idempotently create or update the 'quintin' reviewer row."""
    if not DB_PATH.exists():
        print(f"WARNING: Database not found at {DB_PATH}. "
              "Start the server first so the DB is initialised.")
        return

    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(str(DB_PATH)) as conn:
        # Ensure email column exists (tolerates older schema)
        try:
            conn.execute("ALTER TABLE reviewers ADD COLUMN email TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

        # UPSERT — preserve original created_at; update name + email on conflict
        conn.execute(
            """
            INSERT INTO reviewers (reviewer_id, name, email, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(reviewer_id) DO UPDATE SET
                name  = excluded.name,
                email = COALESCE(excluded.email, reviewers.email)
            """,
            (REVIEWER_ID, REVIEWER_NAME, email, ts),
        )
        conn.commit()

    print(f"[DB] Reviewer upserted: reviewer_id={REVIEWER_ID!r}, name={REVIEWER_NAME!r}, email={email!r}")

# ---------------------------------------------------------------------------
# Email builder
# ---------------------------------------------------------------------------

def _build_email(recipient: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "AI Sports Analyst \u2013 Weekly Test Run"
    msg["From"]    = SMTP_USER
    msg["To"]      = recipient

    plain = (
        f"Hi {REVIEWER_NAME},\n\n"
        "This is a test of the AI Sports Analyst Feedback Platform.\n\n"
        "Open your analyst dashboard here:\n"
        f"  {DASHBOARD_LINK}\n\n"
        "You can:\n"
        "  \u2022 Review AI predictions across MLB, NBA, and FIFA\n"
        "  \u2022 Submit your own pick before the game\n"
        "  \u2022 Record postgame reflections and see if you beat the model\n\n"
        "This is a development test email.\n"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#0b0f14;font-family:'Segoe UI',system-ui,sans-serif;color:#e2e8f0">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0b0f14;padding:40px 20px">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background:#151c28;border:1px solid #1e2d42;border-radius:16px;overflow:hidden">

          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(135deg,#0d1a2d,#111820);
                       padding:32px 36px 24px;border-bottom:1px solid #1e2d42">
              <div style="font-size:32px;margin-bottom:8px">&#x1F916;</div>
              <h1 style="margin:0;font-size:22px;font-weight:800;color:#f8fafc;letter-spacing:-0.5px">
                AI Sports Analyst
              </h1>
              <p style="margin:4px 0 0;font-size:13px;color:#64748b">
                Weekly Prediction Review &mdash; Test Run
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:32px 36px">
              <p style="margin:0 0 16px;font-size:16px;color:#cbd5e1">
                Hi <strong style="color:#f8fafc">{REVIEWER_NAME}</strong>,
              </p>
              <p style="margin:0 0 20px;font-size:15px;color:#94a3b8;line-height:1.6">
                This is a system integration test for the analyst feedback platform.
                Click below to open your personalised dashboard, review this week&rsquo;s AI
                predictions, submit your picks, and record postgame reflections.
              </p>

              <!-- CTA button -->
              <table cellpadding="0" cellspacing="0" style="margin:28px 0">
                <tr>
                  <td style="background:#3b82f6;border-radius:10px">
                    <a href="{DASHBOARD_LINK}"
                       style="display:inline-block;padding:14px 32px;font-size:15px;
                              font-weight:700;color:#ffffff;text-decoration:none;
                              letter-spacing:-0.2px">
                      Open Analyst Dashboard &rarr;
                    </a>
                  </td>
                </tr>
              </table>

              <!-- Feature list -->
              <table cellpadding="0" cellspacing="0" width="100%"
                     style="background:#0f1928;border:1px solid #1e2d42;
                            border-radius:10px;padding:20px">
                <tr>
                  <td style="padding:0 20px">
                    <p style="margin:0 0 12px;font-size:12px;font-weight:700;
                               color:#3b82f6;text-transform:uppercase;letter-spacing:1.2px">
                      What you can do
                    </p>
                    <ul style="margin:0;padding:0 0 0 18px;color:#94a3b8;
                                font-size:13px;line-height:1.8">
                      <li>Review AI predictions for MLB &#x26BE;, NBA &#x1F3C0;, and FIFA &#x26BD;</li>
                      <li>See confidence bars and feature-weight explanations</li>
                      <li>Submit your pregame pick and scouting notes</li>
                      <li>Record postgame reflections &mdash; did you beat the model?</li>
                    </ul>
                  </td>
                </tr>
              </table>

              <p style="margin:24px 0 0;font-size:12px;color:#334155;line-height:1.6">
                If the button doesn&rsquo;t work, copy this link into your browser:<br>
                <a href="{DASHBOARD_LINK}" style="color:#3b82f6;word-break:break-all">{DASHBOARD_LINK}</a>
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding:16px 36px 24px;border-top:1px solid #1e2d42">
              <p style="margin:0;font-size:11px;color:#1e3a5f;text-align:center">
                Development test email &mdash; Sports Analytics Platform
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))
    return msg

# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_email(recipient: str) -> None:
    msg = _build_email(recipient)
    print(f"[SMTP] Connecting to {SMTP_HOST}:{SMTP_PORT} …")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipient, msg.as_string())
    print(f"[SMTP] Email sent to {recipient!r}")
    print(f"[LINK] Dashboard link: {DASHBOARD_LINK}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _require("SMTP_USER", SMTP_USER)
    _require("SMTP_PASS", SMTP_PASS)

    recipient = TEST_EMAIL or SMTP_USER
    if not recipient:
        print("ERROR: TEST_EMAIL (or SMTP_USER) must be set to a valid email address.")
        sys.exit(1)

    print(f"[CONFIG] SMTP:          {SMTP_USER} → {SMTP_HOST}:{SMTP_PORT}")
    print(f"[CONFIG] Recipient:     {recipient}")
    print(f"[CONFIG] Database:      {DB_PATH}")
    print(f"[CONFIG] Base URL:      {FEEDBACK_BASE_URL}")
    print()

    upsert_reviewer(recipient)
    send_email(recipient)
    print()
    print("Done. Open the link from the email to start the end-to-end test.")


if __name__ == "__main__":
    main()
