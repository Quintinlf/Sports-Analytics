"""Personalized weekly reviewer email digest."""
from __future__ import annotations

import json
import logging
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from scripts.db_utils import create_database_engine, ensure_default_reviewers, sql_bool_true

logger = logging.getLogger("weekly_feedback_distribution")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

DEFAULT_EMAIL_DAYS = [0, 3]  # Sunday and Wednesday (UTC cron weekdays)
EMAIL_TYPE_WEEKLY_DIGEST = "weekly_digest"

EMAIL_SEND_LOG_DDL = """
CREATE TABLE IF NOT EXISTS email_send_log (
    reviewer_id TEXT NOT NULL,
    email_type TEXT NOT NULL,
    send_date TEXT NOT NULL,
    email TEXT,
    sent_at TEXT NOT NULL,
    PRIMARY KEY (reviewer_id, email_type, send_date)
)
"""


def ensure_email_send_log(engine: Engine) -> None:
    """Create the idempotency log table if missing (SQLite + Postgres)."""
    with engine.begin() as conn:
        conn.execute(text(EMAIL_SEND_LOG_DDL))


def already_sent(
    engine: Engine,
    reviewer_id: str,
    email_type: str,
    send_date: str,
) -> bool:
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT 1 FROM email_send_log
                WHERE reviewer_id = :rid AND email_type = :etype AND send_date = :sdate
                LIMIT 1
                """
            ),
            {"rid": reviewer_id, "etype": email_type, "sdate": send_date},
        ).first()
    return row is not None


def record_send(
    engine: Engine,
    reviewer_id: str,
    email_type: str,
    send_date: str,
    email: str,
) -> None:
    """Persist a successful send. Unique key makes retries/no-ops safe."""
    ts = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO email_send_log (reviewer_id, email_type, send_date, email, sent_at)
                VALUES (:rid, :etype, :sdate, :email, :sent_at)
                ON CONFLICT (reviewer_id, email_type, send_date) DO NOTHING
                """
            ),
            {
                "rid": reviewer_id,
                "etype": email_type,
                "sdate": send_date,
                "email": email,
                "sent_at": ts,
            },
        )


def _parse_sports(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(s).upper() for s in parsed]
    except Exception:
        pass
    return []


def _parse_email_days(raw: str | None) -> List[int]:
    if not raw:
        return list(DEFAULT_EMAIL_DAYS)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(d) for d in parsed if str(d).isdigit() or isinstance(d, int)]
    except Exception:
        pass
    return list(DEFAULT_EMAIL_DAYS)


def _should_send_today(email_days: List[int], weekday: Optional[int] = None) -> bool:
    day = weekday if weekday is not None else datetime.utcnow().weekday()
    # Python weekday: Mon=0 … Sun=6; cron uses Sun=0 … Sat=6
    cron_day = (day + 1) % 7
    return cron_day in email_days


def load_reviewers(engine, allowlist: List[str] | None = None, weekday: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT r.reviewer_id, r.name, r.email,
               rp.favorite_sports, rp.emails_enabled, rp.email_days
        FROM reviewers r
        LEFT JOIN reviewer_preferences rp ON rp.reviewer_id = r.reviewer_id
        WHERE r.email IS NOT NULL
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()
    result = []
    allow = {e.strip().lower() for e in (allowlist or []) if e.strip()}
    for row in rows:
        if row["emails_enabled"] is not None and not bool(row["emails_enabled"]):
            continue
        email = (row["email"] or "").strip()
        if allow and email.lower() not in allow:
            continue
        days = _parse_email_days(row["email_days"])
        if not _should_send_today(days, weekday=weekday):
            continue
        result.append(
            {
                "reviewer_id": row["reviewer_id"],
                "name": row["name"],
                "email": row["email"],
                "favorite_sports": _parse_sports(row["favorite_sports"]) or ["MLB"],
            }
        )
    return result


def load_reviewer_stats(engine, reviewer_id: str) -> Dict[str, Any]:
    agree_true = sql_bool_true("pr.agree_with_model", engine)
    beat_true = sql_bool_true("ro.reviewer_beat_model", engine)
    with engine.begin() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM prediction_reviews WHERE reviewer_id = :rid"),
            {"rid": reviewer_id},
        ).scalar() or 0
        agreed = conn.execute(
            text(f"SELECT COUNT(*) FROM prediction_reviews pr WHERE pr.reviewer_id = :rid AND {agree_true}"),
            {"rid": reviewer_id},
        ).scalar() or 0
        beat_ai = conn.execute(
            text(
                f"""
                SELECT COUNT(*) FROM review_outcomes ro
                JOIN prediction_reviews pr ON ro.review_id = pr.review_id
                WHERE pr.reviewer_id = :rid AND {beat_true}
                """
            ),
            {"rid": reviewer_id},
        ).scalar() or 0
        pending_pregame = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM predictions p
                WHERE p.prediction_status = 'UPCOMING'
                  AND NOT EXISTS (
                    SELECT 1 FROM prediction_reviews pr
                    WHERE pr.prediction_id = p.prediction_id AND pr.reviewer_id = :rid
                  )
                """
            ),
            {"rid": reviewer_id},
        ).scalar() or 0
        pending_postgame = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM prediction_reviews pr
                JOIN predictions p ON p.prediction_id = pr.prediction_id
                LEFT JOIN review_outcomes ro ON ro.review_id = pr.review_id
                WHERE pr.reviewer_id = :rid
                  AND p.actual_winner IS NOT NULL
                  AND ro.review_id IS NULL
                """
            ),
            {"rid": reviewer_id},
        ).scalar() or 0
        pending_case_studies = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM prediction_reviews pr
                JOIN review_outcomes ro ON ro.review_id = pr.review_id
                LEFT JOIN analyst_case_studies cs ON cs.review_id = pr.review_id
                WHERE pr.reviewer_id = :rid AND {beat_true} AND cs.case_id IS NULL
                """
            ),
            {"rid": reviewer_id},
        ).scalar() or 0
    agree_pct = round((agreed / total) * 100, 1) if total else 0.0
    return {
        "agreement_pct": agree_pct,
        "beat_ai_count": int(beat_ai),
        "pending_pregame": int(pending_pregame),
        "pending_postgame": int(pending_postgame),
        "pending_case_studies": int(pending_case_studies),
    }


def load_featured_research_question(engine) -> Optional[Dict[str, Any]]:
    active_true = sql_bool_true("active", engine)
    featured_true = sql_bool_true("featured", engine)
    with engine.begin() as conn:
        row = conn.execute(
            text(
                f"""
                SELECT question_id, title, body_markdown, knowledge_area
                FROM analyst_questions
                WHERE context = 'research' AND {active_true} AND {featured_true}
                ORDER BY sort_order ASC, created_at DESC
                LIMIT 1
                """
            )
        ).mappings().first()
    return dict(row) if row else None


def load_predictions(engine, sport: str, status: str, limit: int = 6) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT sport, away_team, home_team, game_date, confidence_level
                FROM predictions
                WHERE sport = :sport AND prediction_status = :status
                ORDER BY game_date DESC
                LIMIT :limit
                """
            ),
            {"sport": sport, "status": status, "limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def render_email(
    reviewer: Dict[str, Any],
    stats: Dict[str, Any],
    upcoming: List[Dict[str, Any]],
    completed: List[Dict[str, Any]],
    base_url: str,
    research: Optional[Dict[str, Any]] = None,
) -> str:
    favorite = reviewer["favorite_sports"][0] if reviewer["favorite_sports"] else "MLB"
    cta_url = f"{base_url}/feedback?reviewer_id={reviewer['reviewer_id']}&sport={favorite}"

    def _rows(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "<li>No games available in this section.</li>"
        return "".join(
            f"<li>{g['game_date']} - {g['away_team']} @ {g['home_team']} ({g.get('confidence_level','N/A')})</li>"
            for g in items
        )

    research_block = ""
    if research:
        area = research.get("knowledge_area") or "Research"
        body_preview = (research.get("body_markdown") or "")[:280].replace("\n", "<br>")
        research_block = f"""
        <h3>Current Research Question</h3>
        <p><strong>{research['title']}</strong> <em>({area})</em></p>
        <p>{body_preview}…</p>
        <p><a href="{cta_url}">Answer on your dashboard</a></p>
        """

    case_block = ""
    if stats.get("pending_case_studies", 0) > 0:
        n = stats["pending_case_studies"]
        case_block = f"""
        <p style="background:#ecfdf5;padding:12px;border-radius:8px">
          <strong>{n} beat-AI case {'study' if n == 1 else 'studies'} pending</strong> —
          you found cases where the model was wrong. Help teach it why.
          <a href="{cta_url}">Submit case study</a>
        </p>
        """

    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#f7f7f7;padding:20px">
      <div style="max-width:680px;margin:0 auto;background:#fff;padding:20px;border-radius:10px">
        <h2>Analyst Digest: {reviewer['name']}</h2>
        <p>Favorite sports: {", ".join(reviewer["favorite_sports"])}</p>
        <ul>
          <li>Agreement %: {stats['agreement_pct']}%</li>
          <li>Beat AI count: {stats['beat_ai_count']}</li>
          <li>Pending pregame reviews: {stats['pending_pregame']}</li>
          <li>Pending postgame reflections: {stats['pending_postgame']}</li>
          <li>Pending case studies: {stats.get('pending_case_studies', 0)}</li>
        </ul>
        {case_block}
        {research_block}
        <h3>Upcoming Predictions</h3>
        <ul>{_rows(upcoming)}</ul>
        <h3>Completed Predictions</h3>
        <ul>{_rows(completed)}</ul>
        <p><a href="{cta_url}">Open your reviewer dashboard</a></p>
      </div>
    </body></html>
    """


def send_email(to_email: str, subject: str, html_content: str) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    from_email = os.getenv("FEEDBACK_EMAIL_FROM") or smtp_user
    if not (smtp_host and smtp_user and smtp_pass and from_email):
        raise RuntimeError("Missing SMTP configuration. Set SMTP_HOST/SMTP_USER/SMTP_PASS/FEEDBACK_EMAIL_FROM.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [to_email], msg.as_string())


def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    base_url = (os.getenv("FEEDBACK_BASE_URL") or "").rstrip("/")
    if not db_url:
        logger.error("DATABASE_URL is required.")
        sys.exit(1)
    if not base_url:
        logger.error("FEEDBACK_BASE_URL is required.")
        sys.exit(1)

    engine = create_database_engine(db_url)
    ensure_default_reviewers(engine)
    ensure_email_send_log(engine)

    allowlist_raw = os.getenv("WEEKLY_EMAIL_ALLOWLIST", "").strip()
    allowlist = [e.strip() for e in allowlist_raw.split(",") if e.strip()] or None
    reviewers = load_reviewers(engine, allowlist=allowlist)
    if not reviewers:
        logger.warning("No reviewers eligible for digest today (check email_days / allowlist).")
        sys.exit(0)

    research = load_featured_research_question(engine)
    send_date = datetime.utcnow().strftime("%Y-%m-%d")

    sent = 0
    failed = 0
    skipped = 0
    for reviewer in reviewers:
        favorite = reviewer["favorite_sports"][0] if reviewer["favorite_sports"] else "MLB"
        try:
            if already_sent(engine, reviewer["reviewer_id"], EMAIL_TYPE_WEEKLY_DIGEST, send_date):
                skipped += 1
                logger.info(
                    "Skipping digest for %s (%s) — already sent on %s",
                    reviewer["name"],
                    reviewer["email"],
                    send_date,
                )
                continue
            stats = load_reviewer_stats(engine, reviewer["reviewer_id"])
            upcoming = load_predictions(engine, "SOCCER" if favorite == "FIFA" else favorite, "UPCOMING")
            completed = load_predictions(engine, "SOCCER" if favorite == "FIFA" else favorite, "FINAL")
            html = render_email(reviewer, stats, upcoming, completed, base_url, research=research)
            send_email(
                to_email=reviewer["email"],
                subject=f"AI Analyst Digest - {send_date}",
                html_content=html,
            )
            # Record only after successful SMTP so failed sends can retry.
            record_send(
                engine,
                reviewer["reviewer_id"],
                EMAIL_TYPE_WEEKLY_DIGEST,
                send_date,
                reviewer["email"],
            )
            sent += 1
            logger.info("Sent reviewer digest to %s (%s)", reviewer["name"], reviewer["email"])
        except Exception as exc:
            failed += 1
            logger.error(
                "Failed to send digest to %s (%s): %s",
                reviewer["name"],
                reviewer["email"],
                exc,
                exc_info=True,
            )

    logger.info(
        "Completed weekly distribution. emails_sent=%s emails_failed=%s emails_skipped=%s",
        sent,
        failed,
        skipped,
    )
    if sent == 0 and failed > 0 and skipped == 0:
        logger.error("All email sends failed for %s reviewer(s).", len(reviewers))
        sys.exit(1)


if __name__ == "__main__":
    main()
