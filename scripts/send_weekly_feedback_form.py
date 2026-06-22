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
from typing import Any, Dict, List

from sqlalchemy import create_engine, text

logger = logging.getLogger("weekly_feedback_distribution")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


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


def load_reviewers(engine) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT r.reviewer_id, r.name, r.email,
               rp.favorite_sports, rp.emails_enabled
        FROM reviewers r
        LEFT JOIN reviewer_preferences rp ON rp.reviewer_id = r.reviewer_id
        WHERE r.email IS NOT NULL
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()
    result = []
    for row in rows:
        if row["emails_enabled"] is not None and not bool(row["emails_enabled"]):
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
    with engine.begin() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM prediction_reviews WHERE reviewer_id = :rid"),
            {"rid": reviewer_id},
        ).scalar() or 0
        agreed = conn.execute(
            text("SELECT COUNT(*) FROM prediction_reviews WHERE reviewer_id = :rid AND agree_with_model = 1"),
            {"rid": reviewer_id},
        ).scalar() or 0
        beat_ai = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM review_outcomes ro
                JOIN prediction_reviews pr ON ro.review_id = pr.review_id
                WHERE pr.reviewer_id = :rid AND ro.reviewer_beat_model = 1
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
    agree_pct = round((agreed / total) * 100, 1) if total else 0.0
    return {
        "agreement_pct": agree_pct,
        "beat_ai_count": int(beat_ai),
        "pending_pregame": int(pending_pregame),
        "pending_postgame": int(pending_postgame),
    }


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


def render_email(reviewer: Dict[str, Any], stats: Dict[str, Any], upcoming: List[Dict[str, Any]], completed: List[Dict[str, Any]], base_url: str) -> str:
    favorite = reviewer["favorite_sports"][0] if reviewer["favorite_sports"] else "MLB"
    cta_url = f"{base_url}/feedback?reviewer_id={reviewer['reviewer_id']}&sport={favorite}"

    def _rows(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "<li>No games available in this section.</li>"
        return "".join(
            f"<li>{g['game_date']} - {g['away_team']} @ {g['home_team']} ({g.get('confidence_level','N/A')})</li>"
            for g in items
        )

    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#f7f7f7;padding:20px">
      <div style="max-width:680px;margin:0 auto;background:#fff;padding:20px;border-radius:10px">
        <h2>Weekly Analyst Digest: {reviewer['name']}</h2>
        <p>Favorite sports: {", ".join(reviewer["favorite_sports"])}</p>
        <ul>
          <li>Agreement %: {stats['agreement_pct']}%</li>
          <li>Beat AI count: {stats['beat_ai_count']}</li>
          <li>Pending pregame reviews: {stats['pending_pregame']}</li>
          <li>Pending postgame reflections: {stats['pending_postgame']}</li>
        </ul>
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
    base_url = os.getenv("FEEDBACK_BASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is required.")
        sys.exit(1)
    if not base_url:
        logger.error("FEEDBACK_BASE_URL is required.")
        sys.exit(1)

    engine = create_engine(db_url)
    reviewers = load_reviewers(engine)
    if not reviewers:
        logger.warning("No reviewers with email enabled were found.")
        sys.exit(0)

    sent = 0
    for reviewer in reviewers:
        favorite = reviewer["favorite_sports"][0] if reviewer["favorite_sports"] else "MLB"
        stats = load_reviewer_stats(engine, reviewer["reviewer_id"])
        upcoming = load_predictions(engine, "SOCCER" if favorite == "FIFA" else favorite, "UPCOMING")
        completed = load_predictions(engine, "SOCCER" if favorite == "FIFA" else favorite, "FINAL")
        html = render_email(reviewer, stats, upcoming, completed, base_url)
        send_email(
            to_email=reviewer["email"],
            subject=f"Weekly AI Analyst Digest - {datetime.utcnow().strftime('%Y-%m-%d')}",
            html_content=html,
        )
        sent += 1
        logger.info("Sent reviewer digest to %s (%s)", reviewer["name"], reviewer["email"])

    logger.info("Completed weekly distribution. emails_sent=%s", sent)


if __name__ == "__main__":
    main()
