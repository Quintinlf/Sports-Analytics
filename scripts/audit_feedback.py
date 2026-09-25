"""Phase 2 — audit reviewer feedback / challenge outcome integrity.

Read-only report against the configured DATABASE_URL. Always exits 0.

Usage:
  python -m scripts.audit_feedback
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.config  # noqa: F401 — load .env before database URL resolution

from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.analyst_challenge import evaluate_challenge
from scripts.db_utils import (
    _table_exists,
    create_database_engine,
    format_database_target,
    resolve_database_url,
    sql_case_bool_true,
)

KNOWN_REVIEWER_IDS = (
    "quintin",
    "lamar",
    "melissa",
    "alex",
    "timothy",
    "anderson",
    "luis",
)


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return bool(value)


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _per_reviewer_stats(engine: Engine) -> None:
    _section("Per-reviewer stats")
    if not _table_exists(engine, "reviewers"):
        print("reviewers table missing — skip")
        return

    agree_expr = sql_case_bool_true("pr.agree_with_model", engine)
    if engine.dialect.name == "postgresql":
        disagree_expr = "CASE WHEN pr.agree_with_model IS FALSE THEN 1 ELSE 0 END"
    else:
        disagree_expr = (
            "CASE WHEN pr.agree_with_model IS NOT NULL "
            "AND pr.agree_with_model = 0 THEN 1 ELSE 0 END"
        )
    beat_expr = sql_case_bool_true("ro.reviewer_beat_model", engine)

    with engine.connect() as conn:
        by_id: Dict[str, Dict[str, Any]] = {}
        for r in conn.execute(
            text(
                """
                SELECT r.reviewer_id, r.name, r.email,
                       r.analyst_role, rp.emails_enabled
                FROM reviewers r
                LEFT JOIN reviewer_preferences rp ON rp.reviewer_id = r.reviewer_id
                """
            )
        ).mappings().all():
            by_id[str(r["reviewer_id"]).lower()] = dict(r)

        has_reviews = _table_exists(engine, "prediction_reviews")
        has_outcomes = _table_exists(engine, "review_outcomes")

        for rid in KNOWN_REVIEWER_IDS:
            row = by_id.get(rid)
            if not row:
                print(f"{rid}: (not present)")
                continue

            if has_reviews:
                join_outcomes = (
                    "LEFT JOIN review_outcomes ro ON ro.review_id = pr.review_id"
                    if has_outcomes
                    else ""
                )
                outcome_count_expr = (
                    "COUNT(ro.review_id)" if has_outcomes else "0"
                )
                beat_sum = (
                    f"COALESCE(SUM({beat_expr}), 0)" if has_outcomes else "0"
                )
                stats = conn.execute(
                    text(
                        f"""
                        SELECT
                            COUNT(pr.review_id) AS review_count,
                            {outcome_count_expr} AS outcome_count,
                            COALESCE(SUM({agree_expr}), 0) AS agree_count,
                            COALESCE(SUM({disagree_expr}), 0) AS disagree_count,
                            {beat_sum} AS beat_ai,
                            MAX(pr.created_at) AS last_review_at
                        FROM prediction_reviews pr
                        {join_outcomes}
                        WHERE pr.reviewer_id = :rid
                        """
                    ),
                    {"rid": row["reviewer_id"]},
                ).mappings().first() or {}
            else:
                stats = {
                    "review_count": 0,
                    "outcome_count": 0,
                    "agree_count": 0,
                    "disagree_count": 0,
                    "beat_ai": 0,
                    "last_review_at": None,
                }

            print(
                f"{row['reviewer_id']}: "
                f"reviews={stats.get('review_count', 0)} "
                f"outcomes={stats.get('outcome_count', 0)} "
                f"agree={stats.get('agree_count', 0)} "
                f"disagree={stats.get('disagree_count', 0)} "
                f"beat_ai={stats.get('beat_ai', 0)} "
                f"last_review_at={stats.get('last_review_at')!s} "
                f"email={row.get('email')!r} "
                f"role={row.get('analyst_role')!r} "
                f"emails_enabled={row.get('emails_enabled')!r}"
            )


def _table_totals(engine: Engine) -> None:
    _section("Table totals")
    with engine.connect() as conn:
        for table in ("prediction_reviews", "review_outcomes", "email_send_log"):
            if not _table_exists(engine, table):
                print(f"{table}: (table missing)")
                continue
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
            print(f"{table}: {count}")


def _sample_outcome_revalidate(engine: Engine, limit: int = 20) -> None:
    _section(f"Outcome re-validation (up to {limit})")
    if not _table_exists(engine, "review_outcomes"):
        print("review_outcomes table missing — skip")
        return
    if not _table_exists(engine, "prediction_reviews"):
        print("prediction_reviews table missing — skip")
        return
    if not _table_exists(engine, "predictions"):
        print("predictions table missing — skip")
        return

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    ro.review_id,
                    ro.model_correct,
                    ro.reviewer_correct,
                    ro.reviewer_beat_model,
                    pr.agree_with_model,
                    pr.reviewer_pick,
                    p.predicted_winner,
                    p.actual_winner,
                    p.correct AS model_correct_flag
                FROM review_outcomes ro
                JOIN prediction_reviews pr ON pr.review_id = ro.review_id
                JOIN predictions p ON p.prediction_id = pr.prediction_id
                WHERE p.actual_winner IS NOT NULL
                ORDER BY ro.resolved_at DESC NULLS LAST
                LIMIT :lim
                """
                if engine.dialect.name == "postgresql"
                else """
                SELECT
                    ro.review_id,
                    ro.model_correct,
                    ro.reviewer_correct,
                    ro.reviewer_beat_model,
                    pr.agree_with_model,
                    pr.reviewer_pick,
                    p.predicted_winner,
                    p.actual_winner,
                    p.correct AS model_correct_flag
                FROM review_outcomes ro
                JOIN prediction_reviews pr ON pr.review_id = ro.review_id
                JOIN predictions p ON p.prediction_id = pr.prediction_id
                WHERE p.actual_winner IS NOT NULL
                ORDER BY ro.resolved_at DESC
                LIMIT :lim
                """
            ),
            {"lim": limit},
        ).mappings().all()

    if not rows:
        print("No settled review_outcomes to check.")
        return

    mismatches = 0
    checked = 0
    for row in rows:
        checked += 1
        expected = evaluate_challenge(
            agree_with_model=bool(row["agree_with_model"]),
            reviewer_pick=row["reviewer_pick"] or "",
            predicted_winner=row["predicted_winner"] or "",
            actual_winner=row["actual_winner"] or "",
            model_correct_flag=row["model_correct_flag"],
        )
        stored = {
            "model_correct": _as_bool(row["model_correct"]),
            "reviewer_correct": _as_bool(row["reviewer_correct"]),
            "reviewer_beat_model": _as_bool(row["reviewer_beat_model"]),
        }
        want = {
            "model_correct": bool(expected["model_correct"]),
            "reviewer_correct": bool(expected["reviewer_correct"]),
            "reviewer_beat_model": bool(expected["reviewer_beat_model"]),
        }
        if (
            stored["model_correct"] != want["model_correct"]
            or stored["reviewer_correct"] != want["reviewer_correct"]
            or stored["reviewer_beat_model"] != want["reviewer_beat_model"]
        ):
            mismatches += 1
            print(
                f"MISMATCH review_id={row['review_id']}: "
                f"stored={stored} expected={want} "
                f"pick={row['reviewer_pick']!r} ai={row['predicted_winner']!r} "
                f"actual={row['actual_winner']!r}"
            )

    print(f"Checked: {checked}")
    print(f"Mismatches: {mismatches}")
    if mismatches == 0:
        print("All sampled outcomes match evaluate_challenge.")


def main() -> None:
    try:
        db_url = resolve_database_url(default=None, required=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        # Audit tool: still exit 0
        print("Exiting 0 (audit tool).")
        sys.exit(0)

    print("Phase 2 feedback audit")
    print(f"Database: {format_database_target(db_url)}")
    engine = create_database_engine(db_url)
    try:
        _per_reviewer_stats(engine)
        _table_totals(engine)
        _sample_outcome_revalidate(engine, limit=20)
        _section("Done")
        print("Audit complete (exit 0).")
    finally:
        engine.dispose()
    sys.exit(0)


if __name__ == "__main__":
    main()
