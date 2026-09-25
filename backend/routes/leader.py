"""Leader observability API — Phase 1.

Endpoints under /api/leader/ (admin key or reviewer with analyst_role=leader).
"""
from __future__ import annotations

import json
import logging
import hmac
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, Query
from sqlalchemy import inspect as sa_inspect, text

from backend.db import engine, get_db_session
from backend.routes.feedback import _parse_snapshot, _reviewer_stats, _ui_sport
from data.prediction_time import display_window, pacific_today
from scripts.db_utils import (
    _column_names,
    database_url_source,
    resolve_database_url,
    sql_case_bool_true,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/leader", tags=["leader-observability"])

_UI_SPORTS = ("NBA", "MLB", "FIFA")
_FEATURE_SNAPSHOT_MAX_CHARS = 8000


def _require_leader(
    session,
    reviewer_id: Optional[str],
    x_admin_key: Optional[str],
) -> None:
    """Leader data (reviewer emails, activity, provenance) needs the secret key.

    A reviewer id is a public identifier -- the page itself puts it in the URL --
    so it can never be proof of access on its own. ``reviewer_id`` is kept in the
    signature so existing callers still work, but only ``X-Admin-Key`` matching
    ADMIN_API_KEY grants access. With no key configured, access is refused.
    """
    expected = os.getenv("ADMIN_API_KEY", "").strip()
    supplied = (x_admin_key or "").strip()
    if expected and supplied and hmac.compare_digest(supplied, expected):
        return
    raise HTTPException(status_code=403, detail="Leader access requires the leader key")


def _db_fingerprint(url: str) -> str:
    parsed = urlparse(url.replace("postgresql+psycopg2://", "postgresql://"))
    host = parsed.hostname or "?"
    db = (parsed.path or "/").lstrip("/") or "?"
    return f"{host}/{db} (via {database_url_source()})"


def _table_exists(table: str) -> bool:
    try:
        return table in sa_inspect(engine).get_table_names()
    except Exception:
        return False


def _map_pipeline_sport(sport: Optional[str]) -> str:
    if not sport:
        return "UNKNOWN"
    upper = sport.upper()
    if upper in ("SOCCER", "FIFA"):
        return "FIFA"
    return upper


def _latest_pipeline_by_sport(session) -> Dict[str, Dict[str, Any]]:
    result = {
        s: {
            "status": "unknown",
            "last_run_at": None,
            "detail": None,
            "predictions_count": 0,
        }
        for s in _UI_SPORTS
    }
    if not _table_exists("pipeline_run_log"):
        return result
    try:
        rows = session.execute(
            text(
                """
                SELECT run_id, sport, status, error_message,
                       predictions_count, run_at
                FROM pipeline_run_log
                ORDER BY run_id DESC
                LIMIT 200
                """
            )
        ).mappings().all()
    except Exception as exc:
        logger.warning("pipeline_run_log query failed: %s", exc)
        return result

    seen = set()
    for row in rows:
        ui = _map_pipeline_sport(row.get("sport"))
        if ui not in result or ui in seen:
            continue
        seen.add(ui)
        result[ui] = {
            "status": str(row.get("status") or "unknown").lower(),
            "last_run_at": str(row["run_at"]) if row.get("run_at") is not None else None,
            "detail": row.get("error_message"),
            "predictions_count": int(row.get("predictions_count") or 0),
        }
        if len(seen) >= len(_UI_SPORTS):
            break
    return result


def _date_col(cols: set[str]) -> str:
    return "game_date_pacific" if "game_date_pacific" in cols else "game_date"


def _upcoming_where(cols: set[str]) -> tuple[str, dict]:
    start, end = display_window(days_ahead=int(os.getenv("DASHBOARD_DAYS_AHEAD", "7")))
    date_expr = _date_col(cols)
    where = f"""
        UPPER(COALESCE(prediction_status, 'UPCOMING')) NOT IN ('SETTLED', 'VOID', 'FINAL')
        AND actual_home_score IS NULL
        AND CAST({date_expr} AS TEXT) >= :win_start
        AND CAST({date_expr} AS TEXT) <= :win_end
    """
    return where, {"win_start": start, "win_end": end}


def _settled_where() -> str:
    return """
        UPPER(COALESCE(prediction_status, '')) IN ('SETTLED', 'FINAL')
        OR actual_home_score IS NOT NULL
    """


def _is_correct_expr(cols: set[str]) -> str:
    if "correct" in cols:
        case_correct = sql_case_bool_true("correct", engine)
        return f"""
            CASE
                WHEN correct IS NOT NULL THEN {case_correct}
                WHEN actual_winner IS NOT NULL AND predicted_winner IS NOT NULL THEN
                    CASE WHEN actual_winner = predicted_winner THEN 1 ELSE 0 END
                ELSE 0
            END
        """
    return """
        CASE
            WHEN actual_winner IS NOT NULL AND predicted_winner IS NOT NULL THEN
                CASE WHEN actual_winner = predicted_winner THEN 1 ELSE 0 END
            ELSE 0
        END
    """


def _truncate_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    if not snap:
        return {}
    raw = json.dumps(snap, default=str)
    if len(raw) <= _FEATURE_SNAPSHOT_MAX_CHARS:
        return snap
    return {
        "_truncated": True,
        "_original_chars": len(raw),
        "keys": sorted(snap.keys())[:40],
        "preview": raw[:_FEATURE_SNAPSHOT_MAX_CHARS],
    }


def _health_payload(session) -> Dict[str, Any]:
    db_url = resolve_database_url()
    fingerprint = _db_fingerprint(db_url)
    today = pacific_today()

    db_ok = False
    try:
        session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    pipelines = _latest_pipeline_by_sport(session)
    cols = _column_names(engine, "predictions") if db_ok else set()
    date_expr = _date_col(cols) if cols else "game_date"
    upcoming_where, window_params = _upcoming_where(cols) if cols else ("1=0", {})

    predictions_today = 0
    stale_predictions = 0
    settled_count = 0
    upcoming_by_sport = {s: 0 for s in _UI_SPORTS}

    if db_ok and cols:
        try:
            created_filter = ""
            if "created_at" in cols:
                created_filter = "OR CAST(created_at AS TEXT) LIKE :today_prefix"
            predictions_today = int(
                session.execute(
                    text(
                        f"""
                        SELECT COUNT(*) FROM predictions
                        WHERE ({upcoming_where})
                           {created_filter}
                        """
                    ),
                    {**window_params, "today_prefix": f"{today}%"},
                ).scalar()
                or 0
            )
            stale_predictions = int(
                session.execute(
                    text(
                        f"""
                        SELECT COUNT(*) FROM predictions
                        WHERE UPPER(COALESCE(prediction_status, 'UPCOMING'))
                              IN ('UPCOMING', 'ACTIVE', '')
                          AND actual_home_score IS NULL
                          AND CAST({date_expr} AS TEXT) < :win_start
                        """
                    ),
                    {"win_start": window_params["win_start"]},
                ).scalar()
                or 0
            )
            settled_count = int(
                session.execute(
                    text(f"SELECT COUNT(*) FROM predictions WHERE {_settled_where()}")
                ).scalar()
                or 0
            )
            sport_rows = session.execute(
                text(
                    f"""
                    SELECT sport, COUNT(*) AS cnt FROM predictions
                    WHERE {upcoming_where}
                    GROUP BY sport
                    """
                ),
                window_params,
            ).mappings().all()
            for row in sport_rows:
                ui = _map_pipeline_sport(row.get("sport"))
                if ui in upcoming_by_sport:
                    upcoming_by_sport[ui] += int(row["cnt"] or 0)
        except Exception as exc:
            logger.warning("health prediction counts failed: %s", exc)

    last_successful_run = None
    if _table_exists("pipeline_run_log"):
        try:
            row = session.execute(
                text(
                    """
                    SELECT run_at FROM pipeline_run_log
                    WHERE LOWER(status) = 'ok'
                    ORDER BY run_id DESC
                    LIMIT 1
                    """
                )
            ).first()
            if row and row[0] is not None:
                last_successful_run = str(row[0])
        except Exception:
            last_successful_run = None

    last_email_run = None
    if _table_exists("email_send_log"):
        try:
            row = session.execute(
                text(
                    """
                    SELECT sent_at FROM email_send_log
                    ORDER BY sent_at DESC
                    LIMIT 1
                    """
                )
            ).first()
            if row and row[0] is not None:
                last_email_run = str(row[0])
        except Exception:
            last_email_run = None

    if not db_ok:
        settlement_status = "unknown"
    elif stale_predictions > 0:
        settlement_status = "lagging"
    else:
        settlement_status = "ok"

    return {
        "db_ok": db_ok,
        "db_fingerprint": fingerprint,
        "pacific_today": today,
        "pipelines": pipelines,
        "predictions_today": predictions_today,
        "stale_predictions": stale_predictions,
        "settled_count": settled_count,
        "settlement_status": settlement_status,
        "last_successful_run": last_successful_run,
        "last_email_run": last_email_run,
        "upcoming_by_sport": upcoming_by_sport,
    }


def _accuracy_bucket(settled: int, correct: int) -> Dict[str, Any]:
    return {
        "settled": settled,
        "correct": correct,
        "accuracy_pct": round((correct / settled * 100) if settled else 0.0, 1),
    }


def _performance_payload(session) -> Dict[str, Any]:
    cols = _column_names(engine, "predictions")
    correct_expr = _is_correct_expr(cols)
    settled_where = _settled_where()
    has_prob = "win_probability" in cols
    has_conf = "confidence_level" in cols

    row = session.execute(
        text(
            f"""
            SELECT
                COUNT(*) AS settled,
                COALESCE(SUM({correct_expr}), 0) AS correct
            FROM predictions
            WHERE {settled_where}
            """
        )
    ).mappings().first() or {}
    settled = int(row.get("settled") or 0)
    correct = int(row.get("correct") or 0)
    overall = _accuracy_bucket(settled, correct)

    if has_prob and settled:
        try:
            brier_rows = session.execute(
                text(
                    f"""
                    SELECT win_probability, predicted_winner, actual_winner
                    FROM predictions
                    WHERE ({settled_where})
                      AND win_probability IS NOT NULL
                      AND actual_winner IS NOT NULL
                    """
                )
            ).mappings().all()
            sq_err = []
            for r in brier_rows:
                prob = float(r["win_probability"])
                outcome = 1.0 if r["actual_winner"] == r["predicted_winner"] else 0.0
                sq_err.append((prob - outcome) ** 2)
            if sq_err:
                overall["brier_approx"] = round(sum(sq_err) / len(sq_err), 4)
        except Exception as exc:
            logger.warning("brier_approx failed: %s", exc)

    by_sport_rows = session.execute(
        text(
            f"""
            SELECT sport,
                   COUNT(*) AS settled,
                   COALESCE(SUM({correct_expr}), 0) AS correct
            FROM predictions
            WHERE {settled_where}
            GROUP BY sport
            """
        )
    ).mappings().all()
    by_sport: List[Dict[str, Any]] = []
    for r in by_sport_rows:
        bucket = _accuracy_bucket(int(r["settled"] or 0), int(r["correct"] or 0))
        bucket["sport"] = _ui_sport(r.get("sport"))
        by_sport.append(bucket)
    by_sport.sort(key=lambda x: x["sport"])

    by_confidence: List[Dict[str, Any]] = []
    if has_conf:
        conf_rows = session.execute(
            text(
                f"""
                SELECT UPPER(COALESCE(confidence_level, 'UNKNOWN')) AS confidence_level,
                       COUNT(*) AS settled,
                       COALESCE(SUM({correct_expr}), 0) AS correct
                FROM predictions
                WHERE {settled_where}
                GROUP BY UPPER(COALESCE(confidence_level, 'UNKNOWN'))
                """
            )
        ).mappings().all()
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        for r in conf_rows:
            bucket = _accuracy_bucket(int(r["settled"] or 0), int(r["correct"] or 0))
            bucket["confidence_level"] = r["confidence_level"]
            by_confidence.append(bucket)
        by_confidence.sort(key=lambda x: order.get(x["confidence_level"], 99))

    return {
        "overall": overall,
        "by_sport": by_sport,
        "by_confidence": by_confidence,
    }


def _reviewers_payload(session) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT reviewer_id, name, email, analyst_role, onboarding_completed_at
            FROM reviewers
            ORDER BY name ASC
            """
        )
    ).mappings().all()

    last_review_map: Dict[str, Optional[str]] = {}
    try:
        lr_rows = session.execute(
            text(
                """
                SELECT reviewer_id, MAX(created_at) AS last_review_at
                FROM prediction_reviews
                GROUP BY reviewer_id
                """
            )
        ).mappings().all()
        for r in lr_rows:
            last_review_map[r["reviewer_id"]] = (
                str(r["last_review_at"]) if r.get("last_review_at") is not None else None
            )
    except Exception:
        last_review_map = {}

    result = []
    for row in rows:
        rid = row["reviewer_id"]
        stats = _reviewer_stats(session, rid)
        result.append(
            {
                "reviewer_id": rid,
                "name": row.get("name"),
                "email": row.get("email"),
                "analyst_role": row.get("analyst_role") or "analyst",
                "onboarding_completed_at": (
                    str(row["onboarding_completed_at"])
                    if row.get("onboarding_completed_at") is not None
                    else None
                ),
                "last_review_at": last_review_map.get(rid),
                "stats": stats,
            }
        )
    return result


def _failures_payload(session) -> Dict[str, Any]:
    pipeline_failures: List[Dict[str, Any]] = []
    recent_pipeline_runs: List[Dict[str, Any]] = []
    email_log: List[Dict[str, Any]] = []

    def _run_row(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": r.get("run_id"),
            "sport": _map_pipeline_sport(r.get("sport")),
            "status": r.get("status"),
            "error_message": r.get("error_message"),
            "predictions_count": int(r.get("predictions_count") or 0),
            "run_at": str(r["run_at"]) if r.get("run_at") is not None else None,
        }

    if _table_exists("pipeline_run_log"):
        try:
            recent_pipeline_runs = [
                _run_row(dict(r))
                for r in session.execute(
                    text(
                        """
                        SELECT run_id, sport, status, error_message,
                               predictions_count, run_at
                        FROM pipeline_run_log
                        ORDER BY run_id DESC
                        LIMIT 25
                        """
                    )
                ).mappings().all()
            ]
            pipeline_failures = [
                _run_row(dict(r))
                for r in session.execute(
                    text(
                        """
                        SELECT run_id, sport, status, error_message,
                               predictions_count, run_at
                        FROM pipeline_run_log
                        WHERE LOWER(status) NOT IN ('ok', 'success')
                        ORDER BY run_id DESC
                        LIMIT 25
                        """
                    )
                ).mappings().all()
            ]
        except Exception as exc:
            logger.warning("pipeline failures query failed: %s", exc)

    if _table_exists("email_send_log"):
        try:
            email_log = [
                {
                    "reviewer_id": r.get("reviewer_id"),
                    "email_type": r.get("email_type"),
                    "send_date": str(r["send_date"]) if r.get("send_date") is not None else None,
                    "email": r.get("email"),
                    "sent_at": str(r["sent_at"]) if r.get("sent_at") is not None else None,
                }
                for r in session.execute(
                    text(
                        """
                        SELECT reviewer_id, email_type, send_date, email, sent_at
                        FROM email_send_log
                        ORDER BY sent_at DESC
                        LIMIT 25
                        """
                    )
                ).mappings().all()
            ]
        except Exception as exc:
            logger.warning("email_send_log query failed: %s", exc)

    return {
        "pipeline_failures": pipeline_failures,
        "recent_pipeline_runs": recent_pipeline_runs,
        "email_log": email_log,
    }


def _provenance_payload(session, prediction_id: int) -> Dict[str, Any]:
    cols = _column_names(engine, "predictions")
    wanted = [
        "prediction_id",
        "sport",
        "home_team",
        "away_team",
        "game_date",
        "start_time_utc",
        "game_date_pacific",
        "predicted_winner",
        "win_probability",
        "confidence_level",
        "model_name",
        "model_version",
        "data_source",
        "is_fallback",
        "prediction_status",
        "game_status",
        "pipeline_run_id",
        "created_at",
        "feature_snapshot",
        "provider_game_id",
    ]
    select_cols = [c for c in wanted if c in cols or c == "prediction_id"]
    # Always include core identity cols that exist on base schema
    for required in ("sport", "home_team", "away_team", "game_date", "predicted_winner"):
        if required in cols and required not in select_cols:
            select_cols.append(required)

    row = session.execute(
        text(
            f"""
            SELECT {", ".join(select_cols)}
            FROM predictions
            WHERE prediction_id = :pid
            """
        ),
        {"pid": prediction_id},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    d = dict(row)
    snap = _parse_snapshot(d.pop("feature_snapshot", None) if "feature_snapshot" in d else None)

    def _s(key: str) -> Optional[str]:
        val = d.get(key)
        return str(val) if val is not None else None

    return {
        "sport": _ui_sport(d.get("sport")),
        "home_team": d.get("home_team"),
        "away_team": d.get("away_team"),
        "game_date": _s("game_date"),
        "start_time_utc": _s("start_time_utc"),
        "game_date_pacific": _s("game_date_pacific"),
        "predicted_winner": d.get("predicted_winner"),
        "win_probability": d.get("win_probability"),
        "confidence_level": d.get("confidence_level"),
        "model_name": d.get("model_name"),
        "model_version": d.get("model_version"),
        "data_source": d.get("data_source"),
        "is_fallback": bool(d.get("is_fallback")) if d.get("is_fallback") is not None else None,
        "prediction_status": d.get("prediction_status"),
        "game_status": d.get("game_status"),
        "pipeline_run_id": d.get("pipeline_run_id"),
        "created_at": _s("created_at"),
        "feature_snapshot": _truncate_snapshot(snap),
        "provider_game_id": d.get("provider_game_id"),
    }


@router.get("/health")
def leader_health(
    reviewer_id: Optional[str] = Query(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    with get_db_session() as session:
        _require_leader(session, reviewer_id, x_admin_key)
        return _health_payload(session)


@router.get("/performance")
def leader_performance(
    reviewer_id: Optional[str] = Query(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    with get_db_session() as session:
        _require_leader(session, reviewer_id, x_admin_key)
        return _performance_payload(session)


@router.get("/reviewers")
def leader_reviewers(
    reviewer_id: Optional[str] = Query(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    with get_db_session() as session:
        _require_leader(session, reviewer_id, x_admin_key)
        return {"reviewers": _reviewers_payload(session)}


@router.get("/failures")
def leader_failures(
    reviewer_id: Optional[str] = Query(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    with get_db_session() as session:
        _require_leader(session, reviewer_id, x_admin_key)
        return _failures_payload(session)


@router.get("/predictions/{prediction_id}/provenance")
def leader_prediction_provenance(
    prediction_id: int,
    reviewer_id: Optional[str] = Query(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    with get_db_session() as session:
        _require_leader(session, reviewer_id, x_admin_key)
        return _provenance_payload(session, prediction_id)
