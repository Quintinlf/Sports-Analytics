"""Phase 0G — single diagnostic command for production integrity.

Checks:
  DATABASE fingerprint, model artifacts, schedule→model→DB→API chain,
  upcoming-view filters, and emits a per-sport provenance record.

Modes:
  python -m scripts.production_smoke
      Read-only against configured DATABASE_URL (does not insert).

  python -m scripts.production_smoke --ingest-check
      Also runs live sport services (network) and reports discovered/
      predicted counts without writing (unless --write).

  python -m scripts.production_smoke --write
      Persist discovered predictions (use carefully against prod).

Exit 0 if critical checks pass; 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.config  # noqa: F401
from sqlalchemy import text

from data.prediction_time import display_window, pacific_today
from scripts.check_model_artifacts import check_all as check_model_artifacts
from scripts.db_utils import (
    _column_names,
    create_database_engine,
    database_url_source,
    insert_prediction,
    resolve_database_url,
)

_FAILURES: List[str] = []


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    _FAILURES.append(msg)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _db_fingerprint(url: str) -> str:
    parsed = urlparse(url.replace("postgresql+psycopg2://", "postgresql://"))
    host = parsed.hostname or "?"
    db = (parsed.path or "/").lstrip("/") or "?"
    return f"{host}/{db} (via {database_url_source()})"


def _expected_db_ok(url: str) -> None:
    expected = os.getenv("EXPECTED_DB_HOST", "").strip()
    if not expected:
        _warn("EXPECTED_DB_HOST unset — skip DB host match check")
        return
    parsed = urlparse(url.replace("postgresql+psycopg2://", "postgresql://"))
    host = parsed.hostname or ""
    if expected.lower() in host.lower() or host.lower() in expected.lower():
        _ok(f"DATABASE_URL host matches EXPECTED_DB_HOST ({expected})")
    else:
        _fail(f"DATABASE_URL host {host!r} does not match EXPECTED_DB_HOST {expected!r}")


def _query_counts(engine) -> Dict[str, Any]:
    cols = _column_names(engine, "predictions")
    date_col = "game_date_pacific" if "game_date_pacific" in cols else "game_date"
    start, end = display_window()
    with engine.connect() as conn:
        by_sport = conn.execute(
            text(
                "SELECT sport, COUNT(*) AS cnt FROM predictions GROUP BY sport ORDER BY sport"
            )
        ).mappings().all()
        upcoming = conn.execute(
            text(
                f"""
                SELECT sport, COUNT(*) AS cnt FROM predictions
                WHERE UPPER(COALESCE(prediction_status, 'UPCOMING'))
                      NOT IN ('SETTLED', 'VOID', 'FINAL')
                  AND actual_home_score IS NULL
                  AND CAST({date_col} AS TEXT) >= :start
                  AND CAST({date_col} AS TEXT) <= :end
                GROUP BY sport
                """
            ),
            {"start": start, "end": end},
        ).mappings().all()
        stale = conn.execute(
            text(
                f"""
                SELECT sport, COUNT(*) AS cnt FROM predictions
                WHERE UPPER(COALESCE(prediction_status, 'UPCOMING'))
                      IN ('UPCOMING', 'ACTIVE', '')
                  AND actual_home_score IS NULL
                  AND CAST({date_col} AS TEXT) < :start
                GROUP BY sport
                """
            ),
            {"start": start},
        ).mappings().all()
        settled = conn.execute(
            text(
                """
                SELECT sport, COUNT(*) AS cnt FROM predictions
                WHERE UPPER(COALESCE(prediction_status, '')) IN ('SETTLED', 'FINAL')
                   OR actual_home_score IS NOT NULL
                GROUP BY sport
                """
            )
        ).mappings().all()
    return {
        "by_sport": {r["sport"]: int(r["cnt"]) for r in by_sport},
        "upcoming_window": {r["sport"]: int(r["cnt"]) for r in upcoming},
        "stale_upcoming": {r["sport"]: int(r["cnt"]) for r in stale},
        "settled": {r["sport"]: int(r["cnt"]) for r in settled},
        "window": (start, end),
    }


def _api_list_counts() -> Dict[str, int]:
    """Hit list_predictions the same way the UI does (in-process)."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    counts: Dict[str, int] = {}
    for sport in ("NBA", "MLB", "FIFA"):
        resp = client.get(f"/api/feedback/predictions?sport={sport}")
        if resp.status_code != 200:
            _fail(f"API /predictions?sport={sport} → HTTP {resp.status_code}")
            counts[sport] = -1
            continue
        counts[sport] = len(resp.json())
        _ok(f"API UI-visible {sport}: {counts[sport]}")
    return counts


def _ingest_probe(write: bool, engine) -> Dict[str, Dict[str, Any]]:
    from scripts.prediction_runner import create_prediction_service

    svc = create_prediction_service()
    records: Dict[str, Dict[str, Any]] = {}
    for service in svc.services:
        name = service.sport_name
        try:
            rows = service.fetch_upcoming_games()
        except Exception as exc:
            records[name] = {
                "discovered": 0,
                "predicted": 0,
                "persisted": 0,
                "error": str(exc),
                "model": getattr(service, "model_name", "?"),
            }
            _fail(f"{name} ingest: {exc}")
            continue
        predicted = [
            r
            for r in rows
            if not r.get("is_fallback") and r.get("model_name")
        ]
        schedule_only = [r for r in rows if r.get("is_fallback")]
        persisted = 0
        if write:
            for r in rows:
                insert_prediction(engine, r)
                persisted += 1
        model = next(
            (r.get("model_name") for r in predicted if r.get("model_name")),
            "schedule-only" if schedule_only else "n/a (empty)",
        )
        records[name] = {
            "discovered": len(rows),
            "predicted": len(predicted),
            "schedule_only": len(schedule_only),
            "persisted": persisted,
            "model": model,
        }
        _ok(
            f"{name} ingest: discovered={len(rows)} predicted={len(predicted)} "
            f"schedule_only={len(schedule_only)}"
        )
    return records


def _print_provenance(
    db_fp: str,
    counts: Dict[str, Any],
    api_counts: Dict[str, int],
    ingest: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    sport_map = [("NBA", "NBA"), ("MLB", "MLB"), ("FIFA", "SOCCER")]
    print("\n======== PHASE 0 PROVENANCE RECORD ========")
    print(f"Pacific today: {pacific_today()}")
    print(f"Display window: {counts['window'][0]} -> {counts['window'][1]}")
    print(f"Current DB:     {db_fp}")
    for ui, db_key in sport_map:
        ing = (ingest or {}).get(ui) or (ingest or {}).get(
            "FIFA" if ui == "FIFA" else ui, {}
        )
        # FIFA service sport_name may be FIFA
        if ui == "FIFA" and not ing:
            ing = (ingest or {}).get("FIFA", {})
        stale = counts["stale_upcoming"].get(db_key, 0) + (
            counts["stale_upcoming"].get("FIFA", 0) if ui == "FIFA" else 0
        )
        settled_n = counts["settled"].get(db_key, 0)
        upcoming = counts["upcoming_window"].get(db_key, 0)
        if ui == "FIFA":
            upcoming += counts["upcoming_window"].get("FIFA", 0)
            settled_n += counts["settled"].get("FIFA", 0)
        ui_vis = api_counts.get(ui, 0)
        settlement = "verified" if settled_n > 0 else (
            "partial" if upcoming > 0 else "missing"
        )
        print(f"\n{ui}")
        print(f"Games discovered:       {ing.get('discovered', 'n/a')}")
        print(f"Games predicted:        {ing.get('predicted', 'n/a')}")
        print(f"Model:                  {ing.get('model', 'n/a')}")
        print(f"Predictions persisted:  {ing.get('persisted', 'n/a (read-only)')}")
        print(f"Current DB:             {db_fp}")
        print(f"UI-visible:             {ui_vis}")
        print(f"Upcoming in window:     {upcoming}")
        print(f"Stale rows:             {stale}")
        print(f"Settlement status:      {settlement}")
    print("\n===========================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingest-check",
        action="store_true",
        help="Call live sport services (network) for discovered/predicted counts",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist ingest rows (implies --ingest-check)",
    )
    args = parser.parse_args()

    print("Production smoke — Sports Analytics")
    url = resolve_database_url()
    db_fp = _db_fingerprint(url)
    _ok(f"Resolved DATABASE_URL -> {db_fp}")
    _expected_db_ok(url)

    if check_model_artifacts() != 0:
        _fail("Model artifacts incomplete")
    else:
        _ok("Model artifacts present for NBA/MLB/FIFA")

    engine = create_database_engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _ok("DB connectivity")
    except Exception as exc:
        _fail(f"DB connectivity: {exc}")
        return 1

    counts = _query_counts(engine)
    _ok(
        f"Upcoming window rows: {dict(counts['upcoming_window'])}; "
        f"stale: {dict(counts['stale_upcoming'])}"
    )

    ingest = None
    if args.ingest_check or args.write:
        ingest = _ingest_probe(write=args.write, engine=engine)
        if args.write:
            counts = _query_counts(engine)

    api_counts = _api_list_counts()

    # Soft consistency: API should not exceed upcoming window totals
    for ui, db_key in [("NBA", "NBA"), ("MLB", "MLB"), ("FIFA", "SOCCER")]:
        window_n = counts["upcoming_window"].get(db_key, 0)
        if ui == "FIFA":
            window_n += counts["upcoming_window"].get("FIFA", 0)
        api_n = api_counts.get(ui, 0)
        if api_n > window_n and window_n >= 0:
            _warn(
                f"{ui}: API returned {api_n} but window count is {window_n} "
                "(limits/filters may differ)"
            )

    _print_provenance(db_fp, counts, api_counts, ingest)

    if _FAILURES:
        print(f"{len(_FAILURES)} failure(s).")
        return 1
    print("Smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
