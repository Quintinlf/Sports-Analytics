"""End-to-end live prediction pipeline verification.

Proves, for NBA, MLB, and FIFA, that ONE real model-generated prediction
flows unchanged through every layer the product actually uses:

    model output -> database record -> API response
        -> dashboard list endpoint (what frontend/feedback/script.js fetches)
        -> weekly email digest (what scripts/send_weekly_feedback_form.py sends)

It does this by calling the exact same service functions the live cron job
and FastAPI app call in production (data/*_predictions_service.py), feeding
them a synthetic matchup between two real, historically-active teams so the
check doesn't depend on today's actual schedule. A trained-model failure
(ModelUnavailableError) is reported as a hard FAIL for that sport — this
script does not fall back to a hardcoded guess, matching the pipeline's own
fail-loud behavior.

Uses an isolated temp SQLite database (via the same engine/session patch
pattern as tests/test_analyst_phase1.py) — it never reads or writes your
configured DATABASE_URL / sports_analytics.db.

Requires network access (nba_api / MLB-StatsAPI / thesportsdb) to build live
features, and the trained model artifacts already present under
machine_learning/models/.

Usage:
    python scripts/verify_live_pipeline.py               # all three sports
    python scripts/verify_live_pipeline.py --sport nba
    python scripts/verify_live_pipeline.py --sport mlb fifa
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.config  # noqa: F401 — load .env before other backend imports
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from scripts.db_utils import create_database_engine, insert_prediction
from data.prediction_errors import ModelUnavailableError

# Names that must NEVER appear on a row this script inserts — their presence
# means the pipeline silently fell back to a hardcoded guess instead of a
# real model prediction (or failing loudly, per the no-placeholder-fallback
# requirement).
PLACEHOLDER_MODEL_NAMES = {
    "NBA-Live-v1",
    "NBA-Historical-Fallback-v1",
    "MLB-Live-v1",
    "MLB-Fallback-v1",
    "FIFA-Live-v1",
    "FIFA-Fallback-v1",
}

_FAILURES: List[str] = []


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _ok(sport: str, message: str) -> None:
    print(f"[ OK ] {sport}: {message}")


def _fail(sport: str, message: str) -> None:
    print(f"[FAIL] {sport}: {message}")
    _FAILURES.append(f"{sport}: {message}")


def _values_match(model_row: Dict[str, Any], observed: Dict[str, Any], keys: List[str]) -> bool:
    ok = True
    for key in keys:
        expected, actual = model_row.get(key), observed.get(key)
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                matched = abs(float(expected) - float(actual)) < 1e-6
            except (TypeError, ValueError):
                matched = False
        else:
            matched = expected == actual
        if not matched:
            print(f"       mismatch on {key!r}: model={expected!r} vs observed={actual!r}")
            ok = False
    return ok


def _verify_row_through_all_layers(
    engine, client, sport: str, model_row: Dict[str, Any], db_sport: str, ui_sport: str
) -> bool:
    model_name = model_row.get("model_name", "")
    if model_name in PLACEHOLDER_MODEL_NAMES or not model_name:
        _fail(sport, f"model_name={model_name!r} is a placeholder/fallback name, not a real trained model")
        return False
    _ok(
        sport,
        f"model produced a real prediction: {model_row['away_team']} @ {model_row['home_team']} "
        f"-> {model_row['predicted_winner']} ({model_row['win_probability']:.3f}, model={model_name})",
    )

    # --- Database layer ---
    prediction_id = insert_prediction(engine, model_row)
    with engine.begin() as conn:
        db_row = conn.execute(
            text("SELECT * FROM predictions WHERE prediction_id = :pid"),
            {"pid": prediction_id},
        ).mappings().first()
    if db_row is None:
        _fail(sport, "insert_prediction() did not produce a readable row")
        return False
    if not _values_match(
        model_row, dict(db_row), ["predicted_winner", "win_probability", "model_name", "confidence_level"]
    ):
        _fail(sport, "database record does not match model output")
        return False
    _ok(sport, "database record matches model output")

    # --- API layer (detail endpoint) ---
    resp = client.get(f"/api/feedback/predictions/{prediction_id}")
    if resp.status_code != 200:
        _fail(sport, f"GET /api/feedback/predictions/{prediction_id} returned {resp.status_code}")
        return False
    api_row = resp.json()
    if not _values_match(model_row, api_row, ["predicted_winner", "win_probability", "confidence_level", "model_name"]):
        _fail(sport, "API detail response does not match model output")
        return False
    _ok(sport, "API detail response matches model output")

    # --- Dashboard layer: the list endpoint frontend/feedback/script.js fetches ---
    list_resp = client.get(f"/api/feedback/predictions?sport={ui_sport}&limit=50")
    if list_resp.status_code != 200:
        _fail(sport, f"GET /api/feedback/predictions?sport={ui_sport} returned {list_resp.status_code}")
        return False
    match = next((r for r in list_resp.json() if r.get("prediction_id") == prediction_id), None)
    if match is None:
        _fail(sport, "prediction did not appear in the dashboard list endpoint")
        return False
    if not _values_match(model_row, match, ["predicted_winner", "confidence_level"]):
        _fail(sport, "dashboard list endpoint does not match model output")
        return False
    _ok(sport, "dashboard list endpoint (same one the frontend fetches) matches model output")

    # --- Email layer: the same query + template scripts/send_weekly_feedback_form.py uses ---
    from scripts.send_weekly_feedback_form import load_predictions, render_email

    email_rows = load_predictions(engine, db_sport, "UPCOMING", limit=50)
    email_match = next(
        (
            r for r in email_rows
            if r["home_team"] == model_row["home_team"] and r["away_team"] == model_row["away_team"]
        ),
        None,
    )
    if email_match is None:
        _fail(sport, "prediction did not appear in the weekly email digest query (load_predictions)")
        return False
    if email_match.get("confidence_level") != model_row.get("confidence_level"):
        _fail(sport, "email digest confidence_level does not match model output")
        return False

    html = render_email(
        reviewer={"reviewer_id": "verify", "name": "Verify", "favorite_sports": [ui_sport]},
        stats={
            "agreement_pct": 0, "beat_ai_count": 0, "pending_pregame": 0,
            "pending_postgame": 0, "pending_case_studies": 0,
        },
        upcoming=email_rows,
        completed=[],
        base_url="http://localhost:8000",
    )
    if model_row["home_team"] not in html or model_row["away_team"] not in html:
        _fail(sport, "rendered email HTML does not contain the verified matchup")
        return False
    _ok(sport, "email digest (same query + template as scripts/send_weekly_feedback_form.py) matches model output")

    return True


def verify_nba(engine, client) -> bool:
    sport = "NBA"
    from data.nba_predictions_service import NBALivePredictionService, _load_nba_model

    try:
        ensemble = _load_nba_model()
    except ModelUnavailableError as exc:
        _fail(sport, f"model failed to load: {exc}")
        return False

    service = NBALivePredictionService()
    candidate_pairs = [
        ("Los Angeles Lakers", "Boston Celtics"),
        ("Golden State Warriors", "Miami Heat"),
        ("Denver Nuggets", "Milwaukee Bucks"),
    ]
    row = None
    for home, away in candidate_pairs:
        # _predict_one is the same private helper build_prediction_rows() and
        # the historical fallback both call in production — using it directly
        # lets this check target a specific matchup instead of whatever
        # nba_api's live schedule happens to return today.
        row = service._predict_one(
            ensemble=ensemble,
            home_team=home,
            away_team=away,
            game_date=_today(),
            provider_game_id="VERIFY-NBA",
            prediction_status="UPCOMING",
        )
        if row is not None:
            break

    if row is None:
        _fail(sport, "could not build live features for any candidate matchup (insufficient recent game history)")
        return False

    return _verify_row_through_all_layers(engine, client, sport, row, db_sport="NBA", ui_sport="NBA")


def verify_mlb(engine, client) -> bool:
    sport = "MLB"
    from data.mlb_predictions_service import MLBLivePredictionService, _load_mlb_model

    try:
        model = _load_mlb_model()
    except ModelUnavailableError as exc:
        _fail(sport, f"model failed to load: {exc}")
        return False

    service = MLBLivePredictionService()
    synthetic_games = [
        {
            "home_name": home, "away_name": away,
            "game_datetime": f"{_today()}T18:00:00Z",
            "game_id": "VERIFY-MLB", "league": "MLB",
        }
        for home, away in [
            ("New York Yankees", "Boston Red Sox"),
            ("Los Angeles Dodgers", "San Francisco Giants"),
        ]
    ]
    rows = service.build_prediction_rows(synthetic_games, model)
    if not rows:
        _fail(sport, "could not build live features for any candidate matchup (insufficient recent game history)")
        return False

    return _verify_row_through_all_layers(engine, client, sport, rows[0], db_sport="MLB", ui_sport="MLB")


def verify_fifa(engine, client) -> bool:
    sport = "FIFA"
    from data.fifa_predictions_service import FIFALivePredictionService, _load_fifa_model

    try:
        bundle = _load_fifa_model()
    except ModelUnavailableError as exc:
        _fail(sport, f"model failed to load: {exc}")
        return False

    teams = bundle["squad_profiles"]["team"].unique().tolist()
    if len(teams) < 2:
        _fail(sport, "trained squad_profiles has fewer than 2 teams; can't build a synthetic fixture")
        return False

    service = FIFALivePredictionService()
    synthetic_fixtures = [
        {"id": "VERIFY-FIFA", "league": "FIFA World Cup", "home_team": teams[i], "away_team": teams[i + 1], "utc_date": _today()}
        for i in range(min(len(teams) - 1, 10))
    ]
    rows = service.build_prediction_rows(synthetic_fixtures, bundle)
    if not rows:
        _fail(sport, "no synthetic fixture had squad profiles the model could score")
        return False

    return _verify_row_through_all_layers(engine, client, sport, rows[0], db_sport="SOCCER", ui_sport="FIFA")


@contextmanager
def _isolated_app_client():
    """Build a FastAPI TestClient wired to a throwaway temp SQLite DB.

    Mirrors the isolation pattern in tests/test_analyst_phase1.py so this
    script never touches the real configured database.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.routes.feedback import init_platform

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "verify_pipeline.db")
        engine = create_database_engine(f"sqlite:///{db_path}")
        init_platform(engine)

        session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        @contextmanager
        def _session():
            db = session_factory()
            try:
                yield db
            finally:
                db.close()

        with patch("backend.routes.feedback.engine", engine), \
             patch("backend.routes.feedback.get_db_session", _session), \
             patch("backend.main.engine", engine):
            client = TestClient(app)
            try:
                yield engine, client
            finally:
                engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the live prediction pipeline end-to-end.")
    parser.add_argument(
        "--sport", nargs="+", choices=["nba", "mlb", "fifa"], default=["nba", "mlb", "fifa"],
        help="Which sport(s) to verify (default: all three).",
    )
    args = parser.parse_args()

    verifiers = {"nba": verify_nba, "mlb": verify_mlb, "fifa": verify_fifa}

    print("=" * 70)
    print("LIVE PREDICTION PIPELINE VERIFICATION")
    print(f"Sports: {', '.join(s.upper() for s in args.sport)}")
    print("Using an isolated temp database — your configured DATABASE_URL is untouched.")
    print("=" * 70)

    with _isolated_app_client() as (engine, client):
        results = {}
        for sport in args.sport:
            print()
            results[sport] = verifiers[sport](engine, client)

    print()
    print("=" * 70)
    for sport, passed in results.items():
        print(f"{sport.upper():6s} {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    if not all(results.values()) or _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1

    print("\nAll layers agree: model output, database, API, dashboard, and email all use the same live prediction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
