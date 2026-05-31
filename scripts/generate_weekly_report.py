"""Weekly report generator.

Aggregates past 7 days of predictions, computes metrics, and writes HTML email.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import text

from reports.weekly_email_template import render_weekly_report
from scripts.db_utils import create_database_engine

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"weekly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def _require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value

def _fetch_weekly_predictions(engine) -> pd.DataFrame:
    cutoff = datetime.utcnow().date() - timedelta(days=7)
    sql = """
    SELECT
        prediction_id,
        game_date,
        home_team,
        away_team,
        win_probability,
        predicted_spread,
        confidence_level,
        model_version,
        actual_home_score,
        actual_away_score
    FROM mlb_predictions
    WHERE game_date >= :cutoff
    ORDER BY game_date ASC;
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"cutoff": cutoff}).mappings().all()
    return pd.DataFrame(rows)


def _compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    if total == 0:
        return {
            "n_predictions": 0,
            "accuracy": 0.0,
            "brier_score": 0.0,
            "calibration_gap": 0.0,
        }

    df = df.copy()
    df["has_actuals"] = df["actual_home_score"].notna() & df["actual_away_score"].notna()
    df_actuals = df[df["has_actuals"]]

    if df_actuals.empty:
        return {
            "n_predictions": total,
            "accuracy": 0.0,
            "brier_score": 0.0,
            "calibration_gap": 0.0,
        }

    df_actuals["home_win"] = df_actuals["actual_home_score"] > df_actuals["actual_away_score"]
    df_actuals["pred_home_win"] = df_actuals["win_probability"] >= 0.5
    accuracy = (df_actuals["home_win"] == df_actuals["pred_home_win"]).mean()

    actual = df_actuals["home_win"].astype(int)
    brier_score = ((df_actuals["win_probability"] - actual) ** 2).mean()

    df_actuals["bucket"] = pd.cut(
        df_actuals["win_probability"],
        bins=[0.0, 0.4, 0.5, 0.6, 0.7, 1.0],
        include_lowest=True,
    )
    calibration_gap = 0.0
    for _, group in df_actuals.groupby("bucket"):
        if group.empty:
            continue
        predicted = group["win_probability"].mean()
        observed = group["home_win"].mean()
        calibration_gap = max(calibration_gap, abs(predicted - observed))

    return {
        "n_predictions": total,
        "accuracy": float(accuracy),
        "brier_score": float(brier_score),
        "calibration_gap": float(calibration_gap),
    }


def _identify_failures(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df.copy()
    df["has_actuals"] = df["actual_home_score"].notna() & df["actual_away_score"].notna()
    df_actuals = df[df["has_actuals"]]
    if df_actuals.empty:
        return []

    df_actuals["home_win"] = df_actuals["actual_home_score"] > df_actuals["actual_away_score"]
    df_actuals["pred_home_win"] = df_actuals["win_probability"] >= 0.5
    df_actuals["correct"] = df_actuals["home_win"] == df_actuals["pred_home_win"]

    failures = df_actuals[~df_actuals["correct"]]
    failures = failures.sort_values("win_probability", ascending=False)
    top = failures.head(5)
    return top.to_dict("records")


def _derive_patterns(df: pd.DataFrame, failures: List[Dict[str, Any]]) -> List[str]:
    patterns: List[str] = []
    total = len(df)
    if total == 0:
        return ["Not enough data to identify patterns."]

    missing_actuals = df["actual_home_score"].isna().sum()
    if missing_actuals:
        patterns.append(f"{missing_actuals} games missing final scores.")

    if failures:
        high_prob = [f for f in failures if (f.get("win_probability") or 0) >= 0.65]
        if high_prob:
            patterns.append("High-confidence bucket (>=0.65) produced misses.")
        patterns.append(f"{len(failures)} top failure(s) in the last 7 days.")
    else:
        patterns.append("No high-confidence failures this week.")

    return patterns


def _derive_feature_targets(df: pd.DataFrame, failures: List[Dict[str, Any]]) -> List[str]:
    targets: List[str] = []
    if not failures:
        return ["No urgent feature gaps identified from failures."]

    avg_prob = sum(float(f.get("win_probability") or 0) for f in failures) / max(len(failures), 1)
    if avg_prob >= 0.65:
        targets.append("Recalibrate high-confidence probabilities (>=0.65).")
    targets.append("Audit bullpen usage and late lineup confirmations for missed games.")
    targets.append("Evaluate park and travel-rest adjustments for close spreads.")
    return targets


def main() -> int:
    try:
        database_url = os.getenv("DATABASE_URL") or "sqlite:///./sports_analytics.db"
        feedback_url = _require_env("FEEDBACK_FORM_URL", "http://localhost:8000/")
        engine = create_database_engine(database_url)

        df = _fetch_weekly_predictions(engine)
        metrics = _compute_metrics(df)
        failures = _identify_failures(df)
        failure_patterns = _derive_patterns(df, failures)
        feature_targets = _derive_feature_targets(df, failures)

        html = render_weekly_report(
            metrics=metrics,
            failures=failures,
            failure_patterns=failure_patterns,
            feature_targets=feature_targets,
            feedback_form_url=feedback_url,
        )

        report_path = Path("reports") / "weekly_email.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html, encoding="utf-8")

        logger.info("Weekly report written to %s", report_path)
        return 0
    except Exception as exc:
        logger.exception("Weekly report failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
