"""HTML template for weekly sport performance report email."""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any, Dict, List

from data.sport_config import format_matchup, get_email_labels


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_weekly_report(
    metrics: Dict[str, Any],
    failures: List[Dict[str, Any]],
    failure_patterns: List[str],
    feature_targets: List[str],
    feedback_form_url: str,
    sport: str = "MLB",
) -> str:
    labels = get_email_labels(sport)
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    weekly_title = labels.get("weekly_title", f"Weekly {sport} Prediction Report")
    icon = labels.get("icon", "")

    failures_rows = ""
    if failures:
        for failure in failures:
            matchup = format_matchup(
                sport,
                str(failure.get("home_team", "N/A")),
                str(failure.get("away_team", "N/A")),
            )
            prob = float(failure.get("win_probability") or 0.0)
            prediction_id = escape(str(failure.get("prediction_id", "")))
            review_link = (
                f"{feedback_form_url}?prediction_id={prediction_id}&sport={sport}"
                if prediction_id
                else feedback_form_url
            )
            failures_rows += (
                "<tr>"
                f"<td>{escape(str(failure.get('game_date', '')))}</td>"
                f"<td>{escape(matchup)}</td>"
                f"<td>{prob:.2f}</td>"
                f"<td>{escape(str(failure.get('confidence_level', 'N/A')))}</td>"
                f"<td><a href=\"{review_link}\">Review</a></td>"
                "</tr>"
            )
    else:
        failures_rows = "<tr><td colspan=\"5\">No high-confidence misses this week.</td></tr>"

    pattern_items = "".join(
        f"<li>{escape(item)}</li>" for item in failure_patterns
    ) or "<li>No patterns identified.</li>"

    target_items = "".join(
        f"<li>{escape(item)}</li>" for item in feature_targets
    ) or "<li>No feature targets identified.</li>"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\" />
  <title>{escape(weekly_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; }}
    .card {{ max-width: 820px; margin: 24px auto; background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }}
    h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    .meta {{ color: #475569; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .metric {{ background: #f1f5f9; border-radius: 10px; padding: 12px; text-align: center; }}
    .metric span {{ display: block; color: #64748b; font-size: 12px; }}
    .metric strong {{ font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
    th {{ background: #f8fafc; font-size: 12px; color: #334155; }}
    ul {{ margin: 10px 0 0 18px; color: #1f2937; }}
    .cta {{ margin-top: 20px; }}
    .cta a {{ display: inline-block; padding: 10px 16px; border-radius: 8px; background: #0f766e; color: #fff; text-decoration: none; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>{icon} {escape(weekly_title)}</h1>
    <div class=\"meta\">Generated {generated_at}</div>

    <div class=\"grid\">
      <div class=\"metric\"><span>Predictions</span><strong>{metrics.get('n_predictions', 0)}</strong></div>
      <div class=\"metric\"><span>Accuracy</span><strong>{_format_percent(metrics.get('accuracy', 0.0))}</strong></div>
      <div class=\"metric\"><span>Brier Score</span><strong>{metrics.get('brier_score', 0.0):.3f}</strong></div>
      <div class=\"metric\"><span>Calibration Gap</span><strong>{_format_percent(metrics.get('calibration_gap', 0.0))}</strong></div>
    </div>

    <h2>Top Failures</h2>
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Matchup</th>
          <th>Win Prob</th>
          <th>Confidence</th>
          <th>Review</th>
        </tr>
      </thead>
      <tbody>
        {failures_rows}
      </tbody>
    </table>

    <h2>Recurring Failure Patterns</h2>
    <ul>
      {pattern_items}
    </ul>

    <h2>Feature Engineering Targets</h2>
    <ul>
      {target_items}
    </ul>

    <div class=\"cta\">
      <p>Share feedback to improve next week's model decisions:</p>
      <a href=\"{escape(feedback_form_url)}\">Open Feedback Form</a>
    </div>
  </div>
</body>
</html>
"""
