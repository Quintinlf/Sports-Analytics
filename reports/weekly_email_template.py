"""HTML template for weekly MLB performance report email."""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any, Dict, List


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_weekly_report(
    metrics: Dict[str, Any],
    failures: List[Dict[str, Any]],
    feedback_form_url: str,
) -> str:
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    failures_rows = ""
    if failures:
        for failure in failures:
            matchup = f"{failure.get('away_team', 'N/A')} at {failure.get('home_team', 'N/A')}"
            prob = float(failure.get("win_probability") or 0.0)
            failures_rows += (
                "<tr>"
                f"<td>{escape(str(failure.get('game_date', '')))}</td>"
                f"<td>{escape(matchup)}</td>"
                f"<td>{prob:.2f}</td>"
                f"<td>{escape(str(failure.get('confidence_level', 'N/A')))}</td>"
                "</tr>"
            )
    else:
        failures_rows = "<tr><td colspan=\"4\">No high-confidence misses this week.</td></tr>"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\" />
  <title>Weekly MLB Prediction Report</title>
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
    .cta {{ margin-top: 20px; }}
    .cta a {{ display: inline-block; padding: 10px 16px; border-radius: 8px; background: #0f766e; color: #fff; text-decoration: none; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Weekly MLB Prediction Report</h1>
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
        </tr>
      </thead>
      <tbody>
        {failures_rows}
      </tbody>
    </table>

    <div class=\"cta\">
      <p>Share feedback to improve next week's model decisions:</p>
      <a href=\"{escape(feedback_form_url)}\">Open Feedback Form</a>
    </div>
  </div>
</body>
</html>
"""
