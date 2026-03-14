"""
HTML report rendering for NBA predictions.

This module intentionally returns HTML as a string only.
No file writes are performed.
"""

from datetime import datetime
from html import escape
from typing import Dict, List


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return 'N/A'


def _fmt_num(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return 'N/A'


def _confidence_badge(level: str) -> str:
    level = (level or 'LOW').upper()
    css = {
        'HIGH': 'badge-high',
        'MEDIUM': 'badge-medium',
        'LOW': 'badge-low',
    }.get(level, 'badge-low')
    return f'<span class="badge {css}">{escape(level)}</span>'


def generate(games_predictions: List[Dict], title: str = 'NBA Game Predictions Report') -> str:
    """
    Render predictions to an HTML report string.

    Parameters
    ----------
    games_predictions : list[dict]
        Each item should include game metadata and model output fields.
    title : str

    Returns
    -------
    str  HTML document.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    rows = []
    for idx, game in enumerate(games_predictions, start=1):
        home = escape(str(game.get('home_team', 'Home')))
        away = escape(str(game.get('away_team', 'Away')))
        game_date = escape(str(game.get('game_date', 'TBD')))

        spread = _fmt_num(game.get('spread'))
        q10 = _fmt_num(game.get('q10'))
        q90 = _fmt_num(game.get('q90'))
        uncertainty = _fmt_num(game.get('uncertainty'), digits=3)
        win_prob = _fmt_pct(game.get('win_prob'))

        confidence = _confidence_badge(str(game.get('confidence', 'LOW')))

        favored = home if (game.get('spread', 0) or 0) >= 0 else away
        favored = escape(str(favored))

        rows.append(
            f"""
            <tr>
                <td>{idx}</td>
                <td>{game_date}</td>
                <td>{away} @ {home}</td>
                <td>{favored}</td>
                <td>{spread}</td>
                <td>{win_prob}</td>
                <td>[{q10}, {q90}]</td>
                <td>{uncertainty}</td>
                <td>{confidence}</td>
            </tr>
            """
        )

    table_rows = ''.join(rows) if rows else (
        '<tr><td colspan="9" class="empty">No predictions available.</td></tr>'
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f4ef;
      --card: #ffffff;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: #e3e8ef;
      --accent: #0f766e;
      --high: #0b8f46;
      --medium: #c77d00;
      --low: #b42318;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", Verdana, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% -10%, #d9f7f3 0%, transparent 45%),
        radial-gradient(circle at 100% 0%, #fdeacc 0%, transparent 40%),
        var(--bg);
      min-height: 100vh;
      padding: 24px;
    }}

    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
    }}

    .header {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
      margin-bottom: 16px;
    }}

    .title {{
      margin: 0;
      font-size: clamp(1.25rem, 1.1rem + 1vw, 2rem);
      letter-spacing: 0.3px;
    }}

    .meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 880px;
    }}

    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 12px 10px;
      text-align: left;
      font-size: 0.93rem;
      white-space: nowrap;
    }}

    th {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      background: #fbfcfd;
    }}

    tr:hover td {{
      background: #fbfffd;
    }}

    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      font-weight: 700;
      color: #fff;
    }}

    .badge-high {{ background: var(--high); }}
    .badge-medium {{ background: var(--medium); }}
    .badge-low {{ background: var(--low); }}

    .empty {{
      text-align: center;
      color: var(--muted);
      font-style: italic;
      padding: 24px;
    }}

    @media (max-width: 700px) {{
      body {{ padding: 14px; }}
      .header {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="header">
      <h1 class="title">{escape(title)}</h1>
      <div class="meta">Generated: {escape(now)} | Games: {len(games_predictions)}</div>
    </section>

    <section class="card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Date</th>
              <th>Matchup</th>
              <th>Favored</th>
              <th>Spread</th>
              <th>Win Prob</th>
              <th>Interval</th>
              <th>Uncertainty</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
"""

    return html
