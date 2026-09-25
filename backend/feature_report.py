"""Which stored features actually track the result? Feature selection on real games.

Every prediction is saved with the ``feature_snapshot`` the model saw. Once games
settle, each numeric feature can be checked against what happened, two ways:

- **home win**: does the feature separate home wins from home losses? Reported as
  a correlation and an AUC (0.5 = no signal, 1.0 = perfect separation).
- **model correct**: is the model right more often when the feature is high?
  That points at conditions where the model is weak.

A permutation p-value says how often shuffled outcomes do as well. With a few
dozen settled games almost nothing will be significant, and the report says so
rather than ranking noise. A feature that separates outcomes almost perfectly is
flagged as likely *leakage* -- the result sneaking into the inputs -- because no
honest pregame feature does that.
"""
from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import text

LEVELS = {"none": 0, "low": 1, "limited": 1, "weak": 1, "moderate": 2, "medium": 2,
          "high": 3, "strong": 3, "full": 3}
MIN_GAMES = 100
_RECORD = re.compile(r"^(\d+)-(\d+)(?:-(\d+))?$")


def to_number(value: Any) -> Optional[float]:
    """Numbers stay numbers; '6-4' becomes a win share; 'moderate' an ordinal."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        m = _RECORD.match(s)
        if m:
            w, second, third = int(m.group(1)), int(m.group(2)), m.group(3)
            if third is None:          # W-L
                return w / (w + second) if w + second else None
            d, l = second, int(third)  # W-D-L
            return (w + 0.5 * d) / (w + d + l) if w + d + l else None
        if s in LEVELS:
            return float(LEVELS[s])
        try:
            return float(s)
        except ValueError:
            return None
    return None


def load_rows(conn) -> List[Dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT sport, home_team, predicted_winner, actual_winner, feature_snapshot
        FROM predictions
        WHERE actual_winner IS NOT NULL AND TRIM(actual_winner) <> '' AND feature_snapshot IS NOT NULL
          AND UPPER(COALESCE(prediction_status, '')) <> 'VOID'
    """)).mappings().all()
    out = []
    for r in rows:
        try:
            snap = json.loads(r["feature_snapshot"]) if isinstance(r["feature_snapshot"], str) else r["feature_snapshot"]
        except (TypeError, ValueError):
            continue
        metrics = (snap or {}).get("metrics") or {}
        feats = {k: to_number(v) for k, v in metrics.items()}
        feats = {k: v for k, v in feats.items() if v is not None}
        norm = lambda x: (x or "").strip().lower()
        out.append({
            "sport": r["sport"],
            "features": feats,
            "home_win": norm(r["actual_winner"]) == norm(r["home_team"]),
            "model_correct": norm(r["actual_winner"]) == norm(r["predicted_winner"]),
        })
    return out


def _corr(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def _auc(x: List[float], y: List[float]) -> Optional[float]:
    pos = [a for a, b in zip(x, y) if b]
    neg = [a for a, b in zip(x, y) if not b]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def _perm_p(x: List[float], y: List[float], trials: int, rng: random.Random) -> Optional[float]:
    obs = _corr(x, y)
    if obs is None:
        return None
    ys = list(y)
    hits = 0
    for _ in range(trials):
        rng.shuffle(ys)
        r = _corr(x, ys)
        if r is not None and abs(r) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (trials + 1)


def assess(rows: Iterable[Dict[str, Any]], trials: int = 1000, seed: int = 11) -> Dict[str, Any]:
    rng = random.Random(seed)
    by_sport: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_sport.setdefault(r["sport"], []).append(r)
    report = {}
    for sport, games in sorted(by_sport.items()):
        names = sorted({k for g in games for k in g["features"]})
        feats = []
        for name in names:
            have = [g for g in games if name in g["features"]]
            x = [g["features"][name] for g in have]
            win = [float(g["home_win"]) for g in have]
            right = [float(g["model_correct"]) for g in have]
            r_win, r_right = _corr(x, win), _corr(x, right)
            auc = _auc(x, win)
            feats.append({
                "feature": name, "n": len(have),
                "home_win_r": r_win, "home_win_auc": auc,
                "home_win_p": _perm_p(x, win, trials, rng),
                "model_correct_r": r_right,
                "model_correct_p": _perm_p(x, right, trials, rng),
                "likely_leakage": auc is not None and (auc >= 0.97 or auc <= 0.03),
            })
        feats.sort(key=lambda f: -(abs(f["home_win_r"]) if f["home_win_r"] is not None else -1))
        report[sport] = {
            "settled_games": len(games),
            "enough_to_select": len(games) >= MIN_GAMES,
            "features": feats,
            "note": (None if len(games) >= MIN_GAMES else
                     f"Only {len(games)} settled games; below {MIN_GAMES} this is a monitor, not a verdict."),
        }
    return report
