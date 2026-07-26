"""Prediction explanation layer — grounded in actual model inputs / live stats.

Produces human-readable ``why_factors`` and ``risk_factors`` stored inside
``predictions.feature_snapshot``. Does not invent SHAP/importance values that
were never computed; factors are derived from feature differentials present at
predict time (rolling form, Elo/matchup signals, squad profile metrics, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Human-readable labels for known model / live-feature keys
# ---------------------------------------------------------------------------

_MLB_PAIR_LABELS: Dict[str, Tuple[str, bool]] = {
    # key -> (label, higher_is_better)
    "R_ROLL": ("Recent scoring (runs / game)", True),
    "RA_ROLL": ("Recent run prevention", False),
    "WIN_STREAK": ("Win streak", True),
    "WIN_RATE_10": ("Last-10 win rate", True),
    "REST_DAYS": ("Rest days", True),
    "IS_BACK_TO_BACK": ("Avoiding a back-to-back", False),
}

_NBA_PAIR_LABELS: Dict[str, Tuple[str, bool]] = {
    "PTS_ROLL": ("Recent scoring", True),
    "FG_PCT_ROLL": ("Field-goal percentage", True),
    "FG3_PCT_ROLL": ("Three-point percentage", True),
    "REB_ROLL": ("Rebounding", True),
    "AST_ROLL": ("Assist rate", True),
    "STL_ROLL": ("Steals", True),
    "BLK_ROLL": ("Blocks", True),
    "TOV_ROLL": ("Turnovers", False),
    "WIN_STREAK": ("Win streak", True),
    "WIN_RATE_10": ("Recent win rate", True),
    "REST_DAYS": ("Rest days", True),
    "IS_BACK_TO_BACK": ("Avoiding a back-to-back", False),
}

# Signed matchup features: positive favors home unless noted
_NBA_SIGNED: Dict[str, Tuple[str, bool]] = {
    "elo_diff": ("Elo rating edge", True),
    "rest_diff": ("Rest advantage", True),
    "home_away_strength_diff": ("Home/away strength edge", True),
    "pace_diff": ("Pace matchup edge", True),
    "schedule_density_diff": ("Schedule density edge", True),
    "last5_point_diff_home": ("Home last-5 point differential", True),
    "last5_point_diff_away": ("Away last-5 point differential", True),
    "last5_win_pct_home": ("Home last-5 win rate", True),
    "last5_win_pct_away": ("Away last-5 win rate", True),
    "rest_days_home": ("Home rest days", True),
    "rest_days_away": ("Away rest days", True),
    "is_back_to_back_home": ("Home back-to-back penalty", False),
    "is_back_to_back_away": ("Away back-to-back penalty", False),
}

# Substring → label for FIFA FBref-derived squad columns (applied case-insensitive)
_FIFA_COLUMN_HINTS: Sequence[Tuple[str, str, bool]] = (
    ("xg", "Expected goals (xG)", True),
    ("gls", "Goals scored", True),
    ("goals", "Goals scored", True),
    ("ast", "Assists", True),
    ("sh", "Shots", True),
    ("sot", "Shots on target", True),
    ("ga", "Goals against", False),
    ("gk", "Goalkeeping metrics", True),
    ("poss", "Possession", True),
    ("pass", "Passing", True),
    ("tkl", "Tackles", True),
    ("int", "Interceptions", True),
    ("press", "Pressing", True),
    ("sca", "Shot-creating actions", True),
    ("gca", "Goal-creating actions", True),
    ("crd", "Cards / discipline", False),
)


def build_snapshot(
    *,
    sport: str,
    data_source: str,
    is_fallback: bool,
    confidence_score: float,
    explanations: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    why_factors: Optional[List[Dict[str, Any]]] = None,
    risk_factors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Canonical feature_snapshot payload (schema_version 2 adds why/risk)."""
    why = list(why_factors or [])
    risks = list(risk_factors or [])
    # Backward-compatible explanations: prefer why_factors when provided.
    if why and not explanations:
        explanations = [
            {
                "label": f.get("label", "Factor"),
                "weight": float(f.get("strength", 0.0) or 0.0),
                "value": f.get("detail") or f.get("label"),
            }
            for f in why
        ]
    return {
        "schema_version": 2,
        "sport": sport,
        "data_source": data_source,
        "is_fallback": bool(is_fallback),
        "confidence_score": round(float(confidence_score), 3),
        "explanations": explanations,
        "why_factors": why,
        "risk_factors": risks,
        "metrics": metrics,
    }


def build_risk_factors(
    *,
    win_probability: float,
    confidence_level: str,
    missing_data_warnings: Optional[Sequence[str]] = None,
    is_fallback: bool = False,
    injury_proxy: Optional[float] = None,
    extra: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Risk / uncertainty flags from prediction metadata (not invented narrative)."""
    risks: List[Dict[str, Any]] = []
    wp = float(win_probability)
    edge = abs(wp - 0.5)
    if edge < 0.05:
        risks.append(
            {
                "code": "close_matchup",
                "label": "Close matchup",
                "detail": f"Win probability is near even ({wp:.0%} vs {1 - wp:.0%}).",
            }
        )
    elif edge < 0.10:
        risks.append(
            {
                "code": "narrow_edge",
                "label": "Narrow model edge",
                "detail": f"Projected win probability is only moderately separated ({wp:.0%}).",
            }
        )

    level = (confidence_level or "").upper()
    if level == "LOW":
        risks.append(
            {
                "code": "low_confidence",
                "label": "Low confidence",
                "detail": "Model confidence label is LOW for this matchup.",
            }
        )

    if is_fallback:
        risks.append(
            {
                "code": "fallback_prediction",
                "label": "Fallback / demo data",
                "detail": "This row is not a full live-model prediction.",
            }
        )

    for code in missing_data_warnings or []:
        risks.append(
            {
                "code": str(code),
                "label": str(code).replace("_", " ").title(),
                "detail": "Input data was missing or incomplete for this signal.",
            }
        )

    if injury_proxy is not None and float(injury_proxy) != 0.0:
        risks.append(
            {
                "code": "injury_uncertainty",
                "label": "Injury uncertainty",
                "detail": f"Injury proxy signal is non-zero ({float(injury_proxy):.2f}).",
            }
        )

    if extra:
        risks.extend(list(extra))
    return risks[:8]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _row_to_dict(features: Any) -> Dict[str, Any]:
    if features is None:
        return {}
    if isinstance(features, Mapping):
        return dict(features)
    # pandas DataFrame / Series
    if hasattr(features, "iloc") and hasattr(features, "columns"):
        if len(features) == 0:
            return {}
        return {str(k): features.iloc[0][k] for k in features.columns}
    if hasattr(features, "to_dict"):
        d = features.to_dict()
        if d and not isinstance(next(iter(d.values())), (int, float, str, type(None))):
            # Series-like already
            return {str(k): v for k, v in d.items()}
        return {str(k): v for k, v in d.items()}
    return {}


def _favor_home(predicted_winner: str, home_team: str, away_team: str) -> Optional[bool]:
    pw = (predicted_winner or "").strip().lower()
    if pw == "draw":
        return None
    if pw == (home_team or "").strip().lower():
        return True
    if pw == (away_team or "").strip().lower():
        return False
    # Winner strings sometimes include "Win" suffix
    if (home_team or "").strip().lower() in pw:
        return True
    if (away_team or "").strip().lower() in pw:
        return False
    return None


def _normalize_strengths(factors: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    if not factors:
        return []
    factors = sorted(factors, key=lambda f: -float(f.get("_raw", 0.0)))[:top_n]
    max_raw = max(float(f.get("_raw", 0.0)) for f in factors) or 1.0
    out: List[Dict[str, Any]] = []
    for f in factors:
        raw = float(f.get("_raw", 0.0))
        item = {
            "label": f["label"],
            "detail": f.get("detail", ""),
            "side": f.get("side", "neutral"),
            "strength": round(raw / max_raw, 3),
            "source_feature": f.get("source_feature"),
        }
        out.append(item)
    return out


def factors_from_home_away_pairs(
    features: Any,
    *,
    predicted_winner: str,
    home_team: str,
    away_team: str,
    pair_labels: Mapping[str, Tuple[str, bool]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Build factors from HOME_X / AWAY_X feature pairs favoring the predicted winner."""
    row = _row_to_dict(features)
    favor_home = _favor_home(predicted_winner, home_team, away_team)
    if favor_home is None:
        return []

    candidates: List[Dict[str, Any]] = []
    for key, (label, higher_is_better) in pair_labels.items():
        home_v = _safe_float(row.get(f"HOME_{key}"))
        away_v = _safe_float(row.get(f"AWAY_{key}"))
        if home_v is None or away_v is None:
            continue
        diff = home_v - away_v
        # Advantage for predicted side
        if higher_is_better:
            advantage = diff if favor_home else -diff
        else:
            advantage = -diff if favor_home else diff
        if advantage <= 1e-9:
            continue
        winner_v = home_v if favor_home else away_v
        loser_v = away_v if favor_home else home_v
        candidates.append(
            {
                "label": label,
                "detail": f"{predicted_winner}: {winner_v:.2f} vs opponent {loser_v:.2f}",
                "side": "home" if favor_home else "away",
                "_raw": abs(advantage),
                "source_feature": key,
            }
        )
    return _normalize_strengths(candidates, top_n=top_n)


def factors_from_nba_features(
    features: Any,
    *,
    predicted_winner: str,
    home_team: str,
    away_team: str,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """NBA factors from HOME_/AWAY_ rolling stats plus signed matchup columns."""
    pair_factors = factors_from_home_away_pairs(
        features,
        predicted_winner=predicted_winner,
        home_team=home_team,
        away_team=away_team,
        pair_labels=_NBA_PAIR_LABELS,
        top_n=8,
    )
    row = _row_to_dict(features)
    favor_home = _favor_home(predicted_winner, home_team, away_team)
    if favor_home is None:
        return pair_factors[:top_n]

    signed: List[Dict[str, Any]] = []
    # Paired last5 win pct / rest already handled somewhat; add elo_diff etc.
    for key, (label, higher_is_better) in _NBA_SIGNED.items():
        val = _safe_float(row.get(key))
        if val is None:
            continue
        # Features that are absolute side stats (last5_win_pct_home) — only count
        # when that side is the predicted winner.
        if key.endswith("_home"):
            if not favor_home:
                continue
            advantage = val if higher_is_better else -val
        elif key.endswith("_away"):
            if favor_home:
                continue
            advantage = val if higher_is_better else -val
        else:
            # Signed home-minus-away style
            advantage = val if favor_home else -val
            if not higher_is_better:
                advantage = -advantage
        if advantage <= 1e-9:
            continue
        signed.append(
            {
                "label": label,
                "detail": f"{key} = {val:.3f} (favors {predicted_winner})",
                "side": "home" if favor_home else "away",
                "_raw": abs(advantage),
                "source_feature": key,
            }
        )

    # Home court is an implicit training signal for home-win models when home wins
    if favor_home:
        signed.append(
            {
                "label": "Home court advantage",
                "detail": f"{home_team} is at home; model win probability leans home.",
                "side": "home",
                "_raw": 0.15,
                "source_feature": "home_court",
            }
        )

    merged = [
        {**f, "_raw": float(f.get("strength", 0) or 0) * 10 + (0.01 if f.get("source_feature") else 0)}
        for f in pair_factors
    ]
    # Re-score pair factors with original relative strength preserved via strength field
    for f, orig in zip(merged, pair_factors):
        f["_raw"] = float(orig.get("strength", 0.5)) + 0.5

    all_factors = signed + [
        {
            "label": f["label"],
            "detail": f["detail"],
            "side": f["side"],
            "_raw": float(f.get("strength", 0.5)) + 0.25,
            "source_feature": f.get("source_feature"),
        }
        for f in pair_factors
    ]
    return _normalize_strengths(all_factors, top_n=top_n)


def _fifa_label_for_column(col: str) -> Tuple[str, bool]:
    low = col.lower()
    for needle, label, hib in _FIFA_COLUMN_HINTS:
        if needle in low:
            return label, hib
    clean = col.replace("_", " ").strip()
    return clean.title() if clean else col, True


def factors_from_squad_profiles(
    home_metrics: Mapping[str, Any],
    away_metrics: Mapping[str, Any],
    *,
    predicted_winner: str,
    home_team: str,
    away_team: str,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Compare raw squad-profile numerics (pre-PCA inputs to the FIFA model)."""
    favor_home = _favor_home(predicted_winner, home_team, away_team)
    if favor_home is None:
        # Draw: show strongest absolute gaps without claiming a winner side
        favor_home = True  # report as home-vs-away gaps; labels stay neutral below

    keys = sorted(set(home_metrics) & set(away_metrics))
    candidates: List[Dict[str, Any]] = []
    for key in keys:
        hv = _safe_float(home_metrics.get(key))
        av = _safe_float(away_metrics.get(key))
        if hv is None or av is None:
            continue
        label, higher_is_better = _fifa_label_for_column(key)
        diff = hv - av
        if predicted_winner.strip().lower() == "draw":
            advantage = abs(diff)
            if advantage <= 1e-9:
                continue
            detail = f"{home_team} {hv:.2f} vs {away_team} {av:.2f}"
            side = "neutral"
        else:
            if higher_is_better:
                advantage = diff if favor_home else -diff
            else:
                advantage = -diff if favor_home else diff
            if advantage <= 1e-9:
                continue
            winner_v = hv if favor_home else av
            loser_v = av if favor_home else hv
            detail = f"{predicted_winner}: {winner_v:.2f} vs opponent {loser_v:.2f}"
            side = "home" if favor_home else "away"
        candidates.append(
            {
                "label": label,
                "detail": detail,
                "side": side,
                "_raw": abs(diff) if predicted_winner.strip().lower() == "draw" else abs(advantage),
                "source_feature": key,
            }
        )
    # Deduplicate by label keeping strongest
    best: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        prev = best.get(c["label"])
        if prev is None or c["_raw"] > prev["_raw"]:
            best[c["label"]] = c
    return _normalize_strengths(list(best.values()), top_n=top_n)


def explain_mlb_prediction(
    *,
    features: Any,
    predicted_winner: str,
    home_team: str,
    away_team: str,
    win_probability: float,
    confidence_level: str,
    missing_data_warnings: Optional[Sequence[str]] = None,
    pitcher_explanations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return why_factors + risk_factors + UI explanations for MLB."""
    why = factors_from_home_away_pairs(
        features,
        predicted_winner=predicted_winner,
        home_team=home_team,
        away_team=away_team,
        pair_labels=_MLB_PAIR_LABELS,
        top_n=5,
    )
    # Pitcher ERA/WHIP are context overlays (not LightGBM inputs). Append only
    # when they favor the predicted winner and we have numeric values.
    for pe in pitcher_explanations or []:
        label = str(pe.get("label") or "")
        value = pe.get("value")
        if value is None:
            continue
        # Existing mlb_context uses fixed weights; keep as supplementary factor
        # only if the label mentions the predicted side or is generic starter ERA.
        why.append(
            {
                "label": label or "Starting pitcher context",
                "detail": str(value),
                "side": "neutral",
                "strength": round(float(pe.get("weight") or 0.2), 3),
                "source_feature": "starting_pitcher_context",
            }
        )
    why = why[:5]
    if why:
        # Renormalize strengths after pitcher append
        max_s = max(float(f.get("strength") or 0) for f in why) or 1.0
        for f in why:
            f["strength"] = round(float(f.get("strength") or 0) / max_s, 3)

    risks = build_risk_factors(
        win_probability=win_probability,
        confidence_level=confidence_level,
        missing_data_warnings=missing_data_warnings,
    )
    explanations = [
        {
            "label": f["label"],
            "weight": f["strength"],
            "value": f.get("detail") or f["label"],
        }
        for f in why
    ]
    return {"why_factors": why, "risk_factors": risks, "explanations": explanations}


def explain_nba_prediction(
    *,
    features: Any,
    predicted_winner: str,
    home_team: str,
    away_team: str,
    win_probability: float,
    confidence_level: str,
) -> Dict[str, Any]:
    row = _row_to_dict(features)
    why = factors_from_nba_features(
        features,
        predicted_winner=predicted_winner,
        home_team=home_team,
        away_team=away_team,
        top_n=5,
    )
    risks = build_risk_factors(
        win_probability=win_probability,
        confidence_level=confidence_level,
        injury_proxy=_safe_float(row.get("injury_proxy")),
    )
    explanations = [
        {
            "label": f["label"],
            "weight": f["strength"],
            "value": f.get("detail") or f["label"],
        }
        for f in why
    ]
    return {"why_factors": why, "risk_factors": risks, "explanations": explanations}


def explain_fifa_prediction(
    *,
    home_metrics: Mapping[str, Any],
    away_metrics: Mapping[str, Any],
    predicted_winner: str,
    home_team: str,
    away_team: str,
    win_probability: float,
    confidence_level: str,
    outcome_probabilities: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    why = factors_from_squad_profiles(
        home_metrics,
        away_metrics,
        predicted_winner=predicted_winner,
        home_team=home_team,
        away_team=away_team,
        top_n=5,
    )
    extra_risks: List[Dict[str, Any]] = []
    if outcome_probabilities:
        draw_p = float(outcome_probabilities.get("DRAW", 0.0) or 0.0)
        if draw_p >= 0.28:
            extra_risks.append(
                {
                    "code": "draw_likely",
                    "label": "Draw is a live outcome",
                    "detail": f"Model assigns {draw_p:.0%} probability to a draw.",
                }
            )
    risks = build_risk_factors(
        win_probability=win_probability,
        confidence_level=confidence_level,
        extra=extra_risks,
    )
    explanations = [
        {
            "label": f["label"],
            "weight": f["strength"],
            "value": f.get("detail") or f["label"],
        }
        for f in why
    ]
    return {"why_factors": why, "risk_factors": risks, "explanations": explanations}
