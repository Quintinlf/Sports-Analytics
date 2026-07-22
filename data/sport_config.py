"""Centralized sport-specific configuration.

Single source of truth for outcome types, feature sets, model labels,
email display text, and feedback categories. No prediction logic here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SPORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "NBA": {
        "name": "Basketball",
        "league_default": "NBA",
        "outcome_types": ["HOME_WIN", "AWAY_WIN"],
        "features": {
            "elo": ["elo_diff"],
            "form_windows": [5, 10],
            "common": [
                "last5_win_pct_home",
                "last5_win_pct_away",
                "last5_point_diff_home",
                "last5_point_diff_away",
                "rest_days_home",
                "rest_days_away",
                "rest_diff",
                "is_back_to_back_home",
                "is_back_to_back_away",
                "home_away_strength_diff",
                "schedule_density_diff",
            ],
            "sport_specific": [
                "pace_diff",
                "injury_proxy",
                "expected_payoff_matrix",
                "optimal_path_delta",
                "signal_consistency_score",
            ],
            "box_score_stats": [
                "PTS",
                "FG_PCT",
                "FG3_PCT",
                "REB",
                "AST",
                "STL",
                "BLK",
                "TOV",
            ],
        },
        "model_type": "ensemble",
        "email": {
            "icon": "🏀",
            "report_title": "NBA Game Predictions Report",
            "weekly_title": "Weekly NBA Prediction Report",
            "matchup_format": "{away} @ {home}",
        },
        "feedback_categories": [
            "injuries",
            "pace",
            "lineup",
            "foul_trouble",
            "rest",
            "home_court",
        ],
        "weekly_feature_hints": [
            "Review pace and rest-day features for high-confidence misses.",
            "Check injury_proxy coverage for late scratch situations.",
        ],
    },
    "MLB": {
        "name": "Baseball",
        "league_default": "MLB",
        "outcome_types": ["HOME_WIN", "AWAY_WIN"],
        "features": {
            "elo": ["elo_diff"],
            "form_windows": [5, 10],
            "common": [
                "rest_diff",
                "last5_runs_for_home",
                "last5_runs_for_away",
                "last5_runs_against_home",
                "last5_runs_against_away",
            ],
            "sport_specific": [
                "pitcher_era_diff",
                "bullpen_fatigue",
                "bullpen_usage",
                "park_factor",
                "weather_impact",
                "lineup_change",
                "platoon_splits",
            ],
            "box_score_stats": [],
        },
        "model_type": "lightgbm",
        "email": {
            "icon": "⚾",
            "report_title": "MLB Game Predictions Report",
            "weekly_title": "Weekly MLB Prediction Report",
            "matchup_format": "{away} at {home}",
        },
        "feedback_categories": [
            "pitcher",
            "bullpen",
            "weather",
            "lineup",
            "park_factor",
            "travel_rest",
            "injury",
        ],
        "weekly_feature_hints": [
            "Audit bullpen usage and late lineup confirmations for missed games.",
            "Evaluate park and travel-rest adjustments for close spreads.",
        ],
    },
    "SOCCER": {
        "name": "Football",
        "league_default": "International",
        "outcome_types": ["HOME_WIN", "DRAW", "AWAY_WIN"],
        "features": {
            "elo": ["elo_diff"],
            "form_windows": [5, 10],
            "common": [
                "last5_goals_for_home",
                "last5_goals_for_away",
                "last5_goals_against_home",
                "last5_goals_against_away",
                "head_to_head_win_rate",
            ],
            "sport_specific": [
                "neutral_venue",
                "tournament_tier",
                "recent_form",
                "tactical_mismatch",
            ],
            "box_score_stats": [],
        },
        "model_type": "xgboost_multiclass",
        "email": {
            "icon": "⚽",
            "report_title": "International Soccer Predictions Report",
            "weekly_title": "Weekly Soccer Prediction Report",
            "matchup_format": "{home} vs {away}",
        },
        "feedback_categories": [
            "injuries",
            "elo",
            "neutral_venue",
            "recent_form",
            "tactical_mismatch",
            "tournament_context",
        ],
        "weekly_feature_hints": [
            "Review neutral-venue and tournament-tier features for international fixtures.",
            "Check head-to-head and recent-form windows (5 and 10 games).",
        ],
    },
}

SPORTS: List[str] = list(SPORT_CONFIG.keys())


def get_config(sport: str) -> Dict[str, Any]:
    """Return configuration dict for a sport (case-insensitive)."""
    key = sport.upper()
    if key not in SPORT_CONFIG:
        raise ValueError(f"Unknown sport: {sport!r}. Valid sports: {SPORTS}")
    return SPORT_CONFIG[key]


def get_outcome_types(sport: str) -> List[str]:
    return list(get_config(sport)["outcome_types"])


def get_form_windows(sport: str) -> List[int]:
    return list(get_config(sport)["features"]["form_windows"])


def get_feature_set(sport: str, category: Optional[str] = None) -> List[str]:
    """Return feature names for a sport, optionally filtered by category."""
    features = get_config(sport)["features"]
    if category is not None:
        return list(features.get(category, []))
    combined: List[str] = []
    for key in ("elo", "common", "sport_specific"):
        combined.extend(features.get(key, []))
    return combined


def get_model_type(sport: str) -> str:
    return str(get_config(sport)["model_type"])


def get_email_labels(sport: str) -> Dict[str, str]:
    return dict(get_config(sport)["email"])


def get_feedback_categories(sport: str) -> List[str]:
    return list(get_config(sport)["feedback_categories"])


def get_weekly_feature_hints(sport: str) -> List[str]:
    return list(get_config(sport).get("weekly_feature_hints", []))


def get_league_default(sport: str) -> str:
    return str(get_config(sport)["league_default"])


def outcome_display_label(outcome_type: str, home_team: str, away_team: str) -> str:
    """Map outcome type code to human-readable option label."""
    if outcome_type == "HOME_WIN":
        return f"{home_team} Win"
    if outcome_type == "AWAY_WIN":
        return f"{away_team} Win"
    if outcome_type == "DRAW":
        return "Draw"
    return outcome_type


def format_matchup(sport: str, home_team: str, away_team: str) -> str:
    labels = get_email_labels(sport)
    fmt = labels.get("matchup_format", "{away} @ {home}")
    return fmt.format(home=home_team, away=away_team)


def binary_home_win_probabilities(home_win_probability: float) -> Dict[str, float]:
    """Map home win probability to HOME_WIN / AWAY_WIN outcome probabilities."""
    home_prob = float(home_win_probability)
    return {
        "HOME_WIN": home_prob,
        "AWAY_WIN": round(1.0 - home_prob, 6),
    }


def build_outcome_options(
    sport: str,
    home_team: str,
    away_team: str,
    probabilities: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build ranked prediction_options payloads from outcome_type probabilities."""
    options: List[Dict[str, Any]] = []
    for outcome_type in get_outcome_types(sport):
        if outcome_type not in probabilities:
            continue
        options.append(
            {
                "outcome_type": outcome_type,
                "option_name": outcome_display_label(outcome_type, home_team, away_team),
                "probability": float(probabilities[outcome_type]),
            }
        )
    options.sort(key=lambda row: row["probability"], reverse=True)
    for rank, row in enumerate(options, start=1):
        row["rank"] = rank
    return options
