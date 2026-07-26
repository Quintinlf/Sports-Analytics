"""Analyst challenge / successful-override tracking.

Maps onto existing prediction_reviews + review_outcomes columns without
renaming the schema. Successful override = analyst disagreed, analyst pick
was correct, and the AI pick was wrong — used to unlock reasoning collection.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

# Shown when the analyst beat the model — collecting reasoning, not ranking.
OVERRIDE_FOLLOWUP_PROMPT = (
    "If your prediction was correct and the AI was wrong, explain what "
    "information the model missed."
)


def compose_analyst_reasoning(
    missing_factors: Optional[Union[Sequence[str], str]] = None,
    pregame_notes: Optional[str] = None,
) -> Optional[str]:
    """Combine pregame disagreement notes + selected factors into one reasoning blob."""
    parts: List[str] = []
    if pregame_notes and str(pregame_notes).strip():
        parts.append(str(pregame_notes).strip())

    factors: List[str] = []
    if isinstance(missing_factors, str):
        raw = missing_factors.strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    factors = [str(x) for x in parsed if str(x).strip()]
                else:
                    factors = [raw]
            except json.JSONDecodeError:
                factors = [raw]
    elif missing_factors:
        factors = [str(x).strip() for x in missing_factors if str(x).strip()]

    if factors:
        parts.append("Missing factors: " + ", ".join(factors))

    if not parts:
        return None
    return "\n\n".join(parts)


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _as_bool_correct(correct: Any, predicted_winner: str, actual_winner: str) -> bool:
    if correct is None:
        return _norm(predicted_winner) == _norm(actual_winner)
    if isinstance(correct, bool):
        return correct
    try:
        return int(correct) == 1
    except (TypeError, ValueError):
        return bool(correct)


def evaluate_challenge(
    *,
    agree_with_model: bool,
    reviewer_pick: str,
    predicted_winner: str,
    actual_winner: str,
    model_correct_flag: Any = None,
    analyst_reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute challenge tracking fields from review + settled prediction."""
    analyst_disagreed = not bool(agree_with_model)
    ai_was_correct = _as_bool_correct(model_correct_flag, predicted_winner, actual_winner)
    analyst_was_correct = _norm(reviewer_pick) == _norm(actual_winner)
    successful_override = (
        analyst_disagreed and analyst_was_correct and not ai_was_correct
    )
    return {
        "analyst_disagreed": analyst_disagreed,
        "analyst_reasoning": analyst_reasoning,
        "final_result": actual_winner,
        "analyst_was_correct": analyst_was_correct,
        "ai_was_correct": ai_was_correct,
        "successful_analyst_override": successful_override,
        # Backward-compatible aliases used elsewhere in the API/UI
        "model_correct": ai_was_correct,
        "reviewer_correct": analyst_was_correct,
        "reviewer_beat_model": successful_override,
        "override_followup_prompt": OVERRIDE_FOLLOWUP_PROMPT if successful_override else None,
    }
