"""Heuristic hand-strength scoring, used by AI opponents to make decisions.

IMPORTANT SCOPE NOTE
--------------------
This is *not* equity and must never be displayed to the player as equity.
It is a cheap 0-1 score that lets a simple opponent behave plausibly without
running simulations on every action.

Real equity — win/tie/loss probability against a range, by enumeration or
Monte Carlo — is a separate engine (Milestone 3). When it lands, opponents
should switch to it and this module becomes a fast fallback for bulk
simulation only. Keeping the two strictly separate is what stops a heuristic
number from ever being shown to the learner as if it were a probability.
"""
from __future__ import annotations

from typing import Sequence

from poker.cards import Card
from poker.evaluator import HandCategory, evaluate

__all__ = ["chen_score", "preflop_strength", "postflop_strength", "hand_strength"]


#: Chen formula high-card points.
_CHEN_POINTS = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}

#: Base score per made-hand category, before within-category refinement.
_CATEGORY_BASE = {
    HandCategory.HIGH_CARD: 0.08,
    HandCategory.PAIR: 0.32,
    HandCategory.TWO_PAIR: 0.55,
    HandCategory.THREE_OF_A_KIND: 0.70,
    HandCategory.STRAIGHT: 0.80,
    HandCategory.FLUSH: 0.86,
    HandCategory.FULL_HOUSE: 0.92,
    HandCategory.FOUR_OF_A_KIND: 0.97,
    HandCategory.STRAIGHT_FLUSH: 1.00,
}

#: Width of the band each category occupies, for refinement by top tiebreaker.
_CATEGORY_SPAN = 0.10


def chen_score(hole_cards: Sequence[Card]) -> float:
    """Bill Chen's starting-hand formula. Roughly -1 (72o) to 20 (AA)."""
    if len(hole_cards) != 2:
        raise ValueError("Chen score needs exactly two hole cards")

    first, second = hole_cards
    high = max(int(first.rank), int(second.rank))
    low = min(int(first.rank), int(second.rank))
    is_pair = high == low
    is_suited = first.suit == second.suit

    score = _CHEN_POINTS.get(high, high / 2.0)

    if is_pair:
        score = max(score * 2.0, 5.0)
    if is_suited:
        score += 2.0

    gap = high - low - 1
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0

    # Straight bonus for low, close cards.
    if gap <= 1 and not is_pair and high < 12:
        score += 1.0

    return score


def preflop_strength(hole_cards: Sequence[Card]) -> float:
    """Normalise the Chen score into a 0-1 band."""
    return _clamp((chen_score(hole_cards) + 1.0) / 21.0)


def postflop_strength(
    hole_cards: Sequence[Card], board: Sequence[Card]
) -> float:
    """Score the made hand, discounting hands the board makes on its own.

    A pair on a paired board where neither hole card plays is worth far less
    than the category alone suggests, so hands that do not use a hole card are
    penalised. This is crude; Milestone 3 replaces it with equity.
    """
    value = evaluate(list(hole_cards) + list(board))
    base = _CATEGORY_BASE[value.category]

    # Refine within the category using the primary tiebreaker rank (2-14).
    top = value.tiebreakers[0] if value.tiebreakers else 2
    refinement = ((top - 2) / 12.0) * _CATEGORY_SPAN
    score = base + refinement

    if len(board) >= 3:
        board_value = evaluate(list(board)) if len(board) >= 5 else None
        hole_contributes = any(card in value.cards for card in hole_cards)
        if not hole_contributes:
            score *= 0.55
        if board_value is not None and value.key <= board_value.key:
            # The board plays; everyone has at least this hand.
            score = min(score, 0.30)

    return _clamp(score)


def hand_strength(
    hole_cards: Sequence[Card], board: Sequence[Card]
) -> float:
    """Dispatch to the preflop or postflop heuristic."""
    if not board:
        return preflop_strength(hole_cards)
    return postflop_strength(hole_cards, board)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
