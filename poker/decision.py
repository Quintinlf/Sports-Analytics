"""The mathematics of the decision the player is facing right now.

Everything here is *exact arithmetic on observed game state*. Pot odds, the
break-even equity threshold, stack-to-pot ratio and the current made hand are
all derived, never estimated and never hardcoded.

Quantities that genuinely require estimation — equity, outs, opponent ranges,
expected value — are deliberately absent. They are reported through
``PENDING_METRICS`` as not-yet-available rather than approximated, because a
plausible-looking fake probability is worse than no probability at all in a
tool whose purpose is to build correct intuition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from poker.engine.rules import LegalActions
from poker.engine.state import HandState, Player, Street
from poker.evaluator import evaluate

__all__ = [
    "DecisionContext",
    "BetOption",
    "build_decision_context",
    "PENDING_METRICS",
]


#: Metrics the interface will gain in later milestones. Surfaced explicitly so
#: the player knows what is missing rather than seeing an invented number.
PENDING_METRICS: List[Dict[str, str]] = [
    {
        "key": "equity",
        "label": "Estimated equity",
        "milestone": "Milestone 3",
        "note": "Requires the Monte Carlo / enumeration equity engine.",
    },
    {
        "key": "outs",
        "label": "Outs",
        "milestone": "Milestone 3",
        "note": "Counted against the opponent's range, not in isolation.",
    },
    {
        "key": "opponent_range",
        "label": "Estimated opponent range",
        "milestone": "Milestone 5",
        "note": "Needs the range model before it can mean anything.",
    },
    {
        "key": "expected_value",
        "label": "EV of each action",
        "milestone": "Milestone 4",
        "note": "EV needs equity; equity comes first.",
    },
]

#: Bet sizes offered as pot fractions, the sizings most commonly used at the table.
_SIZING_FRACTIONS = (
    (0.33, "⅓ pot"),
    (0.50, "½ pot"),
    (0.75, "¾ pot"),
    (1.00, "Pot"),
)


@dataclass(frozen=True)
class BetOption:
    """A concrete, legal bet or raise amount."""

    label: str
    #: Total street commitment to raise to.
    amount: int
    fraction_of_pot: Optional[float]
    is_all_in: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "amount": self.amount,
            "fraction_of_pot": self.fraction_of_pot,
            "is_all_in": self.is_all_in,
        }


@dataclass
class DecisionContext:
    """Exactly-derived facts about the hero's current decision."""

    street: str
    position: str
    pot: int
    to_call: int
    #: Pot size once the hero calls — the amount being played for.
    pot_after_call: int
    #: to_call / pot_after_call. The share of the time a call must win to break even.
    pot_odds: Optional[float]
    stack: int
    effective_stack: int
    #: effective_stack / pot. Low values mean commitment decisions arrive fast.
    stack_to_pot_ratio: Optional[float]
    hole_cards: List[str]
    board: List[str]
    made_hand: Optional[str]
    made_hand_category: Optional[str]
    bet_options: List[BetOption] = field(default_factory=list)
    explanations: List[Dict[str, str]] = field(default_factory=list)
    pending: List[Dict[str, str]] = field(default_factory=lambda: list(PENDING_METRICS))

    @property
    def pot_odds_pct(self) -> Optional[float]:
        return None if self.pot_odds is None else round(self.pot_odds * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "street": self.street,
            "position": self.position,
            "pot": self.pot,
            "to_call": self.to_call,
            "pot_after_call": self.pot_after_call,
            "pot_odds": self.pot_odds,
            "pot_odds_pct": self.pot_odds_pct,
            "stack": self.stack,
            "effective_stack": self.effective_stack,
            "stack_to_pot_ratio": self.stack_to_pot_ratio,
            "hole_cards": list(self.hole_cards),
            "board": list(self.board),
            "made_hand": self.made_hand,
            "made_hand_category": self.made_hand_category,
            "bet_options": [option.to_dict() for option in self.bet_options],
            "explanations": list(self.explanations),
            "pending": list(self.pending),
        }


def build_decision_context(
    state: HandState, player: Player, options: LegalActions
) -> DecisionContext:
    """Derive the decision context for ``player`` from the live hand state."""
    pot = state.pot_total
    to_call = options.call_amount
    pot_after_call = pot + to_call

    pot_odds = (to_call / pot_after_call) if to_call > 0 and pot_after_call else None

    opponents = [p for p in state.players if p is not player and p.is_contesting]
    effective_stack = min(
        [player.stack] + [o.stack + o.committed_street for o in opponents]
    ) if opponents else player.stack

    spr = (effective_stack / pot) if pot > 0 else None

    made_hand = made_category = None
    if len(state.board) >= 3 and len(player.hole_cards) == 2:
        value = evaluate(list(player.hole_cards) + list(state.board))
        made_hand = value.describe()
        made_category = value.category.name

    return DecisionContext(
        street=state.street.name,
        position=state.position_label(state.players.index(player)),
        pot=pot,
        to_call=to_call,
        pot_after_call=pot_after_call,
        pot_odds=pot_odds,
        stack=player.stack,
        effective_stack=effective_stack,
        stack_to_pot_ratio=round(spr, 2) if spr is not None else None,
        hole_cards=[str(card) for card in player.hole_cards],
        board=[str(card) for card in state.board],
        made_hand=made_hand,
        made_hand_category=made_category,
        bet_options=_bet_options(state, player, options, pot, to_call),
        explanations=_explanations(
            state, player, options, pot, to_call, pot_after_call, pot_odds, spr
        ),
    )


def _bet_options(
    state: HandState,
    player: Player,
    options: LegalActions,
    pot: int,
    to_call: int,
) -> List[BetOption]:
    """Legal pot-fraction sizings plus all-in.

    A bet of fraction f raises by f times the pot *as it would stand after the
    call*, which is the standard meaning of a "pot-sized bet".
    """
    if not options.can_aggress:
        return []

    base_pot = pot + to_call
    results: List[BetOption] = []
    seen: set[int] = set()

    for fraction, label in _SIZING_FRACTIONS:
        target = player.committed_street + to_call + int(round(fraction * base_pot))
        target = max(target, options.min_raise_to)
        target = min(target, options.max_raise_to)
        if target in seen or target < options.min_raise_to:
            continue
        seen.add(target)
        results.append(
            BetOption(
                label=label,
                amount=target,
                fraction_of_pot=fraction,
                is_all_in=target >= options.max_raise_to,
            )
        )

    if options.max_raise_to not in seen:
        results.append(
            BetOption(
                label="All-in",
                amount=options.max_raise_to,
                fraction_of_pot=None,
                is_all_in=True,
            )
        )
    else:
        for option in results:
            if option.amount == options.max_raise_to:
                results = [o for o in results if o.amount != options.max_raise_to]
                results.append(
                    BetOption("All-in", options.max_raise_to, option.fraction_of_pot, True)
                )
                break

    return results


def _explanations(
    state: HandState,
    player: Player,
    options: LegalActions,
    pot: int,
    to_call: int,
    pot_after_call: int,
    pot_odds: Optional[float],
    spr: Optional[float],
) -> List[Dict[str, str]]:
    """Short teaching notes attached to the numbers on screen."""
    notes: List[Dict[str, str]] = []

    if to_call > 0 and pot_odds is not None:
        pct = round(pot_odds * 100, 1)
        notes.append(
            {
                "key": "pot_odds",
                "title": "Pot odds",
                "formula": r"\text{required equity} = \frac{\text{call}}{\text{pot} + \text{call}}",
                "substitution": (
                    rf"\frac{{{to_call}}}{{{pot} + {to_call}}} "
                    rf"= \frac{{{to_call}}}{{{pot_after_call}}} = {pct}\%"
                ),
                "body": (
                    f"You must put in {to_call} to play for a pot of "
                    f"{pot_after_call}. That means this call breaks even only if "
                    f"you win at least {pct}% of the time. Below {pct}% it loses "
                    f"money on average; above it, it makes money — regardless of "
                    f"how this particular hand turns out."
                ),
            }
        )
    elif options.can_check:
        notes.append(
            {
                "key": "free_card",
                "title": "Checking is free",
                "formula": "",
                "substitution": "",
                "body": (
                    "Nobody has bet, so you can see the next card without "
                    "risking anything. There are no pot odds to satisfy — the "
                    "question is only whether betting yourself gains more than "
                    "checking."
                ),
            }
        )

    if spr is not None and spr > 0:
        notes.append(
            {
                "key": "spr",
                "title": "Stack-to-pot ratio",
                "formula": r"\text{SPR} = \frac{\text{effective stack}}{\text{pot}}",
                "substitution": rf"\frac{{{min(player.stack, 10**9)}}}{{{pot}}} = {round(spr, 2)}",
                "body": (
                    f"There is {round(spr, 2)}× the pot still behind. "
                    + (
                        "With a low SPR, hands get all-in quickly and decisions "
                        "become commitment decisions."
                        if spr < 4
                        else "With a deep SPR there is room to manoeuvre on later "
                        "streets, so implied odds matter more."
                    )
                ),
            }
        )

    if state.street is Street.PREFLOP:
        notes.append(
            {
                "key": "position",
                "title": "Position",
                "formula": "",
                "substitution": "",
                "body": (
                    "Heads-up, the button posts the small blind and acts first "
                    "before the flop, then acts last on every later street. "
                    "Acting last is a persistent informational advantage."
                ),
            }
        )

    return notes
