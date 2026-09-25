"""Legal action computation and validation for No-Limit Texas Hold'em.

The betting rules that are easiest to get wrong, and how they are handled:

*Minimum raise* — a raise must increase the current bet by at least the size of
the previous bet or raise. Preflop the big blind seeds that increment, so the
first raise must be to at least two big blinds.

*Short all-in does not reopen betting* — if an opponent moves all-in for less
than a full raise, players who already acted may only call or fold. This falls
out of ``Player.has_acted_this_street``: a full raise clears that flag for
everyone else (reopening the action), a short all-in does not.

*Folding when you could check* is legal in real poker and the engine permits
it. It is never correct, so the interface presents it as a secondary action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from poker.engine.state import Action, ActionType, HandState, Player

__all__ = ["LegalActions", "legal_actions", "validate_action", "IllegalAction"]


class IllegalAction(ValueError):
    """Raised when an action violates the betting rules."""


@dataclass(frozen=True)
class LegalActions:
    """What the player to act may legally do right now."""

    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int
    can_bet: bool
    can_raise: bool
    #: Smallest legal total street commitment for a bet/raise.
    min_raise_to: int
    #: Largest legal total street commitment — always the all-in amount.
    max_raise_to: int
    #: True when the only legal aggressive action is an under-sized all-in.
    raise_is_all_in_only: bool

    @property
    def can_aggress(self) -> bool:
        return self.can_bet or self.can_raise

    def to_dict(self) -> Dict[str, object]:
        return {
            "can_fold": self.can_fold,
            "can_check": self.can_check,
            "can_call": self.can_call,
            "call_amount": self.call_amount,
            "can_bet": self.can_bet,
            "can_raise": self.can_raise,
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
            "raise_is_all_in_only": self.raise_is_all_in_only,
        }


def legal_actions(state: HandState, player: Player | None = None) -> LegalActions:
    """Compute the legal action set for ``player`` (defaults to whoever acts)."""
    if player is None:
        player = state.to_act
    if player is None:
        return LegalActions(
            can_fold=False, can_check=False, can_call=False, call_amount=0,
            can_bet=False, can_raise=False, min_raise_to=0, max_raise_to=0,
            raise_is_all_in_only=False,
        )

    to_call = state.amount_to_call(player)
    has_chips = player.stack > 0
    max_raise_to = player.committed_street + player.stack

    facing_bet = to_call > 0
    can_check = not facing_bet
    can_call = facing_bet and has_chips

    # Opening a street versus raising an existing bet.
    can_bet = has_chips and state.current_bet == 0
    can_raise = (
        has_chips
        and state.current_bet > 0
        and player.stack > to_call
        and not player.has_acted_this_street
    )

    if can_bet:
        desired_min = state.config.effective_min_bet
    elif can_raise:
        # A full raise increments by at least the previous raise size.
        increment = state.last_raise_size or state.config.effective_min_bet
        desired_min = state.current_bet + increment
    else:
        desired_min = 0

    min_raise_to = min(desired_min, max_raise_to) if desired_min else 0
    raise_is_all_in_only = bool(desired_min) and desired_min > max_raise_to

    return LegalActions(
        can_fold=True,
        can_check=can_check,
        can_call=can_call,
        call_amount=to_call,
        can_bet=can_bet,
        can_raise=can_raise,
        min_raise_to=min_raise_to,
        max_raise_to=max_raise_to,
        raise_is_all_in_only=raise_is_all_in_only,
    )


def validate_action(state: HandState, player: Player, action: Action) -> Action:
    """Check an action against the rules, returning it normalised.

    Normalisation clamps a bet/raise that exceeds the stack down to exactly
    all-in, so a client asking to "raise to 10000" with 200 behind shoves
    rather than erroring.
    """
    options = legal_actions(state, player)

    if action.type is ActionType.FOLD:
        if not options.can_fold:
            raise IllegalAction("Cannot fold right now")
        return Action(ActionType.FOLD)

    if action.type is ActionType.CHECK:
        if not options.can_check:
            raise IllegalAction(
                f"Cannot check facing a bet of {options.call_amount}"
            )
        return Action(ActionType.CHECK)

    if action.type is ActionType.CALL:
        if not options.can_call:
            raise IllegalAction("Nothing to call")
        return Action(ActionType.CALL, options.call_amount)

    if action.type in (ActionType.BET, ActionType.RAISE):
        if action.type is ActionType.BET and not options.can_bet:
            raise IllegalAction(
                "Cannot bet: there is already a bet this street, use raise"
            )
        if action.type is ActionType.RAISE and not options.can_raise:
            if state.current_bet == 0:
                raise IllegalAction("Cannot raise: no bet to raise, use bet")
            if player.has_acted_this_street:
                raise IllegalAction(
                    "Cannot raise: the action was not reopened by a full raise"
                )
            raise IllegalAction("Cannot raise: not enough chips behind")

        target = min(action.amount, options.max_raise_to)
        if target < options.min_raise_to:
            raise IllegalAction(
                f"{action.type.value} to {action.amount} is below the minimum "
                f"of {options.min_raise_to}"
            )
        return Action(action.type, target)

    raise IllegalAction(f"Unsupported action type: {action.type}")
