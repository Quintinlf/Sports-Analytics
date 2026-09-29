"""A simple rule-based opponent driven entirely by its profile parameters.

Deliberately unsophisticated. Milestone 1's goal is a correct, enjoyable
playable loop; a strong opponent is Milestone 9's problem. What matters here
is that every decision routes through the profile numbers, so swapping in an
archetype changes behaviour without changing code — and so a future opponent
model has something concrete to estimate.

The policy compares a heuristic hand strength against the pot odds it is being
offered. That is the same comparison the interface teaches the player to make,
which makes the opponent's mistakes legible rather than arbitrary.
"""
from __future__ import annotations

from random import Random
from typing import Optional

from poker.engine.rules import LegalActions
from poker.engine.state import Action, ActionType, HandState, Player, Street
from poker.opponents.base import DEFAULT_PROFILE_KEY, OpponentProfile, get_profile
from poker.strength import hand_strength, preflop_cutoff

__all__ = ["HeuristicOpponent", "defend_chance", "raise_band"]


def raise_band(profile: OpponentProfile) -> float:
    """Share of starting hands eligible for a preflop raise, so that eligible x aggression = PFR."""
    return min(1.0, profile.pfr / profile.aggression) if profile.aggression > 0 else 0.0


def defend_chance(profile: OpponentProfile, strength: float, to_call: int, pot_before: int) -> float:
    """Chance of continuing against a bet of ``to_call`` into ``pot_before``.

    A hand that clears both the price (pot odds) and the profile's call threshold always
    continues. A marginal hand -- one that clears only one of the two -- continues unless the
    opponent folds to aggression, at the profile's rate. Below both, it folds. So a calling
    station (threshold 0.12) keeps calling with weak hands the price says to fold, and a rock
    (threshold 0.45) lays down hands the price says to keep.
    """
    if to_call <= 0:
        return 1.0
    pot_after_call = pot_before + to_call
    required = to_call / pot_after_call if pot_after_call else 1.0
    if strength >= max(profile.call_threshold, required):
        return 1.0
    if strength >= profile.call_threshold or strength >= required:
        return 1.0 - profile.fold_to_aggression
    return 0.0


class HeuristicOpponent:
    """Profile-driven opponent policy."""

    def __init__(self, profile: Optional[OpponentProfile] = None) -> None:
        self.profile = profile or get_profile(DEFAULT_PROFILE_KEY)

    def choose_action(
        self,
        state: HandState,
        player: Player,
        options: LegalActions,
        rng: Random,
    ) -> Action:
        strength = hand_strength(player.hole_cards, state.board)

        if state.street is Street.PREFLOP:
            return self._preflop(state, player, options, rng, strength)
        return self._postflop(state, player, options, rng, strength)

    # -- preflop ----------------------------------------------------------

    def _preflop(
        self,
        state: HandState,
        player: Player,
        options: LegalActions,
        rng: Random,
        strength: float,
    ) -> Action:
        profile = self.profile

        # VPIP sets how deep into the strength distribution this opponent
        # plays: a 20% VPIP means the top 20% of the 1326 starting hands.
        # Raises are gated by aggression, so the raising band is widened to
        # pfr / aggression to keep the overall raise rate at PFR.
        plays_hand = strength >= preflop_cutoff(profile.vpip)
        raises_hand = strength >= preflop_cutoff(raise_band(profile))

        if options.can_raise and raises_hand and rng.random() < profile.aggression:
            return self._sized_raise(state, options, rng)

        if options.can_check:
            # Free to see a flop; take it unless strong enough to raise.
            return Action(ActionType.CHECK)

        if plays_hand and options.can_call:
            if self._call_is_defensible(state, player, options, strength, rng):
                return Action(ActionType.CALL, options.call_amount)

        return Action(ActionType.FOLD)

    # -- postflop ---------------------------------------------------------

    def _postflop(
        self,
        state: HandState,
        player: Player,
        options: LegalActions,
        rng: Random,
        strength: float,
    ) -> Action:
        profile = self.profile

        if options.can_check:
            value_bet = strength >= 0.55 and rng.random() < profile.aggression
            bluff = strength < 0.30 and rng.random() < profile.bluff_frequency
            if (value_bet or bluff) and options.can_bet:
                return self._sized_raise(state, options, rng)
            return Action(ActionType.CHECK)

        # Facing a bet.
        strong = strength >= 0.72
        if options.can_raise and strong and rng.random() < profile.aggression:
            return self._sized_raise(state, options, rng)

        if options.can_call and self._call_is_defensible(
            state, player, options, strength, rng
        ):
            return Action(ActionType.CALL, options.call_amount)

        return Action(ActionType.FOLD)

    # -- shared helpers ---------------------------------------------------

    def _call_is_defensible(
        self,
        state: HandState,
        player: Player,
        options: LegalActions,
        strength: float,
        rng: Random,
    ) -> bool:
        """Compare heuristic strength against the pot odds and the profile's own threshold."""
        to_call = options.call_amount
        if to_call <= 0:
            return True
        return rng.random() < defend_chance(self.profile, strength, to_call, state.pot_total)

    def _sized_raise(
        self, state: HandState, options: LegalActions, rng: Random
    ) -> Action:
        """Bet or raise a pot fraction, clamped to what the rules allow."""
        pot = max(state.pot_total, state.config.big_blind)
        jitter = 1.0 + rng.uniform(-0.15, 0.15)
        target = state.current_bet + int(round(pot * self.profile.bet_sizing * jitter))

        target = max(target, options.min_raise_to)
        target = min(target, options.max_raise_to)

        action_type = ActionType.BET if options.can_bet else ActionType.RAISE
        return Action(action_type, target)
