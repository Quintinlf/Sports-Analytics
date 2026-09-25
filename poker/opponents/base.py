"""Opponent interface and behavioural parameters.

Every opponent is defined by a small set of numeric parameters rather than
hard-coded logic branches. That matters for two later milestones: opponent
archetypes become data rather than new classes, and opponent modelling becomes
the problem of *estimating these same numbers* from observed actions — the
model and the opponent share one vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, Protocol

from poker.engine.rules import LegalActions
from poker.engine.state import Action, HandState, Player

__all__ = ["OpponentProfile", "Opponent", "PROFILES", "get_profile"]


@dataclass(frozen=True)
class OpponentProfile:
    """Measurable behavioural parameters for an opponent.

    All frequencies are in [0, 1]. The names match the player-statistics
    vocabulary deliberately, so an opponent's configured VPIP and its
    *observed* VPIP can be compared directly.
    """

    key: str
    name: str
    description: str

    #: Fraction of hands played rather than folded preflop.
    vpip: float = 0.45
    #: Fraction of hands raised preflop.
    pfr: float = 0.25
    #: Willingness to bet or raise when holding a strong hand.
    aggression: float = 0.5
    #: How often a weak hand becomes a bluff.
    bluff_frequency: float = 0.10
    #: Hand strength required to continue against a bet, before pot odds.
    call_threshold: float = 0.35
    #: Tendency to fold marginal hands when facing aggression.
    fold_to_aggression: float = 0.45
    #: Bet size as a fraction of the pot when betting or raising.
    bet_sizing: float = 0.6

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "vpip": self.vpip,
            "pfr": self.pfr,
            "aggression": self.aggression,
            "bluff_frequency": self.bluff_frequency,
            "call_threshold": self.call_threshold,
            "fold_to_aggression": self.fold_to_aggression,
            "bet_sizing": self.bet_sizing,
        }


class Opponent(Protocol):
    """Anything that can act for a non-hero seat."""

    profile: OpponentProfile

    def choose_action(
        self,
        state: HandState,
        player: Player,
        options: LegalActions,
        rng: Random,
    ) -> Action:
        """Pick a legal action for ``player`` given the current hand state.

        Implementations may read ``player.hole_cards`` (their own) and the
        public board, but must not read another player's hole cards.
        """
        ...


#: Archetypes from the project brief. Milestone 1 ships the parameter surface
#: and a single heuristic policy that reads it; adaptive modelling comes later.
PROFILES: Dict[str, OpponentProfile] = {
    profile.key: profile
    for profile in (
        OpponentProfile(
            key="balanced",
            name="Balanced",
            description=(
                "Plays a reasonable range and bets its good hands. "
                "A fair default sparring partner."
            ),
            vpip=0.45, pfr=0.25, aggression=0.55, bluff_frequency=0.10,
            call_threshold=0.35, fold_to_aggression=0.45, bet_sizing=0.60,
        ),
        OpponentProfile(
            key="tight_passive",
            name="Tight-Passive (Rock)",
            description=(
                "Folds most hands and rarely raises. When it does bet, "
                "believe it."
            ),
            vpip=0.20, pfr=0.08, aggression=0.25, bluff_frequency=0.02,
            call_threshold=0.45, fold_to_aggression=0.65, bet_sizing=0.45,
        ),
        OpponentProfile(
            key="loose_aggressive",
            name="Loose-Aggressive",
            description=(
                "Plays many hands and applies constant pressure. "
                "Hard to read, but over-bluffs."
            ),
            vpip=0.70, pfr=0.50, aggression=0.80, bluff_frequency=0.30,
            call_threshold=0.25, fold_to_aggression=0.30, bet_sizing=0.75,
        ),
        OpponentProfile(
            key="calling_station",
            name="Calling Station",
            description=(
                "Calls far too often and almost never raises. "
                "Value bet relentlessly; bluffing is a waste."
            ),
            vpip=0.75, pfr=0.05, aggression=0.15, bluff_frequency=0.03,
            call_threshold=0.12, fold_to_aggression=0.12, bet_sizing=0.40,
        ),
        OpponentProfile(
            key="maniac",
            name="Maniac",
            description=(
                "Raises relentlessly with anything. Wait for a hand, "
                "then let it pay you off."
            ),
            vpip=0.90, pfr=0.75, aggression=0.92, bluff_frequency=0.50,
            call_threshold=0.15, fold_to_aggression=0.18, bet_sizing=0.95,
        ),
    )
}

DEFAULT_PROFILE_KEY = "balanced"


def get_profile(key: str) -> OpponentProfile:
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(
            f"Unknown opponent profile {key!r}; "
            f"available: {sorted(PROFILES)}"
        ) from None
