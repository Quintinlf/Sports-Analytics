"""Data model for a single hand of No-Limit Texas Hold'em.

Chips are integers throughout. Floating point money in a betting engine
produces pots that fail to balance, so it is avoided entirely.

Betting amount convention: for BET and RAISE, ``Action.amount`` is the total
this player will have committed *on the current street* once the action is
applied — the standard "raise to" convention. CALL, CHECK and FOLD ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Sequence

from poker.cards import Card

__all__ = [
    "Street",
    "ActionType",
    "Action",
    "ActionRecord",
    "Player",
    "Pot",
    "GameConfig",
    "HandState",
]


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def cards_dealt(self) -> int:
        """Community cards face-up during this street."""
        return {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
            Street.SHOWDOWN: 5,
        }[self]


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    POST_BLIND = "post_blind"

    @property
    def is_aggressive(self) -> bool:
        return self in (ActionType.BET, ActionType.RAISE)


@dataclass(frozen=True)
class Action:
    """A player decision. ``amount`` is the total street commitment to raise to."""

    type: ActionType
    amount: int = 0

    def __str__(self) -> str:
        if self.type in (ActionType.BET, ActionType.RAISE):
            return f"{self.type.value} to {self.amount}"
        return self.type.value


@dataclass
class ActionRecord:
    """One entry in the hand history.

    Captures enough surrounding state that the hand can be replayed and
    audited without re-running the engine.
    """

    index: int
    street: Street
    player_id: str
    action: Action
    #: Chips actually moved from the stack by this action.
    chips_committed: int
    pot_before: int
    pot_after: int
    stack_after: int
    to_call_before: int
    is_all_in: bool
    board: List[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "street": self.street.name,
            "player_id": self.player_id,
            "action": self.action.type.value,
            "amount": self.action.amount,
            "chips_committed": self.chips_committed,
            "pot_before": self.pot_before,
            "pot_after": self.pot_after,
            "stack_after": self.stack_after,
            "to_call_before": self.to_call_before,
            "is_all_in": self.is_all_in,
            "board": list(self.board),
            "timestamp": self.timestamp,
        }


@dataclass
class Player:
    """A seat at the table for the duration of one hand."""

    player_id: str
    name: str
    stack: int
    is_hero: bool = False
    hole_cards: List[Card] = field(default_factory=list)

    #: Chips committed on the current street only. Reset each street.
    committed_street: int = 0
    #: Chips committed across the whole hand. Drives side-pot construction.
    committed_hand: int = 0

    has_folded: bool = False
    is_all_in: bool = False
    #: False means this player still owes an action before the street can end.
    #: A full raise resets this to False for everyone else, which is exactly
    #: what reopens the betting.
    has_acted_this_street: bool = False

    @property
    def is_active(self) -> bool:
        """Still in the hand and still able to put chips in."""
        return not self.has_folded and not self.is_all_in

    @property
    def is_contesting(self) -> bool:
        """Still eligible to win the pot (may be all-in)."""
        return not self.has_folded

    def bet(self, amount: int) -> int:
        """Move ``amount`` chips from the stack into the pot. Caps at all-in."""
        actual = min(amount, self.stack)
        self.stack -= actual
        self.committed_street += actual
        self.committed_hand += actual
        if self.stack == 0:
            self.is_all_in = True
        return actual

    def reset_for_street(self) -> None:
        self.committed_street = 0
        self.has_acted_this_street = False


@dataclass
class Pot:
    """A main or side pot and the players eligible to win it."""

    amount: int
    eligible_player_ids: List[str]
    is_side_pot: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "amount": self.amount,
            "eligible_player_ids": list(self.eligible_player_ids),
            "is_side_pot": self.is_side_pot,
        }


@dataclass(frozen=True)
class GameConfig:
    """Table stakes and structure."""

    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 200
    #: Minimum bet on any street. In standard NLHE this is the big blind.
    min_bet: Optional[int] = None

    def __post_init__(self) -> None:
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if self.small_blind > self.big_blind:
            raise ValueError("Small blind cannot exceed big blind")
        if self.starting_stack < self.big_blind:
            raise ValueError("Starting stack must cover the big blind")

    @property
    def effective_min_bet(self) -> int:
        return self.min_bet if self.min_bet is not None else self.big_blind

    def to_dict(self) -> Dict[str, object]:
        return {
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stack": self.starting_stack,
            "min_bet": self.effective_min_bet,
        }


@dataclass
class HandState:
    """Everything that defines the current position in a hand."""

    hand_id: str
    config: GameConfig
    players: List[Player]
    button_index: int
    deck_seed: int

    street: Street = Street.PREFLOP
    board: List[Card] = field(default_factory=list)
    action_log: List[ActionRecord] = field(default_factory=list)

    #: Highest street commitment any player has reached. What others must match.
    current_bet: int = 0
    #: Size of the last full bet or raise increment; sets the minimum re-raise.
    last_raise_size: int = 0
    #: Index into ``players`` of whoever must act, or None when no one can.
    to_act_index: Optional[int] = None

    is_complete: bool = False
    #: Chips returned because no one called them. Never enters a pot.
    uncalled_returned: int = 0

    # -- lookups ----------------------------------------------------------

    def player_by_id(self, player_id: str) -> Player:
        for player in self.players:
            if player.player_id == player_id:
                return player
        raise KeyError(f"No player {player_id!r}")

    @property
    def hero(self) -> Player:
        return next(p for p in self.players if p.is_hero)

    @property
    def to_act(self) -> Optional[Player]:
        if self.to_act_index is None:
            return None
        return self.players[self.to_act_index]

    @property
    def is_heads_up(self) -> bool:
        return len(self.players) == 2

    # -- money ------------------------------------------------------------

    @property
    def pot_total(self) -> int:
        """All chips committed this hand, including the current street."""
        return sum(player.committed_hand for player in self.players)

    def amount_to_call(self, player: Player) -> int:
        """Chips this player must add to match the current bet."""
        return max(0, min(self.current_bet - player.committed_street, player.stack))

    @property
    def contesting_players(self) -> List[Player]:
        return [p for p in self.players if p.is_contesting]

    @property
    def active_players(self) -> List[Player]:
        return [p for p in self.players if p.is_active]

    # -- blind positions --------------------------------------------------

    @property
    def small_blind_index(self) -> int:
        """Heads-up, the button posts the small blind."""
        if self.is_heads_up:
            return self.button_index
        return (self.button_index + 1) % len(self.players)

    @property
    def big_blind_index(self) -> int:
        if self.is_heads_up:
            return (self.button_index + 1) % len(self.players)
        return (self.button_index + 2) % len(self.players)

    def position_label(self, index: int) -> str:
        if self.is_heads_up:
            return "BTN/SB" if index == self.button_index else "BB"
        if index == self.button_index:
            return "BTN"
        if index == self.small_blind_index:
            return "SB"
        if index == self.big_blind_index:
            return "BB"
        return "MP"

    # -- pot construction -------------------------------------------------

    def build_pots(self) -> List[Pot]:
        """Split committed chips into main and side pots.

        Works by contribution level: every distinct commitment amount opens a
        layer, and only players who reached that layer can win it. Folded
        players' chips stay in the pot but win nothing.
        """
        contributions = {
            p.player_id: p.committed_hand for p in self.players if p.committed_hand > 0
        }
        if not contributions:
            return []

        levels = sorted(set(contributions.values()))
        pots: List[Pot] = []
        previous = 0

        for level in levels:
            layer = sum(
                min(amount, level) - min(amount, previous)
                for amount in contributions.values()
            )
            eligible = sorted(
                p.player_id
                for p in self.players
                if p.is_contesting and p.committed_hand >= level
            )
            if layer > 0 and eligible:
                pots.append(
                    Pot(
                        amount=layer,
                        eligible_player_ids=eligible,
                        is_side_pot=bool(pots),
                    )
                )
            previous = level

        # Consecutive layers with the same eligible set are one pot.
        merged: List[Pot] = []
        for pot in pots:
            if merged and merged[-1].eligible_player_ids == pot.eligible_player_ids:
                merged[-1].amount += pot.amount
            else:
                merged.append(pot)
        for index, pot in enumerate(merged):
            pot.is_side_pot = index > 0
        return merged

    # -- serialisation ----------------------------------------------------

    def to_dict(self, *, reveal_all: bool = False) -> Dict[str, object]:
        """Serialise for the API.

        ``reveal_all`` must stay False while the hand is live — otherwise the
        response would leak the opponent's hole cards to the browser.
        """
        return {
            "hand_id": self.hand_id,
            "config": self.config.to_dict(),
            "street": self.street.name,
            "board": [str(card) for card in self.board],
            "board_display": [card.display for card in self.board],
            "pot_total": self.pot_total,
            "current_bet": self.current_bet,
            "last_raise_size": self.last_raise_size,
            "button_index": self.button_index,
            "to_act_index": self.to_act_index,
            "is_complete": self.is_complete,
            "deck_seed": self.deck_seed,
            "uncalled_returned": self.uncalled_returned,
            "pots": [pot.to_dict() for pot in self.build_pots()],
            "players": [
                {
                    "player_id": p.player_id,
                    "name": p.name,
                    "stack": p.stack,
                    "is_hero": p.is_hero,
                    "position": self.position_label(index),
                    "committed_street": p.committed_street,
                    "committed_hand": p.committed_hand,
                    "has_folded": p.has_folded,
                    "is_all_in": p.is_all_in,
                    "to_call": self.amount_to_call(p),
                    "hole_cards": (
                        [str(c) for c in p.hole_cards]
                        if (reveal_all or p.is_hero)
                        else None
                    ),
                    "hole_cards_display": (
                        [c.display for c in p.hole_cards]
                        if (reveal_all or p.is_hero)
                        else None
                    ),
                }
                for index, p in enumerate(self.players)
            ],
            "action_log": [record.to_dict() for record in self.action_log],
        }
