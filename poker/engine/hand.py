"""The hand state machine: deal, betting rounds, showdown, payout.

A hand advances only through ``apply_action``. Everything else (dealing the
next street, running out an all-in board, awarding pots) happens as a
consequence of an action, so there is exactly one way for state to change.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from poker.cards import Card, Deck
from poker.engine.rules import LegalActions, legal_actions, validate_action
from poker.engine.state import (
    Action,
    ActionRecord,
    ActionType,
    GameConfig,
    HandState,
    Player,
    Pot,
    Street,
)
from poker.evaluator import HandValue, evaluate

__all__ = ["HandEngine", "HandResult"]


def _verb(player: Player) -> str:
    """Agree the verb with the seat name, since the hero may be called "You"."""
    return "win" if player.name.strip().lower() == "you" else "wins"


@dataclass
class HandResult:
    """Outcome of a completed hand."""

    hand_id: str
    winners: List[str]
    #: Gross chips received from pots, by player id.
    payouts: Dict[str, int]
    #: Stack change across the hand — the number that matters for bankroll.
    net: Dict[str, int]
    went_to_showdown: bool
    #: Evaluated hands, present only at showdown.
    showdown: Dict[str, dict] = field(default_factory=dict)
    board: List[str] = field(default_factory=list)
    pots: List[dict] = field(default_factory=list)
    #: Human-readable one-liner, e.g. "Villain wins 24 with Pair of Aces".
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "hand_id": self.hand_id,
            "winners": list(self.winners),
            "payouts": dict(self.payouts),
            "net": dict(self.net),
            "went_to_showdown": self.went_to_showdown,
            "showdown": dict(self.showdown),
            "board": list(self.board),
            "pots": list(self.pots),
            "summary": self.summary,
        }


class HandEngine:
    """Drives one hand of No-Limit Texas Hold'em."""

    def __init__(
        self,
        config: GameConfig,
        players: Sequence[Player],
        button_index: int,
        *,
        seed: Optional[int] = None,
        hand_id: Optional[str] = None,
    ) -> None:
        if len(players) < 2:
            raise ValueError("A hand needs at least two players")
        if not 0 <= button_index < len(players):
            raise ValueError("button_index out of range")

        self.deck = Deck(seed)
        self._starting_stacks = {p.player_id: p.stack for p in players}
        self.state = HandState(
            hand_id=hand_id or str(uuid.uuid4()),
            config=config,
            players=list(players),
            button_index=button_index,
            deck_seed=self.deck.seed,
        )
        self.result: Optional[HandResult] = None
        self._started = False

    # -- setup ------------------------------------------------------------

    def start(self) -> "HandEngine":
        """Deal hole cards and post blinds. Returns self for chaining."""
        if self._started:
            raise RuntimeError("Hand already started")
        self._started = True

        state = self.state
        for player in state.players:
            player.hole_cards = []
            player.committed_street = 0
            player.committed_hand = 0
            player.has_folded = False
            player.is_all_in = player.stack <= 0
            player.has_acted_this_street = False

        # Deal one card at a time starting left of the button, as at a table.
        order = self._seat_order(start_index=(state.button_index + 1) % len(state.players))
        for _ in range(2):
            for player in order:
                if player.stack > 0:
                    player.hole_cards.append(self.deck.deal_one())

        self._post_blinds()
        state.to_act_index = self._first_to_act_index(Street.PREFLOP)
        return self

    def _post_blinds(self) -> None:
        state = self.state
        small = state.players[state.small_blind_index]
        big = state.players[state.big_blind_index]

        self._post(small, state.config.small_blind)
        self._post(big, state.config.big_blind)

        state.current_bet = max(p.committed_street for p in state.players)
        # The big blind seeds the minimum raise increment for preflop.
        state.last_raise_size = state.config.big_blind

    def _post(self, player: Player, amount: int) -> None:
        state = self.state
        pot_before = state.pot_total
        to_call_before = max(0, state.current_bet - player.committed_street)
        committed = player.bet(amount)
        # Posting a blind is forced, so it does not count as acting — this is
        # what gives the big blind the option to raise a limped pot.
        state.action_log.append(
            ActionRecord(
                index=len(state.action_log),
                street=Street.PREFLOP,
                player_id=player.player_id,
                action=Action(ActionType.POST_BLIND, committed),
                chips_committed=committed,
                pot_before=pot_before,
                pot_after=state.pot_total,
                stack_after=player.stack,
                to_call_before=to_call_before,
                is_all_in=player.is_all_in,
                board=[],
            )
        )

    # -- queries ----------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        return self.state.is_complete

    def legal_actions(self) -> LegalActions:
        return legal_actions(self.state)

    # -- the one mutation -------------------------------------------------

    def apply_action(self, action: Action) -> "HandEngine":
        """Apply one player action and advance the hand as far as it can go."""
        state = self.state
        if state.is_complete:
            raise RuntimeError("Hand is already complete")
        player = state.to_act
        if player is None:
            raise RuntimeError("No player is to act")

        normalised = validate_action(state, player, action)
        pot_before = state.pot_total
        to_call_before = state.amount_to_call(player)
        committed = 0

        if normalised.type is ActionType.FOLD:
            player.has_folded = True
            player.has_acted_this_street = True

        elif normalised.type is ActionType.CHECK:
            player.has_acted_this_street = True

        elif normalised.type is ActionType.CALL:
            committed = player.bet(to_call_before)
            player.has_acted_this_street = True

        else:  # BET or RAISE
            target = normalised.amount
            increment = target - state.current_bet
            committed = player.bet(target - player.committed_street)

            min_full_raise = (
                state.last_raise_size or state.config.effective_min_bet
            )
            is_full_raise = increment >= min_full_raise

            state.current_bet = max(state.current_bet, player.committed_street)
            player.has_acted_this_street = True

            if is_full_raise:
                state.last_raise_size = increment
                # A full raise reopens the betting for everyone else.
                for other in state.players:
                    if other is not player and other.is_active:
                        other.has_acted_this_street = False

        state.action_log.append(
            ActionRecord(
                index=len(state.action_log),
                street=state.street,
                player_id=player.player_id,
                action=normalised,
                chips_committed=committed,
                pot_before=pot_before,
                pot_after=state.pot_total,
                stack_after=player.stack,
                to_call_before=to_call_before,
                is_all_in=player.is_all_in,
                board=[str(card) for card in state.board],
            )
        )

        self._advance()
        return self

    # -- progression ------------------------------------------------------

    def _advance(self) -> None:
        state = self.state

        if len(state.contesting_players) == 1:
            self._finish_without_showdown()
            return

        if not self._betting_round_complete():
            state.to_act_index = self._next_to_act_index()
            return

        # No further betting is possible: run the board out and show down.
        if len(state.active_players) <= 1 and state.street < Street.RIVER:
            self._return_uncalled_bet()
            while state.street < Street.RIVER:
                self._deal_next_street()
            self._showdown()
            return

        if state.street >= Street.RIVER:
            self._showdown()
            return

        self._deal_next_street()
        state.to_act_index = self._first_to_act_index(state.street)
        if state.to_act_index is None:
            # Everyone is all-in; finish out the remaining streets.
            self._advance()

    def _betting_round_complete(self) -> bool:
        active = self.state.active_players
        if not active:
            return True
        return all(
            player.has_acted_this_street
            and player.committed_street == self.state.current_bet
            for player in active
        )

    def _deal_next_street(self) -> None:
        state = self.state
        state.street = Street(state.street + 1)
        needed = state.street.cards_dealt - len(state.board)
        if needed > 0:
            state.board.extend(self.deck.deal(needed))

        for player in state.players:
            player.reset_for_street()
        state.current_bet = 0
        state.last_raise_size = 0

    # -- seating order ----------------------------------------------------

    def _seat_order(self, start_index: int) -> List[Player]:
        players = self.state.players
        count = len(players)
        return [players[(start_index + offset) % count] for offset in range(count)]

    def _first_to_act_index(self, street: Street) -> Optional[int]:
        """Heads-up: button acts first preflop, big blind acts first after."""
        state = self.state
        count = len(state.players)

        if street is Street.PREFLOP:
            if state.is_heads_up:
                start = state.button_index
            else:
                start = (state.big_blind_index + 1) % count
        else:
            if state.is_heads_up:
                start = state.big_blind_index
            else:
                start = (state.button_index + 1) % count

        for offset in range(count):
            index = (start + offset) % count
            if state.players[index].is_active:
                return index
        return None

    def _next_to_act_index(self) -> Optional[int]:
        state = self.state
        count = len(state.players)
        current = state.to_act_index if state.to_act_index is not None else 0
        for offset in range(1, count + 1):
            index = (current + offset) % count
            player = state.players[index]
            if player.is_active and not (
                player.has_acted_this_street
                and player.committed_street == state.current_bet
            ):
                return index
        return None

    # -- settlement -------------------------------------------------------

    def _return_uncalled_bet(self) -> None:
        """Give back the portion of a bet no opponent covered.

        Only the single largest contributor can have an uncalled amount, and
        only by the margin over the next largest.
        """
        state = self.state
        amounts = sorted(
            (p.committed_hand for p in state.players), reverse=True
        )
        if len(amounts) < 2 or amounts[0] <= amounts[1]:
            return

        excess = amounts[0] - amounts[1]
        top = next(p for p in state.players if p.committed_hand == amounts[0])
        top.stack += excess
        top.committed_hand -= excess
        top.committed_street = max(0, top.committed_street - excess)
        state.uncalled_returned += excess
        # Returning chips can un-do an all-in.
        if top.stack > 0:
            top.is_all_in = False

    def _finish_without_showdown(self) -> None:
        state = self.state
        self._return_uncalled_bet()
        winner = state.contesting_players[0]
        pots = state.build_pots()
        total = sum(pot.amount for pot in pots)
        winner.stack += total

        folded = len(state.players) - 1
        self._finalise(
            winners=[winner.player_id],
            payouts={winner.player_id: total},
            went_to_showdown=False,
            showdown={},
            pots=pots,
            summary=(
                f"{winner.name} {_verb(winner)} {total} — "
                f"{'opponent' if folded == 1 else 'opponents'} folded"
            ),
        )

    def _showdown(self) -> None:
        state = self.state
        state.street = Street.SHOWDOWN
        self._return_uncalled_bet()

        values: Dict[str, HandValue] = {
            player.player_id: evaluate(list(player.hole_cards) + list(state.board))
            for player in state.contesting_players
        }

        pots = state.build_pots()
        payouts: Dict[str, int] = {p.player_id: 0 for p in state.players}
        all_winners: List[str] = []

        for pot in pots:
            contenders = [
                pid for pid in pot.eligible_player_ids if pid in values
            ]
            if not contenders:
                continue
            best = max(values[pid] for pid in contenders)
            winners = [pid for pid in contenders if values[pid] == best]

            share, remainder = divmod(pot.amount, len(winners))
            for pid in winners:
                payouts[pid] += share
            # Odd chips go to the first winner left of the button.
            for offset, pid in enumerate(self._odd_chip_order(winners)):
                if offset < remainder:
                    payouts[pid] += 1
            all_winners.extend(w for w in winners if w not in all_winners)

        for player in state.players:
            player.stack += payouts[player.player_id]

        best_overall = max(values.values()) if values else None
        winner_names = [state.player_by_id(pid).name for pid in all_winners]
        if len(all_winners) > 1:
            summary = (
                f"Split pot: {' and '.join(winner_names)} "
                f"both hold {best_overall.describe() if best_overall else ''}"
            )
        elif all_winners:
            pid = all_winners[0]
            winner = state.player_by_id(pid)
            summary = (
                f"{winner.name} {_verb(winner)} "
                f"{payouts[pid]} with {values[pid].describe()}"
            )
        else:
            summary = "No winner"

        self._finalise(
            winners=all_winners,
            payouts={k: v for k, v in payouts.items() if v},
            went_to_showdown=True,
            showdown={
                pid: {
                    **value.to_dict(),
                    "hole_cards": [
                        str(c) for c in state.player_by_id(pid).hole_cards
                    ],
                }
                for pid, value in values.items()
            },
            pots=pots,
            summary=summary,
        )

    def _odd_chip_order(self, winners: Sequence[str]) -> List[str]:
        ordered = self._seat_order(
            start_index=(self.state.button_index + 1) % len(self.state.players)
        )
        return [p.player_id for p in ordered if p.player_id in winners]

    def _finalise(
        self,
        *,
        winners: List[str],
        payouts: Dict[str, int],
        went_to_showdown: bool,
        showdown: Dict[str, dict],
        pots: List[Pot],
        summary: str,
    ) -> None:
        state = self.state
        state.is_complete = True
        state.to_act_index = None

        self.result = HandResult(
            hand_id=state.hand_id,
            winners=winners,
            payouts=payouts,
            net={
                p.player_id: p.stack - self._starting_stacks[p.player_id]
                for p in state.players
            },
            went_to_showdown=went_to_showdown,
            showdown=showdown,
            board=[str(card) for card in state.board],
            pots=[pot.to_dict() for pot in pots],
            summary=summary,
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """Client-facing state. Opponent cards appear only once the hand ends."""
        payload = self.state.to_dict(reveal_all=self.state.is_complete)
        payload["legal_actions"] = (
            self.legal_actions().to_dict() if not self.state.is_complete else None
        )
        payload["result"] = self.result.to_dict() if self.result else None
        payload["starting_stacks"] = dict(self._starting_stacks)
        return payload
