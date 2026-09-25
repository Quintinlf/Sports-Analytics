"""Betting rules, pot mechanics, and hand progression.

These cover the cases a poker engine most often gets wrong: heads-up blind
order, the big blind's option, the minimum-raise ladder, short all-ins that
must not reopen betting, side pots, split pots, and uncalled bet returns.
"""
from __future__ import annotations

import unittest
from random import Random

from poker.cards import parse_cards
from poker.engine import (
    Action,
    ActionType,
    GameConfig,
    HandEngine,
    Player,
    Street,
)
from poker.engine.rules import IllegalAction, legal_actions
from poker.engine.state import HandState
from poker.session import PokerSession

CONFIG = GameConfig(small_blind=1, big_blind=2, starting_stack=200)


def seats(hero_stack: int = 200, villain_stack: int = 200):
    return [
        Player("hero", "Hero", hero_stack, is_hero=True),
        Player("villain", "Villain", villain_stack),
    ]


def new_hand(hero_stack=200, villain_stack=200, button=0, config=CONFIG, seed=1):
    return HandEngine(
        config, seats(hero_stack, villain_stack), button, seed=seed
    ).start()


def chips_in_play(engine: HandEngine) -> int:
    """Total chips visible in the game.

    While a hand is live, chips sit either in a stack or in the pot. Once the
    hand settles the pot is paid into the stacks, but ``committed_hand`` is
    deliberately retained as the historical record of what was wagered — so
    after settlement only the stacks may be counted.
    """
    if engine.state.is_complete:
        return sum(p.stack for p in engine.state.players)
    return sum(p.stack for p in engine.state.players) + engine.state.pot_total


# ---------------------------------------------------------------------------
# Blinds and position
# ---------------------------------------------------------------------------

class TestHeadsUpBlindsAndPosition(unittest.TestCase):
    def test_button_posts_the_small_blind(self) -> None:
        engine = new_hand(button=0)
        hero = engine.state.player_by_id("hero")
        villain = engine.state.player_by_id("villain")
        self.assertEqual(hero.committed_street, 1)
        self.assertEqual(villain.committed_street, 2)

    def test_button_acts_first_preflop(self) -> None:
        engine = new_hand(button=0)
        self.assertEqual(engine.state.to_act.player_id, "hero")

    def test_big_blind_acts_first_postflop(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.CALL))
        engine.apply_action(Action(ActionType.CHECK))
        self.assertEqual(engine.state.street, Street.FLOP)
        self.assertEqual(engine.state.to_act.player_id, "villain")

    def test_button_alternates_between_hands(self) -> None:
        session = PokerSession(config=CONFIG, seed=3)
        session.start_hand()
        first = session.button_index
        while not session.engine.is_complete:
            legal = session.engine.legal_actions()
            session.hero_action("check" if legal.can_check else "fold")
        session.start_hand()
        self.assertNotEqual(session.button_index, first)

    def test_position_labels(self) -> None:
        engine = new_hand(button=0)
        self.assertEqual(engine.state.position_label(0), "BTN/SB")
        self.assertEqual(engine.state.position_label(1), "BB")


class TestBigBlindOption(unittest.TestCase):
    def test_big_blind_may_raise_after_a_limp(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.CALL))
        self.assertEqual(engine.state.to_act.player_id, "villain")
        options = engine.legal_actions()
        self.assertTrue(options.can_check, "BB should be able to check its option")
        self.assertTrue(options.can_raise, "BB should be able to raise its option")

    def test_checking_the_option_advances_to_the_flop(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.CALL))
        engine.apply_action(Action(ActionType.CHECK))
        self.assertEqual(engine.state.street, Street.FLOP)
        self.assertEqual(len(engine.state.board), 3)


# ---------------------------------------------------------------------------
# Betting rules
# ---------------------------------------------------------------------------

class TestMinimumRaise(unittest.TestCase):
    def test_first_preflop_raise_is_two_big_blinds(self) -> None:
        engine = new_hand(button=0)
        self.assertEqual(engine.legal_actions().min_raise_to, 4)

    def test_reraise_must_match_the_previous_increment(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.RAISE, 6))   # +4 over the BB
        self.assertEqual(engine.legal_actions().min_raise_to, 10)
        engine.apply_action(Action(ActionType.RAISE, 10))  # +4 again
        self.assertEqual(engine.legal_actions().min_raise_to, 14)

    def test_large_raise_widens_the_next_minimum(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.RAISE, 20))  # +18 over the BB
        self.assertEqual(engine.legal_actions().min_raise_to, 38)

    def test_below_minimum_raise_is_rejected(self) -> None:
        engine = new_hand(button=0)
        with self.assertRaises(IllegalAction):
            engine.apply_action(Action(ActionType.RAISE, 3))

    def test_postflop_minimum_bet_is_the_big_blind(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.CALL))
        engine.apply_action(Action(ActionType.CHECK))
        self.assertEqual(engine.legal_actions().min_raise_to, CONFIG.big_blind)

    def test_oversized_raise_is_clamped_to_all_in(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.RAISE, 10_000))
        hero = engine.state.player_by_id("hero")
        self.assertEqual(hero.stack, 0)
        self.assertTrue(hero.is_all_in)
        self.assertEqual(hero.committed_street, 200)


class TestIllegalActions(unittest.TestCase):
    def test_cannot_check_facing_a_bet(self) -> None:
        engine = new_hand(button=0)
        with self.assertRaises(IllegalAction):
            engine.apply_action(Action(ActionType.CHECK))

    def test_cannot_call_when_nothing_is_owed(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.CALL))
        with self.assertRaises(IllegalAction):
            engine.apply_action(Action(ActionType.CALL))  # BB owes nothing

    def test_cannot_bet_when_a_bet_already_stands(self) -> None:
        engine = new_hand(button=0)
        with self.assertRaises(IllegalAction):
            engine.apply_action(Action(ActionType.BET, 10))

    def test_cannot_act_after_the_hand_completes(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.FOLD))
        with self.assertRaises(RuntimeError):
            engine.apply_action(Action(ActionType.CHECK))


class TestShortAllInDoesNotReopenBetting(unittest.TestCase):
    def test_all_in_below_a_full_raise_only_allows_call_or_fold(self) -> None:
        # Villain is short enough that shoving is less than a full re-raise.
        engine = new_hand(hero_stack=200, villain_stack=14, button=0)
        engine.apply_action(Action(ActionType.RAISE, 10))   # hero raises to 10
        engine.apply_action(Action(ActionType.RAISE, 14))   # villain shoves 14 (+4 < +8)

        options = engine.legal_actions()
        self.assertEqual(engine.state.to_act.player_id, "hero")
        self.assertTrue(options.can_call)
        self.assertFalse(
            options.can_raise,
            "a short all-in must not reopen the betting",
        )

    def test_full_raise_does_reopen_betting(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.RAISE, 10))
        engine.apply_action(Action(ActionType.RAISE, 18))  # +8 == full raise
        self.assertTrue(engine.legal_actions().can_raise)


# ---------------------------------------------------------------------------
# Pots
# ---------------------------------------------------------------------------

class TestPotConstruction(unittest.TestCase):
    def _state(self, commitments, folded=()):
        players = []
        for pid, amount in commitments.items():
            player = Player(pid, pid.title(), 0)
            player.committed_hand = amount
            player.has_folded = pid in folded
            players.append(player)
        return HandState(
            hand_id="t", config=CONFIG, players=players,
            button_index=0, deck_seed=0,
        )

    def test_single_pot_when_all_commit_equally(self) -> None:
        pots = self._state({"a": 50, "b": 50}).build_pots()
        self.assertEqual(len(pots), 1)
        self.assertEqual(pots[0].amount, 100)
        self.assertEqual(pots[0].eligible_player_ids, ["a", "b"])

    def test_side_pot_created_by_a_short_all_in(self) -> None:
        # a is all-in for 50, b and c continue to 120.
        pots = self._state({"a": 50, "b": 120, "c": 120}).build_pots()
        self.assertEqual(len(pots), 2)

        main, side = pots
        self.assertEqual(main.amount, 150)          # 50 x 3
        self.assertEqual(main.eligible_player_ids, ["a", "b", "c"])
        self.assertFalse(main.is_side_pot)

        self.assertEqual(side.amount, 140)          # 70 x 2
        self.assertEqual(side.eligible_player_ids, ["b", "c"])
        self.assertTrue(side.is_side_pot)

    def test_two_side_pots_from_three_stack_sizes(self) -> None:
        pots = self._state({"a": 30, "b": 80, "c": 200}).build_pots()
        self.assertEqual([p.amount for p in pots], [90, 100, 120])
        self.assertEqual(
            [p.eligible_player_ids for p in pots],
            [["a", "b", "c"], ["b", "c"], ["c"]],
        )

    def test_folded_players_contribute_chips_but_win_nothing(self) -> None:
        pots = self._state({"a": 40, "b": 40, "c": 40}, folded=("c",)).build_pots()
        self.assertEqual(len(pots), 1)
        self.assertEqual(pots[0].amount, 120)
        self.assertEqual(pots[0].eligible_player_ids, ["a", "b"])

    def test_total_chips_across_pots_equals_total_committed(self) -> None:
        commitments = {"a": 17, "b": 93, "c": 250, "d": 93}
        pots = self._state(commitments).build_pots()
        self.assertEqual(sum(p.amount for p in pots), sum(commitments.values()))


class TestUncalledBets(unittest.TestCase):
    def test_uncalled_raise_is_returned_on_a_fold(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.RAISE, 50))
        engine.apply_action(Action(ActionType.FOLD))

        hero = engine.state.player_by_id("hero")
        # Hero risked 50, only 2 was called; 48 comes back plus the 4-chip pot.
        self.assertEqual(engine.state.uncalled_returned, 48)
        self.assertEqual(hero.stack, 202)
        self.assertEqual(engine.result.net["hero"], 2)
        self.assertEqual(engine.result.net["villain"], -2)

    def test_uncalled_portion_of_an_over_shove_is_returned(self) -> None:
        engine = new_hand(hero_stack=200, villain_stack=60, button=0)
        engine.apply_action(Action(ActionType.RAISE, 200))  # hero shoves 200
        engine.apply_action(Action(ActionType.CALL))        # villain can only cover 60

        self.assertEqual(engine.state.uncalled_returned, 140)
        self.assertEqual(sum(p.stack for p in engine.state.players), 260)


# ---------------------------------------------------------------------------
# Showdown
# ---------------------------------------------------------------------------

class TestShowdown(unittest.TestCase):
    def _rigged(self, hero_cards, villain_cards, board):
        """Force a specific showdown, bypassing the shuffle."""
        engine = new_hand(button=0)
        engine.state.player_by_id("hero").hole_cards = parse_cards(hero_cards)
        engine.state.player_by_id("villain").hole_cards = parse_cards(villain_cards)
        engine.state.board = parse_cards(board)
        return engine

    def test_better_hand_wins_the_pot(self) -> None:
        engine = self._rigged("As Ac", "Kd Kh", "2c 7d 9s Ts 3h")
        engine.state.street = Street.RIVER
        engine._showdown()
        self.assertEqual(engine.result.winners, ["hero"])

    def test_identical_hands_split_the_pot(self) -> None:
        engine = self._rigged("As Ac", "Ad Ah", "2c 7d 9s Ts 3h")
        engine.state.street = Street.RIVER
        engine._showdown()
        self.assertEqual(sorted(engine.result.winners), ["hero", "villain"])
        self.assertEqual(engine.result.payouts["hero"], engine.result.payouts["villain"])

    def test_board_plays_and_the_pot_splits(self) -> None:
        # A royal flush on the board beats anything either player holds.
        engine = self._rigged("2c 3d", "4h 5s", "As Ks Qs Js Ts")
        engine.state.street = Street.RIVER
        engine._showdown()
        self.assertEqual(sorted(engine.result.winners), ["hero", "villain"])

    def test_kicker_decides_when_the_board_pairs_both_players(self) -> None:
        engine = self._rigged("Ah Kd", "Ac 2d", "As 8c 4h 9d 3s")
        engine.state.street = Street.RIVER
        engine._showdown()
        self.assertEqual(engine.result.winners, ["hero"])

    def test_unequal_contribution_is_returned_rather_than_split(self) -> None:
        # Heads-up, a lone extra chip was never called, so it goes back
        # instead of being split.
        engine = self._rigged("As Ac", "Ad Ah", "2c 7d 9s Ts 3h")
        engine.state.players[0].committed_hand = 5
        engine.state.players[1].committed_hand = 6
        engine.state.street = Street.RIVER
        engine._showdown()

        self.assertEqual(engine.state.uncalled_returned, 1)
        self.assertEqual(engine.result.payouts, {"hero": 5, "villain": 5})

    def test_odd_chip_awarded_to_first_winner_left_of_the_button(self) -> None:
        # Odd chips only arise with three or more contributors: a 15-chip pot
        # split between two tied winners leaves one chip over.
        players = [
            Player("a", "A", 100, is_hero=True),
            Player("b", "B", 100),
            Player("c", "C", 100),
        ]
        engine = HandEngine(CONFIG, players, button_index=0, seed=5).start()
        engine.state.player_by_id("a").hole_cards = parse_cards("As Ac")
        engine.state.player_by_id("b").hole_cards = parse_cards("Ad Ah")
        engine.state.player_by_id("c").hole_cards = parse_cards("2h 3c")
        engine.state.board = parse_cards("Kd 7d 9s Ts 4h")
        engine.state.player_by_id("c").has_folded = True
        for player in engine.state.players:
            player.committed_hand = 5
        engine.state.street = Street.RIVER
        engine._showdown()

        self.assertEqual(sorted(engine.result.winners), ["a", "b"])
        self.assertEqual(sum(engine.result.payouts.values()), 15)
        # Player b sits immediately left of the button, so it takes the odd chip.
        self.assertEqual(engine.result.payouts["b"], 8)
        self.assertEqual(engine.result.payouts["a"], 7)


class TestAllInRunout(unittest.TestCase):
    def test_all_in_deals_the_full_board_and_shows_down(self) -> None:
        engine = new_hand(hero_stack=50, villain_stack=200, button=0)
        engine.apply_action(Action(ActionType.RAISE, 50))
        engine.apply_action(Action(ActionType.CALL))

        self.assertTrue(engine.is_complete)
        self.assertEqual(len(engine.state.board), 5)
        self.assertTrue(engine.result.went_to_showdown)

    def test_chips_are_conserved_through_an_all_in(self) -> None:
        engine = new_hand(hero_stack=50, villain_stack=200, button=0)
        engine.apply_action(Action(ActionType.RAISE, 50))
        engine.apply_action(Action(ActionType.CALL))
        self.assertEqual(sum(p.stack for p in engine.state.players), 250)


class TestHandProgression(unittest.TestCase):
    def test_streets_advance_in_order(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.CALL))
        engine.apply_action(Action(ActionType.CHECK))
        self.assertEqual((engine.state.street, len(engine.state.board)), (Street.FLOP, 3))

        engine.apply_action(Action(ActionType.CHECK))
        engine.apply_action(Action(ActionType.CHECK))
        self.assertEqual((engine.state.street, len(engine.state.board)), (Street.TURN, 4))

        engine.apply_action(Action(ActionType.CHECK))
        engine.apply_action(Action(ActionType.CHECK))
        self.assertEqual((engine.state.street, len(engine.state.board)), (Street.RIVER, 5))

    def test_fold_ends_the_hand_immediately(self) -> None:
        engine = new_hand(button=0)
        engine.apply_action(Action(ActionType.FOLD))
        self.assertTrue(engine.is_complete)
        self.assertFalse(engine.result.went_to_showdown)
        self.assertEqual(engine.result.winners, ["villain"])

    def test_hole_cards_stay_hidden_until_the_hand_ends(self) -> None:
        engine = new_hand(button=0)
        villain = next(
            p for p in engine.to_dict()["players"] if p["player_id"] == "villain"
        )
        self.assertIsNone(villain["hole_cards"], "opponent cards must not leak")

        engine.apply_action(Action(ActionType.FOLD))
        villain = next(
            p for p in engine.to_dict()["players"] if p["player_id"] == "villain"
        )
        self.assertIsNotNone(villain["hole_cards"])

    def test_every_player_receives_two_hole_cards(self) -> None:
        engine = new_hand(button=0)
        for player in engine.state.players:
            self.assertEqual(len(player.hole_cards), 2)
        self.assertEqual(len(set(
            card for p in engine.state.players for card in p.hole_cards
        )), 4)


class TestInvariantsUnderRandomPlay(unittest.TestCase):
    """Fuzz the engine; chips must never be created or destroyed."""

    def test_chips_conserved_across_many_random_hands(self) -> None:
        rng = Random(20260806)
        hands_played = 0

        for trial in range(25):
            engine = new_hand(button=trial % 2, seed=trial)
            expected = chips_in_play(engine)

            guard = 0
            while not engine.is_complete and guard < 200:
                guard += 1
                options = engine.legal_actions()
                choices = []
                if options.can_check:
                    choices.append(Action(ActionType.CHECK))
                if options.can_call:
                    choices.append(Action(ActionType.CALL))
                if options.can_bet or options.can_raise:
                    choices.append(
                        Action(
                            ActionType.BET if options.can_bet else ActionType.RAISE,
                            rng.randint(options.min_raise_to, options.max_raise_to),
                        )
                    )
                choices.append(Action(ActionType.FOLD))
                engine.apply_action(rng.choice(choices))

                self.assertEqual(
                    chips_in_play(engine), expected, "chips were created or destroyed"
                )

            self.assertTrue(engine.is_complete)
            self.assertEqual(sum(p.stack for p in engine.state.players), expected)
            hands_played += 1

        self.assertEqual(hands_played, 25)

    def test_no_player_ever_goes_negative(self) -> None:
        rng = Random(99)
        for trial in range(25):
            engine = new_hand(button=trial % 2, seed=1000 + trial)
            guard = 0
            while not engine.is_complete and guard < 200:
                guard += 1
                options = engine.legal_actions()
                if options.can_bet or options.can_raise:
                    action = Action(
                        ActionType.BET if options.can_bet else ActionType.RAISE,
                        rng.randint(options.min_raise_to, options.max_raise_to),
                    )
                elif options.can_call:
                    action = Action(ActionType.CALL)
                else:
                    action = Action(ActionType.CHECK)
                engine.apply_action(action)
                for player in engine.state.players:
                    self.assertGreaterEqual(player.stack, 0)


if __name__ == "__main__":
    unittest.main()
