"""Poker API routes, decision mathematics, and session behaviour.

The router is mounted on a bare FastAPI app rather than importing
``backend.main``, so these tests need no database and no application startup.
"""
from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.poker import router
from poker.decision import build_decision_context
from poker.engine import Action, ActionType, GameConfig, HandEngine, Player
from poker.session import LearningMode, PokerSession


def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Decision mathematics
# ---------------------------------------------------------------------------

class TestPotOdds(unittest.TestCase):
    """Pot odds are exact arithmetic and must match by hand."""

    def _context(self, hero_stack=200, villain_stack=200, raise_to=None):
        players = [
            Player("hero", "Hero", hero_stack, is_hero=True),
            Player("villain", "Villain", villain_stack),
        ]
        engine = HandEngine(GameConfig(1, 2, 200), players, 0, seed=11).start()
        if raise_to is not None:
            engine.apply_action(Action(ActionType.RAISE, raise_to))
        state = engine.state
        return build_decision_context(state, state.to_act, engine.legal_actions())

    def test_required_equity_matches_the_textbook_example(self) -> None:
        # Facing 8 into a pot of 24: 8 / (24 + 8) = 25%.
        players = [
            Player("hero", "Hero", 200, is_hero=True),
            Player("villain", "Villain", 200),
        ]
        engine = HandEngine(GameConfig(1, 2, 200), players, 0, seed=11).start()
        hero, villain = engine.state.players
        # Force the classic spot directly.
        hero.committed_hand, hero.committed_street = 8, 0
        villain.committed_hand, villain.committed_street = 16, 8
        engine.state.current_bet = 8

        context = build_decision_context(
            engine.state, hero, engine.legal_actions()
        )
        self.assertEqual(context.pot, 24)
        self.assertEqual(context.to_call, 8)
        self.assertEqual(context.pot_after_call, 32)
        self.assertEqual(context.pot_odds_pct, 25.0)

    def test_small_blind_facing_the_big_blind(self) -> None:
        context = self._context()
        # SB has 1 in, BB has 2 in: pot 3, call 1 -> 1/4 = 25%.
        self.assertEqual(context.pot, 3)
        self.assertEqual(context.to_call, 1)
        self.assertEqual(context.pot_odds_pct, 25.0)

    def test_no_pot_odds_when_nothing_to_call(self) -> None:
        context = self._context()
        self.assertIsNotNone(context.pot_odds)
        # After the SB limps, the BB faces no bet.
        players = [
            Player("hero", "Hero", 200, is_hero=True),
            Player("villain", "Villain", 200),
        ]
        engine = HandEngine(GameConfig(1, 2, 200), players, 0, seed=11).start()
        engine.apply_action(Action(ActionType.CALL))
        context = build_decision_context(
            engine.state, engine.state.to_act, engine.legal_actions()
        )
        self.assertEqual(context.to_call, 0)
        self.assertIsNone(context.pot_odds)

    def test_explanation_states_the_break_even_threshold(self) -> None:
        context = self._context()
        note = next(e for e in context.explanations if e["key"] == "pot_odds")
        self.assertIn("25.0%", note["body"])

    def test_no_equity_is_ever_fabricated(self) -> None:
        """Estimated quantities must be reported as pending, never invented."""
        context = self._context()
        payload = context.to_dict()
        pending_keys = {item["key"] for item in payload["pending"]}
        self.assertIn("equity", pending_keys)
        self.assertIn("expected_value", pending_keys)
        self.assertIn("opponent_range", pending_keys)
        for forbidden in ("equity", "ev", "expected_value", "win_probability"):
            self.assertNotIn(forbidden, payload)


class TestBetSizing(unittest.TestCase):
    def test_sizings_are_legal_amounts(self) -> None:
        players = [
            Player("hero", "Hero", 200, is_hero=True),
            Player("villain", "Villain", 200),
        ]
        engine = HandEngine(GameConfig(1, 2, 200), players, 0, seed=3).start()
        options = engine.legal_actions()
        context = build_decision_context(engine.state, engine.state.to_act, options)

        self.assertTrue(context.bet_options)
        for option in context.bet_options:
            self.assertGreaterEqual(option.amount, options.min_raise_to)
            self.assertLessEqual(option.amount, options.max_raise_to)

    def test_all_in_is_always_offered(self) -> None:
        players = [
            Player("hero", "Hero", 200, is_hero=True),
            Player("villain", "Villain", 200),
        ]
        engine = HandEngine(GameConfig(1, 2, 200), players, 0, seed=3).start()
        options = engine.legal_actions()
        context = build_decision_context(engine.state, engine.state.to_act, options)
        self.assertTrue(any(option.is_all_in for option in context.bet_options))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class TestSession(unittest.TestCase):
    def test_rebuy_restores_stacks_without_touching_recorded_profit(self) -> None:
        session = PokerSession(config=GameConfig(1, 2, 200), seed=1)
        session.hero.stack = 0
        session.villain.stack = 400
        session.hero_net_total = -200

        session._rebuy()

        self.assertEqual(session.hero.stack, 200)
        self.assertEqual(session.villain.stack, 200)
        self.assertEqual(session.rebuys, 1)
        self.assertEqual(
            session.hero_net_total, -200,
            "a rebuy must not erase the recorded loss",
        )

    def test_busting_triggers_a_rebuy_on_the_next_hand(self) -> None:
        session = PokerSession(config=GameConfig(1, 2, 200), seed=1)
        session.start_hand()
        session.engine.state.is_complete = True
        session.hero.stack = 0
        session.villain.stack = 400

        session.start_hand()

        self.assertEqual(session.rebuys, 1)
        self.assertTrue(session.last_rebuy)
        # Both players were restored before blinds were posted.
        self.assertGreaterEqual(
            session.hero.stack + session.hero.committed_hand, 200
        )

    def test_history_records_provenance_for_replay(self) -> None:
        session = PokerSession(config=GameConfig(1, 2, 200), seed=8)
        session.start_hand()
        while not session.engine.is_complete:
            legal = session.engine.legal_actions()
            session.hero_action("check" if legal.can_check else "fold")

        record = session.history[-1].to_dict()
        for field in (
            "hand_id", "deck_seed", "hero_cards", "villain_cards",
            "board", "action_log", "engine_version", "completed_at",
        ):
            self.assertIn(field, record)
        self.assertTrue(record["action_log"])

    def test_cannot_act_out_of_turn(self) -> None:
        session = PokerSession(config=GameConfig(1, 2, 200), seed=8)
        session.start_hand()
        while not session.engine.is_complete:
            legal = session.engine.legal_actions()
            session.hero_action("check" if legal.can_check else "fold")
        with self.assertRaises(RuntimeError):
            session.hero_action("check")

    def test_learning_mode_validation(self) -> None:
        session = PokerSession(config=GameConfig(1, 2, 200), seed=8)
        session.set_learning_mode(LearningMode.CHALLENGE)
        self.assertEqual(session.learning_mode, "challenge")
        with self.assertRaises(ValueError):
            session.set_learning_mode("expert")


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

class TestPokerRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.client = build_client()

    def _new_session(self, **overrides):
        payload = {"opponent_profile": "balanced", "seed": 42, **overrides}
        response = self.client.post("/api/poker/sessions", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_health(self) -> None:
        response = self.client.get("/api/poker/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_lists_opponent_archetypes(self) -> None:
        data = self.client.get("/api/poker/opponents").json()
        keys = {profile["key"] for profile in data["opponents"]}
        self.assertIn("balanced", keys)
        self.assertIn("calling_station", keys)
        for profile in data["opponents"]:
            self.assertIn("vpip", profile)
            self.assertIn("aggression", profile)

    def test_creating_a_session_deals_a_hand(self) -> None:
        data = self._new_session()
        self.assertEqual(data["hand_number"], 1)
        self.assertIsNotNone(data["hand"])
        self.assertEqual(len(data["hand"]["players"]), 2)

    def test_opponent_cards_are_not_leaked_mid_hand(self) -> None:
        data = self._new_session()
        villain = next(p for p in data["hand"]["players"] if not p["is_hero"])
        self.assertIsNone(villain["hole_cards"])

    def test_hero_always_sees_own_cards(self) -> None:
        data = self._new_session()
        hero = next(p for p in data["hand"]["players"] if p["is_hero"])
        self.assertEqual(len(hero["hole_cards"]), 2)

    def test_playing_a_hand_through_the_api(self) -> None:
        data = self._new_session()
        guard = 0
        while not data["hand"]["is_complete"] and guard < 40:
            guard += 1
            if not data["awaiting_hero"]:
                break
            legal = data["hand"]["legal_actions"]
            action = "check" if legal["can_check"] else "call"
            response = self.client.post(
                f"/api/poker/sessions/{data['session_id']}/actions",
                json={"action": action, "amount": 0},
            )
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()

        self.assertTrue(data["hand"]["is_complete"])
        self.assertIsNotNone(data["hand"]["result"])

    def test_illegal_action_returns_422(self) -> None:
        data = self._new_session()
        response = self.client.post(
            f"/api/poker/sessions/{data['session_id']}/actions",
            json={"action": "check", "amount": 0},   # SB faces the big blind
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_action_returns_400(self) -> None:
        data = self._new_session()
        response = self.client.post(
            f"/api/poker/sessions/{data['session_id']}/actions",
            json={"action": "teleport", "amount": 0},
        )
        self.assertEqual(response.status_code, 400)

    def test_dealing_mid_hand_returns_409(self) -> None:
        data = self._new_session()
        response = self.client.post(
            f"/api/poker/sessions/{data['session_id']}/hands"
        )
        self.assertEqual(response.status_code, 409)

    def test_missing_session_returns_404(self) -> None:
        response = self.client.get("/api/poker/sessions/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_blinds_must_be_consistent(self) -> None:
        response = self.client.post(
            "/api/poker/sessions",
            json={"small_blind": 10, "big_blind": 2},
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_opponent_returns_400(self) -> None:
        response = self.client.post(
            "/api/poker/sessions", json={"opponent_profile": "nemesis"}
        )
        self.assertEqual(response.status_code, 400)

    def test_learning_mode_can_be_changed(self) -> None:
        data = self._new_session()
        response = self.client.post(
            f"/api/poker/sessions/{data['session_id']}/learning-mode",
            json={"learning_mode": "challenge"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["learning_mode"], "challenge")

    def test_session_can_be_ended(self) -> None:
        data = self._new_session()
        self.assertEqual(
            self.client.delete(f"/api/poker/sessions/{data['session_id']}").status_code,
            200,
        )
        self.assertEqual(
            self.client.get(f"/api/poker/sessions/{data['session_id']}").status_code,
            404,
        )

    def test_seeded_sessions_are_reproducible(self) -> None:
        first = self._new_session(seed=777)
        second = self._new_session(seed=777)
        hero_first = next(p for p in first["hand"]["players"] if p["is_hero"])
        hero_second = next(p for p in second["hand"]["players"] if p["is_hero"])
        self.assertEqual(hero_first["hole_cards"], hero_second["hole_cards"])


if __name__ == "__main__":
    unittest.main()
