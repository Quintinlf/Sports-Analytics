"""A seat at the table: many hands, persistent stacks, an AI opponent.

Holds the things that outlive a single hand — stacks, the button, the running
result — and drives the opponent so the client only ever submits hero actions.

Milestone 1 keeps sessions in memory. Every payload it produces is already
shaped for persistence (seeded deck, full action log, explicit provenance), so
adding hand history in Milestone 7 is a save call rather than a redesign.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from random import Random
from typing import Any, Dict, List, Optional

from poker.decision import DecisionContext, build_decision_context
from poker.engine import (
    Action,
    ActionType,
    GameConfig,
    HandEngine,
    HandResult,
    Player,
    Street,
)
from poker.opponents import HeuristicOpponent, OpponentProfile, get_profile

__all__ = ["PokerSession", "LearningMode", "HandSummary"]

HERO_ID = "hero"
VILLAIN_ID = "villain"

#: Engine + rules version stamped onto every hand for provenance.
ENGINE_VERSION = "1.0.0-m1"


class LearningMode:
    """How much of the mathematics the interface reveals before a decision."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    CHALLENGE = "challenge"
    ANALYSIS = "analysis"

    ALL = (BEGINNER, INTERMEDIATE, CHALLENGE, ANALYSIS)

    DESCRIPTIONS = {
        BEGINNER: "Show the maths while you decide.",
        INTERMEDIATE: "Hidden by default — reveal it when you want it.",
        CHALLENGE: "Decide with no assistance.",
        ANALYSIS: "Hidden while deciding, revealed once the hand ends.",
    }

    @classmethod
    def validate(cls, mode: str) -> str:
        if mode not in cls.ALL:
            raise ValueError(
                f"Unknown learning mode {mode!r}; available: {list(cls.ALL)}"
            )
        return mode


@dataclass
class HandSummary:
    """Lightweight record of a completed hand, kept for the session log."""

    hand_number: int
    hand_id: str
    deck_seed: int
    hero_cards: List[str]
    villain_cards: List[str]
    board: List[str]
    hero_net: int
    went_to_showdown: bool
    summary: str
    action_log: List[dict]
    engine_version: str = ENGINE_VERSION
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand_number": self.hand_number,
            "hand_id": self.hand_id,
            "deck_seed": self.deck_seed,
            "hero_cards": list(self.hero_cards),
            "villain_cards": list(self.villain_cards),
            "board": list(self.board),
            "hero_net": self.hero_net,
            "went_to_showdown": self.went_to_showdown,
            "summary": self.summary,
            "action_log": list(self.action_log),
            "engine_version": self.engine_version,
            "completed_at": self.completed_at,
        }


class PokerSession:
    """One player, one AI opponent, an ongoing sequence of hands."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        config: Optional[GameConfig] = None,
        opponent_profile: str = "balanced",
        hero_name: str = "You",
        seed: Optional[int] = None,
        learning_mode: str = LearningMode.BEGINNER,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or GameConfig()
        self.profile: OpponentProfile = get_profile(opponent_profile)
        self.opponent = HeuristicOpponent(self.profile)
        self.learning_mode = LearningMode.validate(learning_mode)
        self.created_at = datetime.now(timezone.utc).isoformat()

        self._rng = Random(seed) if seed is not None else Random()
        self.seed = seed

        self.hero = Player(HERO_ID, hero_name, self.config.starting_stack, is_hero=True)
        self.villain = Player(VILLAIN_ID, self.profile.name, self.config.starting_stack)

        self.hand_number = 0
        #: Hero seat index 0, villain 1. Button alternates each hand.
        self.button_index = 0
        self.engine: Optional[HandEngine] = None
        self.history: List[HandSummary] = []

        #: Cumulative hero profit in chips, unaffected by rebuys.
        self.hero_net_total = 0
        self.rebuys = 0
        self.last_rebuy = False

    # -- lifecycle --------------------------------------------------------

    def start_hand(self) -> "PokerSession":
        """Deal a new hand, rebuying first if a stack cannot post its blind."""
        if self.engine is not None and not self.engine.is_complete:
            raise RuntimeError("Finish the current hand before dealing another")

        self.last_rebuy = False
        if min(self.hero.stack, self.villain.stack) < self.config.big_blind:
            self._rebuy()

        if self.hand_number > 0:
            self.button_index = 1 - self.button_index

        self.hand_number += 1
        self.engine = HandEngine(
            self.config,
            [self.hero, self.villain],
            self.button_index,
            seed=self._rng.randrange(2**63),
        ).start()

        self._run_opponent()
        return self

    def _rebuy(self) -> None:
        """Reset both stacks. Cumulative profit is tracked separately, so a
        rebuy never flatters the running result."""
        self.hero.stack = self.config.starting_stack
        self.villain.stack = self.config.starting_stack
        self.rebuys += 1
        self.last_rebuy = True

    def set_learning_mode(self, mode: str) -> str:
        self.learning_mode = LearningMode.validate(mode)
        return self.learning_mode

    # -- actions ----------------------------------------------------------

    def hero_action(self, action_type: str, amount: int = 0) -> "PokerSession":
        """Apply the hero's action, then let the opponent respond."""
        engine = self._require_engine()
        if engine.is_complete:
            raise RuntimeError("Hand is already complete")

        actor = engine.state.to_act
        if actor is None or not actor.is_hero:
            raise RuntimeError("It is not your turn to act")

        try:
            parsed = ActionType(action_type)
        except ValueError:
            raise ValueError(
                f"Unknown action {action_type!r}; expected one of "
                f"{[a.value for a in ActionType if a is not ActionType.POST_BLIND]}"
            ) from None
        if parsed is ActionType.POST_BLIND:
            raise ValueError("Blinds are posted automatically")

        engine.apply_action(Action(parsed, amount))
        self._run_opponent()
        self._record_if_complete()
        return self

    def _run_opponent(self) -> None:
        """Act for the opponent until the hero is to act or the hand ends."""
        engine = self._require_engine()
        # Bounded to protect against a policy that somehow never terminates.
        for _ in range(200):
            if engine.is_complete:
                break
            actor = engine.state.to_act
            if actor is None or actor.is_hero:
                break
            action = self.opponent.choose_action(
                engine.state, actor, engine.legal_actions(), self._rng
            )
            engine.apply_action(action)
        self._record_if_complete()

    def _record_if_complete(self) -> None:
        engine = self.engine
        if engine is None or not engine.is_complete or engine.result is None:
            return
        if self.history and self.history[-1].hand_id == engine.state.hand_id:
            return

        result: HandResult = engine.result
        hero_net = result.net.get(HERO_ID, 0)
        self.hero_net_total += hero_net

        self.history.append(
            HandSummary(
                hand_number=self.hand_number,
                hand_id=engine.state.hand_id,
                deck_seed=engine.state.deck_seed,
                hero_cards=[str(c) for c in self.hero.hole_cards],
                villain_cards=[str(c) for c in self.villain.hole_cards],
                board=result.board,
                hero_net=hero_net,
                went_to_showdown=result.went_to_showdown,
                summary=result.summary,
                action_log=[r.to_dict() for r in engine.state.action_log],
            )
        )

    def _require_engine(self) -> HandEngine:
        if self.engine is None:
            raise RuntimeError("No hand in progress; deal one first")
        return self.engine

    # -- view -------------------------------------------------------------

    def decision_context(self) -> Optional[DecisionContext]:
        engine = self.engine
        if engine is None or engine.is_complete:
            return None
        actor = engine.state.to_act
        if actor is None or not actor.is_hero:
            return None
        return build_decision_context(engine.state, actor, engine.legal_actions())

    def to_dict(self) -> Dict[str, Any]:
        engine = self.engine
        context = self.decision_context()

        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "engine_version": ENGINE_VERSION,
            "config": self.config.to_dict(),
            "opponent": self.profile.to_dict(),
            "learning_mode": self.learning_mode,
            "learning_mode_options": [
                {"key": key, "description": LearningMode.DESCRIPTIONS[key]}
                for key in LearningMode.ALL
            ],
            "hand_number": self.hand_number,
            "button_index": self.button_index,
            "hero_net_total": self.hero_net_total,
            "rebuys": self.rebuys,
            "last_rebuy": self.last_rebuy,
            "hands_played": len(self.history),
            "hand": engine.to_dict() if engine else None,
            "decision": context.to_dict() if context else None,
            "awaiting_hero": bool(
                engine
                and not engine.is_complete
                and engine.state.to_act is not None
                and engine.state.to_act.is_hero
            ),
            "recent_hands": [h.to_dict() for h in self.history[-10:]],
        }
