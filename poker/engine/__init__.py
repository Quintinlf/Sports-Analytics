"""No-Limit Texas Hold'em game engine.

``state`` holds the data model, ``rules`` decides what is legal, and ``hand``
drives a single hand from deal to showdown as an explicit state machine.
"""
from __future__ import annotations

from poker.engine.hand import HandEngine, HandResult
from poker.engine.rules import LegalActions, legal_actions
from poker.engine.state import (
    Action,
    ActionType,
    ActionRecord,
    GameConfig,
    HandState,
    Player,
    Pot,
    Street,
)

__all__ = [
    "Action",
    "ActionType",
    "ActionRecord",
    "GameConfig",
    "HandEngine",
    "HandResult",
    "HandState",
    "LegalActions",
    "Player",
    "Pot",
    "Street",
    "legal_actions",
]
