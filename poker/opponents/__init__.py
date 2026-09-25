"""AI opponents.

``base`` defines the behavioural parameter surface every opponent exposes, so
later milestones can measure an opponent the same way they measure the player,
and can adapt those parameters in response to observed behaviour.
"""
from __future__ import annotations

from poker.opponents.base import Opponent, OpponentProfile, PROFILES, get_profile
from poker.opponents.heuristic import HeuristicOpponent

__all__ = [
    "Opponent",
    "OpponentProfile",
    "PROFILES",
    "get_profile",
    "HeuristicOpponent",
]
