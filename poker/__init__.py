"""Interactive poker learning laboratory.

A playable No-Limit Texas Hold'em engine whose mathematical internals are
designed to be inspected rather than hidden.

Deliberately dependency-free (stdlib only) so the engine can run inside the
lean web tier described by requirements-web.txt, which excludes numpy/scipy.
Numeric dependencies belong in the offline simulation lab, not the request path.
"""
from __future__ import annotations

__all__ = ["cards", "evaluator", "engine", "opponents"]
