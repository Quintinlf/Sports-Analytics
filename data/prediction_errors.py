"""Shared exception types for the live prediction pipeline.

ModelUnavailableError signals that a trained model failed to load (missing
pointer file, missing artifact, corrupt pickle, etc.) — a configuration/
deployment problem, not "no games today." Prediction services must let this
propagate out of fetch_upcoming_games() uncaught so UnifiedPredictionService
logs it loudly instead of silently swapping in a hardcoded guess.
"""
from __future__ import annotations


class ModelUnavailableError(RuntimeError):
    """Raised when a sport's trained model cannot be loaded for inference."""
