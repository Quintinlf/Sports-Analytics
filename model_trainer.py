"""Deprecated shim: legacy `model_trainer` has been archived.

The full historical implementation now lives at:
  experimental/legacy/model_trainer_legacy.py

Preferred imports (modular stack):
  - training.trainer.ModelTrainer
  - machine_learning.gp_model.GaussianProcessPredictor
  - machine_learning.gp_model.train_gp_models
  - ensemble.ensemble_predictor.EnsemblePredictor

This shim keeps a small subset of the old import surface working while
encouraging migration.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "model_trainer is deprecated; use training.trainer / machine_learning.gp_model / ensemble.ensemble_predictor",
    DeprecationWarning,
    stacklevel=2,
)

from machine_learning.gp_model import GaussianProcessPredictor, train_gp_models  # noqa: F401
from training.trainer import ModelTrainer  # noqa: F401
from ensemble.ensemble_predictor import EnsemblePredictor  # noqa: F401

__all__ = [
    "GaussianProcessPredictor",
    "train_gp_models",
    "ModelTrainer",
    "EnsemblePredictor",
]
