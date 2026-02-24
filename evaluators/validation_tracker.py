# evaluators/validation_tracker.py
# Re-exports PredictionValidator from backtest_validation so that any code
# using `from evaluators.validation_tracker import PredictionValidator` works.

from evaluators.backtest_validation import PredictionValidator  # noqa: F401

__all__ = ["PredictionValidator"]
