"""
Elo Rating Model Module

Provides EloModel — a simple Elo-based win-probability / spread predictor
that can be used as a lightweight component of the ensemble.

Based on the PredictiveModel.calculate_elo_ratings() logic from model_trainer.py,
wrapped in a fit/predict interface consistent with the other model modules.
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Empirical constant: 1 Elo point ≈ this many NBA points of spread
_ELO_TO_POINTS = 0.033  # e.g. 30 Elo diff → ~1 pt spread


class EloModel:
    """
    Elo rating model for NBA teams.

    After calling fit() on a historical games DataFrame, use
    predict_win_probability() and predict_spread() for upcoming matchups.

    Parameters
    ----------
    k_factor : float
        Controls how fast ratings adjust (20 is standard for NBA).
    initial_rating : float
        Starting Elo for all teams.
    home_advantage : float
        Constant Elo bonus added to the home team when computing win probability.
        Empirically ~100 Elo ≈ 3.5 point advantage.
    """

    def __init__(
        self,
        k_factor: float = 20.0,
        initial_rating: float = 1500.0,
        home_advantage: float = 100.0,   # Elo units; ~3.5 pts
    ):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.ratings: Dict[int, float] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, games: pd.DataFrame) -> 'EloModel':
        """
        Build Elo ratings from historical game-log rows.

        The DataFrame must have columns: TEAM_ID, OPP_TEAM_ID, WL
        (one row per team per game, as returned by LeagueGameFinder).
        """
        self.ratings = {}

        for _, row in games.iterrows():
            team_id = int(row['TEAM_ID'])
            opp_id = int(row.get('OPP_TEAM_ID', 0))
            won = row['WL'] == 'W'

            self.ratings.setdefault(team_id, self.initial_rating)
            self.ratings.setdefault(opp_id, self.initial_rating)

            expected = 1.0 / (1.0 + 10.0 ** ((self.ratings[opp_id] - self.ratings[team_id]) / 400.0))
            actual = 1.0 if won else 0.0
            self.ratings[team_id] += self.k_factor * (actual - expected)
            self.ratings[opp_id] += self.k_factor * ((1.0 - actual) - (1.0 - expected))

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_win_probability(
        self,
        home_team_id: int,
        away_team_id: int,
    ) -> float:
        """
        Return probability that *home_team_id* wins the game.

        Home-court advantage is applied as a bonus to the home team's Elo.
        """
        home_elo = self.ratings.get(int(home_team_id), self.initial_rating) + self.home_advantage
        away_elo = self.ratings.get(int(away_team_id), self.initial_rating)
        return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))

    def predict_spread(
        self,
        home_team_id: int,
        away_team_id: int,
    ) -> float:
        """
        Convert Elo difference to an approximate point spread.

        Positive → home team favoured.
        """
        home_elo = self.ratings.get(int(home_team_id), self.initial_rating) + self.home_advantage
        away_elo = self.ratings.get(int(away_team_id), self.initial_rating)
        return (home_elo - away_elo) * _ELO_TO_POINTS

    def get_rating(self, team_id: int) -> float:
        """Return the current Elo rating for *team_id*."""
        return self.ratings.get(int(team_id), self.initial_rating)

    def get_all_ratings(self) -> Dict[int, float]:
        """Return a copy of the full ratings dict."""
        return dict(self.ratings)

    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as fh:
            pickle.dump({
                'ratings': self.ratings,
                'k_factor': self.k_factor,
                'initial_rating': self.initial_rating,
                'home_advantage': self.home_advantage,
                'saved_at': datetime.now().isoformat(),
            }, fh)
        print(f"  Saved -> {filepath}", flush=True)

    @classmethod
    def load(cls, filepath: str) -> 'EloModel':
        with open(filepath, 'rb') as fh:
            data = pickle.load(fh)
        inst = cls(
            k_factor=data['k_factor'],
            initial_rating=data['initial_rating'],
            home_advantage=data['home_advantage'],
        )
        inst.ratings = data['ratings']
        inst.is_fitted = True
        return inst
