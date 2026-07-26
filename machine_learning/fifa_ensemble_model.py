"""
FIFA Ensemble Model (inference-safe module)

Holds FIFAEnsembleModel in its own module with no soccerdata dependency, so
that unpickling machine_learning/models/fifa_ensemble.pkl at inference time
(e.g. in data/fifa_predictions_service.py, or the daily GitHub Actions job)
doesn't transitively require soccerdata to be installed. Pickle resolves
custom classes by importing their defining module, so keeping this class out
of training/fifa_trainer.py (which does import soccerdata, for ingestion)
matters even though this class never calls soccerdata itself.

This mirrors the existing separation between machine_learning/lightgbm_models.py
(inference-safe) and training/mlb_trainer.py (which imports statsapi).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class FIFAEnsembleModel:
    """
    Bundles the fitted scaler, PCA transformer, and classifier together —
    all three are needed at inference time (mirrors how LGBMWinPredictor
    bundles its scaler with the model rather than saving them separately).
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self.squad_feature_cols: List[str] = []
        self.classes_: List[str] = []
        self.is_fitted = False

    def _team_squad_vector(self, squad_profiles: pd.DataFrame, team: str) -> Optional[np.ndarray]:
        rows = squad_profiles[squad_profiles['team'] == team]
        if rows.empty:
            return None
        # Most recent tournament appearance on file for that team.
        row = rows.sort_values('season').iloc[-1]
        return row[self.squad_feature_cols].to_numpy(dtype=float)

    def build_match_features(
        self, squad_profiles: pd.DataFrame, home_team: str, away_team: str
    ) -> Optional[np.ndarray]:
        if not self.is_fitted or self.pca is None:
            raise ValueError("Not fitted. Call fit() first.")
        home_vec = self._team_squad_vector(squad_profiles, home_team)
        away_vec = self._team_squad_vector(squad_profiles, away_team)
        if home_vec is None or away_vec is None:
            return None
        combined = np.concatenate([home_vec, away_vec]).reshape(1, -1)
        scaled = self.scaler.transform(combined)
        return self.pca.transform(scaled)

    def fit(
        self,
        squad_profiles: pd.DataFrame,
        match_results: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict:
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import train_test_split

        self.squad_feature_cols = [
            c for c in squad_profiles.columns if c not in ('league', 'season', 'team')
        ]

        X_rows, y_rows = [], []
        for _, match in match_results.iterrows():
            home_rows = squad_profiles[
                (squad_profiles['team'] == match['home_team'])
                & (squad_profiles['league'] == match['league'])
                & (squad_profiles['season'].astype(str) == str(match['season']))
            ]
            away_rows = squad_profiles[
                (squad_profiles['team'] == match['away_team'])
                & (squad_profiles['league'] == match['league'])
                & (squad_profiles['season'].astype(str) == str(match['season']))
            ]
            if home_rows.empty or away_rows.empty:
                continue
            home_vec = home_rows.iloc[0][self.squad_feature_cols].to_numpy(dtype=float)
            away_vec = away_rows.iloc[0][self.squad_feature_cols].to_numpy(dtype=float)
            X_rows.append(np.concatenate([home_vec, away_vec]))
            y_rows.append(match['outcome'])

        if len(X_rows) < 30:
            raise ValueError(f'Not enough matched squad/result rows for FIFA training: {len(X_rows)}')

        X = np.vstack(X_rows)
        y = np.array(y_rows)

        if verbose:
            print(f"  Training on {len(X)} matches, {X.shape[1]} raw squad-diff features", flush=True)

        X_scaled = self.scaler.fit_transform(X)
        n_components = min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        if verbose:
            explained = self.pca.explained_variance_ratio_.sum()
            print(f"  PCA: {n_components} components, {explained:.1%} variance explained", flush=True)

        can_stratify = min(pd.Series(y).value_counts()) >= 2
        X_train, X_val, y_train, y_val = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None,
        )

        self.classifier.fit(X_train, y_train)
        self.classes_ = list(self.classifier.classes_)
        self.is_fitted = True

        val_pred = self.classifier.predict(X_val)
        val_proba = self.classifier.predict_proba(X_val)
        accuracy = accuracy_score(y_val, val_pred)
        try:
            loss = log_loss(y_val, val_proba, labels=self.classifier.classes_)
        except ValueError:
            loss = None

        if verbose:
            loss_str = f"  log_loss: {loss:.4f}" if loss is not None else ""
            print(f"  Validation accuracy: {accuracy:.1%}{loss_str}", flush=True)

        return {
            'n_samples': len(X),
            'n_components': n_components,
            'accuracy': accuracy,
            'log_loss': loss,
        }

    def squad_metric_maps(
        self, squad_profiles: pd.DataFrame, home_team: str, away_team: str
    ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
        """Return raw squad-profile numerics for each team (pre-PCA model inputs)."""
        if not self.is_fitted or not self.squad_feature_cols:
            return None
        home_vec = self._team_squad_vector(squad_profiles, home_team)
        away_vec = self._team_squad_vector(squad_profiles, away_team)
        if home_vec is None or away_vec is None:
            return None
        home_map = {
            str(col): float(val)
            for col, val in zip(self.squad_feature_cols, home_vec)
        }
        away_map = {
            str(col): float(val)
            for col, val in zip(self.squad_feature_cols, away_vec)
        }
        return home_map, away_map

    def predict_match(
        self, squad_profiles: pd.DataFrame, home_team: str, away_team: str
    ) -> Optional[Dict[str, float]]:
        """Return {'HOME_WIN': p, 'DRAW': p, 'AWAY_WIN': p}, or None if either
        team has no squad profile on file (e.g. didn't qualify for a tracked
        tournament)."""
        features = self.build_match_features(squad_profiles, home_team, away_team)
        if features is None:
            return None
        proba = self.classifier.predict_proba(features)[0]
        return {cls: float(p) for cls, p in zip(self.classes_, proba)}
