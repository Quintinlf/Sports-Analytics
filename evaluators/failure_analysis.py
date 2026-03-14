"""
Failure Analysis

Queries prediction_results JOIN predictions JOIN prediction_features
to identify high-error games and correlate features with failures.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from data.database.database_handler import SportsAnalyticsDB


class FailureAnalyzer:
    """
    Identify systematic error patterns and feature correlations for
    games where the model predicted poorly.

    Usage
    -----
        fa = FailureAnalyzer()
        report = fa.analyze(db, threshold=8.0)
        print(report)
    """

    def analyze(
        self,
        db: SportsAnalyticsDB,
        threshold: float = 8.0,
    ) -> Dict:
        """
        Full error-pattern analysis.

        Parameters
        ----------
        db : SportsAnalyticsDB
        threshold : float
            Absolute spread error (points) above which a game counts as a
            "large error" game for feature correlation.

        Returns
        -------
        dict  with keys: overall_metrics, bias_analysis, confidence_analysis,
              problem_areas, worst_predictions, best_predictions,
              feature_correlations (if features available).
        """
        df = self._load_matched_predictions(db)
        if df.empty:
            return {'error': 'No matched predictions found.'}

        # ---- overall ----
        n = len(df)
        win_acc = df['correct_winner'].mean()
        mae = df['spread_error'].mean()
        rmse = float(np.sqrt((df['spread_error'] ** 2).mean()))

        # ---- bias ----
        errors = df['predicted_spread'] - df['actual_spread']
        mean_bias = float(errors.mean())
        error_skew = float(stats.skew(errors))
        error_kurtosis = float(stats.kurtosis(errors))

        # ---- per-confidence ----
        conf_analysis: Dict[str, Dict] = {}
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            sub = df[df['confidence_level'] == level]
            if len(sub) > 0:
                conf_analysis[level] = {
                    'count': len(sub),
                    'win_accuracy': float(sub['correct_winner'].mean()),
                    'mae': float(sub['spread_error'].mean()),
                }

        # ---- worst / best ----
        worst = (
            df.nlargest(5, 'spread_error')
            [['home_team', 'away_team', 'predicted_spread', 'actual_spread', 'spread_error']]
            .to_dict('records')
        )
        best = (
            df.nsmallest(5, 'spread_error')
            [['home_team', 'away_team', 'predicted_spread', 'actual_spread', 'spread_error']]
            .to_dict('records')
        )

        # ---- feature correlations ----
        feat_corr: Dict[str, float] = {}
        large_error_games = df[df['spread_error'] >= threshold]
        if not large_error_games.empty:
            feat_corr = self.correlate_features(large_error_games, db)

        return {
            'overall_metrics': {
                'n_predictions': n,
                'win_accuracy': float(win_acc),
                'mae': float(mae),
                'rmse': rmse,
            },
            'bias_analysis': {
                'mean_bias': mean_bias,
                'bias_direction': 'home-favoring' if mean_bias > 0 else 'away-favoring',
                'error_skew': error_skew,
                'error_kurtosis': error_kurtosis,
            },
            'confidence_analysis': conf_analysis,
            'problem_areas': {
                'large_error_games': len(large_error_games),
                'overconfident_errors': int(
                    ((df['confidence_level'] == 'HIGH') & (~df['correct_winner'])).sum()
                ),
            },
            'worst_predictions': worst,
            'best_predictions': best,
            'feature_correlations': feat_corr,
        }

    def correlate_features(
        self,
        large_error_games: pd.DataFrame,
        db: SportsAnalyticsDB,
    ) -> Dict[str, float]:
        """
        For each feature stored in prediction_features, compute the
        point-biserial correlation between its value and whether the game
        was a large-error game.

        Returns
        -------
        dict  feature_name -> correlation coefficient (|r|, sorted desc)
        """
        import json as _json

        all_ids = self._load_matched_predictions(db)['prediction_id'].tolist()
        large_ids = set(large_error_games['prediction_id'].tolist())

        if not all_ids:
            return {}

        if db.conn is None:
            return {}

        cursor = db.conn.cursor()
        cursor.execute(
            f"""
            SELECT prediction_id, feature_snapshot
            FROM prediction_features
            WHERE prediction_id IN ({','.join('?' * len(all_ids))})
            """,
            all_ids,
        )
        rows = cursor.fetchall()

        if not rows:
            return {}

        records = []
        for pid, snap_raw in rows:
            try:
                snap = _json.loads(snap_raw) if isinstance(snap_raw, str) else snap_raw
                snap['_prediction_id'] = pid
                snap['_is_large_error'] = int(pid in large_ids)
                records.append(snap)
            except Exception:
                continue

        if not records:
            return {}

        feat_df = pd.DataFrame(records).set_index('_prediction_id')
        label = feat_df.pop('_is_large_error')

        correlations: Dict[str, float] = {}
        for col in feat_df.columns:
            series = pd.to_numeric(feat_df[col], errors='coerce').dropna()
            common = series.index.intersection(label.index)
            if len(common) < 5:
                continue
            y = pd.to_numeric(label.loc[common], errors='coerce')
            x = pd.to_numeric(series.loc[common], errors='coerce')
            r = x.corr(y)
            if pd.notna(r):
                correlations[col] = round(abs(float(r)), 4)

        return dict(
            sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        )

    def generate_report(self, db: SportsAnalyticsDB, threshold: float = 8.0) -> str:
        """Return a markdown-formatted failure analysis report."""
        result = self.analyze(db, threshold=threshold)
        if 'error' in result:
            return f"# Failure Analysis\n\n{result['error']}\n"

        om = result['overall_metrics']
        ba = result['bias_analysis']
        lines = [
            '# Failure Analysis Report\n',
            '## Overall Metrics',
            f"- Predictions: {om['n_predictions']}",
            f"- Win Accuracy: {om['win_accuracy']:.1%}",
            f"- MAE: {om['mae']:.2f} pts",
            f"- RMSE: {om['rmse']:.2f} pts",
            '',
            '## Bias',
            f"- Mean Bias: {ba['mean_bias']:+.2f} pts ({ba['bias_direction']})",
            f"- Error Skew: {ba['error_skew']:.3f}",
            '',
            '## Confidence Breakdown',
        ]
        for level, stats_dict in result['confidence_analysis'].items():
            lines.append(
                f"- {level}: n={stats_dict['count']}, "
                f"acc={stats_dict['win_accuracy']:.1%}, "
                f"MAE={stats_dict['mae']:.2f}"
            )

        lines += ['', '## Problem Areas']
        pa = result['problem_areas']
        lines.append(f"- Large-error games (>={threshold} pts): {pa['large_error_games']}")
        lines.append(f"- HIGH-confidence wrong calls: {pa['overconfident_errors']}")

        fc = result.get('feature_correlations')
        if fc:
            lines += ['', '## Top Feature Correlations With Large Errors']
            for feat, r in list(fc.items())[:10]:
                lines.append(f"- {feat}: |r|={r:.4f}")

        return '\n'.join(lines) + '\n'

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_matched_predictions(self, db: SportsAnalyticsDB) -> pd.DataFrame:
        if db.conn is None:
            return pd.DataFrame()

        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT
                p.prediction_id,
                p.home_team,
                p.away_team,
                p.predicted_spread,
                p.win_probability,
                p.confidence_level,
                p.model_version,
                r.actual_spread,
                r.correct_winner,
                ABS(p.predicted_spread - r.actual_spread) AS spread_error
            FROM predictions p
            JOIN prediction_results r ON p.prediction_id = r.prediction_id
            ORDER BY p.game_date
            """
        )
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=columns)
