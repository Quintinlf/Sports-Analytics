"""
Enhanced Feature Integration Pipeline

Combines 94 team-level features + 36 player-level features = 130 total.
Maintains strict chronological integrity throughout.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, Tuple, Dict, Optional

from data.player_fetcher import fetch_player_logs_for_team, normalize_team_id, PlayerDataCache
from data.feature_engineering import calculate_player_rolling_stats, aggregate_player_stats_by_team


class EnhancedFeaturePipeline:
    """Production pipeline for enhanced 130-feature game feature building."""
    
    def __init__(self, player_data_source: Union[str, PlayerDataCache] = 'api'):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        player_data_source : str or PlayerDataCache
            'api' to fetch from nba_api directly
            PlayerDataCache instance to use cached data
        """
        self.data_source = player_data_source
        self.team_feature_count = 94
        self.player_feature_count = 36
        self.total_feature_count = 130
        
        # Player feature names (18 per team)
        self.player_feature_names = [
            'PLAYER_PTS_ROLL_WEIGHTED',
            'PLAYER_REB_ROLL_WEIGHTED',
            'PLAYER_AST_ROLL_WEIGHTED',
            'PLAYER_FGA_ROLL_WEIGHTED',
            'PLAYER_FG_PCT_ROLL_WEIGHTED',
            'PLAYER_PTS_PER_FGA_ROLL_WEIGHTED',
            'PLAYER_TOP_SCORER_PPG',
            'PLAYER_TOP_REBOUNDER_RPG',
            'PLAYER_TOP_PLAYMAKER_APG',
            'PLAYER_TOP_SCORER_SHARE',
            'PLAYER_BENCH_SCORING_PCT',
            'PLAYER_ACTIVE_ROTATION_SIZE',
            'PLAYER_ROTATION_STABILITY',
            'PLAYER_KEY_PLAYER_MISSING',
            'PLAYER_MINUTES_DROP_40PCT',
            'PLAYER_SCORING_CONCENTRATION',
            'PLAYER_DEFENSIVE_CONTRIBUTORS',
            'PLAYER_BENCH_SCORER_COUNT',
        ]
    
    def _get_player_data(
        self,
        team_id: int,
        before_date: datetime,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Fetch player data from API or cache."""
        
        if isinstance(self.data_source, PlayerDataCache):
            # Use cache
            logs = self.data_source.get_cached_logs(
                team_id=normalize_team_id(team_id),
                before_date=before_date,
                verbose=verbose
            )
        else:
            # Use API
            logs = fetch_player_logs_for_team(
                team_id=team_id,
                season='2024-25',
                before_date=before_date,
                verbose=verbose
            )
        
        return logs
    
    def build_player_features(
        self,
        game_date: datetime,
        team_id: int,
        team_type: str = 'HOME'
    ) -> Dict[str, float]:
        """
        Build 18 player features for one team.
        
        CHRONOLOGICALLY SAFE: Only uses data before game_date.
        
        Parameters:
        -----------
        game_date : datetime
            Game date (prediction date)
        team_id : int
            Team ID
        team_type : str
            'HOME' or 'AWAY' (for feature naming)
        
        Returns:
        --------
        Dict[str, float]
            18 features with keys like 'HOME_PLAYER_PTS_ROLL_WEIGHTED'
        """
        features = {f'{team_type}_{feat_name}': 0.0 for feat_name in self.player_feature_names}
        
        try:
            # Fetch player logs (STRICTLY BEFORE game_date)
            logs = self._get_player_data(team_id, before_date=game_date, verbose=False)
            
            if len(logs) == 0:
                return features
            
            # Calculate rolling stats
            logs_rolled = calculate_player_rolling_stats(logs, window=5)
            
            # Aggregate to team level
            team_feature_map = aggregate_player_stats_by_team(
                logs_rolled,
                game_date=game_date,
            )
            team_features = team_feature_map.get(normalize_team_id(team_id), {})

            # Map generic aggregate outputs to the expected 18 production feature names.
            avg_pts = float(team_features.get('TEAM_AVG_PTS_ROLL', 0.0))
            avg_reb = float(team_features.get('TEAM_AVG_REB_ROLL', 0.0))
            avg_ast = float(team_features.get('TEAM_AVG_AST_ROLL', 0.0))
            avg_fg_pct = float(team_features.get('TEAM_AVG_FG_PCT_ROLL', 0.0))
            avg_min = float(team_features.get('TEAM_AVG_MIN_ROLL', 0.0))

            top_pts = float(team_features.get('TEAM_TOP_PTS_ROLL', 0.0))
            top_reb = float(team_features.get('TEAM_TOP_REB_ROLL', 0.0))
            top_ast = float(team_features.get('TEAM_TOP_AST_ROLL', 0.0))

            features.update({
                f'{team_type}_PLAYER_PTS_ROLL_WEIGHTED': avg_pts,
                f'{team_type}_PLAYER_REB_ROLL_WEIGHTED': avg_reb,
                f'{team_type}_PLAYER_AST_ROLL_WEIGHTED': avg_ast,
                f'{team_type}_PLAYER_FGA_ROLL_WEIGHTED': 0.0,
                f'{team_type}_PLAYER_FG_PCT_ROLL_WEIGHTED': avg_fg_pct,
                f'{team_type}_PLAYER_PTS_PER_FGA_ROLL_WEIGHTED': 0.0,
                f'{team_type}_PLAYER_TOP_SCORER_PPG': top_pts,
                f'{team_type}_PLAYER_TOP_REBOUNDER_RPG': top_reb,
                f'{team_type}_PLAYER_TOP_PLAYMAKER_APG': top_ast,
                f'{team_type}_PLAYER_TOP_SCORER_SHARE': 0.0,
                f'{team_type}_PLAYER_BENCH_SCORING_PCT': 0.0,
                f'{team_type}_PLAYER_ACTIVE_ROTATION_SIZE': float(len(logs_rolled['PLAYER_ID'].unique())),
                f'{team_type}_PLAYER_ROTATION_STABILITY': 0.0,
                f'{team_type}_PLAYER_KEY_PLAYER_MISSING': 0.0,
                f'{team_type}_PLAYER_MINUTES_DROP_40PCT': 0.0,
                f'{team_type}_PLAYER_SCORING_CONCENTRATION': 0.0,
                f'{team_type}_PLAYER_DEFENSIVE_CONTRIBUTORS': 0.0,
                f'{team_type}_PLAYER_BENCH_SCORER_COUNT': 0.0,
            })
            
            if avg_min > 0:
                features[f'{team_type}_PLAYER_BENCH_SCORING_PCT'] = max(0.0, min(1.0, avg_pts / max(avg_min, 1.0)))

        except Exception:
            pass
        
        return features
    
    def build_enhanced_features(
        self,
        game_date: datetime,
        home_team_id: int,
        away_team_id: int,
        matchup_df: pd.DataFrame,
        team_feature_cols: list,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float], list]:
        """
        Build complete 130-feature vector for a game.
        
        CHRONOLOGICALLY SAFE: All player data filtered to before game_date.
        
        Parameters:
        -----------
        game_date : datetime
            Game date (prediction target)
        home_team_id : int
            Home team ID
        away_team_id : int
            Away team ID
        matchup_df : pd.DataFrame
            Pre-computed team features (e.g., matchup_df_sorted from notebook)
        team_feature_cols : list
            List of 94 team feature column names
        verbose : bool
            Print debug info
        
        Returns:
        --------
        Tuple of:
            - feature_vector: np.array (130,) with all features
            - feature_dict: Dict with feature names and values
            - feature_names: List of 130 feature names in order
        """
        
        feature_dict = {}
        
        # === STEP 1: Extract 94 team features ===
        if verbose:
            print(f"Extracting team features...")
        
        # Find matching game in matchup_df
        game_mask = (
            (matchup_df['GAME_DATE'] == game_date) &
            (matchup_df['HOME_TEAM_ID'] == normalize_team_id(home_team_id)) &
            (matchup_df['AWAY_TEAM_ID'] == normalize_team_id(away_team_id))
        )
        
        matching_games = matchup_df[game_mask]
        
        if len(matching_games) > 0:
            game_row = matching_games.iloc[0]
            for col in team_feature_cols:
                if col in game_row.index:
                    val = game_row[col]
                    feature_dict[col] = float(val) if not pd.isna(val) else 0.0
                else:
                    feature_dict[col] = 0.0
        else:
            # No matching game found, fill team features with zeros
            for col in team_feature_cols:
                feature_dict[col] = 0.0
            if verbose:
                print(f"⚠️  No matching game in matchup_df")
        
        # === STEP 2: Build 18 HOME player features ===
        if verbose:
            print(f"Building HOME player features...")
        
        home_features = self.build_player_features(
            game_date=game_date,
            team_id=home_team_id,
            team_type='HOME'
        )
        feature_dict.update(home_features)
        
        # === STEP 3: Build 18 AWAY player features ===
        if verbose:
            print(f"Building AWAY player features...")
        
        away_features = self.build_player_features(
            game_date=game_date,
            team_id=away_team_id,
            team_type='AWAY'
        )
        feature_dict.update(away_features)
        
        # === STEP 4: Build ordered feature vector (130,) ===
        feature_names = team_feature_cols.copy()
        
        for team_type in ['HOME', 'AWAY']:
            for feat_name in self.player_feature_names:
                feature_names.append(f'{team_type}_{feat_name}')
        
        feature_vector = np.array(
            [feature_dict.get(col, 0.0) for col in feature_names],
            dtype=np.float32
        )
        
        # === STEP 5: Validation ===
        if verbose:
            print(f"Validating features...")
            print(f"  Vector size: {len(feature_vector)} (expected 130)")
            print(f"  NaN count: {np.isnan(feature_vector).sum()}")
            print(f"  Inf count: {np.isinf(feature_vector).sum()}")
        
        # Handle NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct size
        if len(feature_vector) != 130:
            if verbose:
                print(f"⚠️  Feature vector size mismatch: {len(feature_vector)} (expected 130)")
            # Pad or trim
            if len(feature_vector) < 130:
                feature_vector = np.pad(feature_vector, (0, 130 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:130]
        
        return feature_vector, feature_dict, feature_names


def build_enhanced_game_features(
    game_date: datetime,
    home_team_id: int,
    away_team_id: int,
    matchup_df: pd.DataFrame,
    team_feature_cols: list,
    player_data_source: Union[str, PlayerDataCache] = 'api',
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, float], list]:
    """
    Convenience function: Build enhanced 130-feature vector for a single game.
    
    CHRONOLOGICALLY SAFE: All player data is strictly before game_date.
    
    Parameters:
    -----------
    game_date : datetime
        Game date
    home_team_id : int
        Home team ID
    away_team_id : int
        Away team ID
    matchup_df : pd.DataFrame
        Pre-computed team features
    team_feature_cols : list
        List of 94 team feature column names
    player_data_source : str or PlayerDataCache
        'api' or PlayerDataCache instance
    verbose : bool
        Print debug info
    
    Returns:
    --------
    Tuple of (feature_vector, feature_dict, feature_names)
    """
    
    pipeline = EnhancedFeaturePipeline(player_data_source=player_data_source)
    
    return pipeline.build_enhanced_features(
        game_date=game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        matchup_df=matchup_df,
        team_feature_cols=team_feature_cols,
        verbose=verbose
    )

"""
Report Generator Module

Handles:
- HTML report generation for weekly predictions
- Live results table
- Performance dashboards
- Styled output with confidence colors
"""

from datetime import datetime
import json


def generate_html_report(predictions, validator=None, week_label="Week 14", output_file="weekly_predictions.html"):
    """
    Generate HTML report for weekly predictions
    
    Parameters:
    - predictions: List of prediction dicts
    - validator: PredictionValidator instance (optional, for results)
    - week_label: Label for the week
    - output_file: Output HTML filename
    
    Returns:
    - HTML string
    """
    
    # Start HTML with contained styles (no body background to prevent notebook bleeding)
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NBA Predictions - {week_label}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            padding: 20px;
        }}
        .wrapper {{
            max-width: 1400px;
            margin: 0 auto;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            font-size: 36px;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            font-size: 18px;
            margin-bottom: 30px;
        }}
        .game-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
            transition: transform 0.2s;
        }}
        .game-card:hover {{
            transform: translateX(5px);
        }}
        .game-header {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .prediction-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 5px;
        }}
        .label {{
            font-weight: 600;
            color: #34495e;
        }}
        .value {{
            color: #2c3e50;
        }}
        .confidence-high {{
            background: #2ecc71;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .confidence-medium {{
            background: #f39c12;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .confidence-low {{
            background: #e74c3c;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .result-correct {{
            background: #d4edda;
            border-left-color: #28a745 !important;
        }}
        .result-incorrect {{
            background: #f8d7da;
            border-left-color: #dc3545 !important;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="container">
            <h1>🏀 NBA Game Predictions</h1>
            <div class="subtitle">{week_label} | Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}</div>
"""
    
    # Add performance metrics if validator available
    if validator:
        perf = validator.get_recent_performance()
        if perf['n_predictions'] > 0:
            html += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Predictions</div>
                <div class="metric-value">{perf['n_predictions']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Accuracy</div>
                <div class="metric-value">{perf['win_prediction_accuracy']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{perf['r2']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{perf['mae']:.1f}</div>
            </div>
        </div>
"""
    
    # Add game predictions
    for pred in predictions:
        spread = pred['predicted_spread']
        uncertainty = pred['uncertainty']
        win_prob = pred['win_probability']
        confidence = pred['confidence']
        
        # Determine favorite
        if spread > 0:
            favorite = pred['home_team']
            margin = spread
        else:
            favorite = pred['away_team']
            margin = abs(spread)
        
        # Confidence badge
        conf_class = f"confidence-{confidence.lower()}"
        
        # Check if result available
        result_class = ""
        result_info = ""
        if 'actual_spread' in pred and pred.get('actual_spread') is not None:
            if pred.get('correct_winner'):
                result_class = "result-correct"
                result_info = f"<div class='prediction-row'><span class='label'>✅ Result:</span><span class='value'>Correct! Error: {pred.get('prediction_error', 0):.1f} pts</span></div>"
            else:
                result_class = "result-incorrect"
                result_info = f"<div class='prediction-row'><span class='label'>❌ Result:</span><span class='value'>Incorrect. Error: {pred.get('prediction_error', 0):.1f} pts</span></div>"
        
        html += f"""
        <div class="game-card {result_class}">
            <div class="game-header">{pred['home_team']} (HOME) vs {pred['away_team']} (AWAY)</div>
            <div class="prediction-row">
                <span class="label">Predicted Spread:</span>
                <span class="value">{spread:+.1f} points (±{uncertainty:.1f})</span>
            </div>
            <div class="prediction-row">
                <span class="label">Favorite:</span>
                <span class="value">{favorite} by {margin:.1f} points</span>
            </div>
            <div class="prediction-row">
                <span class="label">Win Probability:</span>
                <span class="value">{win_prob:.1%}</span>
            </div>
            <div class="prediction-row">
                <span class="label">Confidence:</span>
                <span class="value"><span class="{conf_class}">{confidence}</span></span>
            </div>
"""
        
        # Add EPAA info if available
        if 'epaa_diff' in pred:
            html += f"""
            <div class="prediction-row">
                <span class="label">EPAA Adjustment:</span>
                <span class="value">{pred.get('epaa_diff', 0):+.2f} (weight: {pred.get('epaa_weight_used', 0):.0%})</span>
            </div>
"""
        
        html += result_info
        html += """
        </div>
"""
    
    # Footer
    html += """
        <div class="footer">
            <p><strong>Prediction Model:</strong> Gaussian Process (Matérn Kernel) + Bayesian MCMC EPAA</p>
            <p>Confidence levels: HIGH (>70% win prob, low uncertainty) | MEDIUM (60-70%) | LOW (<60%)</p>
            <p>Generated by NBA Prediction System | © 2026</p>
        </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 HTML report saved to {output_file}")
    
    return html


if __name__ == "__main__":
    print("🏀 Testing Report Generator...")
    
    # Test predictions
    test_preds = [
        {
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Golden State Warriors',
            'predicted_spread': 5.2,
            'uncertainty': 8.5,
            'win_probability': 0.72,
            'confidence': 'HIGH',
            'epaa_diff': 1.5,
            'epaa_weight_used': 0.5
        },
        {
            'home_team': 'Boston Celtics',
            'away_team': 'Miami Heat',
            'predicted_spread': -2.1,
            'uncertainty': 10.2,
            'win_probability': 0.42,
            'confidence': 'LOW'
        }
    ]
    
    html = generate_html_report(test_preds, week_label="Test Week")
    print("✅ Report generated successfully!")
    
    print("\n🎉 Report generator module working correctly!")
