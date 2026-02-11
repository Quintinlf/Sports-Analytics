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
