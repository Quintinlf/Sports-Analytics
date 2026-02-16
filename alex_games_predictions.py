"""
Predictions for Alex's Games - February 19, 2026
Simple script to run predictions for specific games
"""

import sys
import os
from datetime import datetime

# Add paths
parent_dir = r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics'
sys.path.insert(0, parent_dir)

from machine_learning.iterative_predictor import IterativePredictor
from machine_learning.game_parser import parse_game_data_from_text
from machine_learning.database_handler import SportsAnalyticsDB


# Game data for February 19, 2026
GAME_DATA_CSV = """Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
Thu Feb 19 2026,7:00p,Atlanta Hawks,,Philadelphia 76ers,,,,,,Xfinity Mobile Arena,
Thu Feb 19 2026,7:00p,Indiana Pacers,,Washington Wizards,,,,,,Capital One Arena,
Thu Feb 19 2026,7:30p,Detroit Pistons,,New York Knicks,,,,,,Madison Square Garden (IV),
Thu Feb 19 2026,8:00p,Toronto Raptors,,Chicago Bulls,,,,,,United Center,
Thu Feb 19 2026,8:30p,Phoenix Suns,,San Antonio Spurs,,,,,,Moody Center,
Thu Feb 19 2026,10:00p,Boston Celtics,,Golden State Warriors,,,,,,Chase Center,
Thu Feb 19 2026,10:00p,Orlando Magic,,Sacramento Kings,,,,,,Golden 1 Center,
Thu Feb 19 2026,10:30p,Denver Nuggets,,Los Angeles Clippers,,,,,,Intuit Dome,"""


def print_prediction_summary(prediction, game_num, total_games):
    """Pretty print a single prediction"""
    print(f"\n{'='*70}")
    print(f"Game {game_num}/{total_games}: {prediction['away_team']} @ {prediction['home_team']}")
    print(f"{'='*70}")
    print(f"🏆 Predicted Winner: {prediction['predicted_winner']}")
    print(f"📊 Predicted Spread: {prediction['predicted_spread']:.1f}")
    print(f"🎯 Win Probability: {prediction['win_probability']:.1%}")
    print(f"💪 Confidence: {prediction['confidence_score']:.3f} ({prediction['confidence_level']})")
    print(f"🔄 Iterations: {prediction['iteration_count']}")
    print(f"{'='*70}")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🏀 NBA PREDICTIONS FOR FEBRUARY 19, 2026")
    print("="*70)
    print(f"📅 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Parse game data
    print("📋 Parsing game data...")
    parsed_data = parse_game_data_from_text(GAME_DATA_CSV, verbose=False)
    upcoming_games = parsed_data['upcoming_list']
    
    print(f"✅ Found {len(upcoming_games)} games to predict\n")
    for i, game in enumerate(upcoming_games, 1):
        print(f"  {i}. {game['away_team']} @ {game['home_team']} ({game.get('start_time', 'TBD')})")
    
    # Initialize database and predictor
    print("\n🔧 Initializing prediction system...")
    db = SportsAnalyticsDB("sports_analytics.db")
    
    predictor = IterativePredictor(
        confidence_threshold=0.6,
        max_iterations=10,
        db_path="sports_analytics.db",
        verbose=True
    )
    
    # Load models
    print("\n🤖 Loading prediction models...")
    try:
        predictor.load_models(force_retrain=False)
        print("✅ Models loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("⚠️  Attempting to train new models...")
        predictor.load_models(force_retrain=True)
    
    # Run predictions
    print("\n" + "="*70)
    print("🎯 RUNNING PREDICTIONS")
    print("="*70)
    
    predictions = []
    
    for i, game in enumerate(upcoming_games, 1):
        try:
            print(f"\n[{i}/{len(upcoming_games)}] Predicting: {game['away_team']} @ {game['home_team']}")
            
            prediction = predictor.predict_game(
                away_team=game['away_team'],
                home_team=game['home_team'],
                game_date=game['game_date']
            )
            
            predictions.append(prediction)
            
            # Store in database
            db.insert_prediction(
                game_date=game['game_date'],
                away_team=game['away_team'],
                home_team=game['home_team'],
                predicted_winner=prediction['predicted_winner'],
                predicted_spread=prediction['predicted_spread'],
                win_probability=prediction['win_probability'],
                confidence_score=prediction['confidence_score'],
                confidence_level=prediction['confidence_level'],
                model_version='iterative_lgbm',
                iteration_count=prediction['iteration_count']
            )
            
            # Print summary
            print_prediction_summary(prediction, i, len(upcoming_games))
            
        except Exception as e:
            print(f"\n❌ ERROR predicting game: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    if predictions:
        print(f"\nSuccessfully predicted {len(predictions)} out of {len(upcoming_games)} games\n")
        
        # Statistics
        confidences = [p['confidence_score'] for p in predictions]
        iterations = [p['iteration_count'] for p in predictions]
        
        print("📈 Performance Statistics:")
        print(f"   Average Confidence: {sum(confidences)/len(confidences):.3f}")
        print(f"   Average Iterations: {sum(iterations)/len(iterations):.2f}")
        print(f"   High Confidence (≥0.6): {sum(1 for c in confidences if c >= 0.6)} games")
        print(f"   Medium Confidence (0.3-0.6): {sum(1 for c in confidences if 0.3 <= c < 0.6)} games")
        print(f"   Low Confidence (<0.3): {sum(1 for c in confidences if c < 0.3)} games")
        
        print("\n💾 All predictions saved to sports_analytics.db")
    else:
        print("\n❌ No predictions were generated")
    
    # Cleanup
    predictor.close()
    db.close()
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
