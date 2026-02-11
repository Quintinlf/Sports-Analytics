"""
Main Execution Script: Intelligent Iterative NBA Predictions
Runs comprehensive predictions with confidence-driven retraining
"""

import sys
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'machine_learning'))

from iterative_predictor import IterativePredictor
from game_parser import parse_game_data_from_text
from database_handler import SportsAnalyticsDB


# Game data from user (Feb 1-28, 2026)
GAME_DATA_CSV = """Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
Sun Feb 1 2026,3:30p,Milwaukee Bucks,79,Boston Celtics,107,Box Score,,19156,2:09,TD Garden,
Sun Feb 1 2026,6:00p,Brooklyn Nets,77,Detroit Pistons,130,Box Score,,19899,2:10,Little Caesars Arena,
Sun Feb 1 2026,6:00p,Chicago Bulls,91,Miami Heat,134,Box Score,,19700,2:11,Kaseya Center,
Sun Feb 1 2026,6:00p,Utah Jazz,100,Toronto Raptors,107,Box Score,,18749,2:20,Scotiabank Arena,
Sun Feb 1 2026,6:00p,Sacramento Kings,112,Washington Wizards,116,Box Score,,13102,2:15,Capital One Arena,
Sun Feb 1 2026,7:00p,Los Angeles Lakers,100,New York Knicks,112,Box Score,,19812,2:11,Madison Square Garden (IV),
Sun Feb 1 2026,8:00p,Los Angeles Clippers,117,Phoenix Suns,93,Box Score,,17071,2:26,Mortgage Matchup Center,
Sun Feb 1 2026,9:00p,Cleveland Cavaliers,130,Portland Trail Blazers,111,Box Score,,17240,2:05,Moda Center,
Sun Feb 1 2026,9:00p,Orlando Magic,103,San Antonio Spurs,112,Box Score,,18354,2:18,Frost Bank Center,
Sun Feb 1 2026,9:30p,Oklahoma City Thunder,121,Denver Nuggets,111,Box Score,,19900,2:18,Ball Arena,
Mon Feb 2 2026,3:00p,New Orleans Pelicans,95,Charlotte Hornets,102,Box Score,,17263,2:18,Spectrum Center,
Mon Feb 2 2026,7:00p,Houston Rockets,118,Indiana Pacers,114,Box Score,,16511,2:21,Gainbridge Fieldhouse,
Mon Feb 2 2026,7:30p,Minnesota Timberwolves,128,Memphis Grizzlies,137,Box Score,,14005,2:31,FedExForum,
Mon Feb 2 2026,10:00p,Philadelphia 76ers,128,Los Angeles Clippers,113,Box Score,,17927,2:18,Intuit Dome,
Tue Feb 3 2026,7:00p,Denver Nuggets,121,Detroit Pistons,124,Box Score,,19976,2:35,Little Caesars Arena,
Tue Feb 3 2026,7:00p,Utah Jazz,131,Indiana Pacers,122,Box Score,,16678,2:02,Gainbridge Fieldhouse,
Tue Feb 3 2026,7:00p,New York Knicks,132,Washington Wizards,101,Box Score,,17822,2:16,Capital One Arena,
Tue Feb 3 2026,7:30p,Los Angeles Lakers,125,Brooklyn Nets,109,Box Score,,18248,2:10,Barclays Center,
Tue Feb 3 2026,7:30p,Atlanta Hawks,127,Miami Heat,115,Box Score,,19700,2:28,Kaseya Center,
Tue Feb 3 2026,8:00p,Boston Celtics,110,Dallas Mavericks,100,Box Score,,19132,2:15,American Airlines Center,
Tue Feb 3 2026,8:00p,Chicago Bulls,115,Milwaukee Bucks,131,Box Score,,17341,2:03,Fiserv Forum,
Tue Feb 3 2026,8:00p,Orlando Magic,92,Oklahoma City Thunder,128,Box Score,,18203,2:11,Paycom Center,
Tue Feb 3 2026,10:00p,Philadelphia 76ers,113,Golden State Warriors,94,Box Score,,18064,2:05,Chase Center,
Tue Feb 3 2026,11:00p,Phoenix Suns,130,Portland Trail Blazers,125,Box Score,,16092,2:22,Moda Center,
Wed Feb 4 2026,7:00p,Denver Nuggets,127,New York Knicks,134,Box Score,2OT,19812,2:58,Madison Square Garden (IV),
Wed Feb 4 2026,7:30p,Minnesota Timberwolves,128,Toronto Raptors,126,Box Score,,18775,2:19,Scotiabank Arena,
Wed Feb 4 2026,8:00p,Boston Celtics,114,Houston Rockets,93,Box Score,,18055,2:08,Toyota Center,
Wed Feb 4 2026,8:00p,New Orleans Pelicans,137,Milwaukee Bucks,141,Box Score,OT,14343,2:34,Fiserv Forum,
Wed Feb 4 2026,9:30p,Oklahoma City Thunder,106,San Antonio Spurs,116,Box Score,,18354,2:12,Frost Bank Center,
Wed Feb 4 2026,10:00p,Memphis Grizzlies,129,Sacramento Kings,125,Box Score,,15017,2:24,Golden 1 Center,
Wed Feb 4 2026,10:30p,Cleveland Cavaliers,124,Los Angeles Clippers,91,Box Score,,17927,1:58,Intuit Dome,
Thu Feb 5 2026,7:00p,Washington Wizards,126,Detroit Pistons,117,Box Score,,19401,2:13,Little Caesars Arena,
Thu Feb 5 2026,7:00p,Brooklyn Nets,98,Orlando Magic,118,Box Score,,18093,2:25,Kia Center,
Thu Feb 5 2026,7:30p,Utah Jazz,119,Atlanta Hawks,121,Box Score,,15412,2:17,State Farm Arena,
Thu Feb 5 2026,7:30p,Chicago Bulls,107,Toronto Raptors,123,Box Score,,18795,2:06,Scotiabank Arena,
Thu Feb 5 2026,8:00p,Charlotte Hornets,109,Houston Rockets,99,Box Score,,18055,2:07,Toyota Center,
Thu Feb 5 2026,8:30p,San Antonio Spurs,135,Dallas Mavericks,123,Box Score,,19413,2:13,American Airlines Center,
Thu Feb 5 2026,10:00p,Philadelphia 76ers,115,Los Angeles Lakers,119,Box Score,,18731,2:20,Crypto.com Arena,
Thu Feb 5 2026,10:00p,Golden State Warriors,101,Phoenix Suns,97,Box Score,,17071,2:12,Mortgage Matchup Center,
Fri Feb 6 2026,7:30p,Miami Heat,96,Boston Celtics,98,Box Score,,19156,2:24,TD Garden,
Fri Feb 6 2026,7:30p,New York Knicks,80,Detroit Pistons,118,Box Score,,20062,2:17,Little Caesars Arena,
Fri Feb 6 2026,8:00p,Indiana Pacers,99,Milwaukee Bucks,105,Box Score,,17341,2:07,Fiserv Forum,
Fri Feb 6 2026,8:00p,New Orleans Pelicans,119,Minnesota Timberwolves,115,Box Score,,18978,2:14,Target Center,
Fri Feb 6 2026,10:00p,Memphis Grizzlies,115,Portland Trail Blazers,135,Box Score,,16895,2:05,Moda Center,
Fri Feb 6 2026,10:00p,Los Angeles Clippers,114,Sacramento Kings,111,Box Score,,16665,2:27,Golden 1 Center,
Sat Feb 7 2026,3:00p,Washington Wizards,113,Brooklyn Nets,127,Box Score,,17548,2:10,Barclays Center,
Sat Feb 7 2026,3:30p,Houston Rockets,112,Oklahoma City Thunder,106,Box Score,,18203,2:37,Paycom Center,
Sat Feb 7 2026,6:00p,Dallas Mavericks,125,San Antonio Spurs,138,Box Score,,18617,2:18,Frost Bank Center,
Sat Feb 7 2026,7:00p,Utah Jazz,117,Orlando Magic,120,Box Score,,19203,2:23,Kia Center,
Sat Feb 7 2026,7:30p,Charlotte Hornets,126,Atlanta Hawks,119,Box Score,,17492,2:23,State Farm Arena,
Sat Feb 7 2026,8:00p,Denver Nuggets,136,Chicago Bulls,120,Box Score,,20939,2:17,United Center,
Sat Feb 7 2026,8:30p,Golden State Warriors,99,Los Angeles Lakers,105,Box Score,,18997,2:20,Crypto.com Arena,
Sat Feb 7 2026,9:00p,Philadelphia 76ers,109,Phoenix Suns,103,Box Score,,17071,2:30,Mortgage Matchup Center,
Sat Feb 7 2026,10:00p,Memphis Grizzlies,115,Portland Trail Blazers,122,Box Score,,16273,2:07,Moda Center,
Sat Feb 7 2026,10:00p,Cleveland Cavaliers,132,Sacramento Kings,126,Box Score,,16212,2:14,Golden 1 Center,
Sun Feb 8 2026,12:30p,New York Knicks,,Boston Celtics,,,,,,TD Garden,
Sun Feb 8 2026,2:00p,Miami Heat,,Washington Wizards,,,,,,Capital One Arena,
Sun Feb 8 2026,3:00p,Los Angeles Clippers,,Minnesota Timberwolves,,,,,,Target Center,
Sun Feb 8 2026,3:00p,Indiana Pacers,,Toronto Raptors,,,,,,Scotiabank Arena,
Mon Feb 9 2026,7:00p,Detroit Pistons,,Charlotte Hornets,,,,,,Spectrum Center,
Mon Feb 9 2026,7:30p,Chicago Bulls,,Brooklyn Nets,,,,,,Barclays Center,
Mon Feb 9 2026,7:30p,Utah Jazz,,Miami Heat,,,,,,Kaseya Center,
Mon Feb 9 2026,7:30p,Milwaukee Bucks,,Orlando Magic,,,,,,Kia Center,
Mon Feb 9 2026,8:00p,Atlanta Hawks,,Minnesota Timberwolves,,,,,,Target Center,
Mon Feb 9 2026,8:00p,Sacramento Kings,,New Orleans Pelicans,,,,,,Smoothie King Center,
Mon Feb 9 2026,9:00p,Cleveland Cavaliers,,Denver Nuggets,,,,,,Ball Arena,
Mon Feb 9 2026,10:00p,Memphis Grizzlies,,Golden State Warriors,,,,,,Chase Center,
Mon Feb 9 2026,10:00p,Oklahoma City Thunder,,Los Angeles Lakers,,,,,,Crypto.com Arena,
Mon Feb 9 2026,10:00p,Philadelphia 76ers,,Portland Trail Blazers,,,,,,Moda Center,
Tue Feb 10 2026,7:30p,Indiana Pacers,,New York Knicks,,,,,,Madison Square Garden (IV),
Tue Feb 10 2026,8:00p,Los Angeles Clippers,,Houston Rockets,,,,,,Toyota Center,
Tue Feb 10 2026,9:00p,Dallas Mavericks,,Phoenix Suns,,,,,,Mortgage Matchup Center,
Tue Feb 10 2026,10:30p,San Antonio Spurs,,Los Angeles Lakers,,,,,,Crypto.com Arena,
Wed Feb 11 2026,7:00p,Atlanta Hawks,,Charlotte Hornets,,,,,,Spectrum Center,
Wed Feb 11 2026,7:00p,Washington Wizards,,Cleveland Cavaliers,,,,,,Rocket Arena,
Wed Feb 11 2026,7:00p,Milwaukee Bucks,,Orlando Magic,,,,,,Kia Center,
Wed Feb 11 2026,7:30p,Chicago Bulls,,Boston Celtics,,,,,,TD Garden,
Wed Feb 11 2026,7:30p,Indiana Pacers,,Brooklyn Nets,,,,,,Barclays Center,
Wed Feb 11 2026,7:30p,New York Knicks,,Philadelphia 76ers,,,,,,Xfinity Mobile Arena,
Wed Feb 11 2026,7:30p,Detroit Pistons,,Toronto Raptors,,,,,,Scotiabank Arena,
Wed Feb 11 2026,8:00p,Los Angeles Clippers,,Houston Rockets,,,,,,Toyota Center,
Wed Feb 11 2026,8:00p,Portland Trail Blazers,,Minnesota Timberwolves,,,,,,Target Center,
Wed Feb 11 2026,8:00p,Miami Heat,,New Orleans Pelicans,,,,,,Smoothie King Center,
Wed Feb 11 2026,9:00p,Memphis Grizzlies,,Denver Nuggets,,,,,,Ball Arena,
Wed Feb 11 2026,9:00p,Oklahoma City Thunder,,Phoenix Suns,,,,,,Mortgage Matchup Center,
Wed Feb 11 2026,9:00p,Sacramento Kings,,Utah Jazz,,,,,,Delta Center,
Wed Feb 11 2026,10:00p,San Antonio Spurs,,Golden State Warriors,,,,,,Chase Center,
Thu Feb 12 2026,7:30p,Milwaukee Bucks,,Oklahoma City Thunder,,,,,,Paycom Center,
Thu Feb 12 2026,9:00p,Portland Trail Blazers,,Utah Jazz,,,,,,Delta Center,
Thu Feb 12 2026,10:00p,Dallas Mavericks,,Los Angeles Lakers,,,,,,Crypto.com Arena,
Thu Feb 19 2026,7:00p,Houston Rockets,,Charlotte Hornets,,,,,,Spectrum Center,
Thu Feb 19 2026,7:00p,Brooklyn Nets,,Cleveland Cavaliers,,,,,,Rocket Arena,
Thu Feb 19 2026,7:00p,Atlanta Hawks,,Philadelphia 76ers,,,,,,Xfinity Mobile Arena,
Thu Feb 19 2026,7:00p,Indiana Pacers,,Washington Wizards,,,,,,Capital One Arena,
Thu Feb 19 2026,7:30p,Detroit Pistons,,New York Knicks,,,,,,Madison Square Garden (IV),
Thu Feb 19 2026,8:00p,Toronto Raptors,,Chicago Bulls,,,,,,United Center,
Thu Feb 19 2026,8:30p,Phoenix Suns,,San Antonio Spurs,,,,,,Moody Center,
Thu Feb 19 2026,10:00p,Boston Celtics,,Golden State Warriors,,,,,,Chase Center,
Thu Feb 19 2026,10:00p,Orlando Magic,,Sacramento Kings,,,,,,Golden 1 Center,
Thu Feb 19 2026,10:30p,Denver Nuggets,,Los Angeles Clippers,,,,,,Intuit Dome,
Fri Feb 20 2026,7:00p,Cleveland Cavaliers,,Charlotte Hornets,,,,,,Spectrum Center,
Fri Feb 20 2026,7:00p,Utah Jazz,,Memphis Grizzlies,,,,,,FedExForum,
Fri Feb 20 2026,7:00p,Indiana Pacers,,Washington Wizards,,,,,,Capital One Arena,
Fri Feb 20 2026,7:30p,Miami Heat,,Atlanta Hawks,,,,,,State Farm Arena,
Fri Feb 20 2026,7:30p,Dallas Mavericks,,Minnesota Timberwolves,,,,,,Target Center,
Fri Feb 20 2026,8:00p,Milwaukee Bucks,,New Orleans Pelicans,,,,,,Smoothie King Center,
Fri Feb 20 2026,8:00p,Brooklyn Nets,,Oklahoma City Thunder,,,,,,Paycom Center,
Fri Feb 20 2026,10:00p,Los Angeles Clippers,,Los Angeles Lakers,,,,,,Crypto.com Arena,
Fri Feb 20 2026,10:00p,Denver Nuggets,,Portland Trail Blazers,,,,,,Moda Center,
Sat Feb 21 2026,5:00p,Orlando Magic,,Phoenix Suns,,,,,,Mortgage Matchup Center,
Sat Feb 21 2026,7:00p,Philadelphia 76ers,,New Orleans Pelicans,,,,,,Smoothie King Center,
Sat Feb 21 2026,8:00p,Detroit Pistons,,Chicago Bulls,,,,,,United Center,
Sat Feb 21 2026,8:00p,Memphis Grizzlies,,Miami Heat,,,,,,Kaseya Center,
Sat Feb 21 2026,8:00p,Sacramento Kings,,San Antonio Spurs,,,,,,Moody Center,
Sat Feb 21 2026,8:30p,Houston Rockets,,New York Knicks,,,,,,Madison Square Garden (IV),
Sun Feb 22 2026,1:00p,Cleveland Cavaliers,,Oklahoma City Thunder,,,,,,Paycom Center,
Sun Feb 22 2026,3:30p,Brooklyn Nets,,Atlanta Hawks,,,,,,State Farm Arena,
Sun Feb 22 2026,3:30p,Denver Nuggets,,Golden State Warriors,,,,,,Chase Center,
Sun Feb 22 2026,3:30p,Toronto Raptors,,Milwaukee Bucks,,,,,,Fiserv Forum,
Sun Feb 22 2026,5:00p,Dallas Mavericks,,Indiana Pacers,,,,,,Gainbridge Fieldhouse,
Sun Feb 22 2026,6:00p,Charlotte Hornets,,Washington Wizards,,,,,,Capital One Arena,
Sun Feb 22 2026,6:30p,Boston Celtics,,Los Angeles Lakers,,,,,,Crypto.com Arena,
Sun Feb 22 2026,7:00p,Philadelphia 76ers,,Minnesota Timberwolves,,,,,,Target Center,
Sun Feb 22 2026,8:00p,New York Knicks,,Chicago Bulls,,,,,,United Center,
Sun Feb 22 2026,8:00p,Portland Trail Blazers,,Phoenix Suns,,,,,,Mortgage Matchup Center,
Sun Feb 22 2026,9:00p,Orlando Magic,,Los Angeles Clippers,,,,,,Intuit Dome,
Mon Feb 23 2026,7:00p,San Antonio Spurs,,Detroit Pistons,,,,,,Little Caesars Arena,
Mon Feb 23 2026,8:00p,Sacramento Kings,,Memphis Grizzlies,,,,,,FedExForum,
Mon Feb 23 2026,9:30p,Utah Jazz,,Houston Rockets,,,,,,Toyota Center,
Tue Feb 24 2026,7:00p,Philadelphia 76ers,,Indiana Pacers,,,,,,Gainbridge Fieldhouse,
Tue Feb 24 2026,7:30p,Washington Wizards,,Atlanta Hawks,,,,,,State Farm Arena,
Tue Feb 24 2026,7:30p,Dallas Mavericks,,Brooklyn Nets,,,,,,Barclays Center,
Tue Feb 24 2026,7:30p,New York Knicks,,Cleveland Cavaliers,,,,,,Rocket Arena,
Tue Feb 24 2026,7:30p,Oklahoma City Thunder,,Toronto Raptors,,,,,,Scotiabank Arena,
Tue Feb 24 2026,8:00p,Charlotte Hornets,,Chicago Bulls,,,,,,United Center,
Tue Feb 24 2026,8:00p,Miami Heat,,Milwaukee Bucks,,,,,,Fiserv Forum,
Tue Feb 24 2026,8:00p,Golden State Warriors,,New Orleans Pelicans,,,,,,Smoothie King Center,
Tue Feb 24 2026,9:00p,Boston Celtics,,Phoenix Suns,,,,,,Mortgage Matchup Center,
Tue Feb 24 2026,10:00p,Minnesota Timberwolves,,Portland Trail Blazers,,,,,,Moda Center,
Tue Feb 24 2026,10:30p,Orlando Magic,,Los Angeles Lakers,,,,,,Crypto.com Arena,
Wed Feb 25 2026,7:00p,Oklahoma City Thunder,,Detroit Pistons,,,,,,Little Caesars Arena,
Wed Feb 25 2026,7:30p,Golden State Warriors,,Memphis Grizzlies,,,,,,FedExForum,
Wed Feb 25 2026,7:30p,San Antonio Spurs,,Toronto Raptors,,,,,,Scotiabank Arena,
Wed Feb 25 2026,8:00p,Sacramento Kings,,Houston Rockets,,,,,,Toyota Center,
Wed Feb 25 2026,8:00p,Cleveland Cavaliers,,Milwaukee Bucks,,,,,,Fiserv Forum,
Wed Feb 25 2026,10:00p,Boston Celtics,,Denver Nuggets,,,,,,Ball Arena,
Thu Feb 26 2026,7:00p,Charlotte Hornets,,Indiana Pacers,,,,,,Gainbridge Fieldhouse,
Thu Feb 26 2026,7:00p,Miami Heat,,Philadelphia 76ers,,,,,,Xfinity Mobile Arena,
Thu Feb 26 2026,7:30p,Washington Wizards,,Atlanta Hawks,,,,,,State Farm Arena,
Thu Feb 26 2026,7:30p,San Antonio Spurs,,Brooklyn Nets,,,,,,Barclays Center,
Thu Feb 26 2026,7:30p,Houston Rockets,,Orlando Magic,,,,,,Kia Center,
Thu Feb 26 2026,8:00p,Portland Trail Blazers,,Chicago Bulls,,,,,,United Center,
Thu Feb 26 2026,8:30p,Sacramento Kings,,Dallas Mavericks,,,,,,American Airlines Center,
Thu Feb 26 2026,9:00p,Los Angeles Lakers,,Phoenix Suns,,,,,,Mortgage Matchup Center,
Thu Feb 26 2026,9:00p,New Orleans Pelicans,,Utah Jazz,,,,,,Delta Center,
Thu Feb 26 2026,10:00p,Minnesota Timberwolves,,Los Angeles Clippers,,,,,,Intuit Dome,
Fri Feb 27 2026,7:00p,Cleveland Cavaliers,,Detroit Pistons,,,,,,Little Caesars Arena,
Fri Feb 27 2026,7:30p,Brooklyn Nets,,Boston Celtics,,,,,,TD Garden,
Fri Feb 27 2026,8:00p,New York Knicks,,Milwaukee Bucks,,,,,,Fiserv Forum,
Fri Feb 27 2026,8:30p,Memphis Grizzlies,,Dallas Mavericks,,,,,,American Airlines Center,
Fri Feb 27 2026,9:30p,Denver Nuggets,,Oklahoma City Thunder,,,,,,Paycom Center,
Sat Feb 28 2026,1:00p,Portland Trail Blazers,,Charlotte Hornets,,,,,,Spectrum Center,
Sat Feb 28 2026,3:00p,Houston Rockets,,Miami Heat,,,,,,Kaseya Center,
Sat Feb 28 2026,7:00p,Toronto Raptors,,Washington Wizards,,,,,,Capital One Arena,
Sat Feb 28 2026,8:30p,Los Angeles Lakers,,Golden State Warriors,,,,,,Chase Center,
Sat Feb 28 2026,9:30p,New Orleans Pelicans,,Utah Jazz,,,,,,Delta Center,"""


def print_prediction_summary(prediction: Dict):
    """Pretty print a single prediction"""
    print(f"{prediction['away_team']} @ {prediction['home_team']}")
    print(f"   Predicted Winner: {prediction['predicted_winner']} ")
    print(f"   Spread: {prediction['predicted_spread']:.1f}")
    print(f"   Win Probability: {prediction['win_probability']:.1%}")
    print(f"   Confidence: {prediction['confidence_score']:.3f} ({prediction['confidence_level']})")
    print(f"   Iterations: {prediction['iteration_count']}")
    print()


def validate_completed_games(predictor: IterativePredictor, completed_games: List[Dict], db: SportsAnalyticsDB):
    """Validate model on completed games and store results"""
    print("\n" + "=" * 70)
    print("✅ VALIDATING ON COMPLETED GAMES (Feb 1-7, 2026)")
    print("=" * 70 + "\n")
    
    validation_results = []
    correct_winners = 0
    total_error = 0
    
    for game in completed_games[:10]:  # Validate first 10 for speed
        print(f"Validating: {game['away_team']} @ {game['home_team']} ({game['game_date']})")
        
        # Make prediction
        prediction = predictor.predict_with_retraining(
            home_team=game['home_team'],
            away_team=game['away_team'],
            game_date=game['game_date'],
            game_id=game['game_id']
        )
        
        if prediction:
            # Save prediction to DB
            pred_id = db.insert_prediction(prediction)
            
            # Calculate error
            error = abs(prediction['predicted_spread'] - game['actual_spread'])
            correct_winner = (prediction['predicted_winner'] == game['actual_winner'])
            
            within_ci = False
            if prediction.get('ci_lower') and prediction.get('ci_upper'):
                within_ci = (prediction['ci_lower'] <= game['actual_spread'] <= prediction['ci_upper'])
            
            # Store result
            result_data = {
                'actual_home_score': game['actual_home_score'],
                'actual_away_score': game['actual_away_score'],
                'actual_spread': game['actual_spread'],
                'actual_winner': game['actual_winner'],
                'prediction_error': error,
                'correct_winner': correct_winner,
                'within_ci': within_ci
            }
            
            db.insert_result(pred_id, result_data)
            
            # Update counters
            if correct_winner:
                correct_winners += 1
            total_error += error
            
            validation_results.append({
                'game': game,
                'prediction': prediction,
                'error': error,
                'correct_winner': correct_winner
            })
            
            print(f"   ✓ Error: {error:.1f} pts | Winner: {'✓' if correct_winner else '✗'}\n")
    
    # Print summary
    print("=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Games Validated: {len(validation_results)}")
    print(f"Winner Accuracy: {correct_winners}/{len(validation_results)} ({correct_winners/len(validation_results)*100:.1f}%)")
    print(f"Average Error: {total_error/len(validation_results):.2f} points")
    print("=" * 70 + "\n")
    
    return validation_results


def predict_upcoming_games(predictor: IterativePredictor, upcoming_games: List[Dict], db: SportsAnalyticsDB):
    """Make predictions on upcoming games"""
    print("\n" + "=" * 70)
    print("🔮 PREDICTING UPCOMING GAMES (Feb 8-28, 2026)")
    print("=" * 70 + "\n")
    
    predictions = []
    
    for idx, game in enumerate(upcoming_games, 1):
        print(f"\n[{idx}/{len(upcoming_games)}] {game['game_date']}")
        
        # Make prediction
        prediction = predictor.predict_with_retraining(
            home_team=game['home_team'],
            away_team=game['away_team'],
            game_date=game['game_date'],
            game_id=game['game_id']
        )
        
        if prediction:
            # Save to database
            pred_id = db.insert_prediction(prediction)
            prediction['db_id'] = pred_id
            predictions.append(prediction)
            
            print_prediction_summary(prediction)
    
    return predictions


def print_final_statistics(predictor: IterativePredictor, all_predictions: List[Dict]):
    """Print comprehensive final statistics"""
    print("\n" + "=" * 70)
    print("📈 FINAL STATISTICS")
    print("=" * 70)
    
    # Predictor stats
    stats = predictor.get_statistics()
    print(f"\n🤖 Predictor Performance:")
    print(f"   Total Predictions: {stats['total_predictions']}")
    print(f"   Retraining Triggered: {stats['retraining_triggered']}")
    print(f"   Retraining Rate: {stats['retraining_triggered']/stats['total_predictions']*100:.1f}%")
    
    # Confidence distribution
    if all_predictions:
        confidences = [p['confidence_score'] for p in all_predictions]
        iterations = [p['iteration_count'] for p in all_predictions]
        
        print(f"\n📊 Confidence Distribution:")
        print(f"   Average: {sum(confidences)/len(confidences):.3f}")
        print(f"   Min: {min(confidences):.3f}")
        print(f"   Max: {max(confidences):.3f}")
        
        high_conf = sum(1 for c in confidences if c >= 0.6)
        med_conf = sum(1 for c in confidences if 0.3 <= c < 0.6)
        low_conf = sum(1 for c in confidences if c < 0.3)
        
        print(f"   HIGH (≥0.6): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
        print(f"   MEDIUM (0.3-0.6): {med_conf} ({med_conf/len(confidences)*100:.1f}%)")
        print(f"   LOW (<0.3): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
        
        print(f"\n🔄 Iteration Statistics:")
        print(f"   Average Iterations: {sum(iterations)/len(iterations):.2f}")
        print(f"   Max Iterations: {max(iterations)}")
        print(f"   Single Pass (1 iter): {sum(1 for i in iterations if i == 1)}")
        print(f"   Multiple Iterations: {sum(1 for i in iterations if i > 1)}")
    
    print("\n" + "=" * 70)
    print("✅ EXECUTION COMPLETE")
    print("=" * 70)
    print(f"💾 Database: sports_analytics.db")
    print(f"📊 All predictions stored and ready for analysis")
    print("=" * 70 + "\n")


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("🚀 INTELLIGENT ITERATIVE NBA PREDICTIONS")
    print("=" * 70)
    print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Confidence Threshold: 0.6")
    print(f"🔄 Max Iterations: 10")
    print(f"📊 Using 3-season training data (2022-23, 2023-24, 2024-25)")
    print("=" * 70 + "\n")
    
    # Parse game data
    parsed_data = parse_game_data_from_text(GAME_DATA_CSV)
    completed_games = parsed_data['completed_list']
    upcoming_games = parsed_data['upcoming_list']
    
    print(f"\n📋 Games Parsed:")
    print(f"   Completed (with results): {len(completed_games)}")
    print(f"   Upcoming (to predict): {len(upcoming_games)}")
    
    # Initialize database
    db = SportsAnalyticsDB("sports_analytics.db")
    
    # Initialize predictor
    predictor = IterativePredictor(
        confidence_threshold=0.6,
        max_iterations=10,
        db_path="sports_analytics.db",
        verbose=True
    )
    
    # Load/train models
    predictor.load_models(force_retrain=False)
    
    # Validate on completed games (sample)
    validation_results = validate_completed_games(predictor, completed_games, db)
    
    # Predict upcoming games
    upcoming_predictions = predict_upcoming_games(predictor, upcoming_games, db)
    
    # Combine all predictions
    all_predictions = [v['prediction'] for v in validation_results] + upcoming_predictions
    
    # Print final statistics
    print_final_statistics(predictor, all_predictions)
    
    # Cleanup
    predictor.close()
    db.close()
    
    print("✨ All done! Check sports_analytics.db for stored predictions.\n")


if __name__ == "__main__":
    main()
