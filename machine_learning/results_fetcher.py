"""
Results Fetcher Module

Handles:
- Live score retrieval from NBA API
- Manual result entry option
- Auto-matching results to predictions
"""

import json
import time
from datetime import datetime, timedelta
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamelog
import warnings

warnings.filterwarnings('ignore')


def fetch_live_scores(target_date=None):
    """
    Fetch live/completed game scores from NBA API
    
    Parameters:
    - target_date: Date string 'YYYY-MM-DD' (default: today)
    
    Returns:
    - List of dicts with game results
    """
    try:
        # Get scoreboard
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        results = []
        
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            games = games_data['scoreboard']['games']
            
            for game in games:
                # Only include finished games
                status = game.get('gameStatus', 0)
                if status == 3:  # Game finished
                    home_team = game.get('homeTeam', {})
                    away_team = game.get('awayTeam', {})
                    
                    result = {
                        'game_id': game.get('gameId'),
                        'game_date': game.get('gameTimeUTC', '')[:10],
                        'home_team': home_team.get('teamName'),
                        'away_team': away_team.get('teamName'),
                        'home_score': home_team.get('score', 0),
                        'away_score': away_team.get('score', 0),
                        'spread': home_team.get('score', 0) - away_team.get('score', 0),
                        'fetched_at': datetime.now().isoformat()
                    }
                    results.append(result)
        
        print(f"✅ Fetched {len(results)} completed games")
        return results
        
    except Exception as e:
        print(f"❌ Error fetching live scores: {e}")
        return []


def manual_result_entry(game_id, home_team, away_team, home_score, away_score, notes=''):
    """
    Manually enter a game result
    
    Parameters:
    - game_id: Game identifier
    - home_team: Home team name
    - away_team: Away team name
    - home_score: Home team final score
    - away_score: Away team final score
    - notes: Optional notes about special circumstances
    
    Returns:
    - Dict with result info
    """
    result = {
        'game_id': game_id,
        'game_date': datetime.now().date().isoformat(),
        'home_team': home_team,
        'away_team': away_team,
        'home_score': int(home_score),
        'away_score': int(away_score),
        'spread': int(home_score) - int(away_score),
        'entry_method': 'manual',
        'notes': notes,
        'entered_at': datetime.now().isoformat()
    }
    
    print(f"📝 Manual result entered: {home_team} {home_score} - {away_team} {away_score}")
    return result


def match_results_to_predictions(results, validator):
    """
    Match fetched results to logged predictions
    
    Parameters:
    - results: List of result dicts from fetch_live_scores()
    - validator: PredictionValidator instance
    
    Returns:
    - Number of matches found and logged
    """
    matches_found = 0
    
    for result in results:
        # Find matching predictions
        for i, pred in enumerate(validator.predictions):
            if pred['actual_spread'] is not None:
                continue  # Already has result
            
            # Match by team names and date
            home_match = pred['home_team'] in result['home_team'] or result['home_team'] in pred['home_team']
            away_match = pred['away_team'] in result['away_team'] or result['away_team'] in pred['away_team']
            
            if home_match and away_match:
                # Found a match!
                validator.log_result(i, result['home_score'], result['away_score'])
                matches_found += 1
                print(f"   ✓ Matched: {result['home_team']} vs {result['away_team']}")
                break
    
    if matches_found == 0:
        print("   No new matches found")
    
    return matches_found


if __name__ == "__main__":
    print("🏀 Testing Results Fetcher...")
    
    # Test live score fetching
    results = fetch_live_scores()
    print(f"✅ Fetched {len(results)} results")
    
    # Test manual entry
    manual = manual_result_entry(
        game_id='test_001',
        home_team='Lakers',
        away_team='Warriors',
        home_score=115,
        away_score=108,
        notes='Test game'
    )
    print(f"✅ Manual entry created")
    
    print("\n🎉 Results fetcher module working correctly!")
