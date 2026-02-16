"""
NBA Game Predictions - February 19, 2026
Simplified prediction system
"""

from datetime import datetime

# Game schedule
GAMES = [
    ("Atlanta Hawks", "Philadelphia 76ers", "7:00p"),
    ("Indiana Pacers", "Washington Wizards", "7:00p"),
    ("Detroit Pistons", "New York Knicks", "7:30p"),
    ("Toronto Raptors", "Chicago Bulls", "8:00p"),
    ("Phoenix Suns", "San Antonio Spurs", "8:30p"),
    ("Boston Celtics", "Golden State Warriors", "10:00p"),
    ("Orlando Magic", "Sacramento Kings", "10:00p"),
    ("Denver Nuggets", "Los Angeles Clippers", "10:30p"),
]

# Team strength ratings (based on 2024-25 season performance)
TEAM_STRENGTH = {
    'Boston Celtics': 95,
    'Cleveland Cavaliers': 92,
    'Oklahoma City Thunder': 91,
    'Denver Nuggets': 90,
    'Milwaukee Bucks': 88,
    'Phoenix Suns': 87,
    'Los Angeles Clippers': 86,
    'Golden State Warriors': 85,
    'Los Angeles Lakers': 84,
    'Miami Heat': 82,
    'Philadelphia 76ers': 81,
    'New York Knicks': 80,
    'Sacramento Kings': 78,
    'Dallas Mavericks': 77,
    'Indiana Pacers': 76,
    'Chicago Bulls': 75,
    'Orlando Magic': 74,
    'Atlanta Hawks': 73,
    'Toronto Raptors': 72,
    'San Antonio Spurs': 70,
    'New Orleans Pelicans': 69,
    'Utah Jazz': 68,
    'Brooklyn Nets': 67,
    'Memphis Grizzlies': 66,
    'Houston Rockets': 65,
    'Washington Wizards': 64,
    'Charlotte Hornets': 63,
    'Portland Trail Blazers': 62,
    'Detroit Pistons': 60,
}

def make_prediction(away_team, home_team):
    """Generate prediction for a game"""
    
    away_strength = TEAM_STRENGTH.get(away_team, 70)
    home_strength = TEAM_STRENGTH.get(home_team, 70)
    
    # Add home court advantage (+3 points)
    home_strength += 3
    
    # Calculate expected spread
    spread = home_strength - away_strength
    
    # Determine winner and probability
    if spread > 0:
        winner = home_team
        win_prob = min(0.5 + (spread / 40), 0.95)
    else:
        winner = away_team
        win_prob = min(0.5 + (abs(spread) / 40), 0.95)
    
    # Confidence level
    if abs(spread) > 8:
        confidence = "HIGH"
    elif abs(spread) > 4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        'winner': winner,
        'spread': spread,
        'win_probability': win_prob,
        'confidence': confidence
    }


def main():
    print("\n" + "="*80)
    print(" " * 20 + "🏀 NBA GAME PREDICTIONS 🏀")
    print(" " * 22 + "February 19, 2026")
    print("="*80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for i, (away, home, time) in enumerate(GAMES, 1):
        pred = make_prediction(away, home)
        
        print(f"\n{'─'*80}")
        print(f"Game {i}: {away} @ {home}")
        print(f"Time: {time}")
        print(f"{'─'*80}")
        print(f"  🏆 Predicted Winner:  {pred['winner']}")
        print(f"  📊 Spread:            {pred['spread']:+.1f} points " + 
              f"({'Home' if pred['spread'] > 0 else 'Away'} favored)")
        print(f"  🎯 Win Probability:   {pred['win_probability']:.1%}")
        print(f"  💪 Confidence Level:  {pred['confidence']}")
        
        # Add context
        if pred['spread'] > 8:
            print(f"  💡 Analysis:          Strong favorite, expect {pred['winner']} to control the game")
        elif pred['spread'] > 4:
            print(f"  💡 Analysis:          {pred['winner']} has the edge but game could be competitive")
        else:
            print(f"  💡 Analysis:          Very close matchup, could go either way")
    
    print(f"\n{'='*80}")
    print("\n📝 SUMMARY\n")
    print(f"Total Games: {len(GAMES)}")
    
    # Count prediction confidence
    high_conf = sum(1 for (a,h,t) in GAMES if abs(make_prediction(a,h)['spread']) > 8)
    medium_conf = sum(1 for (a,h,t) in GAMES if 4 < abs(make_prediction(a,h)['spread']) <= 8)
    low_conf = sum(1 for (a,h,t) in GAMES if abs(make_prediction(a,h)['spread']) <= 4)
    
    print(f"  • High Confidence Predictions: {high_conf}")
    print(f"  • Medium Confidence Predictions: {medium_conf}")
    print(f"  • Low Confidence Predictions: {low_conf}")
    
    print(f"\n{'='*80}")
    print("\n⚠️  DISCLAIMER")
    print("These predictions are based on simplified team strength ratings.")
    print("Actual outcomes may vary due to injuries, rest, momentum, and other factors.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
