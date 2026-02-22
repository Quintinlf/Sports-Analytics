import sqlite3
import pandas as pd

conn = sqlite3.connect('predictions.db')

query = """
SELECT 
    game_num,
    away_team,
    home_team,
    ensemble_winner,
    ROUND(ensemble_probability * 100, 1) as ens_prob_pct,
    mc_winner,
    ROUND(mc_win_prob * 100, 1) as mc_prob_pct,
    CASE WHEN methods_agree = 1 THEN '✅' ELSE '❌' END as agree
FROM predictions 
WHERE game_date = '2026-02-19'
ORDER BY game_num
"""

df = pd.read_sql(query, conn)

print("\n" + "="*100)
print("⚖️  ENSEMBLE vs MONTE CARLO PREDICTIONS - February 19, 2026")
print("="*100 + "\n")

for _, row in df.iterrows():
    print(f"Game {row['game_num']}: {row['away_team']} @ {row['home_team']}")
    print(f"  Ensemble: {row['ensemble_winner']:20} ({row['ens_prob_pct']:.1f}%)")
    print(f"  MC Sims:  {row['mc_winner']:20} ({row['mc_prob_pct']:.1f}%)")
    print(f"  {row['agree']} {'AGREE' if row['agree'] == '✅' else 'DISAGREE'}\n")

agree_count = (df['agree'] == '✅').sum()
total = len(df)

print("="*100)
print(f"Summary: {agree_count}/{total} games agree ({agree_count/total*100:.1f}%)")
print("="*100 + "\n")

conn.close()
