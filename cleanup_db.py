"""Clean up duplicate predictions in the database, keeping only the latest run."""
import sqlite3

conn = sqlite3.connect('predictions.db')

# Check current state
before = conn.execute("SELECT COUNT(*) FROM predictions WHERE game_date = '2026-02-19'").fetchone()[0]
print(f"Before cleanup: {before} rows for Feb 19, 2026")

# Delete all but the most recent predictions for each game on Feb 19
conn.execute("""
DELETE FROM predictions 
WHERE game_date = '2026-02-19' 
AND id NOT IN (
    SELECT MAX(id) 
    FROM predictions 
    WHERE game_date = '2026-02-19'
    GROUP BY game_num
)
""")

conn.commit()

# Check after cleanup
after = conn.execute("SELECT COUNT(*) FROM predictions WHERE game_date = '2026-02-19'").fetchone()[0]
print(f"After cleanup: {after} rows for Feb 19, 2026")
print(f"Removed {before - after} duplicate rows\n")

conn.close()
print("✅ Database cleaned")
