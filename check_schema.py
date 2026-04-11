import sqlite3

conn = sqlite3.connect('sports_analytics.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(predictions)")
columns = cursor.fetchall()
print(f'Total columns: {len(columns)}')
for i, col in enumerate(columns, 1):
    print(f'{i:2d}. {col[1]:35s} ({col[2]})')
conn.close()
