import json

# Read the notebook
with open('machine_learning/basketball_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Schedule mappings for Feb 24
feb24_schedule = """Tue Feb 24 2026,7:00p,Philadelphia 76ers,,Indiana Pacers
Tue Feb 24 2026,7:30p,Washington Wizards,,Atlanta Hawks
Tue Feb 24 2026,7:30p,Dallas Mavericks,,Brooklyn Nets
Tue Feb 24 2026,7:30p,New York Knicks,,Cleveland Cavaliers
Tue Feb 24 2026,7:30p,Oklahoma City Thunder,,Toronto Raptors
Tue Feb 24 2026,8:00p,Charlotte Hornets,,Chicago Bulls
Tue Feb 24 2026,8:00p,Miami Heat,,Milwaukee Bucks
Tue Feb 24 2026,8:00p,Golden State Warriors,,New Orleans Pelicans
Tue Feb 24 2026,9:00p,Boston Celtics,,Phoenix Suns
Tue Feb 24 2026,10:00p,Minnesota Timberwolves,,Portland Trail Blazers
Tue Feb 24 2026,10:30p,Orlando Magic,,Los Angeles Lakers"""

# Iterate through cells to find and update schedule
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Join source in case it's a list of strings
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Check if this cell has the schedule
        if 'schedule_text = """' in source_str and 'Sun Feb 22 2026' in source_str:
            # Also update the header comment if present
            source_str = source_str.replace('Feb 22, 2026', 'Feb 24, 2026')
            source_str = source_str.replace('Sun Feb 22 2026,1:00p,Cleveland Cavaliers,,Oklahoma City Thunder\nSun Feb 22 2026,3:30p,Brooklyn Nets,,Atlanta Hawks\nSun Feb 22 2026,3:30p,Denver Nuggets,,Golden State Warriors\nSun Feb 22 2026,3:30p,Toronto Raptors,,Milwaukee Bucks\nSun Feb 22 2026,5:00p,Dallas Mavericks,,Indiana Pacers\nSun Feb 22 2026,6:00p,Charlotte Hornets,,Washington Wizards\nSun Feb 22 2026,6:30p,Boston Celtics,,Los Angeles Lakers\nSun Feb 22 2026,7:00p,Philadelphia 76ers,,Minnesota Timberwolves\nSun Feb 22 2026,8:00p,New York Knicks,,Chicago Bulls\nSun Feb 22 2026,8:00p,Portland Trail Blazers,,Phoenix Suns\nSun Feb 22 2026,9:00p,Orlando Magic,,Los Angeles Clippers', feb24_schedule)
            
            # Re-assign the source
            if isinstance(cell['source'], list):
                cell['source'] = source_str.split('\n')
                # Add newlines back except for the last line
                for i in range(len(cell['source']) - 1):
                    cell['source'][i] += '\n'
            else:
                cell['source'] = source_str
            
            print(f"✅ Updated schedule in cell")

# Write the notebook back
with open('machine_learning/basketball_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("✅ Schedule updated to Feb 24 in notebook")
