with open('data/database/database_handler.py', 'r') as f:
    lines = f.readlines()
    
# Find INSERT INTO predictions
for i, line in enumerate(lines):
    if 'INSERT INTO predictions' in line:
        print(f"Found INSERT at line {i+1}")
        # Print 20 lines from there
        for j in range(20):
            if i+j < len(lines):
                l = lines[i+j]
                qcount = l.count('?')
                if qcount > 0:
                    print(f"  Line {i+j+1} ({qcount} ?'s): {l.rstrip()[:100]}")
                elif 'VALUES' in l or j < 5:
                    print(f"  Line {i+j+1}: {l.rstrip()[:100]}")
        break
