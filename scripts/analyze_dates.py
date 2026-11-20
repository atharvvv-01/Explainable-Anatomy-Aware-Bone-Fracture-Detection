import os, re
from collections import defaultdict

date_counts = defaultdict(lambda: {'Fracture': 0, 'Non fracture': 0})
for split in ['Fracture', 'Non fracture']:
    for f in os.listdir(os.path.join('Dataset/Dataset', split)):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
        d = m.group(1) if m else 'no_date'
        if d == '2203-05-19': d = '2026-05-19'
        date_counts[d][split] += 1

print('Date breakdown:')
for d, counts in sorted(date_counts.items()):
    frac = counts['Fracture']
    non = counts['Non fracture']
    total = frac + non
    print(f"{d}: Fracture={frac:4d}, Non-fracture={non:4d}, Total={total:4d}")
