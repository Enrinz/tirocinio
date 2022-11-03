import random

with open('input_1_balanced.csv', 'r') as r, open('input_1_balanced_shuffled.csv', 'w') as w:
    data = r.readlines()
    header, rows = data[0], data[1:]
    random.shuffle(rows)
    rows = '\n'.join([row.strip() for row in rows])
    w.write(header + rows)