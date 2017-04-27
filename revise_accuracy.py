#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

with open(args.filename) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    heading = lines[i].strip()
    table = lines[i+8:i+11]
    out = tuple(map(int, table[0].strip().split()[1:]))
    beg = tuple(map(int, table[1].strip().split()[1:]))
    ins = tuple(map(int, table[2].strip().split()[1:]))
    acc = (beg[1] + ins[2]) / (sum(beg) + sum(ins))
    print(heading)
    print()
    print(f"Accuracy: {acc}")
    print()
    print('-' * 80)
    i += 15
