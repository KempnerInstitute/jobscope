#!/usr/bin/env python3
"""Keep a representative sample of a summary table's job rows, on stdin.

A partition selection is tens or hundreds of rows, which makes a screenshot too tall
to read. The rows that matter for a screenshot are a *spread* -- red beside green --
so this keeps the ones nearest an even sweep of the metric in --column and elides the
rest, then passes the summary block through untouched.

Usage:  ... | trim_rows.py [--column N] [--keep N]
"""
import re
import sys

column = int(sys.argv[sys.argv.index("--column") + 1]) if "--column" in sys.argv else 8
keep = int(sys.argv[sys.argv.index("--keep") + 1]) if "--keep" in sys.argv else 6

lines = sys.stdin.read().split("\n")
plain = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in lines]
rows = [i for i, l in enumerate(plain) if re.match(r"^\d+(_\d+)?\s", l)]
block = next((i for i, l in enumerate(plain) if re.match(r"^1\. ", l)), len(plain))

def value(i):
    fields = plain[i].split()
    return float(fields[column - 1]) if len(fields) >= column and \
        re.match(r"^-?[\d.]+$", fields[column - 1]) else -1.0

if len(rows) <= keep:
    sys.stdout.write("\n".join(lines))
    sys.exit()

have = sorted(v for v in (value(i) for i in rows) if v >= 0)
targets = [have[round(k * (len(have) - 1) / (keep - 1))] for k in range(keep)]
picked: list = []
for want in targets:
    candidates = [i for i in rows if i not in picked and value(i) >= 0]
    if candidates:
        picked.append(min(candidates, key=lambda i: abs(value(i) - want)))

out = lines[:rows[0]] + [lines[i] for i in sorted(picked)] \
    + ["... %d more rows ..." % (len(rows) - len(picked))] + lines[block - 1:]
sys.stdout.write("\n".join(out))
