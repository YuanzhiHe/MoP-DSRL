#!/usr/bin/env python3
"""One line per sweep: complete (weighting, policy) groups out of target.

Raw row counts mislead, because a level split across jobs writes overlapping
files; these are deduplicated groups. The penalty-weight sweep is keyed by
weight as well, that key having once been the difference between ten groups and
twenty.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

SWEEPS = (("multipolicy", "pm_skew*.json", ("skew", "bb_seed"), 20, 20),
          ("bmd-uniform", "pu_skew*.json", ("skew", "bb_seed"), 5, 20),
          ("lambda", "pl_lam*.json", ("lam", "skew", "bb_seed"), 10, 20))

out = []
for tag, pat, fields, need, total in SWEEPS:
    rows = D.load(pat, floor=0.0)
    c = collections.Counter(tuple(r.get(f) for f in fields) for r in rows)
    out.append(f"{tag}={sum(1 for v in c.values() if v >= need)}/{total}")
print("  ".join(out))
