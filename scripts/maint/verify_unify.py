#!/usr/bin/env python3
"""The shared reference-unification must reproduce the number the paper reports.

The manuscript quotes 0.56 standard deviations for the pooled Pushing margin
against the learned-skill arm, which came from a one-off re-scoring script. If
the loader is to replace that script it has to land on the same value, and it
must leave the multi-policy sweeps alone, where every arm in a group already
shares the reference the driver measured once.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

LEGACY = ["pushrw_skew*.json", "bmdP_skew*.json"]

raw = D.load(LEGACY, floor=0.10)
fixed = D.unify_reference(raw)
print(f"legacy Pushing: {len(raw)} rows, {D.unify_reference.rescored} re-scored")


def pooled_d(rows, a, b):
    x = np.array([r["tv_to_ref"] for r in rows if r["arm"] == a])
    y = np.array([r["tv_to_ref"] for r in rows if r["arm"] == b])
    s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    t, p2 = stats.ttest_ind(x, y, equal_var=False)
    return (y.mean() - x.mean()) / s, (p2 / 2 if t < 0 else 1 - p2 / 2), len(x), len(y)

print(f"\n{'comparison':<18}{'raw':>8}{'unified':>10}   paper")
paper = {"dsrl": 0.88, "bmd": 0.56, "mi": 0.85}
ok = True
for other in ("dsrl", "bmd", "mi"):
    d0, _, _, _ = pooled_d(raw, "mop", other)
    d1, p1, n1, n2 = pooled_d(fixed, "mop", other)
    hit = abs(round(d1, 2) - paper[other]) < 5e-3
    ok &= hit
    print(f"  mop vs {other:<10}{d0:+.2f}{d1:+10.2f}   {paper[other]:+.2f}  "
          f"{'match' if hit else 'MISMATCH'}  p={p1:.4f} n={n1}/{n2}")

pm = D.load("pm_skew*.json")
pm_fixed = D.unify_reference(pm)
touched = D.unify_reference.rescored
print(f"\nmulti-policy Pushing: {len(pm)} rows, {touched} re-scored "
      f"({'left alone, as expected' if touched == 0 else 'UNEXPECTED'})")
ok &= touched == 0

av = D.load("mb_skew*.json")
D.unify_reference(av)
print(f"Avoiding: {len(av)} rows, {D.unify_reference.rescored} re-scored")
ok &= D.unify_reference.rescored == 0

print("\nVERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
