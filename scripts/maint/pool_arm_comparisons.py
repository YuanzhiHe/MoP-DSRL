#!/usr/bin/env python3
"""Pool the multi-backbone sweep across levels, and test MoP against each baseline.

The per-level tests are underpowered for the comparison that matters. The
headline had been MoP against `mi`, an entropy objective handed the ground-truth
labels the published method has to discover; `bmd` implements the mechanism
itself. Pooling raises power, and the trend is estimated with backbone as the
unit, since backbone-to-backbone variation in C_ref is 71% of the variation
between skew levels.
"""
import glob
import json

import numpy as np
from scipy import stats

rows = []
for f in sorted(glob.glob("runs/_d3il/mb_skew*.json")):
    rows += json.load(open(f))
rows = [r for r in rows if r["SR"] >= 0.15]


def tvu(r):
    q = np.asarray(r["C_ref"], float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum())


def gn(r):
    d = tvu(r)
    return (r["tv_to_uniform"] - r["tv_to_ref"]) / d if d > 1e-9 else np.nan


print("=== pooled over all levels, for power ===")
for other in ("bmd", "mi", "dsrl"):
    a = np.array([gn(r) for r in rows if r["arm"] == "mop"])
    b = np.array([gn(r) for r in rows if r["arm"] == other])
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    t, p2 = stats.ttest_ind(a, b, equal_var=False)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"  mop vs {other:<5} diff={a.mean() - b.mean():+.3f}  "
          f"d={(a.mean() - b.mean()) / pooled:+.2f}  p(1-sided)={p1:.4f}  "
          f"n={len(a)}/{len(b)}")

print("\n=== trend in the mop-minus-baseline gap, backbone as the unit ===")
bbs = sorted({r.get("bb_seed") for r in rows} - {None})
skews = sorted({r["skew"] for r in rows})
for other in ("bmd", "mi"):
    xs, ys = [], []
    for s in skews:
        for bb in bbs:
            v = [r for r in rows if r["skew"] == s and r.get("bb_seed") == bb
                 and r["arm"] == "mop"]
            w = [r for r in rows if r["skew"] == s and r.get("bb_seed") == bb
                 and r["arm"] == other]
            if len(v) < 2 or len(w) < 2:
                continue
            xs.append(np.mean([tvu(r) for r in v]))
            ys.append(np.nanmean([gn(r) for r in v]) - np.nanmean([gn(r) for r in w]))
    sl, ic, rr, pp, se = stats.linregress(xs, ys)
    print(f"  mop - {other:<5}: n={len(xs)} backbones  slope={sl:+.3f}+-{se:.3f}  "
          f"r={rr:+.3f}  p={pp:.4f}")
