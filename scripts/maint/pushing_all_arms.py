#!/usr/bin/env python3
"""Pushing with every arm, including the learned-skill baseline."""
import glob
import json

import numpy as np
from scipy import stats

rows = []
for f in sorted(glob.glob("runs/_d3il/pushrw_skew*.json")) + \
         sorted(glob.glob("runs/_d3il/bmdP_skew*.json")):
    rows += json.load(open(f))
rows = [r for r in rows if r["SR"] >= 0.10]

REF = {"0.00": 0.327, "0.10": 0.120, "0.25": 0.537, "0.50": 0.770}


def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


def gn(r):
    d = tvu(r)
    return (r["tv_to_uniform"] - r["tv_to_ref"]) / d if d and np.isfinite(d) and d > 1e-9 else np.nan


skews = sorted({r["skew"] for r in rows}, key=lambda s: REF.get(s, 0))
print(f"{'KL(Cref,u)':>10} {'TV(Cref,u)':>11}  " + "  ".join(f"{a:>17}" for a in ("dsrl", "mop", "mi", "bmd")))
for s in skews:
    sr = [r for r in rows if r["skew"] == s]
    tv = np.nanmean([tvu(r) for r in sr])
    cells = []
    for a in ("dsrl", "mop", "mi", "bmd"):
        v = np.array([gn(r) for r in sr if r["arm"] == a])
        v = v[np.isfinite(v)]
        cells.append(f"{v.mean():+.3f}+-{v.std(ddof=1)/np.sqrt(len(v)):.3f}".rjust(17)
                     if len(v) > 1 else f"{'--':>17}")
    print(f"{REF.get(s, float('nan')):>10.3f} {tv:>11.3f}  " + "  ".join(cells))

print("\npooled over levels:")
for other in ("bmd", "mi", "dsrl"):
    a = np.array([gn(r) for r in rows if r["arm"] == "mop"])
    b = np.array([gn(r) for r in rows if r["arm"] == other])
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        continue
    t, p2 = stats.ttest_ind(a, b, equal_var=False)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"  mop vs {other:<5} diff={a.mean()-b.mean():+.3f}  d={(a.mean()-b.mean())/pooled:+.2f}  "
          f"p={p1:.4f}  n={len(a)}/{len(b)}")

print("\nsuccess rate:")
for s in skews:
    sr = [r for r in rows if r["skew"] == s]
    out = []
    for a in ("dsrl", "mop", "mi", "bmd"):
        v = np.array([r["SR"] for r in sr if r["arm"] == a])
        out.append(f"{a}={v.mean():.3f}" if len(v) else f"{a}=--")
    print("  " + "  ".join(out))
