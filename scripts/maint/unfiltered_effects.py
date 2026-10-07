#!/usr/bin/env python3
"""Effect sizes with and without the success-rate exclusion, both tasks.

The exclusion conditions on an outcome the treatment affects, so the manuscript
reports both analyses and claims only what survives each. These are the numbers
that go into that section.
"""
import glob, json
import numpy as np
from scipy import stats

def load(pats, floor=None):
    rows = []
    for pat in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{pat}")):
            rows += json.load(open(f))
    return rows if floor is None else [r for r in rows if r["SR"] >= floor]

def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan

def gn(r):
    d = tvu(r)
    return (r["tv_to_uniform"] - r["tv_to_ref"]) / d if d and np.isfinite(d) and d > 1e-9 else np.nan

def cmp(rows, a, b):
    x = np.array([gn(r) for r in rows if r["arm"] == a])
    y = np.array([gn(r) for r in rows if r["arm"] == b])
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    t, p2 = stats.ttest_ind(x, y, equal_var=False)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return (x.mean() - y.mean()) / pooled, p1, len(x), len(y)

TASKS = [("Avoiding", ["mb_skew*.json"], 0.15),
         ("Pushing", ["pushrw_skew*.json", "bmdP_skew*.json"], 0.10)]

for name, pats, floor in TASKS:
    print(f"=== {name} ===")
    for lab, fl in ((f"filtered SR>={floor}", floor), ("unfiltered", None)):
        rows = load(pats, fl)
        out = []
        for other in ("mi", "dsrl", "bmd"):
            d, pv, na, nb = cmp(rows, "mop", other)
            out.append(f"vs {other}: d={d:+.2f} p={pv:.4f} n={na}/{nb}")
        print(f"  {lab:<18} " + "   ".join(out))
    rows = load(pats, None)
    print("  attrition:", end=" ")
    for a in ("dsrl", "mop", "mi", "bmd"):
        v = [r for r in rows if r["arm"] == a]
        if not v: continue
        drop = sum(1 for r in v if r["SR"] < floor)
        print(f"{a}={drop/len(v):.1%}", end="  ")
    print("\n")

print("=== behavioural diversity, all arms, Avoiding by skew ===")
rows = load(["mb_skew*.json"], 0.15)
skews = sorted({r["skew"] for r in rows}, key=lambda s: np.nanmean([tvu(r) for r in rows if r["skew"] == s]))
print(f"  {'TV(Cref,u)':>10}  " + "  ".join(f"{a:>22}" for a in ("dsrl", "mop", "mi", "bmd")))
print(f"  {'':>10}  " + "  ".join(f"{'modes / H(Cphi)':>22}" for _ in range(4)))
for s_ in skews:
    sr = [r for r in rows if r["skew"] == s_]
    cells = []
    for a in ("dsrl", "mop", "mi", "bmd"):
        v = [r for r in sr if r["arm"] == a]
        cells.append(f"{np.mean([r['n_modes'] for r in v]):.2f} / {np.mean([r['H_modes'] for r in v]):.3f}".rjust(22))
    print(f"  {np.nanmean([tvu(r) for r in sr]):>10.3f}  " + "  ".join(cells))
