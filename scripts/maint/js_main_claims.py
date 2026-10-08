#!/usr/bin/env python3
"""Do the main conclusions hold when the matching penalty uses JS instead of TV?"""
import glob, json
import numpy as np
from scipy import stats

def load(pats, floor):
    rows = []
    for pat in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{pat}")):
            rows += json.load(open(f))
    return [r for r in rows if r["SR"] >= floor]

def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan

def gn(r):
    d = tvu(r)
    return (r["tv_to_uniform"] - r["tv_to_ref"]) / d if d and np.isfinite(d) and d > 1e-9 else np.nan

def cmp(x, y):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    t, p2 = stats.ttest_ind(x, y, equal_var=False)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return (x.mean() - y.mean()) / pooled, p1, len(x), len(y)

for task, basepat, jspat, floor in (
        ("Avoiding", ["mb_skew*.json"], ["js_skew*.json"], 0.15),
        ("Pushing", ["pushrw_skew*.json", "bmdP_skew*.json"], ["jsP_skew*.json"], 0.10)):
    base = load(basepat, floor)
    js = [r for r in load(jspat, floor) if r["arm"] == "mop"]
    print(f"=== {task} ===")
    for div, mop in (("TV", [r for r in base if r["arm"] == "mop"]), ("JS", js)):
        x = np.array([gn(r) for r in mop])
        out = []
        for other in ("mi", "dsrl", "bmd"):
            y = np.array([gn(r) for r in base if r["arm"] == other])
            if len(y) < 2: continue
            d, p, na, nb = cmp(x, y)
            out.append(f"vs {other}: d={d:+.2f} p={p:.4f}")
        print(f"  {div} matching   " + "   ".join(out))
    print()
