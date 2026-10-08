#!/usr/bin/env python3
"""Compare the two bounded divergences on the full sweep, not one skew level.

The earlier comparison ran at a single Avoiding level with one backbone. This
uses the complete design: 20 backbones for Avoiding, both tasks, with the
baselines reused since they do not depend on the divergence.
"""
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

def stat(rows):
    v = np.array([gn(r) for r in rows]); v = v[np.isfinite(v)]
    return v, np.array([r["SR"] for r in rows]), np.array([r["n_modes"] for r in rows])

for task, tvpat, jspat, floor in (
        ("Avoiding", ["mb_skew*.json"], ["js_skew*.json"], 0.15),
        ("Pushing", ["pushrw_skew*.json"], ["jsP_skew*.json"], 0.10)):
    tv_rows = [r for r in load(tvpat, floor) if r["arm"] == "mop"]
    js_rows = [r for r in load(jspat, floor) if r["arm"] == "mop"]
    print(f"=== {task} ===")
    a, sra, ma = stat(tv_rows)
    b, srb, mb = stat(js_rows)
    t, p2 = stats.ttest_ind(b, a, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"  fidelity F   TV {a.mean():+.3f}+-{a.std(ddof=1)/np.sqrt(len(a)):.3f}   "
          f"JS {b.mean():+.3f}+-{b.std(ddof=1)/np.sqrt(len(b)):.3f}   "
          f"d(JS-TV)={(b.mean()-a.mean())/pooled:+.2f}  p={p2:.4f}")
    print(f"  success      TV {sra.mean():.3f}   JS {srb.mean():.3f}")
    print(f"  modes        TV {ma.mean():.2f}    JS {mb.mean():.2f}")
    print(f"  n            TV {len(a)}  JS {len(b)}")

    # per level
    allr = tv_rows + js_rows
    skews = sorted({r["skew"] for r in allr},
                   key=lambda s: np.nanmean([tvu(r) for r in allr if r["skew"] == s]))
    print(f"  {'TV(Cref,u)':>10}  {'TV anchor':>16}  {'JS anchor':>16}")
    for s_ in skews:
        x = np.array([gn(r) for r in tv_rows if r["skew"] == s_]); x = x[np.isfinite(x)]
        y = np.array([gn(r) for r in js_rows if r["skew"] == s_]); y = y[np.isfinite(y)]
        if not len(x) or not len(y): continue
        d = np.nanmean([tvu(r) for r in allr if r["skew"] == s_])
        print(f"  {d:>10.3f}  {x.mean():+8.3f}+-{x.std(ddof=1)/np.sqrt(len(x)):.3f}  "
              f"{y.mean():+8.3f}+-{y.std(ddof=1)/np.sqrt(len(y)):.3f}")
    print()
