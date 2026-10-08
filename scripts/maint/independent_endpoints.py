#!/usr/bin/env python3
"""Score every arm on endpoints that no arm optimises.

Reference matching descends total variation to the reference, so reporting only
that distance measures the method on its own objective. The stored histograms
allow other distances to be computed after the fact. None of the four arms
trains on any of these:

  chi-square to the reference
  largest per-mode absolute error
  Hellinger distance

Entropy maximisation trains on H(C_phi) and the learned-skill arm on a
variational bound, so neither descends these either.
"""
import glob, json
import numpy as np
from scipy import stats

def load(pats, floor):
    rows = []
    for p in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{p}")): rows += json.load(open(f))
    return [r for r in rows if r["SR"] >= floor]

def chi2(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = q > 1e-9
    return float(np.sum((p[m] - q[m]) ** 2 / q[m]))

def maxerr(p, q):
    return float(np.max(np.abs(np.asarray(p, float) - np.asarray(q, float))))

def hellinger(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

METRICS = [("chi-square", chi2), ("max per-mode error", maxerr), ("Hellinger", hellinger)]
ARMS = ("dsrl", "mi", "bmd", "mop")

for task, pats, floor in (("Avoiding", ["mb_skew*.json"], 0.15),
                          ("Pushing", ["pushrw_skew*.json", "bmdP_skew*.json"], 0.10)):
    rows = load(pats, floor)
    rows = [r for r in rows if r.get("C_phi") and r.get("C_ref")]
    print(f"########## {task}  (n={len(rows)}) ##########")
    for name, fn in METRICS:
        vals = {a: np.array([fn(r["C_phi"], r["C_ref"]) for r in rows if r["arm"] == a])
                for a in ARMS}
        line = "  ".join(f"{a}={vals[a].mean():.4f}" for a in ARMS if len(vals[a]))
        print(f"  {name:<20} {line}")
        x = vals["mop"]
        out = []
        for o in ("mi", "dsrl", "bmd"):
            y = vals[o]
            if len(y) < 2: continue
            t, p2 = stats.ttest_ind(x, y, equal_var=False)
            p1 = p2 / 2 if t < 0 else 1 - p2 / 2
            pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            out.append(f"vs {o}: d={(y.mean()-x.mean())/pooled:+.2f} p={p1:.4f}")
        print(f"  {'':<20} " + "   ".join(out))
    print()
