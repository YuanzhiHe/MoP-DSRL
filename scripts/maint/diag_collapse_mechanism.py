#!/usr/bin/env python3
"""Does collapse track how informative the reward is for that frozen policy?

Advantage and mode coefficient are each standardised to unit deviation and then
combined as adv + lam * coef, so lam sets their relative weight. Standardisation
rescales a sparse reward to unit deviation as readily as a dense one, so where
success is rare the advantage direction is noisier at the same nominal weight
and the mode term carries more of the update. If that is what drives the
collapses, then within a task the frozen policies whose task is hardest, as
measured by the arm with no mode term at all, should be the ones where the
regularised arms collapse most.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

for task, pat, floor in (("Avoiding", "mb_skew*.json", 0.15),
                         ("Pushing", "pm_skew*.json", 0.10)):
    rows = D.unify_reference(D.load(pat, floor=0.0))
    units, x, y = [], [], []
    for sk in sorted({r["skew"] for r in rows}):
        for bb in sorted({r.get("bb_seed") for r in rows} - {None}):
            sel = [r for r in rows if r["skew"] == sk and r.get("bb_seed") == bb]
            base = [r["SR"] for r in sel if r["arm"] == "dsrl"]
            mop = [r for r in sel if r["arm"] == "mop"]
            if len(base) < 3 or len(mop) < 3:
                continue
            units.append((sk, bb))
            x.append(float(np.mean(base)))
            y.append(float(np.mean([r["SR"] < floor for r in mop])))
    x, y = np.array(x), np.array(y)
    print(f"\n=== {task}: {len(x)} (weighting, frozen policy) units ===")
    if x.size < 4:
        print("  too few units"); continue
    print(f"  baseline success   {x.min():.3f} to {x.max():.3f}, mean {x.mean():.3f}")
    print(f"  our collapse rate  {y.min():.3f} to {y.max():.3f}, mean {y.mean():.3f}")
    if y.std() < 1e-9:
        print("  collapse rate is constant, nothing to correlate")
        continue
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    print(f"  Pearson  r={r:+.3f}  p={p:.4f}")
    print(f"  Spearman r={rs:+.3f}  p={ps:.4f}")
    print(f"  prediction is a negative correlation: "
          f"{'held' if r < 0 and p < 0.05 else 'not supported at p<0.05'}")
    lo = y[x <= np.median(x)].mean(); hi = y[x > np.median(x)].mean()
    print(f"  collapse rate, harder half {lo:.3f} vs easier half {hi:.3f}")
