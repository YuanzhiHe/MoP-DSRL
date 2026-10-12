#!/usr/bin/env python3
"""Pushing on the two weightings whose sweep is complete, so the read is not of partial data."""
import sys, collections
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

allr = D.unify_reference(D.load("pm_skew*.json", floor=0.0))
print("skill prior used by the learned-skill arm:",
      collections.Counter(r.get("bmd_prior") for r in allr if r["arm"] == "bmd"))
old = D.unify_reference(D.load(["pushrw_skew*.json", "bmdP_skew*.json"], floor=0.0))
print("prior in the earlier Pushing runs:      ",
      collections.Counter(r.get("bmd_prior") for r in old if r["arm"] == "bmd"))

COMPLETE = {"0.00", "0.10"}
rows = [r for r in allr if r["skew"] in COMPLETE]
print(f"\n=== weightings {sorted(COMPLETE)}, 5 frozen policies x 5 seeds x 4 arms, complete ===")
print("  n per arm:", {a: sum(1 for r in rows if r["arm"] == a) for a in ("dsrl","mi","bmd","mop")})

for lab, floor in (("filtered at 0.10", 0.10), ("unfiltered", 0.0)):
    sel = [r for r in rows if r["SR"] >= floor]
    x = np.array([r["tv_to_ref"] for r in sel if r["arm"] == "mop"])
    out = []
    for o in ("dsrl", "mi", "bmd"):
        y = np.array([r["tv_to_ref"] for r in sel if r["arm"] == o])
        s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        t, p2 = stats.ttest_ind(x, y, equal_var=False)
        out.append(f"{o} d={(y.mean()-x.mean())/s:+.2f} p={(p2/2 if t<0 else 1-p2/2):.3f}")
    print(f"  {lab:<18} n_mop={x.size:3d}   " + "   ".join(out))

print("\n  clustered over the ten (weighting, frozen policy) units")
for o in ("dsrl", "mi", "bmd"):
    for lab, floor in (("filtered", 0.10), ("unfiltered", 0.0)):
        d = []
        for sk in sorted(COMPLETE):
            for bb in range(5):
                sel = [r for r in rows if r["skew"] == sk and r.get("bb_seed") == bb
                       and r["SR"] >= floor]
                m = [r["tv_to_ref"] for r in sel if r["arm"] == "mop"]
                y = [r["tv_to_ref"] for r in sel if r["arm"] == o]
                if len(m) > 1 and len(y) > 1:
                    d.append(np.mean(y) - np.mean(m))
        d = np.array(d)
        if d.size < 3:
            print(f"    vs {o:<5} {lab:<11} too few units"); continue
        sem = d.std(ddof=1)/np.sqrt(d.size); t, p2 = stats.ttest_1samp(d, 0)
        print(f"    vs {o:<5} {lab:<11} diff={d.mean():+.3f} "
              f"[{d.mean()-2.262*sem:+.3f},{d.mean()+2.262*sem:+.3f}] "
              f"p={(p2/2 if t>0 else 1-p2/2):.4f} wins={int((d>0).sum())}/{d.size}")

print("\n  task success, unfiltered")
for sk in sorted(COMPLETE):
    s = [r for r in rows if r["skew"] == sk]
    print(f"    skew {sk}: " + "  ".join(
        f"{a}={np.mean([r['SR'] for r in s if r['arm']==a]):.3f}" for a in ("dsrl","mi","bmd","mop")))
