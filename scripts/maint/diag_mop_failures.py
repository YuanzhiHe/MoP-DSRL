#!/usr/bin/env python3
"""What do the Pushing runs that fall below the success floor actually look like?

The floor excludes 34 percent of reference-matching runs on Pushing against 6
percent of the learned-skill ones. Either the penalty collapses those runs onto
one mode, or it spreads them so wide that the rewarded mode is rarely reached.
The two call for different readings, so this separates them.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

FLOOR = 0.10
rows = D.unify_reference(D.load("pm_skew*.json", floor=0.0))
rows = [r for r in rows if r["skew"] in ("0.00", "0.10")]      # complete weightings

print(f"{'arm':<6}{'group':<10}{'n':>4}{'mean SR':>9}{'tv_ref':>9}"
      f"{'modes>0':>9}{'collapsed':>11}{'entropy':>9}")
for arm in ("dsrl", "mi", "bmd", "mop"):
    for lab, keep in (("above", True), ("below", False)):
        sel = [r for r in rows if r["arm"] == arm and ((r["SR"] >= FLOOR) == keep)]
        if not sel:
            print(f"{arm:<6}{lab:<10}{0:>4}"); continue
        nm = [sum(1 for c in r["C_phi"] if c > 0.01) for r in sel]
        ent = []
        for r in sel:
            p = np.asarray(r["C_phi"], float); p = p[p > 0]
            ent.append(float(-(p * np.log(p)).sum()) if p.size else 0.0)
        print(f"{arm:<6}{lab:<10}{len(sel):>4}"
              f"{np.mean([r['SR'] for r in sel]):>9.3f}"
              f"{np.nanmean([r['tv_to_ref'] for r in sel]):>9.3f}"
              f"{np.mean(nm):>9.2f}"
              f"{np.mean([bool(r.get('collapsed')) for r in sel]):>11.2f}"
              f"{np.mean(ent):>9.3f}")

print("\nreference entropy for context:",
      f"{np.mean([-(np.asarray(r['C_ref'],float)[np.asarray(r['C_ref'],float)>0] * np.log(np.asarray(r['C_ref'],float)[np.asarray(r['C_ref'],float)>0])).sum() for r in rows]):.3f} nats",
      f"(uniform over 4 groups = {np.log(4):.3f})")

# Does reward restricted to one mode fight the penalty?
print("\nrewarded-mode share of the policy histogram, above vs below the floor")
for arm in ("bmd", "mop"):
    for lab, keep in (("above", True), ("below", False)):
        sel = [r for r in rows if r["arm"] == arm and ((r["SR"] >= FLOOR) == keep)]
        if not sel: continue
        share = [max(r["C_phi"]) for r in sel]
        print(f"  {arm:<5}{lab:<7} largest single-mode share {np.mean(share):.3f}  n={len(sel)}")
