#!/usr/bin/env python3
"""Does the success floor select unevenly between arms, and does it decide the result?"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

for tag, pat, floor in (("Avoiding", "mb_skew*.json", 0.15),
                        ("Pushing (multi-policy)", "pm_skew*.json", 0.10)):
    allr = D.unify_reference(D.load(pat, floor=0.0))
    print(f"\n=== {tag}, floor {floor} ===")
    for a in ("dsrl", "mi", "bmd", "mop"):
        t = [r for r in allr if r["arm"] == a]
        k = [r for r in t if r["SR"] >= floor]
        print(f"  {a:<5} kept {len(k):3d}/{len(t):3d} "
              f"({100*(1-len(k)/max(len(t),1)):4.1f}% excluded)  mean SR {np.mean([r['SR'] for r in t]):.3f}")
    for lab, rows in (("filtered", [r for r in allr if r["SR"] >= floor]),
                      ("UNfiltered", allr)):
        x = np.array([r["tv_to_ref"] for r in rows if r["arm"] == "mop"])
        line = []
        for o in ("dsrl", "mi", "bmd"):
            y = np.array([r["tv_to_ref"] for r in rows if r["arm"] == o])
            if x.size < 3 or y.size < 3:
                line.append(f"{o}=--"); continue
            s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            t_, p2 = stats.ttest_ind(x, y, equal_var=False)
            p1 = p2/2 if t_ < 0 else 1-p2/2
            line.append(f"{o} d={(y.mean()-x.mean())/s:+.2f} p={p1:.3f}")
        print(f"  {lab:<11} " + "   ".join(line))

# Is the mode term costing task success on Pushing?
pm = D.unify_reference(D.load("pm_skew*.json", floor=0.0))
print("\n=== task success by arm and weighting (Pushing, unfiltered) ===")
for sk in sorted({r["skew"] for r in pm}):
    s = [r for r in pm if r["skew"] == sk]
    print(f"  skew {sk}: " + "  ".join(
        f"{a}={np.mean([r['SR'] for r in s if r['arm']==a]):.3f}" for a in ("dsrl","mi","bmd","mop")))
