#!/usr/bin/env python3
"""Check the manuscript's Pushing and weight figures against the run files.

Stale numbers reached the paper twice before, once because a script and the text
computed the same quantity two different ways. Every value asserted here is
recomputed from the runs and compared with what main.tex says.
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

TEX = (Path(__file__).resolve().parents[2] / "paper" / "main.tex").read_text()


def clustered(rows, other, floor):
    sel = [r for r in rows if r["SR"] >= floor]
    d = []
    for sk in sorted({r["skew"] for r in sel}):
        for bb in sorted({r.get("bb_seed") for r in sel} - {None}):
            g = [r for r in sel if r["skew"] == sk and r.get("bb_seed") == bb]
            m = [r["tv_to_ref"] for r in g if r["arm"] == "mop"]
            y = [r["tv_to_ref"] for r in g if r["arm"] == other]
            if len(m) > 1 and len(y) > 1:
                d.append(np.mean(y) - np.mean(m))
    d = np.array(d)
    sem = d.std(ddof=1) / np.sqrt(d.size)
    t, p2 = stats.ttest_1samp(d, 0)
    return d.mean(), 2.093 * sem, (p2 / 2 if t > 0 else 1 - p2 / 2), int((d > 0).sum()), d.size


pm = [r for r in D.unify_reference(D.load("pm_skew*.json", floor=0.0))
      if r["arm"] != "bmd"]
pu = D.unify_reference(D.load("pu_skew*.json", floor=0.0))
rows = pm + pu

print(f"{'claim':<34}{'computed':>12}{'in main.tex':>14}  match")
ok = True
for other, claim in (("mi", "0.095"), ("dsrl", "0.072"), ("bmd", "0.032")):
    m, h, p1, w, n = clustered(rows, other, 0.10)
    found = f"{m:.3f}" == claim
    ok &= found
    print(f"  Pushing vs {other:<20}{m:>12.3f}{claim:>14}  "
          f"{'yes' if found else 'NO'}   {w}/{n}  p={p1:.3f}")
    assert claim in TEX or not found, f"{claim} absent from main.tex"

# exclusion rates
for arm, claim in (("dsrl", "20.2"), ("mop", "32.2"), ("bmd", "8.9"), ("mi", "26.4")):
    tot = [r for r in rows if r["arm"] == arm]
    pct = 100 * (1 - sum(1 for r in tot if r["SR"] >= 0.10) / len(tot))
    found = f"{pct:.1f}" == claim and f"{claim}\\%" in TEX
    ok &= found
    print(f"  excluded, {arm:<23}{pct:>11.1f}%{claim+'%':>14}  "
          f"{'yes' if found else 'NO'}")

print("\nVERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
