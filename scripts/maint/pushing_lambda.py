#!/usr/bin/env python3
"""Apply the pre-registered penalty-weight rule to the Pushing sweep.

The rule is fixed in docs/PREREG_pushing_lambda.md and is executed here without
reference to the comparison: the largest weight whose collapse rate on the
reference matching arm stays at or below 0.15, with the lowest collapse rate as
the fallback. The weight it selects is applied unchanged to mode entropy.

Distance to the reference is printed after the selection, never used in it.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

FLOOR = 0.10              # the pre-registered Pushing success floor
MAX_COLLAPSE = 0.15       # the pre-registered ceiling on the collapse rate
LEVELS = ("0.00", "0.10")
WEIGHTS = (0.25, 0.50, 1.0)


def rows_at(lam):
    """Runs at one weight. 1.0 comes from the main sweep, the rest from this one."""
    pats = ["pm_skew*.json"] if lam == 1.0 else [f"pl_lam{lam:.2f}_skew*.json"]
    rs = D.unify_reference(D.load(pats, floor=0.0))
    return [r for r in rs if r["skew"] in LEVELS and abs(r.get("lam", 0.0) - lam) < 1e-9]


def main() -> int:
    print("=== step 1: collapse rate, the only quantity the rule reads ===")
    print(f"{'weight':>8}  " + "  ".join(f"{a:>22}" for a in ("reference matching", "mode entropy")))
    collapse = {}
    for lam in WEIGHTS:
        rs = rows_at(lam)
        cells = []
        for arm in ("mop", "mi"):
            a = [r for r in rs if r["arm"] == arm]
            if not a:
                cells.append("        no data      ")
                continue
            bad = sum(1 for r in a if r["SR"] < FLOOR)
            if arm == "mop":
                collapse[lam] = (bad / len(a), len(a))
            cells.append(f"{bad}/{len(a)} = {bad / len(a):.2f}".rjust(22))
        print(f"{lam:>8.2f}  " + "  ".join(cells))

    have = {k: v for k, v in collapse.items() if v[1] >= 40}
    if not have:
        print("\nthe sweep is not complete enough to apply the rule "
              f"(runs per weight: {[(k, v[1]) for k, v in collapse.items()]})")
        return 1

    passing = [k for k, v in sorted(have.items()) if v[0] <= MAX_COLLAPSE]
    chosen = max(passing) if passing else min(have, key=lambda k: have[k][0])
    why = (f"largest weight at or below a collapse rate of {MAX_COLLAPSE}"
           if passing else "no weight met the ceiling, so the lowest collapse rate")
    print(f"\n=== step 2: the rule selects weight {chosen:.2f} ({why}) ===")

    print("\n=== step 3: the comparison at the selected weight, reported either way ===")
    sel = [r for r in rows_at(chosen) if r["SR"] >= FLOOR]
    # The learned-skill baseline is taken from the uniform-prior sweep, which is
    # the protocol the Avoiding main table uses; the fitted prior belongs in the
    # robustness section on both tasks, not in the headline comparison.
    bmd_pat = "pu_skew*.json" if list(D.load("pu_skew*.json", floor=0.0)) else "pm_skew*.json"
    if bmd_pat == "pm_skew*.json":
        print("  NOTE: uniform-prior runs absent, falling back to the fitted-prior "
              "learned-skill rows, which is not the Avoiding protocol")
    base = [r for r in D.unify_reference(D.load("pm_skew*.json", floor=0.0))
            if r["skew"] in LEVELS and r["arm"] == "dsrl" and r["SR"] >= FLOOR]
    base += [r for r in D.unify_reference(D.load(bmd_pat, floor=0.0))
             if r["skew"] in LEVELS and r["arm"] == "bmd" and r["SR"] >= FLOOR]
    pool = sel + base
    x = np.array([r["tv_to_ref"] for r in pool if r["arm"] == "mop"])
    print(f"  reference matching: n={x.size}, mean {x.mean():.3f}, "
          f"task success {np.mean([r['SR'] for r in pool if r['arm'] == 'mop']):.3f}")
    for o, lab in (("dsrl", "no mode term"), ("mi", "mode entropy"), ("bmd", "learned skills")):
        y = np.array([r["tv_to_ref"] for r in pool if r["arm"] == o])
        if y.size < 3:
            print(f"  vs {lab:<16} no data"); continue
        s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        t, p2 = stats.ttest_ind(x, y, equal_var=False)
        print(f"  vs {lab:<16} their mean {y.mean():.3f}  d={(y.mean() - x.mean()) / s:+.2f}  "
              f"p={(p2 / 2 if t < 0 else 1 - p2 / 2):.4f}  n={y.size}")

    print("\n  clustered over (weighting, frozen policy) units")
    for o, lab in (("dsrl", "no mode term"), ("mi", "mode entropy"), ("bmd", "learned skills")):
        d = []
        for sk in LEVELS:
            for bb in range(5):
                m = [r["tv_to_ref"] for r in pool
                     if r["skew"] == sk and r.get("bb_seed") == bb and r["arm"] == "mop"]
                y = [r["tv_to_ref"] for r in pool
                     if r["skew"] == sk and r.get("bb_seed") == bb and r["arm"] == o]
                if len(m) > 1 and len(y) > 1:
                    d.append(np.mean(y) - np.mean(m))
        d = np.array(d)
        if d.size < 3:
            print(f"    vs {lab:<16} too few units ({d.size})"); continue
        sem = d.std(ddof=1) / np.sqrt(d.size)
        t, p2 = stats.ttest_1samp(d, 0)
        print(f"    vs {lab:<16} diff={d.mean():+.3f} "
              f"[{d.mean() - 2.262 * sem:+.3f},{d.mean() + 2.262 * sem:+.3f}] "
              f"p={(p2 / 2 if t > 0 else 1 - p2 / 2):.4f} wins={int((d > 0).sum())}/{d.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
