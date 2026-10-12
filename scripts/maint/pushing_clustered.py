#!/usr/bin/env python3
"""Backbone-clustered paired analysis for Pushing, matching the Avoiding one.

Until the multi-policy sweep, Pushing was reported by pooling runs, which treats
runs sharing a frozen policy as independent when they are not. The sweep trains
five frozen policies per weighting so the same clustered analysis the manuscript
runs on Avoiding can be run here: each frozen policy contributes one paired
difference, and the test is over those twenty units rather than over runs.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

FLOOR = 0.10          # the pre-registered Pushing threshold; Avoiding uses 0.15
ARMS = {"dsrl": "no mode term", "mi": "mode entropy", "bmd": "learned skills"}
T95 = 2.093          # t(19), two-sided 95%


def tv_uniform(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


def main() -> int:
    # The learned-skill arm in the multi-policy sweep was run with the fitted
    # skill prior. On Avoiding the main table uses the uniform prior and reports
    # the fitted one as a robustness check, so the same split applies here: the
    # headline comparison takes that arm from the uniform-prior sweep, and the
    # fitted rows are reported separately at the end.
    rows = [r for r in D.unify_reference(D.load("pm_skew*.json", floor=FLOOR))
            if r["arm"] != "bmd"]
    fitted = [r for r in D.unify_reference(D.load("pm_skew*.json", floor=FLOOR))
              if r["arm"] == "bmd"]
    uniform = D.unify_reference(D.load("pu_skew*.json", floor=FLOOR))
    if uniform:
        rows += uniform
    else:
        print("NOTE: no uniform-prior runs yet; the learned-skill arm below uses "
              "the fitted prior, which is not the protocol Avoiding reports")
        rows += fitted
    print(f"Pushing multi-policy: {len(rows)} runs above the success floor, "
          f"learned-skill prior = {'uniform' if uniform else 'fitted'}")

    skews = sorted({r["skew"] for r in rows})
    bbs = sorted({r.get("bb_seed") for r in rows} - {None})
    print(f"weightings {skews}  frozen policies {bbs}\n")

    # measured skew per weighting, which is how the manuscript orders levels
    print(f"{'weighting':>10} {'measured skew':>14}  " +
          "  ".join(f"{a:>14}" for a in ("dsrl", "mi", "bmd", "mop")))
    for sk in skews:
        sr = [r for r in rows if r["skew"] == sk]
        sk_m = np.nanmean([tv_uniform(r) for r in sr])
        cells = []
        for a in ("dsrl", "mi", "bmd", "mop"):
            v = [r["tv_to_ref"] for r in sr if r["arm"] == a]
            cells.append(f"{np.mean(v):.3f}({len(v):2d})" if v else "   --    ")
        print(f"{sk:>10} {sk_m:>14.3f}  " + "  ".join(f"{c:>14}" for c in cells))

    print(f"\n{'comparison':<22}{'diff':>8}{'95% CI':>20}{'p':>11}{'wins':>8}")
    for other, label in ARMS.items():
        diffs = []
        for sk in skews:
            for bb in bbs:
                sel = [r for r in rows if r["skew"] == sk and r.get("bb_seed") == bb]
                m = [r["tv_to_ref"] for r in sel if r["arm"] == "mop"]
                o = [r["tv_to_ref"] for r in sel if r["arm"] == other]
                if len(m) > 1 and len(o) > 1:
                    diffs.append(np.mean(o) - np.mean(m))
        d = np.array(diffs)
        if d.size < 3:
            print(f"  vs {label:<18} too few units ({d.size})")
            continue
        sem = d.std(ddof=1) / np.sqrt(d.size)
        t, p2 = stats.ttest_1samp(d, 0)
        p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        lo, hi = d.mean() - T95 * sem, d.mean() + T95 * sem
        print(f"  vs {label:<18}{d.mean():>+8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{p1:>11.5f}"
              f"{f'{int((d > 0).sum())}/{d.size}':>8}")

    print("\npooled over runs, for comparison with the clustered result above")
    x = np.array([r["tv_to_ref"] for r in rows if r["arm"] == "mop"])
    for other, label in ARMS.items():
        y = np.array([r["tv_to_ref"] for r in rows if r["arm"] == other])
        s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        t, p2 = stats.ttest_ind(x, y, equal_var=False)
        p1 = p2 / 2 if t < 0 else 1 - p2 / 2
        print(f"  vs {label:<18}d={(y.mean() - x.mean()) / s:+.2f}  "
              f"p={p1:.2e}  n={len(x)}/{len(y)}")

    sr = np.mean([r["SR"] for r in rows if r["arm"] == "mop"])
    print(f"\nreference matching task success {sr:.3f}")

    if uniform and fitted:
        print("\nthe learned-skill arm under the fitted prior, for comparison")
        d = []
        for sk in skews:
            for bb in bbs:
                m = [r["tv_to_ref"] for r in rows
                     if r["skew"] == sk and r.get("bb_seed") == bb and r["arm"] == "mop"]
                y = [r["tv_to_ref"] for r in fitted
                     if r["skew"] == sk and r.get("bb_seed") == bb]
                if len(m) > 1 and len(y) > 1:
                    d.append(np.mean(y) - np.mean(m))
        d = np.array(d)
        if d.size >= 3:
            sem = d.std(ddof=1) / np.sqrt(d.size)
            t, p2 = stats.ttest_1samp(d, 0)
            print(f"  vs learned skills, fitted prior  diff={d.mean():+.3f} "
                  f"p={(p2 / 2 if t > 0 else 1 - p2 / 2):.4f} "
                  f"wins={int((d > 0).sum())}/{d.size}")

    # The floor can exclude arms at different rates, which would make the
    # comparison a selected one. Report what each arm loses to it.
    allr = D.unify_reference(D.load("pm_skew*.json", floor=0.0))
    print("\nexclusion by the success floor, per arm")
    for a in ("dsrl", "mi", "bmd", "mop"):
        tot = [r for r in allr if r["arm"] == a]
        kept = [r for r in tot if r["SR"] >= FLOOR]
        print(f"  {a:<5} kept {len(kept):3d}/{len(tot):3d}  "
              f"({100 * (1 - len(kept) / max(len(tot), 1)):4.1f}% excluded)  "
              f"mean SR {np.mean([r['SR'] for r in tot]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
