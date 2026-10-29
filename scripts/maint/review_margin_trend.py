#!/usr/bin/env python3
"""Does the margin widen with reference skew, and against which baseline?

The abstract and conclusion attach that trend to the comparison with no mode
term. The identity predicts it for the comparison with entropy maximisation,
which is a different claim, so this computes both from the runs.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D


def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


for task, rows, floor in (
        ("Avoiding", D.unify_reference(D.load("mb_skew*.json", floor=0.15)), 0.15),
        ("Pushing", [r for r in D.unify_reference(D.load("pm_skew*.json", floor=0.10))
                     if r["arm"] != "bmd"]
                   + D.unify_reference(D.load("pu_skew*.json", floor=0.10)), 0.10)):
    sks = sorted({r["skew"] for r in rows},
                 key=lambda s: np.nanmean([tvu(r) for r in rows if r["skew"] == s]))
    xs = [np.nanmean([tvu(r) for r in rows if r["skew"] == s]) for s in sks]
    print(f"\n=== {task}: margin over each baseline, by measured skew ===")
    print(f"{'skew':>7}" + "".join(f"{b:>18}" for b in
                                   ("vs no mode term", "vs mode entropy",
                                    "vs learned skills")))
    cols = {b: [] for b in ("dsrl", "mi", "bmd")}
    for s, x in zip(sks, xs):
        sr = [r for r in rows if r["skew"] == s]
        m = np.mean([r["tv_to_ref"] for r in sr if r["arm"] == "mop"])
        line = f"{x:>7.3f}"
        for b in ("dsrl", "mi", "bmd"):
            v = [r["tv_to_ref"] for r in sr if r["arm"] == b]
            d = (np.mean(v) - m) if v else np.nan
            cols[b].append(d)
            line += f"{d:>18.3f}"
        print(line)
    for b, lab in (("dsrl", "no mode term"), ("mi", "mode entropy"),
                   ("bmd", "learned skills")):
        y = np.array(cols[b]); x = np.array(xs)
        sl, ic, r, p, se = stats.linregress(x, y)
        trend = "widens" if sl > 0 else "NARROWS"
        print(f"  vs {lab:<16} slope={sl:+.3f} per unit skew, r={r:+.2f}  -> {trend}")
