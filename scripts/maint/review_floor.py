#!/usr/bin/env python3
"""How much of each comparison sits inside the sampling floor?

Two finite histograms of the same distribution differ by a nonzero amount, so
below that floor the arms are not separable. The figure draws the band; this
counts how many points fall into it, which is the question a reader asks of the
rightmost Pushing column.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

FLOOR = {"Avoiding": 0.055, "Pushing": 0.079}


def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


for task, rows in (
        ("Avoiding", D.unify_reference(D.load("mb_skew*.json", floor=0.15))),
        ("Pushing", [r for r in D.unify_reference(D.load("pm_skew*.json", floor=0.10))
                     if r["arm"] != "bmd"]
                   + D.unify_reference(D.load("pu_skew*.json", floor=0.10)))):
    f = FLOOR[task]
    sks = sorted({r["skew"] for r in rows},
                 key=lambda s: np.nanmean([tvu(r) for r in rows if r["skew"] == s]))
    print(f"\n=== {task}, floor {f} ===")
    for s in sks:
        sr = [r for r in rows if r["skew"] == s]
        x = np.nanmean([tvu(r) for r in sr])
        parts = []
        for a in ("dsrl", "mi", "bmd", "mop"):
            v = [r["tv_to_ref"] for r in sr if r["arm"] == a]
            m = np.mean(v) if v else np.nan
            parts.append(f"{a}={m:.3f}{'*' if m < f * 1.3 else ' '}")
        print(f"  skew {x:.3f}: " + "  ".join(parts)
              + ("   <- within 1.3x of the floor" if any("*" in p for p in parts) else ""))
