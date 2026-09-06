#!/usr/bin/env python3
"""Aggregate the D3IL head-to-head sweep and test the separation claim.

The prediction under test, established analytically and reproduced in the toy:
KL(p || uniform) = ln K - H(p), so anchoring to the reference and maximizing mode
entropy are the SAME objective when the reference is uniform. mop and mi must
therefore coincide at zero backbone skew and can only diverge as the backbone's
own mode profile departs from uniform.

The discriminating measurement is tv_to_ref -- total variation between the
finetuned policy's mode histogram and the frozen backbone's. mop should hold it
low as skew grows; mi, which pulls toward uniform, should not; dsrl, with no
mode objective at all, should drift furthest.

Reports mean +- s.e. per (skew, arm) and a Welch t-test of mop vs mi at each
skew, since the claim is specifically about where those two separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ["dsrl", "mop", "mi"]
METRICS = ["SR", "H_modes", "mc_at_08", "tv_to_ref", "tv_to_uniform"]


def load(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(run_dir.glob("full_skew*.json")):
        rows += json.loads(f.read_text())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/_d3il"))
    args = ap.parse_args()

    rows = load(args.run_dir)
    if not rows:
        print(f"no results under {args.run_dir}")
        return 1
    skews = sorted({r["skew"] for r in rows}, key=float)
    print(f"runs: {len(rows)}  skews: {skews}  "
          f"seeds/cell: {len([r for r in rows if r['skew'] == skews[0] and r['arm'] == 'mop'])}")

    def cell(skew, arm, metric):
        return np.array([r[metric] for r in rows if r["skew"] == skew and r["arm"] == arm],
                        dtype=float)

    for metric in METRICS:
        print(f"\n=== {metric} (mean +- s.e.) ===")
        print(f"  {'skew':>6}  " + "  ".join(f"{a:>14}" for a in ARMS))
        for s in skews:
            cells = []
            for a in ARMS:
                v = cell(s, a, metric)
                cells.append(f"{v.mean():6.3f}+-{v.std(ddof=1) / np.sqrt(len(v)):.3f}")
            print(f"  {s:>6}  " + "  ".join(f"{c:>14}" for c in cells))

    print("\n=== mop vs mi on tv_to_ref (Welch t-test) ===")
    print("  the claim: indistinguishable at skew 0, mop lower as skew grows")
    print(f"  {'skew':>6}  {'mop':>12}  {'mi':>12}  {'diff':>8}  {'t':>7}  {'p':>7}")
    for s in skews:
        a, b = cell(s, "mop", "tv_to_ref"), cell(s, "mi", "tv_to_ref")
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {s:>6}  {a.mean():6.3f}+-{a.std(ddof=1) / np.sqrt(len(a)):.3f}  "
              f"{b.mean():6.3f}+-{b.std(ddof=1) / np.sqrt(len(b)):.3f}  "
              f"{a.mean() - b.mean():+8.3f}  {t:7.2f}  {p:7.3f}")

    print("\n=== fidelity gap: tv_to_uniform - tv_to_ref ===")
    print("  positive = the policy sits nearer the backbone's profile than uniform")
    print(f"  {'skew':>6}  " + "  ".join(f"{a:>14}" for a in ARMS))
    for s in skews:
        cells = []
        for a in ARMS:
            g = cell(s, a, "tv_to_uniform") - cell(s, a, "tv_to_ref")
            cells.append(f"{g.mean():6.3f}+-{g.std(ddof=1) / np.sqrt(len(g)):.3f}")
        print(f"  {s:>6}  " + "  ".join(f"{c:>14}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
