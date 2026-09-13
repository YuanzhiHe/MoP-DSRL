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


def load(run_dir: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(run_dir.glob(f"{prefix}*.json")):
        rows += json.loads(f.read_text())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/_d3il"))
    ap.add_argument("--prefix", default="full_skew",
                    help="result-file prefix to aggregate")
    ap.add_argument("--prereg", action="store_true",
                    help="run the pre-registered confirmatory tests "
                         "(docs/PREREG_d3il_replication.md)")
    ap.add_argument("--min-sr", type=float, default=0.0,
                    help="drop runs below this success rate as failed optimizations")
    args = ap.parse_args()

    rows = load(args.run_dir, args.prefix)
    if args.min_sr > 0:
        # Roughly a quarter of seeds fail to optimize at all -- DSRL, which has no
        # mode objective, still produced a 0.052 run -- so a floor separates
        # "the mode objective changed behaviour" from "this seed never learned".
        before = len(rows)
        rows = [r for r in rows if r["SR"] >= args.min_sr]
        print(f"dropped {before - len(rows)}/{before} runs below SR {args.min_sr}")
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
    print(f"  {'skew':>6}  {'mop':>12}  {'mi':>12}  {'diff':>8}  {'t':>7}  {'p':>7}"
          f"   effect size and the smallest resolvable difference")
    for s in skews:
        a, b = cell(s, "mop", "tv_to_ref"), cell(s, "mi", "tv_to_ref")
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
        # smallest difference this many seeds could resolve at 80% power
        mde = 2.8 * pooled / np.sqrt(len(a))
        print(f"  {s:>6}  {a.mean():6.3f}+-{a.std(ddof=1) / np.sqrt(len(a)):.3f}  "
              f"{b.mean():6.3f}+-{b.std(ddof=1) / np.sqrt(len(b)):.3f}  "
              f"{a.mean() - b.mean():+8.3f}  {t:7.2f}  {p:7.3f}"
              f"   d={d:+.2f}  MDE={mde:.3f}  n={len(a)}/{len(b)}")

    print("\n=== mop vs mi on the fidelity gap (Welch t-test) ===")
    print("  gap = tv_to_uniform - tv_to_ref, differenced WITHIN a run, so the")
    print("  per-seed noise that dominates either distance alone cancels")
    print(f"  {'skew':>6}  {'mop':>12}  {'mi':>12}  {'diff':>8}  {'t':>7}  {'p':>7}   effect")
    for s_ in skews:
        a = cell(s_, "mop", "tv_to_uniform") - cell(s_, "mop", "tv_to_ref")
        b = cell(s_, "mi", "tv_to_uniform") - cell(s_, "mi", "tv_to_ref")
        t, p_ = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
        print(f"  {s_:>6}  {a.mean():+6.3f}+-{a.std(ddof=1)/np.sqrt(len(a)):.3f}  "
              f"{b.mean():+6.3f}+-{b.std(ddof=1)/np.sqrt(len(b)):.3f}  "
              f"{a.mean()-b.mean():+8.3f}  {t:7.2f}  {p_:7.4f}   d={d:+.2f}  n={len(a)}/{len(b)}")

    print("\n=== mode preservation vs no regularizer (dsrl) ===")
    print("  the prior question: does noise-space RL destroy the backbone's modes,")
    print("  and does ANY mode objective prevent it?")
    print(f"  {'skew':>6}  {'metric':>10}  {'dsrl':>12}  {'mop':>12}  {'mi':>12}  {'p(mop)':>7}  {'p(mi)':>7}")
    for s_ in skews:
        for metric in ("H_modes", "mc_at_08"):
            d0, m1, m2 = cell(s_, "dsrl", metric), cell(s_, "mop", metric), cell(s_, "mi", metric)
            _, p1 = stats.ttest_ind(m1, d0, equal_var=False)
            _, p2 = stats.ttest_ind(m2, d0, equal_var=False)
            print(f"  {s_:>6}  {metric:>10}  {d0.mean():6.3f}+-{d0.std(ddof=1)/np.sqrt(len(d0)):.3f}  "
                  f"{m1.mean():6.3f}+-{m1.std(ddof=1)/np.sqrt(len(m1)):.3f}  "
                  f"{m2.mean():6.3f}+-{m2.std(ddof=1)/np.sqrt(len(m2)):.3f}  "
                  f"{p1:7.4f}  {p2:7.4f}")

    if args.prereg:
        # docs/PREREG_d3il_replication.md, fixed before the replication ran.
        # One-sided because the direction is predicted, not fitted: mi optimizes
        # mode entropy, whose optimum is the uniform histogram, so it should sit
        # nearer uniform than the backbone.
        print("\n=== CONFIRMATORY (pre-registered) ===")
        print("  H1: fidelity gap greater for mop than mi at skews 0.15 and 0.30")
        print("  one-sided Welch, Holm-corrected over the two tests; both must reject")
        conf = []
        for s_ in ("0.15", "0.30"):
            if s_ not in skews:
                print(f"  skew {s_}: MISSING")
                continue
            a = cell(s_, "mop", "tv_to_uniform") - cell(s_, "mop", "tv_to_ref")
            b = cell(s_, "mi", "tv_to_uniform") - cell(s_, "mi", "tv_to_ref")
            t, p_two = stats.ttest_ind(a, b, equal_var=False)
            p_one = p_two / 2 if t > 0 else 1 - p_two / 2
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
            conf.append((s_, p_one, a.mean() - b.mean(), d, len(a), len(b)))
        # Holm: sort ascending, compare against alpha/(m - i)
        order = sorted(range(len(conf)), key=lambda i: conf[i][1])
        verdicts = {}
        m = len(conf)
        for rank, i in enumerate(order):
            thresh = 0.05 / (m - rank)
            verdicts[i] = conf[i][1] < thresh
            if not verdicts[i]:
                for j in order[rank + 1:]:
                    verdicts[j] = False
                break
        for i, (s_, p_one, diff, d, na, nb) in enumerate(conf):
            print(f"  skew {s_}  diff={diff:+.3f}  d={d:+.2f}  p(one-sided)={p_one:.4f}  "
                  f"n={na}/{nb}  -> {'REJECT H0' if verdicts.get(i) else 'not rejected'}")
        ok = all(verdicts.get(i) for i in range(len(conf))) and len(conf) == 2
        print(f"\n  H1 {'CONFIRMED' if ok else 'NOT CONFIRMED'} "
              f"({'both tests reject' if ok else 'replication failed; the exploratory result is the likelier artifact'})")

        if "0.00" in skews:
            a = cell("0.00", "mop", "tv_to_uniform") - cell("0.00", "mop", "tv_to_ref")
            b = cell("0.00", "mi", "tv_to_uniform") - cell("0.00", "mi", "tv_to_ref")
            t, p_two = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
            print(f"\n  S1 implementation check, skew 0.00 (two-sided): "
                  f"diff={a.mean() - b.mean():+.3f} d={d:+.2f} p={p_two:.3f} -> "
                  f"{'OK, arms coincide as the identity requires' if p_two > 0.05 else 'WARNING: arms differ where they must not'}")

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
