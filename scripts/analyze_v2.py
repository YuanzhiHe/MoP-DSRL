#!/usr/bin/env python3
"""Analysis rebuilt around the objections a reviewer raised against the first one.

Five changes, each answering a specific criticism.

1. SCALE-FREE ENDPOINT. The fidelity gap TV(p,u) - TV(p,q) is bounded by
   +-TV(q,u) through the triangle inequality, so its whole dynamic range grows
   with reference skew whether or not behaviour changes. Reporting it raw makes
   "the effect grows with skew" partly arithmetic. The normalised gap divides by
   TV(C_ref, uniform), putting every level on [-1, 1], and the trend is tested on
   that.

2. BACKBONE AS THE UNIT. One backbone per skew level confounds the level with a
   single behaviour-cloning draw. Where several backbones exist per level, the
   trend is estimated with backbone-level means and backbone-level standard
   errors, not run-level ones.

3. FULL-FAMILY HOLM. Every mop-vs-mi test across tasks and levels forms one
   family and is corrected together, instead of each table being read at nominal
   alpha.

4. SUCCESS RATE ALONGSIDE FIDELITY. The premise is what each arm retains WHILE
   STILL SOLVING THE TASK, so no fidelity number is reported without the success
   rate beside it.

5. ATTRITION MADE VISIBLE. The SR floor conditions on a post-treatment outcome,
   so per-arm attrition is tabulated and tested, and every endpoint is reported
   both filtered and unfiltered.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ["dsrl", "mop", "mi", "bmd"]


def load(run_dir: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(run_dir.glob(f"{prefix}*.json")):
        rows += json.loads(f.read_text())
    return rows


def tv_ref_uniform(r: dict) -> float:
    q = np.asarray(r.get("C_ref", []), dtype=float)
    if q.size == 0:
        return float("nan")
    return float(0.5 * np.abs(q - 1.0 / q.size).sum())


def gap(r: dict) -> float:
    return r["tv_to_uniform"] - r["tv_to_ref"]


def gap_norm(r: dict) -> float:
    d = tv_ref_uniform(r)
    return gap(r) / d if d and np.isfinite(d) and d > 1e-9 else float("nan")


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [False] * len(pvals)
    m = len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] < alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/_d3il"))
    ap.add_argument("--prefix", nargs="+", required=True,
                    help="one or more result-file prefixes to pool")
    ap.add_argument("--min-sr", type=float, default=0.0)
    args = ap.parse_args()

    allrows: list[dict] = []
    for pre in args.prefix:
        rs = load(args.run_dir, pre)
        for r in rs:
            r.setdefault("task", "avoiding")
            r["_prefix"] = pre
        allrows += rs
    if not allrows:
        print("no results")
        return 1

    # --- attrition ---------------------------------------------------------
    print(f"=== attrition at SR floor {args.min_sr} ===")
    kept, dropped = [], []
    for r in allrows:
        (kept if r["SR"] >= args.min_sr else dropped).append(r)
    ca, cd = Counter(r["arm"] for r in kept), Counter(r["arm"] for r in dropped)
    print(f"  {'arm':>6}  {'kept':>6}  {'dropped':>8}  {'rate':>7}")
    tbl = []
    for a in ARMS:
        k, d = ca.get(a, 0), cd.get(a, 0)
        if k + d == 0:
            continue
        print(f"  {a:>6}  {k:>6}  {d:>8}  {d / (k + d):>6.1%}")
        tbl.append([k, d])
    if len(tbl) >= 2:
        chi = stats.chi2_contingency(np.array(tbl))[1] if min(map(min, tbl)) >= 0 else float("nan")
        print(f"  differential attrition across arms: chi-square p = {chi:.3f}")
    print("  (the floor conditions on a post-treatment outcome, so this is")
    print("   collider conditioning, not a nuisance filter)\n")

    for label, rows in (("UNFILTERED", allrows), (f"SR >= {args.min_sr}", kept)):
        if args.min_sr <= 0 and label != "UNFILTERED":
            continue
        print(f"########## {label}  (n={len(rows)}) ##########")
        report(rows)
        print()
    return 0


def report(rows: list[dict]) -> None:
    tasks = sorted({r["task"] for r in rows})
    pvals, keys = [], []

    for task in tasks:
        tr = [r for r in rows if r["task"] == task]
        skews = sorted({r["skew"] for r in tr},
                       key=lambda s: np.mean([tv_ref_uniform(r) for r in tr
                                              if r["skew"] == s]))
        print(f"--- {task} ---")
        print(f"  {'skew':>6} {'TV(Cref,u)':>11}  " +
              "  ".join(f"{a:>20}" for a in ARMS))
        print(f"  {'':>6} {'':>11}  " +
              "  ".join(f"{'gap_norm / SR':>20}" for _ in ARMS))
        for s in skews:
            sr_rows = [r for r in tr if r["skew"] == s]
            d = np.nanmean([tv_ref_uniform(r) for r in sr_rows])
            cells = []
            for a in ARMS:
                v = [r for r in sr_rows if r["arm"] == a]
                if len(v) < 2:
                    cells.append(f"{'--':>20}")
                    continue
                g = np.array([gap_norm(r) for r in v])
                sr = np.array([r["SR"] for r in v])
                cells.append(f"{np.nanmean(g):+6.3f}+-{np.nanstd(g, ddof=1)/np.sqrt(len(g)):.3f}"
                             f" / {sr.mean():.2f}".rjust(20))
            print(f"  {s:>6} {d:>11.3f}  " + "  ".join(cells))

            a = np.array([gap_norm(r) for r in sr_rows if r["arm"] == "mop"])
            b = np.array([gap_norm(r) for r in sr_rows if r["arm"] == "mi"])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) > 1 and len(b) > 1:
                t, p2 = stats.ttest_ind(a, b, equal_var=False)
                pvals.append(p2 / 2 if t > 0 else 1 - p2 / 2)
                keys.append((task, s, a.mean() - b.mean(), len(a), len(b)))

        # backbone-level trend, where several backbones exist per level
        bbs = {r.get("bb_seed") for r in tr}
        if len(bbs - {None}) > 1:
            print("\n  trend on NORMALISED gap, backbone as the unit of analysis:")
            xs, ys = [], []
            for s in skews:
                for bb in sorted(b for b in bbs if b is not None):
                    v = [r for r in tr if r["skew"] == s and r.get("bb_seed") == bb
                         and r["arm"] == "mop"]
                    w = [r for r in tr if r["skew"] == s and r.get("bb_seed") == bb
                         and r["arm"] == "mi"]
                    if len(v) < 2 or len(w) < 2:
                        continue
                    xs.append(np.nanmean([tv_ref_uniform(r) for r in v]))
                    ys.append(np.nanmean([gap_norm(r) for r in v])
                              - np.nanmean([gap_norm(r) for r in w]))
            if len(xs) >= 4:
                sl, ic, rr, pp, se = stats.linregress(xs, ys)
                print(f"    n = {len(xs)} backbone-level points")
                print(f"    slope = {sl:+.3f} +- {se:.3f}   r = {rr:+.3f}   p = {pp:.4f}")
                print("    (run-level SEs would overstate this by treating runs that")
                print("     share a backbone as independent)")
            else:
                print(f"    only {len(xs)} backbone-level points; trend not estimated")
        else:
            print("\n  single backbone per level: the skew trend has as many independent")
            print("  units as levels, so no trend statistic is reported here.")
        print()

    if pvals:
        print("=== mop vs mi, normalised gap, ONE family, Holm-corrected ===")
        rej = holm(pvals)
        print(f"  {'task':>9} {'skew':>6} {'diff':>8} {'p(1-sided)':>11} {'n':>8}  verdict")
        for (task, s, diff, na, nb), p, r in zip(keys, pvals, rej):
            print(f"  {task:>9} {s:>6} {diff:>+8.3f} {p:>11.4f} {na:>3}/{nb:<4}  "
                  f"{'reject' if r else 'not rejected'}")
        print(f"  -> {sum(rej)} of {len(rej)} reject after Holm across the full family")


if __name__ == "__main__":
    raise SystemExit(main())
