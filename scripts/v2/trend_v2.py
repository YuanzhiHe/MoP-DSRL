#!/usr/bin/env python3
"""How the advantage over mode entropy tracks the reference's own skew.

Under a uniform reference, matching a reference and maximising mode entropy
are the same objective, because KL(p || u) = ln K - H(p). The two separate
only as the reference moves away from uniform, so the gap between the two arms
is predicted to grow with the reference's distance from uniform and to vanish
where that distance is small. This script measures that distance on each
frozen policy's own reference and regresses the gap on it, one point per
policy, as section 6 of the pre-registration declares.

The launch weighting is not used as the predictor. The weighting a backbone
was trained under and the skew its reference actually shows are different
quantities, and on Pushing they are not even monotonically related.
"""
from __future__ import annotations

import os
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

# A job runs a frozen copy of scripts/ under runs/_d3il_v2/_src/<job>,
# so this file's own location says nothing about where the run files
# are. The environment names the root; the fallback is a local run.
ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402

RUNS = ROOT / "runs" / "_d3il_v2"
# Name prefix of the run files the analysis reads. The matrix was measured
# again after the environment rule changed, and both cohorts sit in the same
# directory; the prefix is what keeps the two apart, so a bare invocation
# reads the current one.
PREFIX = "v22_"

ARMS = ("mi", "dsrl", "bmd")


def policy_of(r):
    return (r["task"], r["weighting"], r["backbone_seed"])


def skew_of(r):
    """Total variation from the reference to the uniform histogram."""
    q = np.asarray(r["C_ref"], float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum())


def load(task, prefix=PREFIX):
    rows = []
    for f in sorted(glob.glob(str(RUNS / f"{prefix}main_{task}_*.jsonl"))):
        rows += [json.loads(l) for l in Path(f).open() if l.strip()]
    for r in rows:
        v = r.get("tv_to_ref")
        r["closeness_tv"] = None if v is None else -float(v)
        r["skew"] = skew_of(r)
    return rows


def per_policy_gap(rows, other):
    """Gap in closeness between mop and ``other``, and skew, per policy."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    skew = {}
    for r in rows:
        v = r["closeness_tv"]
        if v is None:
            continue
        acc[policy_of(r)][r["arm"]].append(v)
        skew[policy_of(r)] = r["skew"]
    out = []
    for p, arms in sorted(acc.items()):
        if "mop" in arms and other in arms:
            out.append((skew[p], float(np.mean(arms["mop"]))
                        - float(np.mean(arms[other])), p))
    return out


def regress(points):
    x = np.array([a for a, _, _ in points])
    y = np.array([b for _, b, _ in points])
    res = sps.linregress(x, y)
    return dict(n=int(x.size), slope=float(res.slope),
                stderr=float(res.stderr), intercept=float(res.intercept),
                r=float(res.rvalue), p_two_sided=float(res.pvalue),
                df=int(x.size - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PREFIX,
                    help="run-file name prefix; the empty string reads the "
                         "files the earlier environment rule produced")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-doc", type=Path, default=ROOT / "docs" / "v2" / "TREND.md")
    args = ap.parse_args()
    if args.out is None:
        args.out = RUNS / f"{args.prefix}trend_main.json"

    report, doc = {}, ["# Advantage against reference skew", ""]
    for task in ("avoiding", "pushing"):
        rows = load(task, args.prefix)
        if not rows:
            continue
        byw = collections.defaultdict(list)
        for r in rows:
            byw[r["weighting"]].append(r["skew"])
        doc += [f"## {task.capitalize()}", "",
                "| launch weighting | measured reference skew |", "|---|---|"]
        for w in sorted(byw):
            doc.append(f"| {w} | {np.mean(byw[w]):.3f} |")
        doc.append("")
        report[f"{task}/skew by weighting"] = {
            w: float(np.mean(v)) for w, v in sorted(byw.items())}

        doc += ["| comparison | n | slope | se | r | p | per-weighting gap |",
                "|---|---|---|---|---|---|---|"]
        for other in ARMS:
            pts = per_policy_gap(rows, other)
            if len(pts) < 3:
                continue
            reg = regress(pts)
            bins = collections.defaultdict(list)
            for s, g, p in pts:
                bins[p[1]].append((s, g))
            cells = " ".join(
                f"{np.mean([s for s, _ in v]):.2f}:{np.mean([g for _, g in v]):+.3f}"
                for _, v in sorted(bins.items(),
                                   key=lambda kv: np.mean([s for s, _ in kv[1]])))
            report[f"{task}/trend vs {other}"] = dict(
                **reg, points=[(s, g, list(p)) for s, g, p in pts])
            doc.append(f"| mop over {other} | {reg['n']} | {reg['slope']:+.4f} "
                       f"| {reg['stderr']:.4f} | {reg['r']:+.3f} "
                       f"| {reg['p_two_sided']:.4f} | {cells} |")
        doc.append("")
        doc += ["The per-weighting column reads skew:gap, ordered by measured "
                "skew rather than by launch weighting.", ""]

    args.out.write_text(json.dumps(report, indent=1, default=str))
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.write_text("\n".join(doc))
    print("\n".join(doc))
    print(f"wrote {args.out} and {args.out_doc}")


if __name__ == "__main__":
    main()
