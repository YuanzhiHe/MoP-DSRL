#!/usr/bin/env python3
"""What each objective leaves of the pretrained behaviour, one policy at a time.

Every number in the paper is a distance between two histograms. This draws the
histograms themselves for a single frozen policy and a single run seed, so the
reference, the collapse the unregularised objective produces, the flattening
entropy maximisation produces and what reference matching keeps are all visible
side by side rather than summarised as a distance.

The policy is chosen by a rule fixed before looking at any histogram: the most
skewed level of the task, and within it the policy whose measured skew is the
median of its five. The bars are that policy's mean histogram over its five run
seeds, which is the unit the analysis uses; a single seed is one draw and the
five disagree by more than the objectives do. The rewarded routes are drawn
from the run seed, so they differ across the five and are left unmarked.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs" / "_d3il_v2"
PREFIX = "v22_"

ARMS = ("dsrl", "mi", "bmd", "mop")
NAME = {"dsrl": "No mode term", "mi": "Mode entropy",
        "bmd": "Learned skills", "mop": "Reference matching"}
C = {"dsrl": "#8c8c8c", "mi": "#d1622b", "bmd": "#1b9e77", "mop": "#1f6fb4"}
RULE = "#444444"


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(RUNS / pattern))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def skew(c):
    q = np.asarray(c, float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum())


def main() -> int:
    task, weighting = "avoiding", "0.60"
    rows = [r for r in read(f"{PREFIX}main_{task}_{weighting}.jsonl")
            if r.get("status") == "completed"]
    if not rows:
        raise SystemExit(f"no rows for {task} {weighting}")

    per = {}
    for r in rows:
        per.setdefault(r["backbone_seed"], skew(r["C_ref"]))
    bb = sorted(per.items(), key=lambda kv: kv[1])[len(per) // 2][0]
    mine = [r for r in rows if r["backbone_seed"] == bb]
    hist, own = {}, {}
    for arm in ARMS:
        cs = [np.asarray(r["C_phi"], float) for r in mine
              if r["arm"] == arm and r.get("fidelity_defined")]
        if not cs:
            raise SystemExit(f"policy {bb} has no defined {arm} run")
        hist[arm] = np.mean(cs, axis=0)
        own[arm] = float(np.mean([0.5 * np.abs(c - np.asarray(
            mine[0]["C_ref"], float)).sum() for c in cs]))
    sel = {arm: dict(C_phi=hist[arm], n=len(mine)) for arm in ARMS}
    C_ref = np.asarray(mine[0]["C_ref"], float)
    K = C_ref.size

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.4,
        "axes.labelsize": 7.4, "axes.titlesize": 8,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.dpi": 200,
    })
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.85), sharey=True)
    x = np.arange(K)
    top = max(C_ref.max(), max(hist[a].max() for a in ARMS)) * 1.18

    ax = axes[0]
    ax.bar(x, C_ref, width=0.72, color=RULE, zorder=2)
    ax.set_title("Frozen policy\n(the reference)", color=RULE)
    ax.set_ylabel("share of labelled episodes")

    for ax, arm in zip(axes[1:], ARMS):
        c = hist[arm]
        ax.bar(x, c, width=0.72, color=C[arm], zorder=2)
        ax.step(np.r_[x - 0.5, x[-1] + 0.5], np.r_[C_ref, C_ref[-1]],
                where="post", color=RULE, lw=0.9, ls=(0, (3, 2)), zorder=3)
        d = 0.5 * np.abs(c - C_ref).sum()
        ax.set_title(f"{NAME[arm]}\n$D_{{\\mathrm{{TV}}}}={d:.2f}$",
                     color=C[arm])

    # The identity's other endpoint. Entropy maximisation optimises toward this
    # line whatever the reference does, so drawing it says what the orange bars
    # are being pulled onto.
    for ax in axes:
        ax.axhline(1.0 / K, color="#b8860b", lw=0.8, ls=(0, (1, 2)), zorder=1)
    # On the reference panel the last groups are empty, so the label sits
    # there rather than over a bar.
    axes[0].annotate("uniform", xy=(K - 0.45, 1.0 / K), xytext=(0, 3),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=6.2, color="#b8860b")
    for ax in axes:
        ax.set_ylim(0, top)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_xlabel("mode group")

    fig.tight_layout(pad=0.4, w_pad=0.9)
    out = ROOT / "paper" / "figures" / "fig3_histograms"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"policy {task} weighting {weighting} backbone {bb}, five seeds")
    print(f"  reference skew {skew(C_ref):.3f}")
    for arm in ARMS:
        print(f"  {NAME[arm]:18s} TV of the drawn histograms "
              f"{0.5 * np.abs(hist[arm] - C_ref).sum():.3f}"
              f"   mean of the five runs' own distances {own[arm]:.3f}")
    print(f"wrote {out}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
