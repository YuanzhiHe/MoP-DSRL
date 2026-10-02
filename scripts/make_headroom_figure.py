#!/usr/bin/env python3
"""Figure: when mode regularization matters, and when it cannot.

Avoiding shows unregularized RL destroying the backbone's mode structure and both
regularizers preventing it. Pushing shows no such difference -- and the reason is
not that the regularizers failed there. It is that nothing moved: the Pushing
backbone starts near the floor of its entropy range and solves the task 31% of
the time, so RL has neither diversity to destroy nor reward signal to chase.

Stating that as a precondition is more useful than reporting the null. Mode
regularization can only matter where the backbone has entropy to lose AND the
task has performance to gain.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS, OUT = Path("runs/_d3il"), Path("figures")
COL = {"dsrl": "#8c8c8c", "mop": "#1b6ca8", "mi": "#c4562a"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

SPECS = [
    dict(name="Avoiding", glob="rep_skew*.json", min_sr=0.15, K=6, bb_H=2.65),
    dict(name="Pushing", glob="pushrw_skew*.json", min_sr=0.10, K=4, bb_H=1.06),
]


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))

    # (a) where each arm ends up, against the backbone it started from
    ax = axes[0]
    width, pos = 0.24, np.arange(len(SPECS))
    for i, arm in enumerate(("dsrl", "mop", "mi")):
        means, errs = [], []
        for spec in SPECS:
            rows = []
            for f in sorted(RUNS.glob(spec["glob"])):
                rows += json.loads(f.read_text())
            v = np.array([r["H_modes"] for r in rows
                          if r["SR"] >= spec["min_sr"] and r["arm"] == arm])
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(pos + (i - 1) * width, means, width, yerr=errs, capsize=3,
               color=COL[arm], label=arm.upper() if arm != "mop" else "MoP")
    for j, spec in enumerate(SPECS):
        ax.hlines(spec["bb_H"], j - 1.6 * width, j + 1.6 * width,
                  color="#171a1f", lw=1.6, ls="--", zorder=4)
        ax.text(j, spec["bb_H"] + 0.06, "frozen backbone", ha="center",
                fontsize=7.2, color="#171a1f")
        ax.hlines(np.log(spec["K"]), j - 1.6 * width, j + 1.6 * width,
                  color="#999", lw=.9, ls=":", zorder=4)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{s['name']}\n({s['K']} modes)" for s in SPECS])
    ax.set_ylabel(r"$H(C_\phi)$ after training  [nats]")
    ax.set_title("(a) how far RL moves from the backbone", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7.6)
    ax.set_ylim(0, 3.2)

    # (b) the two headrooms that decide whether the question is askable
    ax = axes[1]
    data = []
    for spec in SPECS:
        rows = []
        for f in sorted(RUNS.glob(spec["glob"])):
            rows += json.loads(f.read_text())
        rows = [r for r in rows if r["SR"] >= spec["min_sr"]]
        dsrl_H = np.mean([r["H_modes"] for r in rows if r["arm"] == "dsrl"])
        dsrl_SR = np.mean([r["SR"] for r in rows if r["arm"] == "dsrl"])
        data.append((spec["name"], spec["bb_H"] - dsrl_H, dsrl_SR))
    names = [d[0] for d in data]
    ax.bar(pos - 0.18, [d[1] for d in data], 0.34, color="#1b6ca8",
           label="entropy the backbone can lose  [nats]")
    ax.bar(pos + 0.18, [d[2] for d in data], 0.34, color="#c4562a",
           label="task success RL has to work with")
    for j, d in enumerate(data):
        ax.text(j - 0.18, d[1] + 0.05, f"{d[1]:.2f}", ha="center", fontsize=7.4)
        ax.text(j + 0.18, d[2] + 0.05, f"{d[2]:.2f}", ha="center", fontsize=7.4)
    ax.set_xticks(pos)
    ax.set_xticklabels(names)
    ax.set_title("(b) the headroom the question needs", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=7.4)
    ax.set_ylim(0, 2.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_headroom.png")
    fig.savefig(OUT / "fig5_headroom.pdf")
    print("wrote figures/fig5_headroom.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
