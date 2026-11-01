#!/usr/bin/env python3
"""Main figures for the MoP write-up, built from the pre-registered replication.

Indexed by MEASURED reference skew KL(C_ref || uniform) rather than by the
demonstration weighting. That is the amendment the pre-registration records: the
theory is about the reference's skew, the experimenter only sets the
demonstrations' skew, and behaviour cloning does not preserve the distinction.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("runs/_d3il")
OUT = Path("figures")
MIN_SR = 0.15

# KL(C_ref || uniform), measured over 1200 rollouts per backbone, 6-group partition
REF_KL = {"0.00": 0.127, "0.15": 0.370, "0.30": 0.302, "0.60": 0.798}
COL = {"dsrl": "#8c8c8c", "mop": "#1b6ca8", "mi": "#c4562a"}
LAB = {"dsrl": "DSRL (no mode objective)", "mop": "Reference matching",
       "mi": "MI (maximize mode entropy)"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def load():
    rows = []
    for f in sorted(RUNS.glob("rep_skew*.json")):
        rows += json.loads(f.read_text())
    return [r for r in rows if r["SR"] >= MIN_SR]


def cell(rows, skew, arm, fn):
    return np.array([fn(r) for r in rows if r["skew"] == skew and r["arm"] == arm])


def se(v):
    return v.std(ddof=1) / np.sqrt(len(v))


def main() -> int:
    rows = load()
    skews = sorted({r["skew"] for r in rows}, key=lambda s: REF_KL[s])
    x = [REF_KL[s] for s in skews]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    jit = np.random.default_rng(0)

    def strip(ax, xi, vals, color):
        """Every run as a point, so the reader sees the spread and not only its
        summary. Error bars alone hide bimodality, and this data has it: a
        quarter of seeds fail to optimize at all."""
        w = 0.012 * (max(x) - min(x) + 1e-9)
        ax.scatter(xi + jit.uniform(-w, w, len(vals)), vals, s=7, alpha=.28,
                   color=color, edgecolor="none", zorder=1)

    # (a) distance to the backbone's own mode profile
    ax = axes[0]
    for arm in ("dsrl", "mop", "mi"):
        cells = [cell(rows, s, arm, lambda r: r["tv_to_ref"]) for s in skews]
        for xi, v in zip(x, cells):
            strip(ax, xi, v, COL[arm])
        ax.errorbar(x, [v.mean() for v in cells], yerr=[se(v) for v in cells],
                    marker="o", ms=4.5, capsize=2.5, lw=1.6, color=COL[arm],
                    label=LAB[arm], zorder=3)
    ax.set_xlabel(r"reference skew  KL($C_{ref}\,\|\,$uniform)  [nats]")
    ax.set_ylabel(r"TV$(C_\phi,\ C_{ref})$")
    ax.set_title("(a) departure from the frozen backbone", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # (b) the discriminating statistic
    ax = axes[1]
    for arm in ("dsrl", "mop", "mi"):
        g = [cell(rows, s, arm, lambda r: r["tv_to_uniform"] - r["tv_to_ref"]) for s in skews]
        for xi, v in zip(x, g):
            strip(ax, xi, v, COL[arm])
        ax.errorbar(x, [v.mean() for v in g], yerr=[se(v) for v in g],
                    marker="o", ms=4.5, capsize=2.5, lw=1.6, color=COL[arm],
                    label=LAB[arm], zorder=3)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"reference skew  KL($C_{ref}\,\|\,$uniform)  [nats]")
    ax.set_ylabel(r"TV to uniform $-$ TV to $C_{ref}$")
    ax.set_title("(b) which target the policy ends up near", loc="left")
    ax.text(0.03, 0.95, "above 0: nearer the backbone\nbelow 0: nearer uniform",
            transform=ax.transAxes, va="top", fontsize=7.5, color="#444")

    # (c) mode collapse without any regularizer
    ax = axes[2]
    for arm in ("dsrl", "mop", "mi"):
        h = [cell(rows, s, arm, lambda r: r["H_modes"]) for s in skews]
        for xi, v in zip(x, h):
            strip(ax, xi, v, COL[arm])
        ax.errorbar(x, [v.mean() for v in h], yerr=[se(v) for v in h],
                    marker="o", ms=4.5, capsize=2.5, lw=1.6, color=COL[arm],
                    label=LAB[arm], zorder=3)
    ax.axhline(np.log(6), color="k", lw=0.7, ls=":")
    ax.text(x[0], np.log(6) + 0.03, "uniform over 6 route groups", fontsize=7, color="#444")
    ax.set_xlabel(r"reference skew  KL($C_{ref}\,\|\,$uniform)  [nats]")
    ax.set_ylabel(r"$H(C_\phi)$  [nats]")
    ax.set_title("(c) behavioural diversity retained", loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_main.png")
    fig.savefig(OUT / "fig1_main.pdf")
    print("wrote figures/fig1_main.{png,pdf}")

    # Effect size vs reference skew
    fig2, ax = plt.subplots(figsize=(3.6, 3.0))
    ds = []
    for s in skews:
        a = cell(rows, s, "mop", lambda r: r["tv_to_uniform"] - r["tv_to_ref"])
        b = cell(rows, s, "mi", lambda r: r["tv_to_uniform"] - r["tv_to_ref"])
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        ds.append((a.mean() - b.mean()) / pooled)
    ax.plot(x, ds, marker="s", ms=5, lw=1.5, color="#1b6ca8")
    for xi, di, s in zip(x, ds, skews):
        ax.annotate(f"{di:.2f}", (xi, di), textcoords="offset points",
                    xytext=(5, -9), fontsize=7.5)
    ax.set_xlabel(r"reference skew  KL($C_{ref}\,\|\,$uniform)  [nats]")
    ax.set_ylabel("Cohen's $d$   (MoP $-$ MI)")
    ax.set_title("separation scales with reference skew", loc="left")
    ax.set_ylim(0, max(ds) * 1.2)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig2_effectsize.png")
    fig2.savefig(OUT / "fig2_effectsize.pdf")
    print("wrote figures/fig2_effectsize.{png,pdf}")
    print("\nd by reference skew:", [f"{k:.3f}->{v:.2f}" for k, v in zip(x, ds)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
