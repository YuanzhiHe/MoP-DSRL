#!/usr/bin/env python3
"""Figure: the separation holds on two tasks whose modes mean different things.

Avoiding's behavioural mode is the arm's own path past three obstacle rows.
Pushing's is which box ends up at which target, in what order -- a property of
the objects, not the arm. Plotting both against measured reference skew asks
whether the result is about the method or about one task's geometry.

Both panels use the same x-axis, KL(C_ref || uniform) measured by rolling each
frozen backbone out, because that is what the theory is about: the experimenter
sets the demonstration weighting, and behaviour cloning does not preserve the
distinction.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RUNS = Path("runs/_d3il")
OUT = Path("figures")
COL = {"dsrl": "#8c8c8c", "mop": "#1b6ca8", "mi": "#c4562a"}
LAB = {"dsrl": "DSRL", "mop": r"MoP (anchor to $C_{ref}$)", "mi": r"MI (maximize $H$)"}

TASKS = [
    dict(key="avoiding", title="(a) Avoiding — mode is the arm's own path",
         glob="rep_skew*.json", min_sr=0.15, kmax=6,
         ref={"0.00": 0.127, "0.15": 0.370, "0.30": 0.302, "0.60": 0.798}),
    dict(key="pushing", title="(b) Pushing — mode is where the objects end up",
         glob="push_skew*.json", min_sr=0.10, kmax=4,
         ref={"0.00": 0.327, "0.10": 0.120, "0.25": 0.537, "0.50": 0.770}),
]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    rng = np.random.default_rng(0)

    for ax, spec in zip(axes, TASKS):
        rows = []
        for f in sorted(RUNS.glob(spec["glob"])):
            rows += json.loads(f.read_text())
        rows = [r for r in rows if r["SR"] >= spec["min_sr"]]
        if not rows:
            ax.text(.5, .5, f"no {spec['key']} results yet", transform=ax.transAxes,
                    ha="center", color="#888")
            ax.set_title(spec["title"], loc="left")
            continue
        ref = spec["ref"]
        skews = sorted({r["skew"] for r in rows}, key=lambda s: ref[s])
        x = [ref[s] for s in skews]

        for arm in ("dsrl", "mop", "mi"):
            vals = [np.array([r["tv_to_uniform"] - r["tv_to_ref"]
                              for r in rows if r["skew"] == s and r["arm"] == arm])
                    for s in skews]
            w = 0.010 * (max(x) - min(x))
            for xi, v in zip(x, vals):
                if len(v):
                    ax.scatter(xi + rng.uniform(-w, w, len(v)), v, s=8, alpha=.3,
                               color=COL[arm], edgecolor="none", zorder=1)
            m = [v.mean() if len(v) else np.nan for v in vals]
            e = [v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals]
            ax.errorbar(x, m, yerr=e, marker="o", ms=4.5, capsize=2.5, lw=1.6,
                        color=COL[arm], label=LAB[arm], zorder=3)

        # significance of mop over mi at each level
        for xi, s in zip(x, skews):
            a = np.array([r["tv_to_uniform"] - r["tv_to_ref"]
                          for r in rows if r["skew"] == s and r["arm"] == "mop"])
            b = np.array([r["tv_to_uniform"] - r["tv_to_ref"]
                          for r in rows if r["skew"] == s and r["arm"] == "mi"])
            if len(a) < 2 or len(b) < 2:
                continue
            t, p2 = stats.ttest_ind(a, b, equal_var=False)
            p1 = p2 / 2 if t > 0 else 1 - p2 / 2
            star = "***" if p1 < .001 else "**" if p1 < .01 else "*" if p1 < .05 else "n.s."
            ax.text(xi, max(a.mean(), b.mean()) + 0.085, star, ha="center",
                    fontsize=7.5, color=COL["mop"])

        ax.axhline(0, color="k", lw=.7, ls=":")
        ax.set_xlabel(r"reference skew  KL($C_{ref}\,\|\,$uniform)  [nats]")
        ax.set_title(spec["title"], loc="left")

    axes[0].set_ylabel(r"TV to uniform $-$ TV to $C_{ref}$")
    axes[0].text(0.03, 0.96, "above 0: nearer the backbone\nbelow 0: nearer uniform",
                 transform=axes[0].transAxes, va="top", fontsize=7.2, color="#444")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_twotask.png")
    fig.savefig(OUT / "fig4_twotask.pdf")
    print("wrote figures/fig4_twotask.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
