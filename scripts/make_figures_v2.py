#!/usr/bin/env python3
"""Figures authored at the journal's measured text width.

The sn-jnl text block is 372 pt, or 5.15 in. The previous figure was drawn on an
11.6 in canvas and included at \\textwidth, a 0.44x reduction that rendered 6 pt
annotations at under 3 pt. Everything here is drawn at final size, so no label
is smaller on the page than it is in the source.

Split into two figures because one was carrying a schematic, a premise and three
results panels at once. Figure 1 is the setting and the premise; Figure 2 is the
result.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d3il_runs
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("figures")
W = 5.15                      # sn-jnl text width, measured
# Smallest type on the page. matplotlib's tight bbox makes the saved file a
# little wider than the canvas, so \includegraphics[width=\textwidth] scales
# the result to about 0.986; everything is authored at 6.3 pt or more so that
# nothing lands under the 6 pt floor once scaled.
MIN_PT = 6.3
INK, MUT, RULE = "#1a1d21", "#5d6670", "#c2c9d2"
# Okabe-Ito, colourblind safe, distinguishable in greyscale by lightness
C = {"dsrl": "#999999", "mi": "#D55E00", "bmd": "#009E73", "mop": "#0072B2"}
MK = {"dsrl": "s", "mi": "^", "bmd": "D", "mop": "o"}
NAME = {"dsrl": "No mode term", "mi": "Mode entropy",
        "bmd": "Learned skills", "mop": "Reference matching"}

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "legend.fontsize": 6.3, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "font.family": "DejaVu Sans",
})


def load(pats, floor):
    """Rows for the figures, through the shared deduplicating loader.

    A level split across jobs by frozen policy can produce a cell twice; counting
    it twice would weight that policy above the others in every panel.
    """
    # unify_reference as well, so the figure and the text are computed the same
    # way: without it the two disagreed in the third decimal.
    return d3il_runs.unify_reference(d3il_runs.load(pats, floor))


def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


# ----------------------------------------------------------------- figure 1
def figure_one():
    fig = plt.figure(figsize=(W, 3.15))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.25], hspace=0.40,
                          left=0.14, right=0.985, top=0.94, bottom=0.11)

    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0.10, 0.99); ax.axis("off")
    ax.set_title("(a)  What is trained and what is compared", loc="left", pad=2)

    def box(x, y, w, h, lab, sub=None, fc="#eef1f5", ec=RULE, lw=0.7, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (h * 0.22 if sub else 0), lab, ha="center",
                va="center", fontsize=6.4, color=tc, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                    fontsize=6.3, color=MUT, zorder=3)

    def arrow(p0, p1, color=INK, ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5,
                                     color=color, lw=0.7, linestyle=ls,
                                     shrinkA=1.5, shrinkB=1.5, zorder=1))

    box(0.00, 0.60, 0.23, 0.33, r"noise policy $\mu_\phi$", "trained",
        fc="#d9e8f4", ec=C["mop"], lw=1.2)
    box(0.30, 0.60, 0.26, 0.33, r"generative policy $v_\theta$", "frozen")
    box(0.63, 0.60, 0.21, 0.33, "rollout", "simulator", fc="#fff")
    arrow((0.23, 0.765), (0.30, 0.765)); arrow((0.56, 0.765), (0.63, 0.765))
    ax.text(0.265, 0.55, r"$z$", ha="center", fontsize=6.3, color=MUT)
    ax.text(0.595, 0.55, r"$a$", ha="center", fontsize=6.3, color=MUT)

    box(0.63, 0.14, 0.21, 0.31, r"$C_\phi$", "policy histogram", fc="#fff",
        ec=C["mop"], lw=1.0)
    box(0.00, 0.14, 0.23, 0.31, r"$C_{\mathrm{ref}}$", "reference, frozen")
    arrow((0.735, 0.60), (0.735, 0.45))
    arrow((0.63, 0.295), (0.23, 0.295), color=C["mop"])
    ax.text(0.43, 0.345, r"penalise $D(C_\phi, C_{\mathrm{ref}})$", ha="center",
            fontsize=6.3, color=C["mop"])

    ax = fig.add_subplot(gs[1])
    MEAS = {"unmodified": [], "0.15": [], "0.30": [], "0.60": []}
    rows = load(["mb_skew*.json"], 0.15)
    seen = {}
    for r in rows:
        k = (r["skew"], r.get("bb_seed"))
        if k not in seen and "C_ref_kl_to_uniform" in r:
            seen[k] = r["C_ref_kl_to_uniform"]
    for (sk, _), v in seen.items():
        MEAS[{"0.00": "unmodified"}.get(sk, sk)].append(v)
    rng = np.random.default_rng(1)
    for i, (lab, vals) in enumerate(MEAS.items()):
        v = np.array(vals)
        ax.scatter(v, np.full_like(v, i) + rng.uniform(-.13, .13, len(v)),
                   s=14, color=C["mop"], alpha=.8, edgecolor="none", zorder=3)
        ax.plot([v.mean()], [i], marker="|", ms=11, mew=1.2, color=INK, zorder=4)
    ax.axvline(0, color=C["mi"], lw=1.2, zorder=2)
    ax.annotate("a uniform policy, where\nthe two objectives agree",
                xy=(0.02, 3.25), xytext=(0.30, 3.45), fontsize=6.3, color=C["mi"],
                va="center", arrowprops=dict(arrowstyle="->", color=C["mi"], lw=0.6))
    ax.set_yticks(range(4))
    ax.set_yticklabels(["unmodified", "0.15 nats", "0.30 nats", "0.60 nats"])
    ax.set_ylabel("demonstration\nweighting", fontsize=6.6)
    ax.set_xlabel(r"measured $D_{\mathrm{KL}}(C_{\mathrm{ref}}\,\|\,u)$   (nats)")
    ax.set_xlim(-0.03, 1.38); ax.set_ylim(-0.6, 3.85)
    ax.set_title("(b)  No cloned policy is uniform", loc="left", pad=2)

    fig.savefig(OUT / "fig1_setting.pdf"); fig.savefig(OUT / "fig1_setting.png")
    print("wrote figures/fig1_setting.{pdf,png}")


# ----------------------------------------------------------------- figure 2
def figure_two():
    fig = plt.figure(figsize=(W, 5.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.35], hspace=0.55,
                          left=0.155, right=0.985, top=0.955, bottom=0.085)

    for row, (title, pats, floor) in enumerate([
            ("(a)  Avoiding", ["mb_skew*.json"], 0.15),
            ("(b)  Pushing", ["pm_skew*.json", "pu_skew*.json"], 0.10)]):
        ax = fig.add_subplot(gs[row])
        rows = load(pats, floor)
        if pats[0].startswith("pm_"):
            # The learned-skill arm appears in both sweeps: pm_ ran it under the
            # fitted skill prior, pu_ under the uniform one. Panel (a) plots the
            # uniform prior, so panel (b) plots it too; mixing the protocols
            # across the two panels would make them incomparable.
            rows = [r for r in rows
                    if r["arm"] != "bmd" or r.get("bmd_prior") != "fitted"]
            if not any(r["arm"] == "bmd" for r in rows):
                raise SystemExit(
                    "panel (b): no uniform-prior learned-skill runs found. "
                    "Plotting the fitted-prior ones instead would put the two "
                    "panels under different protocols, so the figure is not built.")
        sks = sorted({r["skew"] for r in rows},
                     key=lambda s: np.nanmean([tvu(r) for r in rows if r["skew"] == s]))
        xs = [np.nanmean([tvu(r) for r in rows if r["skew"] == s]) for s in sks]
        for arm in ("dsrl", "mi", "bmd", "mop"):
            m, e = [], []
            for s in sks:
                v = np.array([r["tv_to_ref"] for r in rows
                              if r["skew"] == s and r["arm"] == arm])
                m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(len(v)))
            ax.errorbar(xs, m, yerr=e, marker=MK[arm], ms=3.4, capsize=1.6,
                        lw=1.4 if arm == "mop" else 0.9, elinewidth=0.7,
                        color=C[arm], label=NAME[arm],
                        zorder=5 if arm == "mop" else 3)
        floor_v = 0.055 if row == 0 else 0.079
        ax.axhspan(0, floor_v, color=RULE, alpha=.45, zorder=0)
        # Anchored at the left on Pushing: at the right the curves run into the
        # band and the label sat on top of three of them.
        lab_x, lab_ha = ((xs[-1], "right") if row == 0 else (xs[0], "left"))
        ax.text(lab_x, floor_v + 0.004, "sampling floor", ha=lab_ha,
                va="bottom", fontsize=6.3, color=MUT)
        ax.set_ylabel(r"$D_{\mathrm{TV}}(C_\phi, C_{\mathrm{ref}})$", fontsize=6.8)
        ax.set_title(title + "    lower is closer to the pretrained policy",
                     loc="left", pad=2)
        ax.set_ylim(0, 0.56)
        if row == 1:
            ax.legend(frameon=False, loc="upper right", ncol=2,
                      handlelength=1.3, columnspacing=0.9, handletextpad=0.4)
        if row == 1:
            ax.set_xlabel(r"reference skew  $D_{\mathrm{TV}}(C_{\mathrm{ref}}, u)$")

    ax = fig.add_subplot(gs[2])

    def paired(rows, arm):
        """One difference per (weighting, frozen policy), in our favour if positive."""
        bbs = sorted({r.get("bb_seed") for r in rows} - {None})
        d = []
        for s in sorted({r["skew"] for r in rows}):
            for bb in bbs:
                sel = [r for r in rows if r["skew"] == s and r.get("bb_seed") == bb]
                v = [r["tv_to_ref"] for r in sel if r["arm"] == "mop"]
                w = [r["tv_to_ref"] for r in sel if r["arm"] == arm]
                if len(v) > 1 and len(w) > 1:
                    d.append(np.mean(w) - np.mean(v))
        return np.array(d)

    AV = load(["mb_skew*.json"], 0.15)
    PU = [r for r in load(["pm_skew*.json", "pu_skew*.json"], 0.10)
          if r["arm"] != "bmd" or r.get("bmd_prior") != "fitted"]
    ys, labs, cols, lo, hi, wins = [], [], [], [], [], []
    for task, rows in (("Pushing", PU), ("Avoiding", AV)):
        for arm in ("bmd", "mi", "dsrl"):
            d = paired(rows, arm)
            if d.size < 3:
                continue
            sem = d.std(ddof=1) / np.sqrt(d.size)
            t = 2.093 if d.size >= 15 else 2.262
            ys.append(d.mean()); lo.append(t * sem); hi.append(t * sem)
            labs.append(f"{task}, vs {NAME[arm].lower()}"); cols.append(C[arm])
            wins.append(f"{int((d > 0).sum())}/{d.size}")
    y = np.arange(len(ys))[::-1]
    for yy, m, l, h, c in zip(y, ys, lo, hi, cols):
        ax.errorbar([m], [yy], xerr=[[l], [h]], fmt="none", ecolor=c,
                    elinewidth=1.1, capsize=2.4, zorder=2)
        ax.scatter([m], [yy], s=26, color=c, zorder=3, edgecolor="none")
    for yy, m, w in zip(y, ys, wins):
        ax.text(0.335, yy, f"{m:+.3f}   {w}", ha="right", va="center",
                fontsize=6.3, color=INK)
    ax.axvline(0, color=RULE, lw=0.7, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.3)
    ax.set_xlim(-0.06, 0.34); ax.set_ylim(-0.7, len(ys) - 0.3)
    ax.set_xlabel("reduction in " r"$D_{\mathrm{TV}}(C_\phi, C_{\mathrm{ref}})$"
                  ",  paired within each frozen policy")
    ax.set_title("(c)  Each frozen policy as one unit "
                 "(95% CI; policies won)", loc="left", pad=2)

    fig.savefig(OUT / "fig2_result.pdf"); fig.savefig(OUT / "fig2_result.png")
    print("wrote figures/fig2_result.{pdf,png}")


if __name__ == "__main__":
    figure_one(); figure_two()
