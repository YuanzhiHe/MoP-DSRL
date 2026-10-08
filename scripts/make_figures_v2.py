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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("figures")
W = 5.15                      # sn-jnl text width, measured
INK, MUT, RULE = "#1a1d21", "#5d6670", "#c2c9d2"
# Okabe-Ito, colourblind safe, distinguishable in greyscale by lightness
C = {"dsrl": "#999999", "mi": "#D55E00", "bmd": "#009E73", "mop": "#0072B2"}
MK = {"dsrl": "s", "mi": "^", "bmd": "D", "mop": "o"}
NAME = {"dsrl": "No mode term", "mi": "Mode entropy",
        "bmd": "Learned skills", "mop": "Reference matching"}

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "legend.fontsize": 6.2, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "font.family": "DejaVu Sans",
})


def load(pats, floor):
    rows = []
    for p in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{p}")):
            rows += json.load(open(f))
    return [r for r in rows if r["SR"] >= floor]


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
                    fontsize=5.4, color=MUT, zorder=3)

    def arrow(p0, p1, color=INK, ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5,
                                     color=color, lw=0.7, linestyle=ls,
                                     shrinkA=1.5, shrinkB=1.5, zorder=1))

    box(0.00, 0.60, 0.23, 0.33, r"noise policy $\mu_\phi$", "trained",
        fc="#d9e8f4", ec=C["mop"], lw=1.2)
    box(0.30, 0.60, 0.26, 0.33, r"generative policy $v_\theta$", "frozen")
    box(0.63, 0.60, 0.21, 0.33, "rollout", "simulator", fc="#fff")
    arrow((0.23, 0.765), (0.30, 0.765)); arrow((0.56, 0.765), (0.63, 0.765))
    ax.text(0.265, 0.55, r"$z$", ha="center", fontsize=6, color=MUT)
    ax.text(0.595, 0.55, r"$a$", ha="center", fontsize=6, color=MUT)

    box(0.63, 0.14, 0.21, 0.31, r"$C_\phi$", "policy histogram", fc="#fff",
        ec=C["mop"], lw=1.0)
    box(0.00, 0.14, 0.23, 0.31, r"$C_{\mathrm{ref}}$", "reference, frozen")
    arrow((0.735, 0.60), (0.735, 0.45))
    arrow((0.63, 0.295), (0.23, 0.295), color=C["mop"])
    ax.text(0.43, 0.345, r"penalise $D(C_\phi, C_{\mathrm{ref}})$", ha="center",
            fontsize=6.2, color=C["mop"])

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
                xy=(0.02, 3.25), xytext=(0.30, 3.45), fontsize=5.8, color=C["mi"],
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
    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.85], hspace=0.62,
                          left=0.155, right=0.985, top=0.955, bottom=0.085)

    for row, (title, pats, floor) in enumerate([
            ("(a)  Avoiding", ["mb_skew*.json"], 0.15),
            ("(b)  Pushing", ["pushrw_skew*.json", "bmdP_skew*.json"], 0.10)]):
        ax = fig.add_subplot(gs[row])
        rows = load(pats, floor)
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
        ax.text(xs[-1], floor_v, "sampling floor", ha="right", va="bottom",
                fontsize=5.4, color=MUT)
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
    rows = load(["mb_skew*.json"], 0.15)
    bbs = sorted({r.get("bb_seed") for r in rows} - {None})
    sks = sorted({r["skew"] for r in rows})
    ys, labs, cols, lo, hi, wins = [], [], [], [], [], []
    for arm in ("dsrl", "mi", "bmd"):
        d = []
        for s in sks:
            for bb in bbs:
                v = [r["tv_to_ref"] for r in rows if r["skew"] == s
                     and r.get("bb_seed") == bb and r["arm"] == "mop"]
                w = [r["tv_to_ref"] for r in rows if r["skew"] == s
                     and r.get("bb_seed") == bb and r["arm"] == arm]
                if len(v) > 1 and len(w) > 1:
                    d.append(np.mean(w) - np.mean(v))
        d = np.array(d); sem = d.std(ddof=1) / np.sqrt(len(d))
        t = 2.093                                   # t_{0.975, 19}
        ys.append(d.mean()); lo.append(t * sem); hi.append(t * sem)
        labs.append(NAME[arm]); cols.append(C[arm])
        wins.append(f"{int((d > 0).sum())}/{len(d)}")
    y = np.arange(len(ys))[::-1]
    for yy, m, l, h, c in zip(y, ys, lo, hi, cols):
        ax.errorbar([m], [yy], xerr=[[l], [h]], fmt="none", ecolor=c,
                    elinewidth=1.1, capsize=2.4, zorder=2)
        ax.scatter([m], [yy], s=26, color=c, zorder=3, edgecolor="none")
    for yy, m, w in zip(y, ys, wins):
        ax.text(0.287, yy, f"{m:+.3f}   {w}", ha="right", va="center",
                fontsize=6.1, color=INK)
    ax.axvline(0, color=RULE, lw=0.7, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([f"vs {l.lower()}" for l in labs], fontsize=6.3)
    ax.set_xlim(0, 0.30); ax.set_ylim(-0.7, len(ys) - 0.3)
    ax.set_xlabel("reduction in " r"$D_{\mathrm{TV}}(C_\phi, C_{\mathrm{ref}})$"
                  ",  paired within each frozen policy")
    ax.set_title("(c)  Each frozen policy as one unit, Avoiding "
                 "(95% CI; policies won)", loc="left", pad=2)

    fig.savefig(OUT / "fig2_result.pdf"); fig.savefig(OUT / "fig2_result.png")
    print("wrote figures/fig2_result.{pdf,png}")


if __name__ == "__main__":
    figure_one(); figure_two()
