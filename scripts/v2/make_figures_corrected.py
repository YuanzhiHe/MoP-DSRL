#!/usr/bin/env python3
"""The paper's two figures, drawn from the corrected cohort.

Authored at the measured text width so the figure is placed at scale one and
the type on the page is the type set here. Every number comes from the run
files the analysis scripts read, so a figure and the table beside it cannot
disagree.
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402

RUNS = ROOT / "runs" / "_d3il_v2"
# Name prefix of the run files the analysis reads. The matrix was measured
# again after the environment rule changed, and both cohorts sit in the same
# directory; the prefix is what keeps the two apart, so a bare invocation
# reads the current one.
PREFIX = "v22_"

OUT = ROOT / "paper" / "figures"
W = 5.4167                    # elsarticle preprint text width, 390 pt, measured
INK, MUT, RULE = "#1a1d21", "#5d6670", "#c2c9d2"
# Okabe-Ito, colourblind safe, distinguishable in greyscale by lightness
C = {"dsrl": "#999999", "mi": "#D55E00", "bmd": "#009E73", "mop": "#0072B2"}
MK = {"dsrl": "s", "mi": "^", "bmd": "D", "mop": "o"}
NAME = {"dsrl": "No mode term", "mi": "Mode entropy",
        "bmd": "Learned skills", "mop": "Reference matching"}
# Read the floor the frozen policies measured, so the band and the text cannot
# drift apart. Each frozen policy is evaluated under its own latent prior and
# scored against its own reference, which puts both histograms under one
# distribution and carries whatever the two measurement paths differ by.
FLOOR = {k: v["overall"]["tv_to_ref"]["mean"] for k, v in json.loads(
    (ROOT / "runs" / "_d3il_v2" / "frozen2_report.json").read_text()).items()
    if "/" not in k}

plt.rcParams.update({
    "font.size": 7.4, "axes.labelsize": 7.4, "axes.titlesize": 8,
    "legend.fontsize": 6.6, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "figure.dpi": 400, "savefig.bbox": None, "savefig.pad_inches": 0.0,
    "font.family": "DejaVu Sans",
})


def load(task):
    rows = []
    for f in sorted(glob.glob(str(RUNS / f"{PREFIX}main_{task}_*.jsonl"))):
        rows += [json.loads(l) for l in Path(f).open() if l.strip()]
    for r in rows:
        v = r.get("tv_to_ref")
        r["closeness_tv"] = None if v is None else -float(v)
    return rows


def skew_tv(r):
    q = np.asarray(r["C_ref"], float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum())


def skew_kl(r):
    q = np.asarray(r["C_ref"], float)
    nz = q[q > 0]
    return float(np.sum(nz * np.log(nz * q.size)))


def policy_of(r):
    return (r["task"], r["weighting"], r["backbone_seed"])


def figure_one():
    fig = plt.figure(figsize=(W, 3.45))
    # The schematic needs only the height its boxes occupy; the rest goes to
    # the forty measured policies, which is what the panel is about.
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.38], hspace=0.17,
                          left=0.165, right=0.985, top=0.95, bottom=0.105)

    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0.10, 0.99)
    ax.axis("off")
    ax.set_title("(a)  What is trained and what is compared", loc="left", pad=2)

    def box(x, y, w, h, lab, sub=None, fc="#eef1f5", ec=RULE, lw=0.7, tc=INK):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (h * 0.22 if sub else 0), lab,
                ha="center", va="center", fontsize=6.4, color=tc, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                    fontsize=6.6, color=MUT, zorder=3)

    def arrow(p0, p1, color=INK):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5,
                                     color=color, lw=0.7, shrinkA=1.5,
                                     shrinkB=1.5, zorder=1))

    box(0.00, 0.60, 0.23, 0.33, r"noise policy $\mu_\phi$", "trained",
        fc="#d9e8f4", ec=C["mop"], lw=1.2)
    box(0.30, 0.60, 0.26, 0.33, r"generative policy $v_\theta$", "frozen")
    box(0.63, 0.60, 0.21, 0.33, "rollout", "simulator", fc="#fff")
    arrow((0.23, 0.765), (0.30, 0.765)); arrow((0.56, 0.765), (0.63, 0.765))
    ax.text(0.265, 0.55, r"$z$", ha="center", fontsize=6.6, color=MUT)
    ax.text(0.595, 0.55, r"$a$", ha="center", fontsize=6.6, color=MUT)
    box(0.63, 0.14, 0.21, 0.31, r"$C_\phi$", "policy histogram", fc="#fff",
        ec=C["mop"], lw=1.0)
    box(0.00, 0.14, 0.23, 0.31, r"$C_{\mathrm{ref}}$", "reference, frozen")
    arrow((0.735, 0.60), (0.735, 0.45))
    arrow((0.63, 0.295), (0.23, 0.295), color=C["mop"])
    ax.text(0.43, 0.345, r"penalise $D(C_\phi, C_{\mathrm{ref}})$",
            ha="center", fontsize=6.6, color=C["mop"])

    ax = fig.add_subplot(gs[1])
    # One point per frozen policy, both tasks, so the panel carries all forty.
    groups, labels = [], []
    for task in ("pushing", "avoiding"):
        per = {}
        for r in load(task):
            per.setdefault(policy_of(r), skew_kl(r))
        byw = collections.defaultdict(list)
        for (_, w, _), v in per.items():
            byw[w].append(v)
        for w in sorted(byw, reverse=True):
            groups.append(np.array(byw[w]))
            labels.append(f"{task[:2].upper()}  {w}")
    rng = np.random.default_rng(1)
    for i, v in enumerate(groups):
        ax.scatter(v, np.full_like(v, i) + rng.uniform(-.14, .14, len(v)),
                   s=13, color=C["mop"], alpha=.85, edgecolor="none", zorder=3)
        ax.plot([v.mean()], [i], marker="|", ms=10, mew=1.2, color=INK, zorder=4)
    ax.axvline(0, color=C["mi"], lw=1.2, zorder=2)
    # Placed in the empty upper right, so the leader crosses no point.
    ax.annotate("a uniform policy, where\nthe two objectives agree",
                xy=(0.02, 7.25), xytext=(0.62, 7.25), fontsize=6.6,
                color=C["mi"], va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=C["mi"], lw=0.6,
                                shrinkA=2, shrinkB=1))
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_ylabel("task and\ndemonstration weighting", fontsize=6.6)
    ax.set_xlabel(r"measured $D_{\mathrm{KL}}(C_{\mathrm{ref}}\,\|\,u)$   (nats)")
    ax.set_xlim(-0.03, 1.40); ax.set_ylim(-0.7, len(groups) + 0.75)
    ax.set_title("(b)  None of the forty cloned policies is uniform",
                 loc="left", pad=2)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_setting.pdf"); fig.savefig(OUT / "fig1_setting.png")
    print("wrote fig1_setting.{pdf,png}")


def figure_two():
    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.06, 1.28], hspace=0.40,
                          left=0.185, right=0.985, top=0.955, bottom=0.085)

    data = {t: load(t) for t in ("avoiding", "pushing")}
    for row, task in enumerate(("avoiding", "pushing")):
        ax = fig.add_subplot(gs[row])
        rows = data[task]
        ws = sorted({r["weighting"] for r in rows},
                    key=lambda w: np.mean([skew_tv(r) for r in rows
                                           if r["weighting"] == w]))
        xs = [np.mean([skew_tv(r) for r in rows if r["weighting"] == w])
              for w in ws]
        for arm in ("dsrl", "mi", "bmd", "mop"):
            m, e = [], []
            for w in ws:
                v = np.array([r["tv_to_ref"] for r in rows
                              if r["weighting"] == w and r["arm"] == arm
                              and r.get("fidelity_defined")])
                m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(v.size))
            ax.errorbar(xs, m, yerr=e, marker=MK[arm], ms=3.4, capsize=1.6,
                        lw=1.4 if arm == "mop" else 0.9, elinewidth=0.7,
                        color=C[arm], label=NAME[arm],
                        zorder=5 if arm == "mop" else 3)
        ax.axhspan(0, FLOOR[task], color=RULE, alpha=.45, zorder=0)
        lab_x, lab_ha = ((xs[-1], "right") if row == 0 else (xs[0], "left"))
        ax.text(lab_x, FLOOR[task] + 0.005, "sampling floor", ha=lab_ha,
                va="bottom", fontsize=6.6, color=MUT)
        ax.set_ylabel(r"$D_{\mathrm{TV}}(C_\phi, C_{\mathrm{ref}})$",
                      fontsize=6.8)
        ax.set_title(f"({'ab'[row]})  {task.capitalize()}"
                     "    lower is closer to the pretrained policy",
                     loc="left", pad=2)
        ax.set_ylim(0, 0.56)
        if row == 0:
            ax.legend(frameon=False, loc="lower left", ncol=2,
                      handlelength=1.3, columnspacing=0.9, handletextpad=0.4)
        else:
            ax.set_xlabel(r"reference skew  $D_{\mathrm{TV}}(C_{\mathrm{ref}}, u)$")

    ax = fig.add_subplot(gs[2])
    ys, labs, cols, half, wins, tasks = [], [], [], [], [], []
    # Avoiding is drawn first so that it occupies the top block, which is where
    # the reader enters the panel.
    for task in ("avoiding", "pushing"):
        scored = [r for r in data[task] if r.get("fidelity_defined")]
        fam = [S.paired(scored, "closeness_tv", arm, "mop", policy_of,
                        lambda r: r["arm"], "greater")
               for arm in ("bmd", "mi", "dsrl")]
        S.holm(fam)
        for arm, c in zip(("bmd", "mi", "dsrl"), fam):
            # The critical value follows this contrast's own count, so a
            # contrast on fewer policies is drawn wider rather than equal.
            ys.append(c.mean); half.append(sps.t.ppf(0.975, c.df) * c.se)
            labs.append(NAME[arm].lower()); cols.append(C[arm])
            tasks.append(task.capitalize()); wins.append(f"{c.wins}/{c.n}")
    y = np.arange(len(ys))[::-1]
    for yy, m, h, c in zip(y, ys, half, cols):
        ax.errorbar([m], [yy], xerr=[[h], [h]], fmt="none", ecolor=c,
                    elinewidth=1.1, capsize=2.4, zorder=2)
        ax.scatter([m], [yy], s=26, color=c, zorder=3, edgecolor="none")
    # The labels sit in a reserved strip to the right of every interval, so a
    # wide interval cannot run under its own number.
    right = max(m + h for m, h in zip(ys, half))
    for yy, m, w in zip(y, ys, wins):
        ax.text(right + 0.015, yy, f"{m:+.3f}   {w}", ha="left", va="center",
                fontsize=6.6, color=INK)
    ax.axvline(0, color=RULE, lw=0.7, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.6)
    for t in sorted(set(tasks)):
        rt = [yy for yy, tt in zip(y, tasks) if tt == t]
        ax.text(0.002, max(rt) + 0.42, t, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.4, color=INK)
        ax.axhline(min(rt) - 0.45, color=RULE, lw=0.6, zorder=0)
    ax.set_xlim(-0.17, right + 0.145); ax.set_ylim(-0.95, len(ys) + 0.1)
    ax.set_xlabel("reduction in " r"$D_{\mathrm{TV}}(C_\phi, C_{\mathrm{ref}})$"
                  ",  paired within each frozen policy")
    ax.set_title("(c)  Reduction per frozen policy (95% CI; policies won)",
                 loc="left", pad=2)

    fig.savefig(OUT / "fig2_result.pdf"); fig.savefig(OUT / "fig2_result.png")
    print("wrote fig2_result.{pdf,png}")


if __name__ == "__main__":
    figure_one()
    figure_two()
