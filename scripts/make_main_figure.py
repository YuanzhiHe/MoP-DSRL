#!/usr/bin/env python3
"""Figure 1: the setting, the identity that motivates it, and the main result."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT = Path("figures")
INK, MUT, RULE = "#1a1d21", "#5d6670", "#c9d0d8"
C = {"dsrl": "#8a9099", "mop": "#1b6ca8", "mi": "#c4562a", "bmd": "#2e8b57"}
NAME = {"dsrl": "No mode term", "mop": "Reference matching",
        "mi": "Mode entropy", "bmd": "Learned skills"}
FROZEN, TRAIN = "#edf1f5", "#d9e8f4"

plt.rcParams.update({
    "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.6,
    "legend.fontsize": 7.4, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 240, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


def gn(r):
    """Distance to the pretrained behaviour distribution; lower is closer."""
    return r["tv_to_ref"]


def load(pats, floor):
    rows = []
    for pat in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{pat}")):
            rows += json.load(open(f))
    return [r for r in rows if r["SR"] >= floor]


def box(ax, x, y, w, h, lab, sub=None, fc=FROZEN, ec=RULE, lw=1.0, fs=7.4, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.055 if sub else 0), lab, ha="center", va="center",
            fontsize=fs, color=tc or INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.075, sub, ha="center", va="center",
                fontsize=6.3, color=MUT, zorder=4)


def arr(ax, p0, p1, color=INK, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls, zorder=2,
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    fig = plt.figure(figsize=(11.6, 6.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.12],
                  hspace=0.42, wspace=0.42, left=0.055, right=0.985,
                  top=0.955, bottom=0.085)

    # ---------------- (a) pipeline ----------------
    ax = fig.add_subplot(gs[0, :2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a)  Only the noise policy is trained; the generative policy is frozen",
                 loc="left", pad=4)
    yt, h = 0.615, 0.235
    box(ax, 0.005, yt, 0.125, h, "observation", "environment state", fc="#fff")
    box(ax, 0.150, yt, 0.165, h, r"noise policy $\mu_\phi$", "the only trained part",
        fc=TRAIN, ec=C["mop"], lw=1.5)
    box(ax, 0.335, yt, 0.175, h, r"generative policy $v_\theta$", "frozen throughout")
    box(ax, 0.530, yt, 0.115, h, "action", "ODE integration", fc="#fff")
    box(ax, 0.665, yt, 0.125, h, "rollout", "simulator", fc="#fff")
    box(ax, 0.810, yt, 0.185, h, "behavioural mode", "geometric label", fc="#fff",
        ec=C["mop"], lw=1.2)
    for a, b in ((0.130, 0.150), (0.315, 0.335), (0.510, 0.530), (0.645, 0.665), (0.790, 0.810)):
        arr(ax, (a, yt + h / 2), (b, yt + h / 2))
    ax.text(0.4225, yt - 0.045, r"$z\sim\mu_\phi(\cdot\mid o)\;\longrightarrow\;a$",
            ha="center", va="top", fontsize=7.0, color=MUT)

    yb, hb = 0.300, 0.200
    box(ax, 0.810, yb, 0.185, hb, r"policy histogram $C_\phi$", "running estimate", fc="#fff",
        ec=C["mop"], lw=1.2)
    arr(ax, (0.9025, yt), (0.9025, yb + hb))
    box(ax, 0.600, yb, 0.165, hb, r"reference $C_{\mathrm{ref}}$", "measured once, frozen")
    arr(ax, (0.810, yb + hb / 2), (0.765, yb + hb / 2), color=MUT, ls=(0, (3, 2)))
    ax.text(0.6825, yb - 0.035, "compared each update", ha="center", va="top",
            fontsize=6.3, color=MUT)

    ax.text(0.005, 0.245, "objectives compared, identical in every other respect",
            fontsize=7.4, color=INK, ha="left", va="bottom")
    rows = [("mop", r"$\max\;\mathbb{E}[r]-\lambda\,D(C_\phi,C_{\mathrm{ref}})$", "matches the pretrained profile"),
            ("bmd", r"$\max\;\mathbb{E}[r]+\lambda\,\hat{I}(k;s)$", "discovers its own skills"),
            ("mi",  r"$\max\;\mathbb{E}[r]+\lambda\,H(C_\phi)$", "drives toward uniform"),
            ("dsrl", r"$\max\;\mathbb{E}[r]$", "no mode term")]
    for i, (k, expr, note) in enumerate(rows):
        yy = 0.195 - i * 0.052
        ax.add_patch(FancyBboxPatch((0.005, yy - 0.016), 0.009, 0.032,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    facecolor=C[k], edgecolor="none", zorder=4))
        ax.text(0.024, yy, NAME[k] + ("  (ours)" if k == "mop" else ""),
                fontsize=6.9, color=C[k], va="center",
                fontweight="bold" if k == "mop" else "normal")
        ax.text(0.245, yy, expr, fontsize=7.3, color=INK, va="center")
        ax.text(0.445, yy, note, fontsize=6.4, color=MUT, va="center")

    # ---------------- (b) measured reference skew ----------------
    ax = fig.add_subplot(gs[0, 2])
    # KL(C_ref || u) measured over 900 rollouts per frozen policy, six route groups
    MEAS = {"unmodified": [0.154, 0.013, 0.165, 0.395, 0.276],
            "0.15": [0.376, 0.215, 0.444, 0.199, 0.371],
            "0.30": [0.290, 0.303, 0.870, 1.028, 0.616],
            "0.60": [0.789, 1.278, 0.666, 0.846, 0.671]}
    rng2 = np.random.default_rng(1)
    for i, (lab, vals) in enumerate(MEAS.items()):
        v = np.array(vals)
        ax.scatter(v, np.full_like(v, i) + rng2.uniform(-.10, .10, len(v)),
                   s=30, color=C["mop"], alpha=.75, edgecolor="none", zorder=3)
        ax.plot([v.mean()], [i], marker="|", ms=14, color=INK, zorder=4)
    ax.axvline(0, color=C["mi"], lw=1.6, zorder=2)
    ax.text(0.012, 3.42, "a uniform pretrained policy,\nwhere the two objectives\nare the same function",
            fontsize=6.6, color=C["mi"], va="top")
    ax.set_yticks(range(len(MEAS)))
    ax.set_yticklabels([f"demonstrations\n{k}" if k == "unmodified" else f"reweighted\nto {k} nats"
                        for k in MEAS], fontsize=6.6)
    ax.set_xlabel(r"measured $D_{\mathrm{KL}}(C_{\mathrm{ref}}\,\|\,u)$  (nats)")
    ax.set_xlim(-0.06, 1.38)
    ax.set_ylim(-0.6, 3.75)
    ax.set_title("(b)  No cloned policy is uniform,\n       so the two objectives differ",
                 loc="left", pad=6)

    # ---------------- (c, d) per-task results ----------------
    specs = [("(c)  Avoiding", ["mb_skew*.json"], 0.15, None),
             ("(d)  Pushing", ["pushrw_skew*.json", "bmdP_skew*.json"], 0.10, None)]
    for j, (title, pats, floor, _) in enumerate(specs):
        ax = fig.add_subplot(gs[1, j])
        rows_ = load(pats, floor)
        skews = sorted({r["skew"] for r in rows_},
                       key=lambda s: np.nanmean([tvu(r) for r in rows_ if r["skew"] == s]))
        xs = [np.nanmean([tvu(r) for r in rows_ if r["skew"] == s]) for s in skews]
        for arm in ("dsrl", "mi", "bmd", "mop"):
            m, e = [], []
            for s in skews:
                v = np.array([gn(r) for r in rows_ if r["skew"] == s and r["arm"] == arm])
                v = v[np.isfinite(v)]
                m.append(v.mean() if len(v) else np.nan)
                e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            ax.errorbar(xs, m, yerr=e, marker="o", ms=4.2, capsize=2.4,
                        lw=2.0 if arm == "mop" else 1.2,
                        color=C[arm],
                        label=NAME[arm] + (" (ours)" if arm == "mop" else ""),
                        zorder=5 if arm == "mop" else 3)
        ax.set_xlabel(r"reference skew  $\mathrm{TV}(C_{\mathrm{ref}},u)$")
        if j == 0:
            ax.set_ylabel(r"$\mathrm{TV}(C_\phi,\,C_{\mathrm{ref}})$", fontsize=8.6)
            ax.legend(frameon=False, loc="upper right", handlelength=1.4,
                      labelspacing=0.3)
        ax.set_title(title + "   lower is closer to the pretrained policy", loc="left", pad=4)
        ax.set_ylim(0, 0.60)

    # ---------------- (e) paired differences across frozen policies ----------
    ax = fig.add_subplot(gs[1, 2])
    rows_ = load(["mb_skew*.json"], 0.15)
    bbs = sorted({r.get("bb_seed") for r in rows_} - {None})
    skews_ = sorted({r["skew"] for r in rows_})
    labels, means, los, his, wins, cols = [], [], [], [], [], []
    for arm in ("dsrl", "mi", "bmd"):
        d = []
        for s_ in skews_:
            for bb in bbs:
                v = [r["tv_to_ref"] for r in rows_
                     if r["skew"] == s_ and r.get("bb_seed") == bb and r["arm"] == "mop"]
                w = [r["tv_to_ref"] for r in rows_
                     if r["skew"] == s_ and r.get("bb_seed") == bb and r["arm"] == arm]
                if len(v) > 1 and len(w) > 1:
                    d.append(np.mean(w) - np.mean(v))
        d = np.array(d)
        sem = d.std(ddof=1) / np.sqrt(len(d))
        labels.append(f"vs {NAME[arm].lower()}")
        means.append(d.mean()); los.append(1.96 * sem); his.append(1.96 * sem)
        wins.append(f"{int((d > 0).sum())}/{len(d)}")
        cols.append(C[arm])
    y = np.arange(len(means))[::-1]
    ax.errorbar(means, y, xerr=[los, his], fmt="none", ecolor=INK, elinewidth=1.1,
                capsize=3, zorder=2)
    ax.scatter(means, y, s=55, c=cols, zorder=3, edgecolor="none")
    for yy, m, w in zip(y, means, wins):
        ax.text(m, yy + 0.26, f"{m:+.3f}   {w} policies", ha="center",
                fontsize=6.9, color=INK)
    ax.axvline(0, color=RULE, lw=0.9, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_ylim(-0.6, len(means) - 0.25)
    ax.set_xlim(0, 0.30)
    ax.set_xlabel("reduction in " r"$\mathrm{TV}(C_\phi,\,C_{\mathrm{ref}})$"
                  "\npaired within each frozen policy, 95% CI")
    ax.set_title("(e)  Each frozen policy as one unit,\n       Avoiding", loc="left", pad=6)

    fig.savefig(OUT / "fig1_main.png")
    fig.savefig(OUT / "fig1_main.pdf")
    print("wrote figures/fig1_main.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
