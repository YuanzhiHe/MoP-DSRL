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
NAME = {"dsrl": "No mode term", "mop": "Reference anchor (ours)",
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
    d = tvu(r)
    return (r["tv_to_uniform"] - r["tv_to_ref"]) / d if d and np.isfinite(d) and d > 1e-9 else np.nan


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

    yb, hb = 0.205, 0.215
    box(ax, 0.810, yb, 0.185, hb, r"policy histogram $C_\phi$", "running estimate", fc="#fff",
        ec=C["mop"], lw=1.2)
    arr(ax, (0.9025, yt), (0.9025, yb + hb))
    box(ax, 0.600, yb, 0.165, hb, r"reference $C_{\mathrm{ref}}$", "measured once, frozen")
    arr(ax, (0.810, yb + hb / 2), (0.765, yb + hb / 2), color=MUT, ls=(0, (3, 2)))

    ax.text(0.005, yb + hb + 0.02, "objectives compared, identical in every other respect",
            fontsize=7.4, color=INK, ha="left", va="bottom")
    rows = [("mop", r"$\max\;\mathbb{E}[r]-\lambda\,D(C_\phi,C_{\mathrm{ref}})$", "holds the pretrained profile"),
            ("bmd", r"$\max\;\mathbb{E}[r]+\lambda\,\hat{I}(k;s)$", "discovers its own skills"),
            ("mi",  r"$\max\;\mathbb{E}[r]+\lambda\,H(C_\phi)$", "drives toward uniform"),
            ("dsrl", r"$\max\;\mathbb{E}[r]$", "no mode term")]
    for i, (k, expr, note) in enumerate(rows):
        yy = yb + hb - 0.055 - i * 0.055
        ax.add_patch(FancyBboxPatch((0.005, yy - 0.016), 0.009, 0.032,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    facecolor=C[k], edgecolor="none", zorder=4))
        ax.text(0.024, yy, NAME[k], fontsize=7.0, color=C[k], va="center")
        ax.text(0.215, yy, expr, fontsize=7.6, color=INK, va="center")
        ax.text(0.405, yy, note, fontsize=6.6, color=MUT, va="center")

    # ---------------- (b) the identity ----------------
    ax = fig.add_subplot(gs[0, 2])
    K = 6
    rs = np.linspace(0.05, 1.0, 160)
    kl, ent = [], []
    for r in rs:
        p = r ** np.arange(K, dtype=float); p /= p.sum()
        kl.append(np.sum(p * np.log(p * K))); ent.append(-np.sum(p * np.log(p)))
    kl, ent = np.array(kl), np.array(ent)
    ax.plot(kl, np.log(K) - ent, lw=4.0, color=C["mop"], alpha=.35,
            solid_capstyle="round", label=r"$\ln K-H(p)$   (entropy objective)")
    ax.plot(kl, kl, lw=1.4, ls=(0, (4, 3)), color=C["mi"],
            label=r"$D_{\mathrm{KL}}(p\,\|\,u)$   (anchoring objective)")
    ax.set_xlabel(r"$D_{\mathrm{KL}}(p\,\|\,u)$  (nats)")
    ax.set_ylabel(r"$\ln K-H(p)$  (nats)")
    ax.set_title("(b)  The two objectives coincide\n       when the reference is uniform",
                 loc="left", pad=4)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.30, 0.30, "the curves lie on top of one another:\nanchoring and entropy can differ only\nonce the pretrained policy is skewed",
            transform=ax.transAxes, fontsize=6.8, color=MUT, va="top")

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
                        color=C[arm], label=NAME[arm],
                        zorder=5 if arm == "mop" else 3)
        ax.axhline(0, color=RULE, lw=0.8, ls=":")
        ax.set_xlabel(r"reference skew  $\mathrm{TV}(C_{\mathrm{ref}},u)$")
        if j == 0:
            ax.set_ylabel("normalised fidelity\n" r"$[\mathrm{TV}(C_\phi,u)-\mathrm{TV}(C_\phi,C_{\mathrm{ref}})]\,/\,\mathrm{TV}(C_{\mathrm{ref}},u)$",
                          fontsize=7.0)
            ax.legend(frameon=False, loc="upper left", handlelength=1.4,
                      bbox_to_anchor=(0.015, 0.99), labelspacing=0.3)
        ax.set_title(title + "   higher is closer to the pretrained policy", loc="left", pad=4)
        ax.set_ylim(-0.08, 1.0)

    # ---------------- (e) pooled effect sizes ----------------
    ax = fig.add_subplot(gs[1, 2])
    tasks = ["Avoiding", "Pushing"]
    d_vals = {"mi": [0.91, 1.14], "dsrl": [0.38, 0.71], "bmd": [0.31, 0.54]}
    p_vals = {"mi": [1e-4, 1e-4], "dsrl": [0.0019, 0.0010], "bmd": [0.0086, 0.0078]}
    x = np.arange(len(tasks)); w = 0.26
    for i, arm in enumerate(("mi", "dsrl", "bmd")):
        off = (i - 1) * w
        ax.bar(x + off, d_vals[arm], w, color=C[arm],
               label=f"vs {NAME[arm].lower()}")
        for xi, (v, pv) in enumerate(zip(d_vals[arm], p_vals[arm])):
            ax.text(xi + off, v + 0.035, f"{v:.2f}", ha="center", fontsize=6.9, color=INK)
            ax.text(xi + off, v + 0.115,
                    "p<0.001" if pv < 1e-3 else f"p={pv:.3f}",
                    ha="center", fontsize=5.9, color=MUT)
    for yy, t in ((0.2, "small"), (0.5, "medium"), (0.8, "large")):
        ax.axhline(yy, color=RULE, lw=0.7, ls=":", zorder=0)
        ax.text(len(tasks) - 0.13, yy + 0.015, t, fontsize=6.2, color=MUT, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_xlim(-0.52, len(tasks) - 0.10)
    ax.set_ylabel("Cohen's $d$  (advantage of the reference anchor)", fontsize=7.2)
    ax.set_ylim(0, 1.52)
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, labelspacing=0.3,
              bbox_to_anchor=(-0.01, 1.0))
    ax.set_title("(e)  Pooled over reference skew, every\n       comparison favours anchoring",
                 loc="left", pad=6)

    fig.savefig(OUT / "fig1_main.png")
    fig.savefig(OUT / "fig1_main.pdf")
    print("wrote figures/fig1_main.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
