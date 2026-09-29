#!/usr/bin/env python3
"""Figure 1: what is frozen, what is trained, and where the three arms diverge.

The experiment is easy to misread as fine-tuning a diffusion policy. It is not:
the backbone's weights never move. Only a noise policy mu_phi(z|o) is trained,
and it steers the frozen decoder by choosing where in its latent space to sample.
Everything the arms disagree about happens after the rollout, on the histogram of
ground-truth behavioural modes -- which is why the three objectives can be drawn
as one shared pipeline with a three-way split at the end.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("figures")
INK, MUTED, RULE = "#171a1f", "#616a78", "#c8cfd8"
FROZEN, TRAINED = "#eef1f5", "#dceaf5"
MOP, MI, DSRL = "#1b6ca8", "#c4562a", "#8c8c8c"

plt.rcParams.update({
    "font.size": 8.5, "figure.dpi": 220, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, sub=None, fc=FROZEN, ec=RULE, lw=1.0, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center",
                fontsize=7, color=MUTED, zorder=3)


def arrow(ax, p0, p1, color=INK, lw=1.1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def main() -> int:
    fig, ax = plt.subplots(figsize=(10.2, 3.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ytop, h = 0.60, 0.20

    box(ax, 0.015, ytop, 0.115, h, r"observation $o$", "env state", fc="#fff")
    box(ax, 0.155, ytop, 0.135, h, r"noise policy $\mu_\phi(z\,|\,o)$", "TRAINED",
        fc=TRAINED, ec=MOP, lw=1.5)
    box(ax, 0.315, ytop, 0.165, h, r"backbone  $v_\theta$", "FROZEN — weights never move")
    box(ax, 0.505, ytop, 0.115, h, r"action $a$", "M-step Euler", fc="#fff")
    box(ax, 0.645, ytop, 0.125, h, "rollout", "D3IL simulator", fc="#fff")
    box(ax, 0.795, ytop, 0.19, h, "ground-truth mode", "geometry, not K-means",
        fc="#fff", ec=MOP, lw=1.3)

    for a, b in ((0.130, 0.155), (0.290, 0.315), (0.480, 0.505),
                 (0.620, 0.645), (0.770, 0.795)):
        arrow(ax, (a, ytop + h / 2), (b, ytop + h / 2))

    ax.text(0.3975, ytop - 0.055, r"$z \sim \mu_\phi \;\longmapsto\; a$   deterministic ODE",
            ha="center", va="top", fontsize=7.4, color=MUTED)

    # histogram, then the three-way split
    ybot, hb = 0.16, 0.185
    box(ax, 0.795, ybot, 0.19, hb, r"mode histogram $C_\phi$",
        "EMA over iterations", fc="#fff", ec=MOP, lw=1.3)
    arrow(ax, (0.890, ytop), (0.890, ybot + hb), rad=0.0)

    box(ax, 0.505, ybot, 0.145, hb, r"reference $C_{ref}$",
        "frozen backbone, measured", fc=FROZEN)
    arrow(ax, (0.795, ybot + hb / 2), (0.650, ybot + hb / 2), color=MUTED, ls=(0, (3, 2)))

    objs = [
        (0.015, DSRL, r"DSRL:  $\max\;\mathbb{E}[r]$", "no mode term"),
        (0.015, None, None, None),
    ]
    box(ax, 0.015, 0.735, 0.33, 0.115, "", None, fc="none", ec="none")  # spacer, unused

    ax.text(0.012, ybot + hb + 0.055, "three objectives, identical everywhere else:",
            fontsize=8, color=INK, ha="left", va="bottom")
    rows = [
        (DSRL, r"DSRL", r"$\max\ \mathbb{E}[r]$", "no mode term"),
        (MOP, r"MoP", r"$\max\ \mathbb{E}[r]-\lambda D(C_\phi\,\|\,C_{ref})$", "anchor to the backbone"),
        (MI, r"MI", r"$\max\ \mathbb{E}[r]+\lambda H(C_\phi)$", "pull toward uniform"),
    ]
    for i, (col, name, expr, note) in enumerate(rows):
        yy = ybot + hb - 0.045 - i * 0.062
        ax.add_patch(FancyBboxPatch((0.012, yy - 0.019), 0.010, 0.038,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    facecolor=col, edgecolor="none", zorder=3))
        ax.text(0.034, yy, name, fontsize=8.2, color=col, va="center", ha="left")
        ax.text(0.088, yy, expr, fontsize=8.6, color=INK, va="center", ha="left")
        ax.text(0.355, yy, note, fontsize=7.4, color=MUTED, va="center", ha="left")

    arrow(ax, (0.505, ybot + hb / 2), (0.475, ybot + hb / 2), color=MUTED)

    ax.text(0.012, 0.045,
            r"$D$ is the anchoring divergence. KL's coefficient $\log(p_m/q_m)+1$ is unbounded as "
            r"$q_m\!\to\!0$; TV and JS are not — see Fig. 3.",
            fontsize=7.4, color=MUTED, ha="left", va="center")

    fig.savefig(OUT / "fig0_method.png")
    fig.savefig(OUT / "fig0_method.pdf")
    print("wrote figures/fig0_method.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
