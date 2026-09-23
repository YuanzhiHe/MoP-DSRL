#!/usr/bin/env python3
"""Figure: why KL anchoring is unstable and what bounds it.

The algorithmic contribution in one page. Panel (a) is analytic -- the
per-episode coefficient each divergence assigns to a visited mode, as that
mode's reference probability shrinks. Panel (b) evaluates the same coefficients
on the reference histogram actually measured from the skewed backbone, so the
range is the one the optimizer really meets. Panel (c), when divergence-sweep
results are present, is the empirical consequence: coefficient magnitude against
retained mode entropy, per run.
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
# C_ref measured from the skew-0.60 backbone over 1200 rollouts, 6 route groups
C_REF_060 = np.array([0.654, 0.213, 0.077, 0.044, 0.012, 0.000])
DCOL = {"kl": "#b3282d", "tv": "#1b6ca8", "js": "#2e8b57"}
DLAB = {"kl": r"KL  $\log(p/q)+1$", "tv": r"TV  $\frac{1}{2}\,\mathrm{sgn}(p-q)$",
        "js": r"JS  $\frac{1}{2}\log(p/m)-\frac{1}{2}$"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def coef(div: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Magnitude of the per-episode coefficient the estimator applies."""
    p, q = np.asarray(p, float) + 1e-8, np.asarray(q, float) + 1e-8
    if div == "kl":
        return np.abs(np.log(p / q) + 1.0)
    if div == "tv":
        return np.abs(0.5 * np.sign(p - q))
    m = 0.5 * (p + q)
    return np.abs(0.5 * np.log(p / m) - 0.5)


def main() -> int:
    has_sweep = any((RUNS / f"div_{k}.json").exists()
                    for k in ("klraw", "tvraw", "jsraw", "klsmooth"))
    ncol = 3 if has_sweep else 2
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.2))

    # (a) analytic: coefficient as the reference probability of a mode shrinks
    ax = axes[0]
    q = np.logspace(-6, -0.4, 400)
    p_fixed = 0.20
    for d in ("kl", "js", "tv"):
        ax.plot(q, coef(d, np.full_like(q, p_fixed), q), lw=1.8,
                color=DCOL[d], label=DLAB[d])
    ax.axhline(1.0, color="#555", lw=.8, ls=":")
    ax.text(1.3e-6, 1.12, "reward advantage lives in [-1, 1]", fontsize=7.2, color="#555")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"reference probability of the visited mode, $q_m$")
    ax.set_ylabel(r"$|\partial D/\partial p_m|$")
    ax.set_title(r"(a) analytic, at $p_m=0.20$", loc="left")
    ax.legend(frameon=False, loc="upper right")

    # (b) the same on the measured reference of the most skewed backbone
    ax = axes[1]
    K = len(C_REF_060)
    idx = np.arange(K)
    width = 0.26
    for i, d in enumerate(("kl", "js", "tv")):
        vals = coef(d, np.full(K, 1.0 / K), C_REF_060)
        ax.bar(idx + (i - 1) * width, vals, width, color=DCOL[d], label=d.upper())
    ax.axhline(1.0, color="#555", lw=.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{v:.3f}" for v in C_REF_060], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel(r"route group, labelled by its $C_{ref}$ mass")
    ax.set_ylabel(r"$|\partial D/\partial p_m|$")
    ax.set_title("(b) on the measured skew-0.60 reference", loc="left")
    ax.set_ylim(0.08, 400)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.annotate("group outside the backbone's support:\nonly KL diverges there",
                xy=(5 - width, coef("kl", 1 / K, C_REF_060[5]) * 1.15),
                xytext=(1.05, 90), fontsize=7.2, color=DCOL["kl"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=DCOL["kl"], lw=.9,
                                connectionstyle="arc3,rad=-0.15"))

    # (c) empirical, when the sweep has landed
    if has_sweep:
        ax = axes[2]
        for key, d, mark in (("klraw", "kl", "o"), ("jsraw", "js", "^"), ("tvraw", "tv", "s")):
            f = RUNS / f"div_{key}.json"
            if not f.exists():
                continue
            rows = [r for r in json.loads(f.read_text()) if r["arm"] == "mop"]
            if not rows:
                continue
            ax.scatter([r["coef_absmax"] for r in rows], [r["H_modes"] for r in rows],
                       s=26, marker=mark, alpha=.8, edgecolor="none",
                       color=DCOL[d], label=f"{d.upper()}, no smoothing")
        f = RUNS / "div_klsmooth.json"
        if f.exists():
            rows = [r for r in json.loads(f.read_text()) if r["arm"] == "mop"]
            ax.scatter([r["coef_absmax"] for r in rows], [r["H_modes"] for r in rows],
                       s=26, marker="o", facecolor="none", edgecolor=DCOL["kl"],
                       lw=1.1, label="KL + smoothing 0.20")
        ax.set_xscale("log")
        ax.set_xlabel(r"largest $|\partial D/\partial p_m|$ met during training")
        ax.set_ylabel(r"$H(C_\phi)$ retained  [nats]")
        ax.set_title("(c) empirical consequence, one point per run", loc="left")
        ax.legend(frameon=False, loc="lower left", fontsize=7.4)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_divergence.png")
    fig.savefig(OUT / "fig3_divergence.pdf")
    print(f"wrote figures/fig3_divergence.{{png,pdf}}  (panels={ncol})")
    print("\nCoefficient at the unsupported group (C_ref = 0.000), p_m = 1/6:")
    for d in ("kl", "js", "tv"):
        print(f"  {d.upper():3s}  {coef(d, 1/6, 0.000):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
