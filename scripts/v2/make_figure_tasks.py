#!/usr/bin/env python3
"""The two tasks as the simulator draws them, and the routes a mode is.

Panels (a) and (c) are frames from the simulator, taken from above so that
which side of an obstacle the arm passed on is visible. Panels (b) and (d) are
the end-effector paths of the frozen policy's own rollouts, coloured by the
route each episode realised. The histogram the paper preserves is the
distribution over those colours.

Reads the bundles written by render_task.py, one per task.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
REND = ROOT / "runs" / "_render"

# Six for Avoiding's mode groups, four for Pushing's modes. Chosen to stay
# distinguishable in grey as well as in colour.
PALETTE = ["#1f6fb4", "#d1622b", "#1b9e77", "#7b5aa6", "#b8860b", "#444444"]


def crop(frame: np.ndarray, keep: float = 0.96) -> np.ndarray:
    """Trim the empty border the square camera leaves around the workspace.

    The goal line sits near the top of the frame, so the window keeps more
    above than below: a tighter symmetric crop cut the line off.
    """
    h, w = frame.shape[:2]
    dw = int(w * (1 - keep) / 2)
    return frame[: int(h * 0.97), dw:w - dw]


def paths_panel(ax, bundle, task, n_show):
    groups = bundle["groups"]
    paths = bundle["paths"]
    order = [g for g in sorted(set(int(g) for g in groups)) if g >= 0]
    shown = {g: 0 for g in order}
    for path, g in zip(paths, groups):
        g = int(g)
        if g < 0 or shown[g] >= n_show:
            continue
        shown[g] += 1
        xy = np.asarray(path, float)
        ax.plot(xy[:, 0], xy[:, 1], lw=0.7, alpha=0.75,
                color=PALETTE[g % len(PALETTE)], zorder=2,
                label=f"group {g + 1}" if shown[g] == 1 else None)
    geom = json.loads(str(bundle["geometry"])) if "geometry" in bundle else {}
    if geom:
        for (px, py) in geom["posts"]:
            ax.add_patch(__import__("matplotlib").patches.Circle(
                (px, py), 0.0125, facecolor="#e03030", edgecolor="none",
                zorder=3))
        ax.axhline(geom["goal_y"], color="#4caf50", lw=1.6, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    # Sorted, and outside the paths: the lower right corner is where every
    # rollout starts.
    h, lab = ax.get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda i: lab[i])
    ax.legend([h[i] for i in order], [lab[i] for i in order],
              loc="upper left", frameon=False, fontsize=5.8,
              handlelength=1.1, borderaxespad=0.2, ncol=1)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.4,
        "axes.labelsize": 7.0, "axes.titlesize": 8,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.dpi": 300,
    })

    tasks = []
    for t in ("avoiding", "pushing"):
        f = REND / f"{t}.npz"
        if f.exists():
            tasks.append((t, np.load(f, allow_pickle=True)))
    if not tasks:
        raise SystemExit(f"no bundles under {REND}")

    fig, axes = plt.subplots(len(tasks), 2, figsize=(7.1, 3.3 * len(tasks)),
                             gridspec_kw=dict(width_ratios=[1, 1.08]))
    axes = np.atleast_2d(axes)
    tag = "abcdefgh"
    for row, (task, bundle) in enumerate(tasks):
        ax = axes[row, 0]
        ax.imshow(crop(bundle["frame"]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({tag[2 * row]})  {task.capitalize()}, seen from above",
                     loc="left")
        ax = axes[row, 1]
        n = int((np.asarray(bundle["modes"]) >= 0).sum())
        paths_panel(ax, bundle, task, n_show=6)
        ax.set_title(f"({tag[2 * row + 1]})  end-effector paths, six rollouts "
                     f"of each group, {n} labelled of "
                     f"{len(bundle['modes'])}", loc="left")

    fig.tight_layout(pad=0.5, h_pad=1.1, w_pad=1.0)
    out = ROOT / "paper" / "figures" / "fig4_tasks"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    for task, bundle in tasks:
        g = np.asarray(bundle["groups"])
        counts = {int(k): int((g == k).sum()) for k in sorted(set(g.tolist()))}
        print(f"{task}: {counts}")
    print(f"wrote {out}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
