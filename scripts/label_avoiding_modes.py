#!/usr/bin/env python3
"""Assign ground-truth behavioural modes to D3IL Avoiding demonstrations.

The Avoiding task's mode is which side of each of three obstacle rows the arm
passes: 2 x 3 x 4 = 24 routes. The environment decides this geometrically in
``ObstacleAvoidanceEnv.check_mode`` from the end-effector (x, y) at the moment it
crosses each row. Demonstrations store the full Cartesian trajectory (``robot ->
c_pos``), so the same rule can be replayed offline -- no simulator needed.

This is what makes D3IL usable for the mode-preservation claim at all: the label
is ground truth from geometry, not a K-means guess over action chunks. It is
also the prerequisite for constructing a skewed demonstration subset
(scripts/make_skewed_demos.py needs a mode -> files map), which the toy showed is
a necessary condition for MoP and an entropy/MI objective to differ.

Constants mirror avoiding.py:96-107 (level_distance 0.18, obstacle_offset 0.075).

    python scripts/label_avoiding_modes.py --data-root <...>/avoiding/data
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LEVEL_DISTANCE = 0.18
OBSTACLE_OFFSET = 0.075

L1_YPOS = -0.1
L2_YPOS = -0.1 + LEVEL_DISTANCE
L3_YPOS = -0.1 + 2 * LEVEL_DISTANCE

L1_XPOS = 0.5
L2_TOP_XPOS = 0.5 - OBSTACLE_OFFSET
L2_BOTTOM_XPOS = 0.5 + OBSTACLE_OFFSET
L3_TOP_XPOS = 0.5 - 2 * OBSTACLE_OFFSET
L3_MID_XPOS = 0.5
L3_BOTTOM_XPOS = 0.5 + 2 * OBSTACLE_OFFSET

N_MODES = 24


def encode(traj_xy: np.ndarray) -> np.ndarray:
    """Replay check_mode over a trajectory, returning the 9-bit encoding."""
    enc = np.zeros(9)
    l1_passed = l2_passed = l3_passed = False
    for x, y in traj_xy:
        if not l1_passed and (y - 0.03 <= L1_YPOS <= y + 0.03):
            enc[0 if x < L1_XPOS else 1] = 1
            l1_passed = True
        if not l2_passed and (y - 0.03 <= L2_YPOS <= y + 0.03):
            if x < L2_TOP_XPOS:
                enc[2] = 1
            elif x < L2_BOTTOM_XPOS:
                enc[3] = 1
            else:
                enc[4] = 1
            l2_passed = True
        if not l3_passed and y >= L3_YPOS:
            if x < L3_TOP_XPOS:
                enc[5] = 1
            elif x < L3_MID_XPOS:
                enc[6] = 1
            elif x < L3_BOTTOM_XPOS:
                enc[7] = 1
            else:
                enc[8] = 1
            l3_passed = True
    return enc


def mode_id(enc: np.ndarray) -> int:
    """9-bit encoding -> mode in {0..23}, or -1 if the route is incomplete."""
    g1, g2, g3 = enc[:2], enc[2:5], enc[5:9]
    if g1.sum() != 1 or g2.sum() != 1 or g3.sum() != 1:
        return -1
    return int(g1.argmax() * 12 + g2.argmax() * 4 + g3.argmax())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path("/shared/scratch/SCWF00111/d3il_data/data/avoiding/data"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.data_root.glob("*.pkl"))
    print(f"demos: {len(files)}")

    by_mode: dict[int, list[str]] = defaultdict(list)
    unlabelled = []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        pos = np.asarray(d["robot"]["c_pos"])[:, :2]
        m = mode_id(encode(pos))
        if m < 0:
            unlabelled.append(f.name)
        else:
            by_mode[m].append(f.name)

    counts = np.array([len(by_mode.get(k, [])) for k in range(N_MODES)], dtype=float)
    covered = int((counts > 0).sum())
    print(f"labelled: {int(counts.sum())}  unlabelled: {len(unlabelled)}")
    print(f"modes covered: {covered}/{N_MODES}")
    print(f"per-mode counts: {Counter(counts.astype(int).tolist())}")
    if counts.sum() > 0:
        p = counts / counts.sum()
        nz = p > 0
        kl = float(np.sum(p[nz] * np.log(p[nz] * N_MODES)))
        ent = float(-(p[nz] * np.log(p[nz])).sum())
        print(f"KL(demo || uniform) = {kl:.4f} nats   H = {ent:.4f} "
              f"(max {np.log(N_MODES):.4f})")
        nzc = counts[counts > 0]
        print(f"max/min over covered modes = {nzc.max() / nzc.min():.2f}x")
    if unlabelled:
        print(f"first unlabelled: {unlabelled[:5]}")

    out = args.out or args.data_root.parent / "all_modes.pkl"
    with open(out, "wb") as fh:
        pickle.dump({k: v for k, v in sorted(by_mode.items())}, fh)
    print(f"wrote mode -> files map to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
