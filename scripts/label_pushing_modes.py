#!/usr/bin/env python3
"""Assign ground-truth behavioural modes to D3IL Pushing demonstrations.

Pushing has two boxes and two targets, and its behavioural mode is which box is
delivered to which target and in what order: rr->gg, gg->rr, rg->gr, gr->rg.
``PushingEnv.check_mode`` decides this online from object positions, tracking
which (box, target) pair came within ``target_min_dist`` first. The
demonstrations store every object's pose for every step, so the same rule
replays offline without the simulator -- the same approach that worked for
Avoiding, except that Avoiding's mode was pure robot geometry while this one
depends on where the OBJECTS ended up.

Adding a second task is the highest-value item for any venue: every result so
far rests on Avoiding alone. Pushing also carries 2000 demonstrations against
Avoiding's 96, so a skewed subset can be selected outright rather than
approximated by sampling weights.

Constants mirror pushing.py:251 (target_min_dist = 0.05).

    python scripts/label_pushing_modes.py
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TARGET_MIN_DIST = 0.05
N_MODES = 4
MODE_NAMES = {0: "rr->gg", 1: "gg->rr", 2: "rg->gr", 3: "gr->rg"}

DATA = Path("/shared/scratch/SCWF00111/d3il_data/data/pushing/all_data")


def mode_of(blob: dict) -> int:
    """Replay check_mode over a demonstration's stored object trajectories."""
    b1 = np.asarray(blob["red-box"]["pos"])[:, :2]
    b2 = np.asarray(blob["green-box"]["pos"])[:, :2]
    g1 = np.asarray(blob["red-target"]["pos"])[:, :2]
    g2 = np.asarray(blob["green-target"]["pos"])[:, :2]

    first_visit = -1
    for t in range(len(b1)):
        d_rr = np.linalg.norm(b1[t] - g1[t])
        d_rg = np.linalg.norm(b1[t] - g2[t])
        d_gr = np.linalg.norm(b2[t] - g1[t])
        d_gg = np.linalg.norm(b2[t] - g2[t])

        visit = -1
        if d_rr <= TARGET_MIN_DIST and first_visit != 0:
            visit = 0
        elif d_rg <= TARGET_MIN_DIST and first_visit != 1:
            visit = 1
        elif d_gr <= TARGET_MIN_DIST and first_visit != 2:
            visit = 2
        elif d_gg <= TARGET_MIN_DIST and first_visit != 3:
            visit = 3

        if first_visit == -1:
            if visit != -1:
                first_visit = visit
        else:
            if first_visit == 0 and visit == 3:
                return 0
            if first_visit == 3 and visit == 0:
                return 1
            if first_visit == 1 and visit == 2:
                return 2
            if first_visit == 2 and visit == 1:
                return 3
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.data_root.glob("*.pkl"))
    print(f"demos: {len(files)}  in {args.data_root}")

    by_mode: dict[int, list[str]] = defaultdict(list)
    unlabelled = []
    for f in files:
        with open(f, "rb") as fh:
            blob = pickle.load(fh)
        m = mode_of(blob)
        if m < 0:
            unlabelled.append(f.name)
        else:
            by_mode[m].append(f.name)

    counts = np.array([len(by_mode.get(k, [])) for k in range(N_MODES)], dtype=float)
    print(f"labelled: {int(counts.sum())}  unlabelled: {len(unlabelled)}")
    print(f"modes covered: {int((counts > 0).sum())}/{N_MODES}")
    for k in range(N_MODES):
        print(f"    mode {k} ({MODE_NAMES[k]}): {int(counts[k])}")
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
