#!/usr/bin/env python3
"""Compare the three sampling paths on one checkpoint, calling the real code.

Three things disagree about how often one frozen policy completes an episode:
the cached reference record, the reference chunk function it was built from,
and the evaluation rollout the fine-tuned cells use. Each row below drives one
of those code paths directly, so whatever separates them shows up as a gap
between rows and not as a gap between a measurement and a reimplementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d3il_v2 as D  # noqa: E402

# A job runs from a frozen copy of the source, so the repository root
# has to come from the driver rather than from this file's location.
ROOT = D.ROOT


def via_ref_chunk(task, weighting, bb, seeds, per, M, clip):
    """The reference path, through _ref_chunk itself."""
    n_ep = n_valid = 0
    for s in seeds:
        _, ne, nv = D._ref_chunk((weighting, per, int(s), M, "cpu", task, bb, clip))
        n_ep += ne
        n_valid += nv
    return n_valid / n_ep, n_ep


def via_rollout(task, weighting, bb, seed, n, M, clip):
    """The evaluation path, through rollout and an untrained NoisePolicy."""
    model, blob = D.load_backbone(D.backbone_path(task, weighting, bb), "cpu")
    torch.manual_seed(seed)
    env = D.TASKS[task]["env"](seed=seed, history=blob.get("history", 1))
    policy = D.NoisePolicy(blob["obs_dim"], blob["act_dim"])
    n_valid = 0
    for _ in range(n):
        with torch.no_grad():
            _, _, raw_mode, _ = D.rollout(env, policy, model, blob, set(),
                                          "cpu", M, clip, {})
        n_valid += raw_mode >= 0
    return n_valid / n, n


def via_rollout_fresh_env(task, weighting, bb, seeds, per, M, clip):
    """The evaluation path, with the reference's short reset runs."""
    model, blob = D.load_backbone(D.backbone_path(task, weighting, bb), "cpu")
    n_ep = n_valid = 0
    for s in seeds:
        torch.manual_seed(int(s))
        env = D.TASKS[task]["env"](seed=int(s), history=blob.get("history", 1))
        policy = D.NoisePolicy(blob["obs_dim"], blob["act_dim"])
        for _ in range(per):
            with torch.no_grad():
                _, _, raw_mode, _ = D.rollout(env, policy, model, blob, set(),
                                              "cpu", M, clip, {})
            n_ep += 1
            n_valid += raw_mode >= 0
    return n_valid / n_ep, n_ep


def main():
    task, weighting = sys.argv[1], sys.argv[2]
    bb, n, M, clip = int(sys.argv[3]), int(sys.argv[4]), 10, 50.0
    rows = [
        ("_ref_chunk, 8-episode envs from 12345",
         lambda: via_ref_chunk(task, weighting, bb, range(12345, 12345 + n // 8), 8, M, clip)),
        ("_ref_chunk, one env, seed 700",
         lambda: via_ref_chunk(task, weighting, bb, [700], n, M, clip)),
        ("rollout, one env, seed 700",
         lambda: via_rollout(task, weighting, bb, 700, n, M, clip)),
        ("rollout, 8-episode envs from 12345",
         lambda: via_rollout_fresh_env(task, weighting, bb, range(12345, 12345 + n // 8), 8, M, clip)),
    ]
    for name, fn in rows:
        c, ne = fn()
        print(f"{name:40s} n={ne:4d} completion={c:.3f}", flush=True)


if __name__ == "__main__":
    D._single_threaded()
    main()
