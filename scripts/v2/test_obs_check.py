#!/usr/bin/env python3
"""The construction bound, exercised before a five-hour stage depends on it."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("MOP_DSRL_ROOT", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "v2")]

import preflight as P  # noqa: E402
from scripts.run_d3il_headtohead import backbone_path  # noqa: E402

bad = 0


def say(name, ok, detail=""):
    global bad
    bad += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def main() -> int:
    blob = torch.load(backbone_path("pushing", "0.50", 0), map_location="cpu",
                      weights_only=False)
    mu = np.asarray(blob["obs_mean"], float).ravel()
    sd = np.asarray(blob["obs_std"], float).ravel()
    bounds = [(1.0 + abs(mu[i])) / sd[i] for i in P.ORIENT_CHANNELS["pushing"]]
    print(f"  construction bounds on 6,7,10,11: "
          + ", ".join(f"{b:.4g}" for b in bounds))

    base = dict(weighting="0.50", backbone_seed=0)
    n = mu.size

    # A rollout that respects the encoding: every orientation channel sits at
    # its bound exactly, and the position channels are far larger.
    ch = [900.0] * n
    for i, b in zip(P.ORIENT_CHANNELS["pushing"], bounds):
        ch[i] = b
    ok, d = P.orientation_within_construction(
        "pushing", [dict(base, obs_absmax_channels=ch)])
    say("a channel at its exact bound passes", ok, d)

    # One orientation channel past its bound by a percent: an encoding fault,
    # and the position channels being huge must not hide it.
    ch2 = list(ch)
    ch2[7] = bounds[1] * 1.01
    ok2, d2 = P.orientation_within_construction(
        "pushing", [dict(base, obs_absmax_channels=ch2)])
    say("a channel past its bound fails", not ok2, d2)

    # The old tan(yaw) regime, which is what the check exists to catch.
    ch3 = list(ch)
    ch3[6] = 1e12
    ok3, d3 = P.orientation_within_construction(
        "pushing", [dict(base, obs_absmax_channels=ch3)])
    say("the pole regime fails", not ok3, d3)

    # A run that never recorded the channels must not pass by absence.
    ok4, d4 = P.orientation_within_construction(
        "pushing", [dict(base, obs_absmax_channels=None)])
    say("a missing record fails rather than passes", not ok4, d4)

    # Avoiding declares no orientation channels, so the check is vacuous there
    # and must say so instead of indexing into something.
    ok5, d5 = P.orientation_within_construction(
        "avoiding", [dict(weighting="0.00", backbone_seed=0,
                          obs_absmax_channels=[5.0] * 4)])
    say("avoiding is vacuous and says so", ok5, d5)

    print("OBS CHECK: " + ("PASS" if not bad else "FAIL"))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
