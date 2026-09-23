#!/usr/bin/env python3
"""Smoke-test the Pushing adapter: shapes, throughput, and mode reporting."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.envs_d3il import make_d3il_env, PUSHING_MODE_NAMES  # noqa: E402

env = make_d3il_env("d3il_pushing", seed=0)
print(f"obs_dim={env.obs_dim}  action_dim={env.action_dim}  n_modes={env.n_modes}")
print(f"mode names: {PUSHING_MODE_NAMES}")

rng = np.random.default_rng(0)
t0, steps, modes = time.time(), 0, []
for _ in range(6):
    o = env.reset()
    assert o.shape[0] == env.obs_dim, (o.shape, env.obs_dim)
    done = False
    while not done:
        o, r, done, info = env.step(rng.normal(0, 0.01, 2))
        steps += 1
    modes.append(info["mode_id"])
dt = time.time() - t0
print(f"{steps} steps in {dt:.1f}s -> {steps / dt:.1f} steps/s")
print(f"modes under random actions: {modes}   (-1 = no route completed, expected)")
