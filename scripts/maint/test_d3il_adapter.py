#!/usr/bin/env python3
"""Smoke-test the D3IL Avoiding adapter: contract, mode labels, throughput."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.envs_d3il import D3ILAvoidingEnv  # noqa: E402


def main() -> int:
    env = D3ILAvoidingEnv(seed=0)
    print(f"obs_dim={env.obs_dim} action_dim={env.action_dim} n_modes={env.n_modes}")

    modes, t0, total_steps = [], time.time(), 0
    for ep in range(6):
        obs = env.reset()
        assert obs.shape == (env.obs_dim,), f"bad obs shape {obs.shape}"
        done, t, info = False, 0, {}
        while not done and t < 200:
            # random walk in the 2-D desired-position delta space
            obs, r, done, info = env.step(np.random.randn(2) * 0.02)
            t += 1
        total_steps += t
        modes.append(info.get("mode_id", -1))
        enc = info.get("mode_encoding", np.zeros(9)).astype(int)
        print(f"  ep{ep}: steps={t} success={info.get('success')} "
              f"mode_id={info.get('mode_id')} enc={enc}")

    seen = sorted({m for m in modes if m >= 0})
    dt = time.time() - t0
    print(f"distinct modes seen: {seen}")
    print(f"throughput: {total_steps / dt:.1f} env-steps/s ({total_steps} steps in {dt:.1f}s)")
    assert obs.shape == (env.obs_dim,)
    print("ADAPTER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
