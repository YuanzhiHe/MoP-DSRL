#!/usr/bin/env python3
"""Quick check: what keys/shapes does the new lerobot-backed LiberoEnv emit?

Our prepare_libero_batch (in Pi05VelocityField) was written against stock
LIBERO obs keys (agentview_image, robot0_eef_pos, ...).  lerobot's wrapper
typically renames them to observation.images.image / observation.state.

We need to confirm the actual keys so the obs adapter can stay aligned.
"""
from __future__ import annotations
import os, sys, pathlib
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import _ipv4_patch  # noqa: F401
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def main():
    from scripts.envs import make_env

    env = make_env("libero_goal", task_id=0, seed=0, max_steps=300)
    print(f"  env.obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    print(f"  task_instruction={env.task_instruction!r}")
    print()
    print("== last_obs_dict keys + shapes ==")
    raw = env.last_obs_dict
    for k in sorted(raw.keys()):
        v = raw[k]
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', '?')}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    print()
    print("== Step once with zero action ==")
    obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
    print(f"  flat obs shape={obs.shape}, reward={reward}, done={done}")
    print(f"  info keys={list(info.keys()) if isinstance(info, dict) else info}")
    print()
    print("== last_obs_dict after step ==")
    raw = env.last_obs_dict
    for k in sorted(raw.keys()):
        v = raw[k]
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', '?')}")

    env.close()


if __name__ == "__main__":
    raise SystemExit(main())
