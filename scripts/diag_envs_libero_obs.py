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
    # Bypass our LiberoEnv wrapper — talk to lerobot vec env directly to
    # capture its raw obs schema before our _flatten_obs intercepts it.
    from lerobot.envs.configs import LiberoEnv as LeRobotLiberoEnvCfg
    from lerobot.envs.factory import make_env as lerobot_make_env

    env_cfg = LeRobotLiberoEnvCfg(task="libero_goal", task_ids=[0], control_mode="relative")
    envs_by_suite = lerobot_make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = envs_by_suite[list(envs_by_suite.keys())[0]][0]
    print("== Vec env action_space =", vec.action_space)
    print("== Vec env observation_space =", vec.observation_space)

    obs, info = vec.reset()
    print("\n== raw vec.reset() returns ==")
    print(f"  obs type: {type(obs).__name__}")
    if isinstance(obs, dict):
        for k, v in obs.items():
            if hasattr(v, "shape"):
                print(f"  obs[{k!r}]: shape={tuple(v.shape)} dtype={getattr(v,'dtype','?')}")
                if hasattr(v, "min"):
                    try:
                        print(f"     range=[{float(v.min()):.4f},{float(v.max()):.4f}]")
                    except Exception: pass
            elif isinstance(v, dict):
                print(f"  obs[{k!r}]: nested dict, keys={list(v.keys())}")
                for k2, v2 in v.items():
                    if hasattr(v2, "shape"):
                        print(f"     [{k2!r}]: shape={tuple(v2.shape)} dtype={getattr(v2,'dtype','?')}")
            else:
                print(f"  obs[{k!r}]: {type(v).__name__} = {v}")
    print(f"\n== info type: {type(info).__name__} ==")
    if isinstance(info, dict):
        for k, v in info.items():
            print(f"  info[{k!r}]: {type(v).__name__} -> {str(v)[:200]}")
    elif isinstance(info, list):
        print(f"  info list len={len(info)}, first elem keys: {list(info[0].keys()) if info else []}")
        if info:
            for k, v in info[0].items():
                print(f"  info[0][{k!r}]: {type(v).__name__} -> {str(v)[:200]}")

    print("\n== step once with zero action ==")
    import numpy as np
    a = np.zeros((1, 7), dtype=np.float32)
    out = vec.step(a)
    if len(out) == 5:
        obs2, reward, term, trunc, info2 = out
        print(f"  reward={reward}, terminated={term}, truncated={trunc}")
    else:
        obs2, reward, done, info2 = out
        print(f"  reward={reward}, done={done}")
    print(f"  info type: {type(info2).__name__}")
    if isinstance(info2, dict):
        for k, v in info2.items():
            print(f"  info[{k!r}]: {type(v).__name__} -> {str(v)[:200]}")

    # === Also dump env_pre transformation ===
    print("\n== env_pre transformer chain ==")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_libero_finetuned")
    policy_cfg.device = "cuda"
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
    print(f"  env_pre type: {type(env_pre).__name__}")
    if hasattr(env_pre, "steps"):
        for i, s in enumerate(env_pre.steps):
            print(f"    step[{i}] = {type(s).__name__}")

    print("\n== Apply env_pre to raw obs ==")
    obs2 = env_pre(obs)
    print(f"  out type: {type(obs2).__name__}")
    if isinstance(obs2, dict):
        for k, v in obs2.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v,'dtype','?')}", end="")
                try:
                    print(f" range=[{float(v.min()):.4f},{float(v.max()):.4f}]")
                except Exception:
                    print()
            elif isinstance(v, dict):
                print(f"  {k}: nested dict keys={list(v.keys())}")
                for k2, v2 in v.items():
                    if hasattr(v2, "shape"):
                        print(f"    {k2}: shape={tuple(v2.shape)}", end="")
                        try:
                            print(f" range=[{float(v2.min()):.4f},{float(v2.max()):.4f}]")
                        except Exception:
                            print()

    vec.close()
    return 0

    # (Old per-script LiberoEnv test, unreachable.)
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
