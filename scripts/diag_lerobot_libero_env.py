#!/usr/bin/env python3
"""Use lerobot's OFFICIAL LiberoEnv (with control_mode=relative) and pi05_libero_finetuned.

Lerobot doc says pi_0.5 trained for LIBERO uses *delta* action (Box(-1,1,7) =
6D eef delta + gripper).  Our LiberoEnv used stock OffScreenRenderEnv which
defaults to absolute OSC — that's why BC=0%.

This script uses the same env construction lerobot-eval uses internally, then
runs pi_0.5 native select_action for 5 episodes.  If success > 50%, we wrap
that env constructor in our codebase.

Usage:  python scripts/diag_lerobot_libero_env.py
"""
from __future__ import annotations
import os, sys, pathlib
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import _ipv4_patch  # noqa: F401,E402
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch


def main():
    print("== Build pi_0.5 (lerobot finetuned) ==")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    REPO = "lerobot/pi05_libero_finetuned"
    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.device = "cuda"
    cfg.n_action_steps = 10
    cfg.validate_features()
    policy = PI05Policy.from_pretrained(REPO, config=cfg)
    policy.eval().to("cuda")
    pre, post = make_pre_post_processors(cfg, pretrained_path=REPO)

    print("\n== Build LIBERO env via lerobot factory (relative control) ==")
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    env_cfg = LiberoEnvCfg(task="libero_goal", task_ids=[0], control_mode="relative")
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)
    envs_by_suite = make_env(env_cfg, n_envs=1, use_async_envs=False)
    print(f"  envs_by_suite keys: {list(envs_by_suite.keys())}")
    suite_name = list(envs_by_suite.keys())[0]
    suite_envs = envs_by_suite[suite_name]
    print(f"  task_ids in suite: {list(suite_envs.keys())}")
    task_id = list(suite_envs.keys())[0]
    env = suite_envs[task_id]
    print(f"  env type: {type(env).__name__}")
    print(f"  action_space: {env.action_space}")

    print("\n== Use lerobot's eval_policy for the gold-standard rollout ==")
    from lerobot.scripts.lerobot_eval import eval_policy
    result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_pre,
        env_postprocessor=env_post,
        preprocessor=pre,
        postprocessor=post,
        n_episodes=5,
    )
    print(f"  result keys: {list(result.keys())}")
    print(f"  aggregated: {result.get('aggregated', result)}")

    env.close()
    print("\nINTERPRETATION:")
    print("  • If pc_success > 50%, the BC pipeline is functional; the bug was control_mode")
    print("  • Then we'd switch our LiberoEnv to use this same factory + control_mode=relative")


if __name__ == "__main__":
    raise SystemExit(main())
