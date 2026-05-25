#!/usr/bin/env python3
"""Take OUR backbone (RectifiedFlowDenoiser wrapping pi_0.5) and feed it to
lerobot's official eval_policy.  This isolates whether the bug is in the
backbone wrapping or in our manual chunked rollout.
"""
from __future__ import annotations
import os, sys, pathlib
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import _ipv4_patch  # noqa: F401
os.environ.setdefault("MUJOCO_GL", "egl")

import torch


def main():
    REPO = "lerobot/pi05_libero_finetuned"
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv as LeRobotLiberoEnvCfg
    from lerobot.envs.factory import make_env as lerobot_make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.device = "cuda"
    cfg.n_action_steps = 10
    cfg.validate_features()

    # ===== Variant A: use pi_0.5 directly =====
    policy = PI05Policy.from_pretrained(REPO, config=cfg)
    policy.eval().to("cuda")
    pre, post = make_pre_post_processors(cfg, pretrained_path=REPO)
    env_cfg = LeRobotLiberoEnvCfg(task="libero_goal", task_ids=[0], control_mode="relative")
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)
    envs_by_suite = lerobot_make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs_by_suite[list(envs_by_suite.keys())[0]][0]

    print("== A) lerobot pi_0.5 native + eval_policy (sanity baseline) ==")
    result = eval_policy(env=env, policy=policy,
                        env_preprocessor=env_pre, env_postprocessor=env_post,
                        preprocessor=pre, postprocessor=post, n_episodes=5)
    print(f"  agg = {result.get('aggregated', result)}")
    env.close()

    # ===== Variant B: wrap pi_0.5 with our RectifiedFlowDenoiser + monkey-patch
    # policy.predict_action_chunk to call our denoise =====
    print("\n== B) Our denoiser-wrapped pi_0.5 + eval_policy ==")
    from scripts.backbones import make_backbone
    denoiser = make_backbone(
        name="pi05", H=50, d_a=7, M=10, device="cuda",
        obs_dim=9, pretrained_path=REPO,
    )
    # Use the inner policy (it's the same instance lerobot wraps).
    policy_b = denoiser.velocity_field._policy
    pre_b, post_b = make_pre_post_processors(cfg, pretrained_path=REPO)
    envs_by_suite_b = lerobot_make_env(env_cfg, n_envs=1, use_async_envs=False)
    env_b = envs_by_suite_b[list(envs_by_suite_b.keys())[0]][0]

    # Monkey-patch: route predict_action_chunk through our denoise.
    z_dim = 50 * 7
    @torch.no_grad()
    def patched(batch, **kwargs):
        bsize = 1
        device = next(policy_b.parameters()).device
        z = torch.randn(bsize, z_dim, device=device, dtype=torch.float32)
        a = denoiser.denoise_with_obs_dict(batch, z)  # [B, 350]
        return a.reshape(bsize, 50, 7)

    orig = policy_b.predict_action_chunk
    policy_b.predict_action_chunk = patched
    try:
        result_b = eval_policy(env=env_b, policy=policy_b,
                               env_preprocessor=env_pre, env_postprocessor=env_post,
                               preprocessor=pre_b, postprocessor=post_b, n_episodes=5)
        print(f"  agg = {result_b.get('aggregated', result_b)}")
    finally:
        policy_b.predict_action_chunk = orig
        env_b.close()


if __name__ == "__main__":
    raise SystemExit(main())
