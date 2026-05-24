#!/usr/bin/env python3
"""Run train.py's evaluate() function directly with BC config (z~N(0,I)) to see
why training eval gets 0% but verify_finetuned.py got Δeef=0.30m.

Calls the exact same code paths as train.py:
  - make_backbone(name='pi05', ...)
  - prepare_libero_batch (lerobot preproc)
  - SACTrainer with target_entropy=NaN, clip=4.0
  - _denoise(..., unnormalize=True) with PolicyAction postproc
  - 50 episodes, 300 max_steps each

Usage:  python scripts/diag_evaluate_path.py
"""
from __future__ import annotations
import os, sys, pathlib
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import _ipv4_patch  # noqa: F401,E402
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def main():
    print("== Build everything exactly like train.py ==")
    from scripts.backbones import make_backbone
    from scripts.envs import make_env
    from scripts.train import _denoise, evaluate
    from mop_dsrl import Config, SACTrainer

    env = make_env("libero_goal", task_id=0, seed=0, max_steps=300)
    obs_dim = env.obs_dim
    print(f"  env.obs_dim={obs_dim}, action_dim={env.action_dim}")

    denoiser = make_backbone(
        name="pi05",
        H=50, d_a=7, M=10, device="cuda",
        obs_dim=obs_dim,
        pretrained_path="lerobot/pi05_libero_finetuned",
    )

    cfg = Config(
        H=50, d_a=7, K=16,
        lambda_=0.4, n_on=8, n_ref=50, n_obs=200,
        total_env_steps=500_000, warmup_env_steps=10_000,
        backbone_family="flow", device="cuda",
    )
    trainer = SACTrainer(cfg, denoiser=denoiser, obs_dim=obs_dim)
    print(f"  trainer alpha={trainer.alpha.item():.4f} clip_radius={cfg.clip_radius:.2f}")

    # ==== 5 episodes via _denoise (mimicking evaluate) ====
    print("\n== Run 3 episodes through train.py's _denoise (use_dict_path=True) ==")
    for ep in range(3):
        obs = env.reset()
        ep_return = 0.0
        success = False
        info = {}
        done = False
        eef0 = env.last_obs_dict["robot0_eef_pos"].copy()
        n_inferences = 0
        n_env_steps = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
            with torch.no_grad():
                z, _ = trainer._sample_z(obs_t)  # SAC policy at init → close to N(0,I)
                a = _denoise(denoiser, obs_t, z, env=env, use_dict_path=True, d_a=7)
            n_inferences += 1
            chunk = a.squeeze(0).cpu().numpy().reshape(50, 7)
            for i in range(10):  # n_action_steps
                obs, reward, done, info = env.step(chunk[i])
                ep_return += float(reward)
                n_env_steps += 1
                if done:
                    break
        eef_end = env.last_obs_dict["robot0_eef_pos"]
        delta = np.linalg.norm(eef_end - eef0)
        success = bool(info.get("success", False))
        print(f"  ep {ep+1}: success={success} n_infer={n_inferences} n_step={n_env_steps} "
              f"return={ep_return:.3f} Δeef={delta:.4f}")

    # ==== Same loop but with pure N(0,I) z (verify_finetuned setup) ====
    print("\n== Run 3 episodes with pure z=N(0,I) (bypass SAC policy) ==")
    for ep in range(3):
        obs = env.reset()
        ep_return = 0.0
        info = {}
        done = False
        eef0 = env.last_obs_dict["robot0_eef_pos"].copy()
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
            with torch.no_grad():
                z = torch.randn(1, 350, device="cuda", dtype=torch.float32)
                a = _denoise(denoiser, obs_t, z, env=env, use_dict_path=True, d_a=7)
            chunk = a.squeeze(0).cpu().numpy().reshape(50, 7)
            for i in range(10):
                obs, reward, done, info = env.step(chunk[i])
                ep_return += float(reward)
                if done:
                    break
        eef_end = env.last_obs_dict["robot0_eef_pos"]
        delta = np.linalg.norm(eef_end - eef0)
        success = bool(info.get("success", False))
        print(f"  ep {ep+1}: success={success} return={ep_return:.3f} Δeef={delta:.4f}")

    print("\nINTERPRETATION:")
    print("  • If 'pure z=N(0,I)' shows Δeef > 0.2m and 'SAC z' shows ~0, then SAC at init")
    print("    is producing z that drifts pi_0.5 OOD (clip_radius=75 may be too big).")
    print("  • If both show ~0, BC pipeline has a different bug we still haven't found.")
    print("  • Compare to verify_finetuned.py which got 0.27m Δeef.")


if __name__ == "__main__":
    raise SystemExit(main())
