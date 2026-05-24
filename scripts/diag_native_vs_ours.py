#!/usr/bin/env python3
"""Compare:
  A. pi_0.5 native select_action (lerobot's own inference loop with action queue)
  B. our manual chunked denoise_with_obs_dict path

If A succeeds (>0% on 5 episodes) but B fails, the difference is in how we
re-infer per chunk (pure RNG z each time vs deterministic chunk caching).

Usage:  python scripts/diag_native_vs_ours.py
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
    from scripts.envs import make_env
    from scripts.backbones import make_backbone

    print("== Build pi_0.5 native (for path A) ==")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    REPO = "lerobot/pi05_libero_finetuned"
    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.device = "cuda"
    cfg.n_action_steps = 10
    cfg.validate_features()
    policy_native = PI05Policy.from_pretrained(REPO, config=cfg)
    policy_native.eval().to("cuda")
    pre_native, post_native = make_pre_post_processors(cfg, pretrained_path=REPO)

    env = make_env("libero_goal", task_id=0, seed=0, max_steps=300)

    def build_batch(o, instr):
        b = {}
        if "agentview_image" in o:
            img = torch.from_numpy(o["agentview_image"].copy()).float() / 255.0
            b["observation.images.image"] = img.permute(2, 0, 1).unsqueeze(0).to("cuda")
        if "robot0_eye_in_hand_image" in o:
            img = torch.from_numpy(o["robot0_eye_in_hand_image"].copy()).float() / 255.0
            b["observation.images.image2"] = img.permute(2, 0, 1).unsqueeze(0).to("cuda")
        b["observation.images.empty_camera_0"] = torch.zeros(1, 3, 224, 224, device="cuda")
        sp = [torch.from_numpy(o["robot0_eef_pos"].copy()).float(),
              torch.from_numpy(o["robot0_eef_quat"].copy()).float(),
              torch.from_numpy(o["robot0_gripper_qpos"][:1].copy()).float()]
        b["observation.state"] = torch.cat(sp, dim=0).unsqueeze(0).to("cuda")
        b["task"] = [instr]
        return b

    from lerobot.processor import PolicyAction

    # ============ PATH A: pi_0.5 native select_action (action queue) ============
    print("\n== PATH A: pi_0.5 NATIVE select_action ==")
    n_succ_a = 0
    for ep in range(5):
        torch.manual_seed(ep); np.random.seed(ep)
        obs = env.reset()
        policy_native.reset()
        info = {}; done = False
        eef0 = env.last_obs_dict["robot0_eef_pos"].copy()
        for step in range(300):
            batch = build_batch(env.last_obs_dict, env.task_instruction)
            batch_pre = pre_native(batch)
            with torch.no_grad():
                action = policy_native.select_action(batch_pre)
            try:
                action_post = post_native(action)
            except Exception:
                action_post = action
            a = action_post.detach().cpu().numpy().reshape(-1)[:7]
            obs, reward, done, info = env.step(a)
            if done: break
        eef_end = env.last_obs_dict["robot0_eef_pos"]
        success = bool(info.get("success", False))
        n_succ_a += int(success)
        print(f"  A ep {ep+1}: success={success} Δeef={np.linalg.norm(eef_end-eef0):.3f} step={step+1}")
    print(f"  PATH A success rate: {n_succ_a}/5 = {n_succ_a/5:.1%}")

    # ============ PATH B: our denoise_with_obs_dict per chunk ============
    print("\n== PATH B: our manual chunked denoise_with_obs_dict ==")
    denoiser = make_backbone(
        name="pi05", H=50, d_a=7, M=10, device="cuda",
        obs_dim=190, pretrained_path=REPO,
    )
    from scripts.train import _denoise

    n_succ_b = 0
    for ep in range(5):
        torch.manual_seed(ep + 100); np.random.seed(ep + 100)
        obs = env.reset()
        info = {}; done = False
        eef0 = env.last_obs_dict["robot0_eef_pos"].copy()
        n_step = 0
        while not done and n_step < 300:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
            with torch.no_grad():
                z = torch.randn(1, 350, device="cuda", dtype=torch.float32)
                a = _denoise(denoiser, obs_t, z, env=env, use_dict_path=True, d_a=7)
            chunk = a.squeeze(0).cpu().numpy().reshape(50, 7)
            for i in range(10):
                obs, reward, done, info = env.step(chunk[i])
                n_step += 1
                if done or n_step >= 300: break
        eef_end = env.last_obs_dict["robot0_eef_pos"]
        success = bool(info.get("success", False))
        n_succ_b += int(success)
        print(f"  B ep {ep+1}: success={success} Δeef={np.linalg.norm(eef_end-eef0):.3f} step={n_step}")
    print(f"  PATH B success rate: {n_succ_b}/5 = {n_succ_b/5:.1%}")

    print("\nCOMPARE:")
    print(f"  Native pi_0.5:  {n_succ_a/5:.1%}  ← upper bound for BC success on this task")
    print(f"  Our manual:     {n_succ_b/5:.1%}")
    if n_succ_a > 0 and n_succ_b == 0:
        print("  → manual chunking is broken; switch eval to use policy.select_action")


if __name__ == "__main__":
    raise SystemExit(main())
