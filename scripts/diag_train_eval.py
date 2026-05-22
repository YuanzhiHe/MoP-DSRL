#!/usr/bin/env python3
"""Replicate exactly what train.py's `evaluate()` does, with the SAME backbone
construction path, to see where in {build_batch → preproc → policy → postproc
→ env.step} we differ from verify_finetuned.py (which got Δeef=0.27m, success
shape).

Usage:  python scripts/diag_train_eval.py
"""
from __future__ import annotations
import os, sys, pathlib
# Insert repo root so `import scripts.backbones` works (mirrors train.py).
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import _ipv4_patch  # noqa: F401,E402
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def main():
    print("== Building backbone via scripts.backbones.make_backbone (same as train.py) ==")
    from scripts.backbones import make_backbone

    # We DON'T know obs_dim until we create env, but make_backbone doesn't use it.
    denoiser = make_backbone(
        name="pi05",
        H=50, d_a=7, M=10, device="cuda",
        obs_dim=190,  # placeholder; not used by pi05
        pretrained_path="lerobot/pi05_libero_finetuned",
    )

    print("\n== Inspect denoiser.velocity_field internals ==")
    vf = denoiser.velocity_field
    print(f"  pretrained_path: {getattr(vf, '_pretrained_path', '<missing>')}")
    print(f"  policy.config._name_or_path: {getattr(vf._policy.config, '_name_or_path', '<missing>')}")

    print("\n== Trigger preproc/postproc build ==")
    pre = vf._get_preprocessor()
    post = vf._get_postprocessor()
    for i, s in enumerate(pre.steps):
        cname = type(s).__name__
        if cname == "NormalizerProcessorStep":
            stats = getattr(s, "stats", {}) or getattr(s, "_stats", {})
            print(f"  preproc[{i}] {cname} stats keys: {list(stats.keys()) if hasattr(stats, 'keys') else stats}")
    for i, s in enumerate(post.steps):
        cname = type(s).__name__
        if cname == "UnnormalizerProcessorStep":
            stats = getattr(s, "stats", {}) or getattr(s, "_stats", {})
            print(f"  postproc[{i}] {cname} stats keys: {list(stats.keys()) if hasattr(stats, 'keys') else stats}")

    print("\n== LIBERO env ==")
    from scripts.envs import make_env
    env = make_env("libero_goal", task_id=0, seed=0, max_steps=300)
    obs = env.reset()
    print(f"  task_instruction: {env.task_instruction!r}")
    print(f"  initial eef_pos: {env.last_obs_dict['robot0_eef_pos']}")

    print("\n== Call denoiser.velocity_field.prepare_libero_batch directly ==")
    batch = vf.prepare_libero_batch(env.last_obs_dict, env.task_instruction)
    print(f"  batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} range=[{v.float().min().item():.4f},{v.float().max().item():.4f}]")
        else:
            print(f"    {k}: {v}")

    print("\n== Call denoiser.denoise_with_obs_dict (this is what train uses) ==")
    z = torch.randn(1, 350, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        a = denoiser.denoise_with_obs_dict(batch, z)
    print(f"  a from denoise_with_obs_dict: shape={tuple(a.shape)} range=[{a.float().min().item():.4f},{a.float().max().item():.4f}]")
    print(f"  a first 14 values: {a.flatten()[:14].cpu().numpy()}")

    print("\n== Apply postproc (lerobot unnormalizer, reshape to [B,H,d_a]) ==")
    try:
        a3 = a.view(1, 50, 7)
        pa = post({"action": a3})
        a3_post = pa["action"] if isinstance(pa, dict) else pa
        print(f"  postproc OK: shape={tuple(a3_post.shape)} range=[{a3_post.float().min().item():.4f},{a3_post.float().max().item():.4f}]")
        print(f"  a_post first 14 values: {a3_post.flatten()[:14].cpu().numpy()}")
        a_post = a3_post.reshape(1, 350)
    except Exception as e:
        print(f"  postproc FAILED: {type(e).__name__}: {e}")
        a_post = a

    print("\n== Run 50 env steps with the chunked action sequence ==")
    obs = env.reset()
    eef0 = env.last_obs_dict["robot0_eef_pos"].copy()
    print(f"  eef_pos reset: {eef0}")

    for step in range(5):
        # Build batch from CURRENT obs
        batch = vf.prepare_libero_batch(env.last_obs_dict, env.task_instruction)
        z = torch.randn(1, 350, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            a = denoiser.denoise_with_obs_dict(batch, z)
        try:
            a3 = a.view(1, 50, 7)
            pa = post({"action": a3})
            a3_post = pa["action"] if isinstance(pa, dict) else pa
            chunk = a3_post.detach().cpu().numpy().reshape(50, 7)
        except Exception as e:
            print(f"  postproc inside loop failed: {e}; using normalized")
            chunk = a.detach().cpu().numpy().reshape(50, 7)
        # Execute 10 actions per chunk
        for i in range(10):
            obs, reward, done, info = env.step(chunk[i])
            if done:
                break
        eef_now = env.last_obs_dict["robot0_eef_pos"]
        delta = np.linalg.norm(eef_now - eef0)
        print(f"  chunk {step}: eef_pos={eef_now} Δfrom_start={delta:.4f}")
        if done:
            print(f"    DONE step={step} success={info.get('success', False)}")
            break

    print("\nINTERPRETATION:")
    print("  • postproc stats keys should include 'action' — if empty, that's the bug")
    print("  • a_post should be in [-1, 1] (normalized) — if it's huge or constant, postproc wrong")
    print("  • Δfrom_start after 5 chunks (50 steps) should be > 0.05 m")


if __name__ == "__main__":
    raise SystemExit(main())
