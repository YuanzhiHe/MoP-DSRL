#!/usr/bin/env python3
"""Verify that lerobot/pi05_libero_finetuned actually works end-to-end on LIBERO.

Loads the *_finetuned* checkpoint (which has populated normalizer stats —
unlike `pi05_libero` which has empty stats and produces 0% on LIBERO), feeds
LIBERO obs through the official preproc, runs select_action, and steps the env
to confirm:

  1. preproc actually normalizes state (state OUT != state IN)
  2. policy outputs non-trivial action
  3. arm moves meaningfully when fed the action

If this works, we change `--pretrained_path lerobot/pi05_libero` →
`lerobot/pi05_libero_finetuned` everywhere and rerun.

Usage:  python scripts/verify_finetuned.py
"""

from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch


def main():
    REPO = "lerobot/pi05_libero_finetuned"

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"== Loading {REPO} ==")
    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.device = "cuda"
    cfg.n_action_steps = 10
    cfg.validate_features()
    print(f"  cfg.input_features keys: {list(cfg.input_features.keys())}")
    print(f"  cfg.output_features keys: {list(cfg.output_features.keys())}")

    policy = PI05Policy.from_pretrained(REPO, config=cfg)
    policy.eval().to("cuda")
    preproc, postproc = make_pre_post_processors(cfg, pretrained_path=REPO)

    # Show normalizer stats actually loaded
    for step in preproc.steps:
        if type(step).__name__ == "NormalizerProcessorStep":
            stats = getattr(step, "stats", None) or getattr(step, "_stats", {})
            print(f"  normalizer stats keys: {list(stats.keys())[:10] if hasattr(stats, 'keys') else type(stats).__name__}")
            if hasattr(stats, "items"):
                for k, v in list(stats.items())[:3]:
                    print(f"    [{k}]")
                    if hasattr(v, "items"):
                        for k2, v2 in v.items():
                            if isinstance(v2, torch.Tensor):
                                print(f"      {k2}: shape={tuple(v2.shape)} mean={v2.float().mean().item():.4f}")

    print("\n== LIBERO env ==")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    b = benchmark.get_benchmark("libero_goal")()
    task = b.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
    obs = env.reset()
    instruction = task.language
    print(f"  task: {task.name}")
    print(f"  instruction: {instruction!r}")
    print(f"  initial eef_pos: {obs['robot0_eef_pos']}")

    def build_batch(o):
        batch = {}
        # The finetuned checkpoint expects image/image2/empty_camera_0 plus state.
        # Use LIBERO's agentview as "image" and wrist as "image2".  empty_camera_0
        # is a zero-image placeholder.
        if "agentview_image" in o:
            img = torch.from_numpy(o["agentview_image"].copy()).float() / 255.0
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            batch["observation.images.image"] = img.unsqueeze(0).to("cuda")
        if "robot0_eye_in_hand_image" in o:
            img = torch.from_numpy(o["robot0_eye_in_hand_image"].copy()).float() / 255.0
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            batch["observation.images.image2"] = img.unsqueeze(0).to("cuda")
        # Empty placeholder camera (3rd camera not available in LIBERO).
        batch["observation.images.empty_camera_0"] = torch.zeros(1, 3, 224, 224, device="cuda")
        state_parts = []
        for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if k in o:
                state_parts.append(torch.from_numpy(o[k].copy()).float().flatten())
        batch["observation.state"] = torch.cat(state_parts, dim=0).unsqueeze(0).to("cuda")
        batch["task"] = [instruction]
        return batch

    batch = build_batch(obs)
    print(f"\n== Step: preproc effect ==")
    print(f"  state BEFORE: {batch['observation.state'].cpu().numpy().flatten()}")
    print(f"  image[0,0,100,100] BEFORE: {batch['observation.images.image'][0,0,100,100].item():.4f}")
    out = preproc(batch)
    print(f"  state AFTER:  {out['observation.state'].cpu().numpy().flatten()}")
    print(f"  image[0,0,100,100] AFTER:  {out['observation.images.image'][0,0,100,100].item():.4f}")

    print(f"\n== policy.select_action ==")
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(out)
    print(f"  raw action: shape={tuple(action.shape)} range=[{action.min().item():.4f},{action.max().item():.4f}]")
    print(f"  raw action: {action.detach().cpu().numpy().flatten()}")

    # postproc: action is a tensor subclass PolicyAction; just call directly
    try:
        action_post = postproc(action)
    except Exception:
        try:
            action_post = postproc({"action": action})
            if isinstance(action_post, dict):
                action_post = action_post["action"]
        except Exception as e:
            print(f"  postproc failed: {e}; using raw")
            action_post = action

    print(f"\n== post action ==")
    print(f"  post action range: [{action_post.min().item():.4f},{action_post.max().item():.4f}]")
    print(f"  post action: {action_post.detach().cpu().numpy().flatten()}")

    print(f"\n== Run 50-step rollout with action chunking ==")
    obs = env.reset()
    policy.reset()
    eef0 = obs["robot0_eef_pos"].copy()
    print(f"  eef_pos at reset: {eef0}")
    for step in range(50):
        batch = build_batch(obs)
        batch_pre = preproc(batch)
        with torch.no_grad():
            action = policy.select_action(batch_pre)
        try:
            action_post = postproc(action)
        except Exception:
            action_post = action
        a = action_post.detach().cpu().numpy().reshape(-1)[:7]
        obs, reward, done, info = env.step(a)
        if step % 10 == 0:
            print(f"  step {step}: eef_pos={obs['robot0_eef_pos']} reward={reward} done={done}")
        if done:
            print(f"  DONE at step {step}, success={info.get('success', False)}")
            break
    eef_final = obs["robot0_eef_pos"]
    print(f"  eef_pos final: {eef_final}")
    print(f"  Δeef total: {np.linalg.norm(eef_final - eef0):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
