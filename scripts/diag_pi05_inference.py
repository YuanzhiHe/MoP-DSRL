#!/usr/bin/env python3
"""Diagnose pi_0.5 inference at a single LIBERO reset state.

Walks through what each layer of pi_0.5 sees and outputs, so we can pinpoint
where the BC=0% bug is between (obs) → (preproc) → (policy) → (postproc) →
(env.step) → (next obs).

Prints:
  1. lerobot module structure: which postproc class is wrapping action_tensor
     (PolicyAction lives in different paths across lerobot versions)
  2. raw obs key shapes/ranges
  3. batch right before preproc → image min/max, state shape
  4. batch right after preproc → same fields, see if image was normalized
  5. policy output action: shape, range
  6. action after postproc: shape, range (== "physical units" if postproc works)
  7. delta in robot eef_pos after 10 env steps with that action — sanity check
     that pi_0.5 is actually moving the arm purposefully (vs flailing randomly)

Usage:
    python scripts/diag_pi05_inference.py
"""

from __future__ import annotations

import os
import sys
import pathlib
import importlib
import inspect

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch


def find_policy_action_class():
    """Search lerobot for the PolicyAction type — module path varies across versions."""
    candidates = [
        "lerobot.processor.policy_processor",
        "lerobot.processor.policies",
        "lerobot.processor",
        "lerobot.processor.pipeline",
        "lerobot.policies.pi05.processor_pi05",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod):
            if name == "PolicyAction" and inspect.isclass(obj):
                print(f"  PolicyAction found at: {mod_name}.PolicyAction")
                return obj
    # Brute force walk
    import pkgutil, lerobot
    for m in pkgutil.walk_packages(lerobot.__path__, prefix="lerobot."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        if hasattr(mod, "PolicyAction"):
            cls = getattr(mod, "PolicyAction")
            if inspect.isclass(cls):
                print(f"  PolicyAction found (walk): {m.name}.PolicyAction")
                return cls
    print("  PolicyAction NOT FOUND in any lerobot module")
    return None


def main() -> int:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    print("== Step 1: locate PolicyAction class ==")
    PolicyAction = find_policy_action_class()

    print("\n== Step 2: load policy + preprocessors ==")
    cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_libero")
    cfg.device = "cuda"
    cfg.n_action_steps = 10
    cfg.validate_features()
    policy = PI05Policy.from_pretrained("lerobot/pi05_libero", config=cfg)
    policy.eval().to("cuda")
    preproc, postproc = make_pre_post_processors(cfg, pretrained_path="lerobot/pi05_libero")
    print(f"  preproc type: {type(preproc).__name__}")
    print(f"  postproc type: {type(postproc).__name__}")
    print(f"  preproc has 'steps': {hasattr(preproc, 'steps')}")
    if hasattr(preproc, "steps"):
        print(f"  preproc steps ({len(preproc.steps)}):")
        for s in preproc.steps:
            print(f"    - {type(s).__name__}")
    if hasattr(postproc, "steps"):
        print(f"  postproc steps ({len(postproc.steps)}):")
        for s in postproc.steps:
            print(f"    - {type(s).__name__}")

    print("\n== Step 3: load LIBERO env + reset ==")
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

    print("\n== Step 4: build raw batch ==")
    def build_batch(o):
        batch = {}
        for src_key, dst_key in [
            ("agentview_image", "observation.images.image"),
            ("robot0_eye_in_hand_image", "observation.images.wrist_image"),
        ]:
            if src_key in o:
                img = torch.from_numpy(o[src_key].copy()).float() / 255.0
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                batch[dst_key] = img.unsqueeze(0).to("cuda")
        state_parts = []
        for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if k in o:
                state_parts.append(torch.from_numpy(o[k].copy()).float().flatten())
        batch["observation.state"] = torch.cat(state_parts, dim=0).unsqueeze(0).to("cuda")
        batch["task"] = [instruction]
        return batch

    batch = build_batch(obs)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  raw {k}: shape={tuple(v.shape)} dtype={v.dtype} range=[{v.min().item():.3f},{v.max().item():.3f}]")
        else:
            print(f"  raw {k}: {v}")

    print("\n== Step 5: run preproc ==")
    try:
        batch_pre = preproc(batch)
        print("  preproc succeeded")
        for k, v in batch_pre.items():
            if isinstance(v, torch.Tensor):
                print(f"  after preproc {k}: shape={tuple(v.shape)} dtype={v.dtype} range=[{v.min().item():.3f},{v.max().item():.3f}]")
            else:
                print(f"  after preproc {k}: {v}")
    except Exception as e:
        print(f"  preproc RAISED: {type(e).__name__}: {e}")
        batch_pre = batch

    print("\n== Step 6: policy.select_action ==")
    policy.reset()
    with torch.no_grad():
        action_tensor = policy.select_action(batch_pre)
    print(f"  raw action: shape={tuple(action_tensor.shape)} dtype={action_tensor.dtype}")
    print(f"  raw action range: [{action_tensor.min().item():.4f}, {action_tensor.max().item():.4f}]")
    print(f"  raw action values: {action_tensor.detach().cpu().numpy().flatten()}")

    print("\n== Step 7: postproc ==")
    if PolicyAction is not None:
        try:
            pa = PolicyAction(action=action_tensor)
            pa_out = postproc(pa)
            action_post = pa_out.action if hasattr(pa_out, "action") else pa_out
            print(f"  postproc OK")
            print(f"  post action: shape={tuple(action_post.shape)} range=[{action_post.min().item():.4f},{action_post.max().item():.4f}]")
            print(f"  post action values: {action_post.detach().cpu().numpy().flatten()}")
        except Exception as e:
            print(f"  postproc(PolicyAction) RAISED: {type(e).__name__}: {e}")
            action_post = action_tensor
    else:
        # Try dict form
        try:
            pa_out = postproc({"action": action_tensor})
            action_post = pa_out["action"] if isinstance(pa_out, dict) else pa_out
            print(f"  postproc(dict) OK")
            print(f"  post action range: [{action_post.min().item():.4f},{action_post.max().item():.4f}]")
        except Exception as e:
            print(f"  postproc(dict) RAISED: {type(e).__name__}: {e}")
            action_post = action_tensor

    print("\n== Step 8: take 10 env steps with that chunk-of-first-action ==")
    a_np = action_post.detach().cpu().numpy().reshape(-1)[:7]
    eef0 = obs["robot0_eef_pos"].copy()
    print(f"  action being fed to env.step: {a_np}")
    print(f"  eef_pos before: {eef0}")
    for i in range(10):
        obs, reward, done, info = env.step(a_np)
        if done:
            break
    eef1 = obs["robot0_eef_pos"].copy()
    print(f"  eef_pos after 10 steps: {eef1}")
    print(f"  Δeef_pos: {eef1 - eef0}  (norm={np.linalg.norm(eef1 - eef0):.4f})")
    print(f"  reward: {reward}  done: {done}")

    print("\nINTERPRETATION:")
    print("  • If preproc output image range is roughly [-2.5, 2.5], ImageNet norm was applied.")
    print("  • If post action range is [-1, 1], LIBERO expects delta actions ∈ [-1, 1] → OK")
    print("  • If Δeef_pos is < 0.001 m the arm isn't actually responding to the action")
    print("    (likely action units / coordinate frame mismatch).")
    print("  • If Δeef_pos is > 0.05 m but reward is 0, arm moves but not toward task goal")
    print("    (likely instruction/image mismatch with pi05_libero training).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
