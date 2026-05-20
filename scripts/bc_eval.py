#!/usr/bin/env python3
"""Standalone BC sanity test: pi_0.5 + LIBERO via lerobot's OFFICIAL pipeline.

The goal of this script is to verify that pi_0.5 can solve LIBERO tasks at
all on this machine, BEFORE we plug the SAC outer loop on top.  It uses
lerobot's own preprocessor / postprocessor chain — same code path lerobot
trains pi_0.5 with — so any mismatch with the train-time distribution will
not be due to our adapter code.

If this script reports success_rate > 0.5 on libero_goal task_id=0, the
LIBERO ↔ pi_0.5 weight loading is correct and any 0% in the main training
driver must be in our custom adapter (img normalization, action de-norm,
etc.).  If this script also reports 0%, the issue is in checkpoint/env
compatibility itself.

Usage:
    python scripts/bc_eval.py --task_id 0 --episodes 10
"""

from __future__ import annotations

# IPv4 patch (UK IPv6 hang workaround)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path", default="lerobot/pi05_libero")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_action_steps", type=int, default=10,
                   help="Execute this many actions per inference (pi_0.5 default).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args()

    # ---- lerobot policy + preprocessors (the OFFICIAL path) ----
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except Exception:
        make_pre_post_processors = None
    try:
        from lerobot.policies.pi05 import make_pi05_pre_post_processors
    except Exception:
        make_pi05_pre_post_processors = None

    print(f"[load] {args.policy_path}", flush=True)
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    cfg.n_action_steps = args.n_action_steps
    cfg.validate_features()
    policy = PI05Policy.from_pretrained(args.policy_path, config=cfg)
    policy.eval()
    policy.to(args.device)

    preproc = postproc = None
    try:
        if make_pre_post_processors is not None:
            preproc, postproc = make_pre_post_processors(cfg, pretrained_path=args.policy_path)
            print("[load] used make_pre_post_processors")
    except Exception as e:
        print(f"[load] make_pre_post_processors failed: {e}")
    if preproc is None and make_pi05_pre_post_processors is not None:
        try:
            preproc, postproc = make_pi05_pre_post_processors(config=cfg, dataset_stats=None)
            print("[load] used make_pi05_pre_post_processors")
        except Exception as e:
            print(f"[load] make_pi05_pre_post_processors failed: {e}")
    if preproc is None:
        print("[warn] no preprocessor available — pi_0.5 will see raw obs (likely 0% success)")

    # ---- LIBERO env ----
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = libero_benchmark.get_benchmark(args.suite)()
    task = bench.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    instruction = getattr(task, "language", None) or task.name.replace("_", " ")
    print(f"[env] suite={args.suite} task={task.name} instruction='{instruction}'")
    print(f"[env] bddl={bddl}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=224, camera_widths=224,
    )

    H = int(cfg.chunk_size)
    d_a = int(cfg.output_features["action"].shape[0])
    print(f"[cfg] H={H} d_a={d_a} max_action_dim={cfg.max_action_dim}")

    # ---- diagnostic: what obs keys does LIBERO actually emit ----
    obs0 = env.reset()
    print(f"[diag] LIBERO obs keys: {sorted(obs0.keys())}")
    for k in sorted(obs0.keys()):
        v = obs0[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype} range=[{v.min():.3f},{v.max():.3f}]")

    # Try to import lerobot's PolicyAction type — postproc requires it.
    PolicyAction = None
    try:
        from lerobot.processor.policy_processor import PolicyAction
        print("[load] PolicyAction available")
    except Exception as e:
        print(f"[warn] PolicyAction import failed: {e}; postproc will be skipped")

    # ---- rollout helper ----
    def build_batch(obs):
        """LIBERO obs dict → lerobot batch dict that the preprocessor will normalize.

        Tries multiple image key conventions; lerobot's pi05_libero was likely
        trained with specific names (e.g. "observation.images.image" vs ".cam_high").
        """
        batch = {}
        if "agentview_image" in obs:
            img = torch.from_numpy(obs["agentview_image"].copy()).float() / 255.0
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            # Flip vertically — robosuite renders bottom-up by default; pi_0.5 trained
            # on top-down (PIL/PNG style).
            img = torch.flip(img, dims=[-2])
            batch["observation.images.image"] = img.unsqueeze(0).to(args.device)
        if "robot0_eye_in_hand_image" in obs:
            img = torch.from_numpy(obs["robot0_eye_in_hand_image"].copy()).float() / 255.0
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            img = torch.flip(img, dims=[-2])
            batch["observation.images.wrist_image"] = img.unsqueeze(0).to(args.device)
        state_parts = []
        for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if k in obs:
                v = obs[k]
                if isinstance(v, np.ndarray):
                    state_parts.append(torch.from_numpy(v.copy()).float().flatten())
        state = torch.cat(state_parts, dim=0).unsqueeze(0).to(args.device) \
            if state_parts else torch.zeros(1, 8, device=args.device)
        batch["observation.state"] = state
        batch["task"] = [instruction]
        return batch

    # ---- inference helper ----
    def get_action(obs):
        batch = build_batch(obs)
        if preproc is not None:
            batch = preproc(batch)
        with torch.no_grad():
            action_tensor = policy.select_action(batch)  # [1, d_a] normalized
        # Try postproc with the correct PolicyAction wrapper.
        if postproc is not None and PolicyAction is not None:
            try:
                pa = PolicyAction(action=action_tensor)
                pa_out = postproc(pa)
                action_tensor = pa_out.action
            except Exception as e:
                if not get_action._warned:
                    print(f"[warn] postproc(PolicyAction) failed: {e}; using raw")
                    get_action._warned = True
        return action_tensor.detach().cpu().numpy().reshape(-1)[:d_a]
    get_action._warned = False

    # Diagnostic: print the action range across 5 random obs.
    print("[diag] sampling 5 actions for range check…")
    for _ in range(5):
        a = get_action(obs0)
        print(f"  action range=[{a.min():.3f}, {a.max():.3f}]  values={a}")

    n_success = 0
    t0 = time.time()
    for ep in range(args.episodes):
        torch.manual_seed(args.seed * 1000 + ep)
        np.random.seed(args.seed * 1000 + ep)
        obs = env.reset()
        policy.reset()  # clear action queue / KV cache
        success = False
        info = {}
        for step in range(args.max_steps):
            a = get_action(obs)
            obs, reward, done, info = env.step(a)
            if done:
                success = bool(info.get("success", False))
                break
        n_success += int(success)
        if args.verbose:
            print(f"  ep {ep+1}/{args.episodes}: success={success} step={step+1}", flush=True)

    sr = n_success / max(args.episodes, 1)
    dt = time.time() - t0
    print()
    print(f"=========================================")
    print(f"  task        : {args.suite}/{task.name}")
    print(f"  policy      : {args.policy_path}")
    print(f"  episodes    : {args.episodes}")
    print(f"  successes   : {n_success}")
    print(f"  success rate: {sr:.3f}")
    print(f"  wall sec    : {dt:.1f}  ({dt/max(args.episodes,1):.1f} s/episode)")
    print(f"=========================================")

    if sr < 0.3:
        print()
        print("[!] success_rate is suspiciously low.  Likely causes:")
        print("    1. policy_preprocessor.json / postprocessor.json not loaded")
        print("       (this script uses lerobot.policies.factory.make_pre_post_processors —")
        print("       if that's None above, the preproc chain is missing entirely)")
        print("    2. LIBERO obs key names mismatch with what pi05_libero expects")
        print("    3. checkpoint was fine-tuned on a different LIBERO setup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
