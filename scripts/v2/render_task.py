#!/usr/bin/env python3
"""A picture of the task, and the routes a frozen policy actually takes.

The paper describes both tasks in words and draws the training loop as a block
diagram. Neither shows what a behavioural mode is. This renders the scene from
the simulator itself and rolls the frozen policy out under its own latent prior,
recording the end-effector path and the route each episode realised, so the
modes can be drawn as the paths they are.

One process builds one environment (see envfactory), so this script handles one
task per invocation and writes a bundle the figure script reads.

    MUJOCO_GL=egl python scripts/v2/render_task.py --task avoiding
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envfactory as EF  # noqa: E402
from run_d3il_v2 import norm_obs  # noqa: E402
from run_d3il_headtohead import (  # noqa: E402
    TASKS, NoisePolicy, backbone_path, coarsen, decode, load_backbone,
)

# Where the camera sits for each task. The cage camera the benchmark ships
# looks along the robot at thirty degrees, which hides which side of a post the
# arm passed on. Both views here look down the z axis, where a route is the
# shape it is.
VIEWS = {
    "avoiding": dict(lookat=(0.5, 0.0, 0.0), distance=0.95,
                     azimuth=90.0, elevation=-89.0),
    "pushing": dict(lookat=(0.5, 0.0, 0.0), distance=0.95,
                    azimuth=90.0, elevation=-89.0),
}


def render(scene, task: str, size: int) -> np.ndarray:
    import mujoco
    cam = mujoco.MjvCamera()
    v = VIEWS[task]
    cam.lookat[:] = v["lookat"]
    cam.distance = v["distance"]
    cam.azimuth = v["azimuth"]
    cam.elevation = v["elevation"]
    # The scene's offscreen framebuffer is declared in the model XML at 640
    # square, and the renderer refuses anything larger. The paper wants a
    # figure that survives print, so the buffer is raised here.
    scene.model.vis.global_.offwidth = max(scene.model.vis.global_.offwidth, size)
    scene.model.vis.global_.offheight = max(scene.model.vis.global_.offheight, size)
    # mujoco 2.3 predates both the renderer's context manager and its close
    # method, and tearing the context down by hand raises inside EGL. The
    # renderer is left to the garbage collector, which is what this version
    # expects.
    r = mujoco.Renderer(scene.model, size, size)
    r.update_scene(scene.data, camera=cam)
    return r.render().copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--weighting", required=True)
    ap.add_argument("--backbone-seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-clip", type=float, default=50.0)
    args = ap.parse_args()

    ck = backbone_path(args.task, args.weighting, args.backbone_seed)
    model, blob = load_backbone(ck, "cpu")
    torch.manual_seed(args.seed)
    env = EF.build(args.task, TASKS, args.seed, blob.get("history", 1))
    # Initialised, never trained: the noise policy is the standard normal the
    # decoder was trained under, so these rollouts are the frozen policy's own.
    policy = NoisePolicy(blob["obs_dim"], blob["act_dim"])

    obs = env.reset()
    frame = render(env._env.scene, args.task, args.size)

    paths, modes = [], []
    for ep in range(args.episodes):
        obs = env.reset()
        xy, done = [], False
        while not done:
            xy.append(np.asarray(env._env.robot_state()[:2], dtype=np.float64))
            o = torch.tensor(norm_obs(obs, blob, args.obs_clip),
                             dtype=torch.float32)[None]
            with torch.no_grad():
                z = policy.dist(o).sample()
                a = decode(model, blob, o, z, 10)[0]
            obs, _, done, info = env.step(a)
        paths.append(np.asarray(xy))
        modes.append(int(info["mode_id"]))
        if (ep + 1) % 20 == 0:
            print(f"  {ep + 1}/{args.episodes} episodes", flush=True)

    # The obstacle field and the goal line, read off the environment rather
    # than copied into the figure script, so the drawing cannot drift from
    # what the rollouts ran against.
    geom = {}
    if args.task == "avoiding":
        e = env._env
        geom = dict(
            posts=[(e.l1_xpos, e.l1_ypos),
                   (e.l2_top_xpos, e.l2_ypos), (e.l2_bottom_xpos, e.l2_ypos),
                   (e.l3_top_xpos, e.l3_ypos), (e.l3_mid_xpos, e.l3_ypos),
                   (e.l3_bottom_xpos, e.l3_ypos)],
            goal_y=e.goal_ypos)

    out = ROOT / "runs" / "_render" / f"{args.task}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, frame=frame,
        paths=np.array(paths, dtype=object), modes=np.asarray(modes),
        groups=np.asarray([coarsen(m, TASKS[args.task]["n_modes"] if
                                   args.task == "pushing" else 6)
                           if m >= 0 else -1 for m in modes]),
        weighting=args.weighting, backbone_seed=args.backbone_seed,
        geometry=json.dumps(geom), allow_pickle=True)
    labelled = int((np.asarray(modes) >= 0).sum())
    print(f"wrote {out}: {labelled} of {len(modes)} episodes carry a route")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
