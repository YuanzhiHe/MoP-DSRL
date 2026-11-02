#!/usr/bin/env python3
"""Measure the normalised observation range and check the bound sits outside it.

Two Pushing channels are unwrapped orientation angles whose demonstration
standard deviations are hundreds of radians. Along a rollout they accumulate
without bound, and the networks that read them saturate. The bound in
``--obs-clip`` exists for that; this script is the evidence that it is wide
enough to be inert on a rollout that is behaving.

Rollouts are driven at z ~ N(0, I), the latent distribution the frozen decoder
was trained under. Each task is measured twice, once with the bound in force
and once without it, because the runaway is a feedback loop: the saturated
observation produces an extreme action, which drives the state further out. The
unbounded pass shows the hazard, the bounded pass shows the range the state
actually occupies once the loop is broken, and the check passes when the second
sits well inside the bound.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

# Jobs run from a frozen copy of scripts/, so the output directory has to come
# from the environment rather than from where this file happens to sit.
ROOT = Path(os.environ.get("MOP_DSRL_ROOT") or Path(__file__).resolve().parents[2])
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "v2")]

import run_d3il_v2 as R                                       # noqa: E402

PLAN = {"pushing": ("0.00", "0.50"), "avoiding": ("0.00", "0.60")}


def measure(task, weightings, episodes, bound):
    args = R.build_parser().parse_args(
        ["--task", task, "--weightings", weightings[0], "--out", "/dev/null"])
    per_channel, per_step = None, []
    for w in weightings:
        model, blob = R.load_backbone(R.backbone_path(task, w, 0), "cpu")
        env = R.TASKS[task]["env"](seed=0, history=blob.get("history", 1))
        rng = np.random.default_rng(0)
        if per_channel is None:
            per_channel = np.zeros(len(blob["obs_std"]))
        for _ in range(episodes):
            obs, done = env.reset(), False
            while not done:
                v = (np.asarray(obs, float) - blob["obs_mean"]) / blob["obs_std"]
                per_channel = np.maximum(per_channel, np.abs(v))
                per_step.append(float(np.abs(v).max()))
                o = torch.tensor(np.clip(v, -bound, bound), dtype=torch.float32)[None]
                z = torch.tensor(rng.standard_normal((1, blob["act_dim"])),
                                 dtype=torch.float32)
                obs, _, done, _ = env.step(R.decode(model, blob, o, z, args.M)[0])
    q = np.asarray(per_step)
    return dict(task=task, weightings=list(weightings), episodes=episodes,
                steps=int(q.size), median=float(np.median(q)),
                p99=float(np.percentile(q, 99)), max=float(q.max()),
                per_channel_max=[float(x) for x in per_channel],
                steps_over_bound=int((q > bound).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--bound", type=float, default=50.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs" / "_d3il_v2" / "observation_range.json")
    a = ap.parse_args()

    report, ok = {"bound": a.bound}, True
    for task, ws in PLAN.items():
        free = measure(task, ws, a.episodes, float("inf"))
        held = measure(task, ws, a.episodes, a.bound)
        ch = np.asarray(free["per_channel_max"])
        free["runaway_channels"] = [int(i) for i in np.where(ch > a.bound)[0]]
        held["headroom"] = a.bound / held["max"] if held["max"] > 0 else None
        report[task] = dict(unbounded=free, bounded=held)
        print(f"{task}")
        print(f"  unbounded  {free['steps']:6d} steps  max |obs| median "
              f"{free['median']:.2f}  p99 {free['p99']:.3g}  max {free['max']:.3g}"
              f"   runaway channels {free['runaway_channels'] or 'none'}")
        print(f"  bounded    {held['steps']:6d} steps  max |obs| median "
              f"{held['median']:.2f}  p99 {held['p99']:.3g}  max {held['max']:.3g}"
              f"   bound is {held['headroom']:.1f}x outside that")
        if held["headroom"] is not None and held["headroom"] < 2.0:
            ok = False
            print("  FAIL: the bound sits too close to the range the state occupies")
    report["verdict"] = "PASS" if ok else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}\nVERDICT: {report['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
