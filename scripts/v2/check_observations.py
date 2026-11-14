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
import glob
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
# Name prefix of the run files the analysis reads. The matrix was measured
# again after the environment rule changed, and both cohorts sit in the same
# directory; the prefix is what keeps the two apart, so a bare invocation
# reads the current one.
PREFIX = "v22_"

sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "v2")]

import run_d3il_v2 as R                                       # noqa: E402

import envfactory as EF  # noqa: E402

PLAN = {"pushing": ("0.00", "0.50"), "avoiding": ("0.00", "0.60")}


def from_runs(paths):
    """What the completed runs themselves met, which is the real evidence.

    A rollout driven at the latent prior samples the runaway unreliably: three
    measurements of two dozen Pushing episodes found it twice and missed it
    once. A fine-tuned noise policy leaves the prior, so the training runs meet
    it far more often, and they are many more episodes. Their recorded
    ``obs_absmax`` is therefore what this check reports on, with the rollout
    measurement kept only to show the range the state occupies once the bound
    breaks the feedback loop.
    """
    out = {}
    for pat in paths:
        for f in sorted(Path(g) for g in glob.glob(pat)):
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out.setdefault(r.get("task", "?"), []).append(r)
    summary = {}
    by_arm = {}
    for task, rows in out.items():
        # By arm as well as by task. Symmetric clipping leaves a paired
        # contrast alone, because it moves both members of the pair; clipping
        # that falls on one arm and not the other is the case that would put
        # a difference in the table that the bound had made.
        per = {}
        for arm in sorted({r.get("arm") for r in rows}):
            sel = [r for r in rows if r.get("arm") == arm]
            c = np.asarray([r.get("obs_clipped_frac", 0.0) for r in sel], float)
            a = np.asarray([r.get("obs_absmax", 0.0) for r in sel], float)
            per[arm] = dict(runs=len(sel), clipped_frac_mean=float(c.mean()),
                            obs_absmax_median=float(np.median(a)))
        by_arm[task] = per
    for task, rows in out.items():
        am = np.asarray([r.get("obs_absmax", 0.0) for r in rows], float)
        cf = np.asarray([r.get("obs_clipped_frac", 0.0) for r in rows], float)
        summary[task] = dict(
            runs=len(rows),
            obs_absmax_median=float(np.median(am)),
            obs_absmax_max=float(am.max()) if am.size else None,
            runs_touching_bound=int((cf > 0).sum()),
            clipped_frac_median=float(np.median(cf)),
            clipped_frac_max=float(cf.max()) if cf.size else None,
            nonfinite_gradients=int(sum(r.get("n_nonfinite_grad", 0)
                                        for r in rows)),
            by_arm=by_arm[task])
    return summary


def measure(task, weightings, episodes, bound):
    args = R.build_parser().parse_args(
        ["--task", task, "--weightings", weightings[0], "--out", "/dev/null"])
    per_channel, per_step = None, []
    for w in weightings:
        model, blob = R.load_backbone(R.backbone_path(task, w, 0), "cpu")
        # One environment per process, like every other path. Looping over
        # weightings here built several in one process, so every weighting
        # after the first was measured on a reused environment: the probe
        # refuses that now rather than reporting it.
        env = EF.build(task, R.TASKS, 0, blob.get("history", 1))
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
    ap.add_argument("--runs", nargs="*",
                    default=[str(ROOT / "runs" / "_d3il_v2"
                                 / f"{PREFIX}calibration_*.jsonl"),
                             str(ROOT / "runs" / "_d3il_v2"
                                 / f"{PREFIX}main_*.jsonl")],
                    help="completed runs to read obs_absmax from")
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
    seen = from_runs(a.runs)
    report["from_completed_runs"] = seen
    if seen:
        print("\nwhat the completed runs met")
        for task, m in sorted(seen.items()):
            print(f"  {task:9s} {m['runs']:4d} runs  obs_absmax median "
                  f"{m['obs_absmax_median']:.3g} max {m['obs_absmax_max']:.3g}; "
                  f"{m['runs_touching_bound']}/{m['runs']} touch the bound, "
                  f"median {m['clipped_frac_median']:.2e} of steps; "
                  f"{m['nonfinite_gradients']} non-finite gradients")
            if m["nonfinite_gradients"]:
                ok = False
                print("           FAIL: a run stepped on a non-finite gradient")
    report["verdict"] = "PASS" if ok else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}\nVERDICT: {report['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
