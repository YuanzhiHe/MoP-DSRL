#!/usr/bin/env python3
"""What makes the first environment of a worker process complete fewer episodes.

A worker that builds one environment and evaluates 400 episodes on it reports a
completion rate about two tenths below a worker that builds a fresh environment
every eight episodes, and the gap survives swapping the latent sampler. These
configurations separate the candidate causes: whether the deficit belongs to
the first episodes of any environment, to the first environment of a process,
or to the act of constructing one at all.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d3il_v2 as D  # noqa: E402

# A job runs from a frozen copy of the source, so the repository root
# has to come from the driver rather than from this file's location.
ROOT = D.ROOT


def _work(payload):
    kind, task, weighting, bb, n, M, clip = payload
    D._single_threaded()
    model, blob = D.load_backbone(D.backbone_path(task, weighting, bb), "cpu")
    torch.manual_seed(700)

    def make():
        return D.TASKS[task]["env"](seed=700, history=blob.get("history", 1))

    def run(env, k, policy):
        valid = 0
        for _ in range(k):
            with torch.no_grad():
                _, _, raw_mode, _ = D.rollout(env, policy, model, blob, set(),
                                              "cpu", M, clip, {})
            valid += raw_mode >= 0
        return valid

    policy = D.NoisePolicy(blob["obs_dim"], blob["act_dim"])
    if kind == "plain":                      # the configuration in use now
        return run(make(), n, policy) / n
    if kind == "discard_first_50":           # a deficit in an environment's first episodes
        env = make()
        run(env, 50, policy)
        return run(env, n, policy) / n
    if kind == "throwaway_env_used":         # a deficit in a process's first environment
        run(make(), 5, policy)
        return run(make(), n, policy) / n
    if kind == "throwaway_env_unused":       # constructing one is enough
        make()
        return run(make(), n, policy) / n
    raise ValueError(kind)


def main():
    task, weighting = sys.argv[1], sys.argv[2]
    bb, n, M, clip = int(sys.argv[3]), int(sys.argv[4]), 10, 50.0
    kinds = ["plain", "discard_first_50", "throwaway_env_used",
             "throwaway_env_unused"]
    out = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=D._MP) as ex:
        for kind, c in zip(kinds, ex.map(
                _work, [(k, task, weighting, bb, n, M, clip) for k in kinds])):
            out[kind] = c
            print(f"{kind:22s} completion={c:.3f}", flush=True)
    dest = ROOT / "runs" / "_d3il_v2" / f"warmup_probe_{task}_{weighting}_bb{bb}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
