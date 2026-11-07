#!/usr/bin/env python3
"""The three sampling paths again, each in a process of its own.

The reference runs every chunk in a fresh worker and an evaluated cell runs in
a fresh worker too, so a probe that drives several configurations in one
process shares whatever global random state the environment reads and cannot
be compared against either. Each configuration below is dispatched through the
same spawn context the drivers use, which is the only way the numbers it
reports are the numbers those drivers would produce.
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
    kind, task, weighting, bb, seeds, per, M, clip = payload
    D._single_threaded()
    n_ep = n_valid = 0
    if kind == "ref_chunk":
        for s in seeds:
            _, ne, nv = D._ref_chunk(
                (weighting, per, int(s), M, "cpu", task, bb, clip))
            n_ep += ne
            n_valid += nv
    else:
        model, blob = D.load_backbone(
            D.backbone_path(task, weighting, bb), "cpu")
        for s in seeds:
            torch.manual_seed(int(s))
            env = D.TASKS[task]["env"](seed=int(s),
                                       history=blob.get("history", 1))
            policy = D.NoisePolicy(blob["obs_dim"], blob["act_dim"])
            for _ in range(per):
                with torch.no_grad():
                    _, _, raw_mode, _ = D.rollout(env, policy, model, blob,
                                                  set(), "cpu", M, clip, {})
                n_ep += 1
                n_valid += raw_mode >= 0
    return n_valid / n_ep, n_ep


def main():
    task, weighting = sys.argv[1], sys.argv[2]
    bb, n, M, clip = int(sys.argv[3]), int(sys.argv[4]), 10, 50.0
    chunks = list(range(12345, 12345 + n // 8))
    jobs = {
        "ref path: _ref_chunk, 8-ep envs from 12345":
            ("ref_chunk", task, weighting, bb, chunks, 8, M, clip),
        "ref code, one env, seed 700":
            ("ref_chunk", task, weighting, bb, [700], n, M, clip),
        "eval path: rollout, one env, seed 700":
            ("rollout", task, weighting, bb, [700], n, M, clip),
        "eval code, 8-ep envs from 12345":
            ("rollout", task, weighting, bb, chunks, 8, M, clip),
    }
    names = list(jobs)
    out = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=D._MP) as ex:
        for name, (c, ne) in zip(names, ex.map(_work,
                                               [jobs[k] for k in names])):
            out[name] = dict(completion=c, episodes=ne)
            print(f"{name:44s} n={ne:4d} completion={c:.3f}", flush=True)
    dest = (ROOT / "runs" / "_d3il_v2"
            / f"path_probe_{task}_{weighting}_bb{bb}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
