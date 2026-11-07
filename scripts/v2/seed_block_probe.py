#!/usr/bin/env python3
"""Why the reference measurement and the evaluation disagree on completion.

The reference draws its 900 episodes as 113 environments of 8 episodes each,
seeded from 12345, while an evaluated cell draws its 500 from one environment
seeded in the 700 block. The completion rates the two report agree on the
unweighted checkpoints and differ by about a tenth on every weighted one, so
one of those two differences carries it. Holding the latent sampler fixed and
varying one factor at a time separates them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d3il_v2 as D  # noqa: E402

# A job runs from a frozen copy of the source, so the repository root
# has to come from the driver rather than from this file's location.
ROOT = D.ROOT


def episodes(task, weighting, bb, env_seeds, per_seed, M=10, clip=50.0):
    """Run per_seed episodes under each env seed, through the reference path."""
    D._single_threaded()
    model, blob = D.load_backbone(D.backbone_path(task, weighting, bb), "cpu")
    counts = np.zeros(D.TASKS[task]["n_modes"], dtype=np.int64)
    n_ep = n_valid = n_succ = 0
    for s in env_seeds:
        env = D.TASKS[task]["env"](seed=int(s), history=blob.get("history", 1))
        rng = np.random.default_rng(int(s))
        for _ in range(per_seed):
            obs, done = env.reset(), False
            while not done:
                o = torch.tensor(D.norm_obs(obs, blob, clip),
                                 dtype=torch.float32)[None]
                z = torch.tensor(rng.standard_normal((1, blob["act_dim"])),
                                 dtype=torch.float32)
                obs, _, done, info = env.step(D.decode(model, blob, o, z, M)[0])
            n_ep += 1
            n_succ += bool(info["success"])
            if info["mode_id"] >= 0:
                counts[int(info["mode_id"])] += 1
                n_valid += 1
    return dict(episodes=n_ep, completion=n_valid / n_ep,
                success=n_succ / n_ep, counts=counts.tolist())


def main():
    task, weighting = sys.argv[1], sys.argv[2]
    bb, n = int(sys.argv[3]), int(sys.argv[4])
    arms = {
        # How an evaluated cell draws its episodes.
        "eval: one env, seed 700": (range(700, 701), n),
        # How the reference draws them.
        "ref: 8-episode envs from 12345": (range(12345, 12345 + n // 8), 8),
        # Seed block held at the reference, reset run held at the evaluation.
        "one env, seed 12345": (range(12345, 12346), n),
        # Reset run held at the reference, seed block held at the evaluation.
        "8-episode envs from 700": (range(700, 700 + n // 8), 8),
    }
    out = {}
    for name, (seeds, per) in arms.items():
        r = episodes(task, weighting, bb, seeds, per)
        out[name] = r
        print(f"{name:34s} n={r['episodes']:4d} "
              f"completion={r['completion']:.3f} success={r['success']:.3f}",
              flush=True)
    dest = (ROOT / "runs" / "_d3il_v2"
            / f"seed_block_probe_{task}_{weighting}_bb{bb}.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
