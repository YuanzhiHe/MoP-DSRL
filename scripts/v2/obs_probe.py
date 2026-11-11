#!/usr/bin/env python3
"""Where the Pushing observation becomes large, channel by channel.

The slice reported a normalised observation reaching 1.2e3 on Pushing with the
bound engaging in a few percent of steps. The encoding was already ruled out:
the re-cloned backbones hold a standard deviation near 0.5 on every sine and
cosine channel, so no channel is being divided by a near-zero scale. This asks
the remaining question, which is whether the large values are the simulator
leaving its working range and which channel carries them.

It rolls out the frozen policy, which trains against nothing, so anything it
reaches belongs to the environment rather than to an objective.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(os.environ.get("MOP_DSRL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_d3il_headtohead import (  # noqa: E402
    TASKS, backbone_path, decode, load_backbone,
)
import envfactory as EF  # noqa: E402

NAMES = ["des_x", "des_y", "rob_x", "rob_y",
         "b1_x", "b1_y", "b1_sin", "b1_cos",
         "b2_x", "b2_y", "b2_sin", "b2_cos"]


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "pushing"
    weighting = sys.argv[2] if len(sys.argv) > 2 else "0.50"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    device = "cpu"
    ck = backbone_path(task, weighting, 0)
    model, blob = load_backbone(ck, device)
    mu = np.asarray(blob["obs_mean"], float).ravel()
    sd = np.asarray(blob["obs_std"], float).ravel()
    print(f"task={task} w={weighting} dim={mu.size} episodes={episodes}")
    env = EF.build(task, TASKS, 900, blob.get("history", 1))

    worst = np.zeros(mu.size)
    raw_worst = np.zeros(mu.size)
    step_of = np.zeros(mu.size, dtype=int)
    over50 = np.zeros(mu.size, dtype=int)
    steps = 0
    per_ep = []
    for ep in range(episodes):
        obs = env.reset()
        done, t, ep_max = False, 0, 0.0
        while not done:
            v = (np.asarray(obs, float).ravel() - mu) / sd
            # The frame may be stacked; fold history back onto the channels.
            vv = v.reshape(-1, mu.size) if v.size % mu.size == 0 else v[None]
            a = np.abs(vv).max(axis=0)
            raw = np.asarray(obs, float).ravel().reshape(-1, mu.size)[-1]
            for i in range(mu.size):
                if a[i] > worst[i]:
                    worst[i], step_of[i], raw_worst[i] = a[i], t, raw[i]
                if a[i] > 50:
                    over50[i] += 1
            ep_max = max(ep_max, float(a.max()))
            steps += 1
            t += 1
            o = torch.tensor(np.clip(np.nan_to_num(v), -50, 50),
                             dtype=torch.float32, device=device)[None]
            z = torch.zeros(1, blob["act_dim"], device=device)
            obs, r, done, info = env.step(decode(model, blob, o, z, 10)[0])
        per_ep.append((ep, t, ep_max, bool(info.get("success")),
                       int(info.get("mode_id", -1))))

    print(f"\n{steps} steps over {episodes} episodes")
    print(f"{'ch':>3} {'name':8} {'|z|max':>12} {'raw there':>12} "
          f"{'step':>5} {'steps>50':>9}")
    for i in range(mu.size):
        nm = NAMES[i] if i < len(NAMES) else f"c{i}"
        print(f"{i:3d} {nm:8} {worst[i]:12.4g} {raw_worst[i]:12.4g} "
              f"{step_of[i]:5d} {over50[i]:9d}")
    print("\nper episode: ep, steps, max|z|, success, mode")
    for row in per_ep:
        print("  " + " ".join(str(x) for x in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
