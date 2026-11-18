#!/usr/bin/env python3
"""Where a backbone cloning run spends its time, and what a re-clone would cost.

Twenty Pushing backbones were cloned at 14.6 minutes each. The training loop
draws a weighted minibatch with ``rng.choice(n, size=batch, p=p)``, which walks
the whole probability vector on every one of its twenty thousand calls, so the
draw can cost more than the gradient step it feeds. This separates the two and
prices the inverse-CDF draw that samples the same distribution in time
logarithmic in n.

Nothing here writes a backbone. It reports timings so the re-clone can be
scheduled against measurement.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# A job runs a frozen copy of scripts/ under runs/_d3il_v2/_src/<job>,
# so this file's own location says nothing about where the run files
# are. The environment names the root; the fallback is a local run.
ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "scripts"))

import train_d3il_backbone as T  # noqa: E402


def timed(fn, n_rep):
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    return (time.perf_counter() - t0) / n_rep


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "pushing"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    batch, hidden, layers = 512, 256, 4

    files = sorted(T.TASK_DATA[task].glob("*.pkl"))
    t0 = time.perf_counter()
    obs, act, owner = T.load_pairs(files, history=1, task=task)
    load_s = time.perf_counter() - t0
    owner = np.asarray(owner)
    n = len(act)
    print(f"task={task}  demos={len(files)}  transitions={n}  "
          f"obs{obs.shape}  load {load_s:.1f}s")

    counts = np.bincount(owner, minlength=len(files)).astype(float)
    per_demo = np.ones(len(files)) / len(files)
    p = per_demo[owner] / np.maximum(counts[owner], 1)
    p = p / p.sum()

    rng = np.random.default_rng(0)
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    choice_s = timed(lambda: rng.choice(n, size=batch, p=p), 50)
    searchsorted_s = timed(
        lambda: np.searchsorted(cdf, rng.random(batch)), 200)

    o_mu, o_sd = obs.mean(0), obs.std(0) + 1e-6
    a_mu, a_sd = act.mean(0), act.std(0) + 1e-6
    obs_n = torch.tensor((obs - o_mu) / o_sd)
    act_n = torch.tensor((act - a_mu) / a_sd)
    model = T.VelocityMLP(obs_dim=obs.shape[1], hidden=hidden, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = rng.choice(n, size=batch, p=p)

    def grad_step():
        i = torch.tensor(idx)
        a1 = act_n[i]
        a0 = torch.randn_like(a1)
        t = torch.rand(len(i), 1)
        x_t = (1 - t) * a0 + t * a1
        loss = ((model(x_t, t, obs_n[i]) - (a1 - a0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    grad_step()                                   # warm up the graph
    grad_s = timed(grad_step, steps)

    torch.set_num_threads(1)
    grad_s1 = timed(grad_step, steps)

    print()
    print(f"  weighted draw, rng.choice(p=p)   {choice_s * 1e3:8.3f} ms/step")
    print(f"  weighted draw, inverse CDF       "
          f"{searchsorted_s * 1e3:8.3f} ms/step")
    print(f"  gradient step, default threads   {grad_s * 1e3:8.3f} ms/step")
    print(f"  gradient step, one thread        {grad_s1 * 1e3:8.3f} ms/step")
    print()
    for nm, draw, g in (("as written", choice_s, grad_s),
                        ("inverse CDF", searchsorted_s, grad_s),
                        ("inverse CDF, one thread", searchsorted_s, grad_s1)):
        tot = 20000 * (draw + g) + load_s
        print(f"  20000 steps, {nm:<24} {tot / 60:6.2f} min per backbone, "
              f"{20 * tot / 60:7.1f} min for twenty in series")
    print(f"\n  draw is {choice_s / max(grad_s, 1e-9):.1f}x the gradient step "
          f"as written, {searchsorted_s / max(grad_s, 1e-9):.2f}x by CDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
