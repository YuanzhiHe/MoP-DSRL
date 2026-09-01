#!/usr/bin/env python3
"""Roll out a trained Avoiding backbone and measure its behavioural-mode spread.

This is the gate the whole D3IL move exists to pass. On LIBERO the frozen
backbone put every sample in one cluster -- C_ref entropy 0.000 across 45 runs --
so KL(C_phi || C_ref) had nothing to preserve and stayed exactly 0 for ~18k
updates per run. Before spending any GPU time on the head-to-head, confirm this
backbone actually spreads mass over several of the 24 routes, and that a skewed
demonstration distribution produces a correspondingly skewed policy.

Reports success rate, the induced mode histogram, its entropy and coverage, and
KL to uniform -- the last being the quantity that has to be non-zero for
KL-to-reference and an entropy/MI objective to differ at all.

    python scripts/eval_backbone_modes.py --ckpt <...>/avoiding_skew0.30.pt -n 200
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.envs_d3il import D3ILAvoidingEnv, N_MODES_AVOIDING  # noqa: E402
from scripts.train_avoiding_backbone import VelocityMLP  # noqa: E402


def load_backbone(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = VelocityMLP(obs_dim=blob["obs_dim"], act_dim=blob["act_dim"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


@torch.no_grad()
def sample_action(model, blob, obs: np.ndarray, z: np.ndarray, M: int, device: str):
    """Integrate the rectified flow from noise z to an action (M Euler steps)."""
    o = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                     dtype=torch.float32, device=device)[None]
    x = torch.tensor(z, dtype=torch.float32, device=device)[None]
    dt = 1.0 / M
    for k in range(M):
        t = torch.full((1, 1), k * dt, device=device)
        x = x + dt * model(x, t, o)
    a = x.cpu().numpy()[0] * blob["act_std"] + blob["act_mean"]
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("-n", "--episodes", type=int, default=200)
    ap.add_argument("--M", type=int, default=10, help="Euler steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, blob = load_backbone(args.ckpt, args.device)
    env = D3ILAvoidingEnv(seed=args.seed, history=blob.get('history', 1))
    print(f"backbone: {args.ckpt.name}  trained on weights={blob.get('weights_file')}")

    modes, succ = [], []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            z = rng.standard_normal(blob["act_dim"])
            a = sample_action(model, blob, obs, z, args.M, args.device)
            obs, r, done, info = env.step(a)
        modes.append(info["mode_id"])
        succ.append(int(info["success"]))

    modes = np.asarray(modes)
    valid = modes >= 0
    print(f"episodes={args.episodes}  success_rate={np.mean(succ):.3f}  "
          f"routes completed={valid.sum()}/{args.episodes}")
    if valid.sum() == 0:
        print("NO completed routes -- backbone cannot solve the task; C_ref undefined")
        return 1

    hist = np.bincount(modes[valid], minlength=N_MODES_AVOIDING).astype(float)
    p = hist / hist.sum()
    nz = p > 0
    ent = float(-(p[nz] * np.log(p[nz])).sum())
    kl_u = float(np.sum(p[nz] * np.log(p[nz] * N_MODES_AVOIDING)))
    print(f"modes visited: {int(nz.sum())}/{N_MODES_AVOIDING}")
    print(f"H(modes) = {ent:.4f} nats (max {np.log(N_MODES_AVOIDING):.4f})")
    print(f"KL(policy modes || uniform) = {kl_u:.4f} nats")
    print(f"top modes: {Counter(modes[valid].tolist()).most_common(6)}")
    print()
    if ent < 0.1:
        print("FAIL: backbone is effectively unimodal -- R_mode would be inert, "
              "exactly the LIBERO failure. Do not proceed to the head-to-head.")
        return 1
    print("PASS: backbone is multimodal; C_ref will be non-degenerate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
