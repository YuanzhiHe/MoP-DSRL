#!/usr/bin/env python3
"""Train a multimodal rectified-flow backbone on D3IL demonstrations.

This is the frozen pi_theta for the noise-space RL experiments. It replaces the
plan to freeze D3IL's own BESO agent: BESO would drag in that repo's hydra
config stack, whereas a rectified-flow velocity field plugs straight into this
project's existing RectifiedFlowDenoiser, so z -> action needs no adapter at all
and the M-step Euler integration is the same code path already used for pi_0.5.

Observation / action follow the simulator's convention
(d3il/simulation/avoiding_sim.py:52-77): the policy sees 4 dims -- the current
desired xy target concatenated with the achieved xy -- and emits the 2-D delta
applied to that target.

The point of the backbone is to be MULTIMODAL: several routes must retain
probability mass, otherwise the reference histogram C_ref collapses to a point
mass and the mode regularizer is inert (exactly what happened on LIBERO, where
C_ref entropy was 0.000 in 45 runs). Training therefore reports the induced mode
distribution, and --weights lets the demonstration distribution be skewed, which
the toy showed is required for KL-to-reference and an entropy/MI objective to
differ at all.

    python scripts/train_d3il_backbone.py --weights <...>/demo_weights_skew0.30.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TASK_DATA = {
    "avoiding": Path("/shared/scratch/SCWF00111/d3il_data/data/avoiding/data"),
    "pushing": Path("/shared/scratch/SCWF00111/d3il_data/data/pushing/all_data"),
}
DATA = TASK_DATA["avoiding"]


class VelocityMLP(nn.Module):
    """v_theta(x_t, t, obs) for rectified flow over the 2-D action."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = 256, layers: int = 4):
        super().__init__()
        dims = [obs_dim + act_dim + 1] + [hidden] * layers
        net: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(a, b), nn.SiLU()]
        net += [nn.Linear(hidden, act_dim)]
        self.net = nn.Sequential(*net)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return self.net(torch.cat([x_t, t, obs], dim=-1))


def _frame_avoiding(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Avoiding: the policy sees its desired xy and the achieved xy."""
    des = np.asarray(d["robot"]["des_c_pos"])[:, :2]
    pos = np.asarray(d["robot"]["c_pos"])[:, :2]
    return np.concatenate([des[:-1], pos[:-1]], axis=1), des[1:] - des[:-1]


def _frame_pushing(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pushing: desired xy, robot xy, and each box's xy plus its yaw tangent.

    Mirrors PushingEnv.get_observation, which is what the env adapter presents,
    so the backbone is trained on exactly the observation it will be rolled out
    with. The boxes enter because this task's mode is a property of where the
    OBJECTS end up, not of the arm's own path.
    """
    des = np.asarray(d["robot"]["des_c_pos"])[:, :2]
    pos = np.asarray(d["robot"]["c_pos"])[:, :2]
    parts = [des[:-1], pos[:-1]]
    for box in ("red-box", "green-box"):
        bp = np.asarray(d[box]["pos"])[:, :2]
        bq = np.asarray(d[box]["quat"])
        # yaw from the quaternion, then the tangent the env reports
        yaw = np.arctan2(2 * (bq[:, 0] * bq[:, 3] + bq[:, 1] * bq[:, 2]),
                         1 - 2 * (bq[:, 2] ** 2 + bq[:, 3] ** 2))
        parts += [bp[:-1], np.tan(yaw[:-1])[:, None]]
    return np.concatenate(parts, axis=1), des[1:] - des[:-1]


FRAME_FN = {"avoiding": _frame_avoiding, "pushing": _frame_pushing}


def load_pairs(files: list[Path], history: int = 1, task: str = "avoiding"
               ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(obs, action[2]) pairs plus the demo index each pair came from.

    With ``history`` > 1 the observation stacks the last ``history`` frames of
    (desired xy, achieved xy). Avoiding requires the policy to COMMIT to one of
    24 routes, and a single-frame observation cannot express that commitment --
    a Markovian policy averages over the routes still reachable and drives into
    an obstacle, which is why D3IL's own agents condition on a window.
    """
    obs_l, act_l, owner = [], [], []
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        # action_t is the increment applied to the desired target at t
        frame, act = FRAME_FN[task](d)
        if history > 1:
            pad = np.repeat(frame[:1], history - 1, axis=0)
            padded = np.concatenate([pad, frame], axis=0)
            obs = np.concatenate(
                [padded[k:k + len(frame)] for k in range(history)], axis=1)
        else:
            obs = frame
        obs_l.append(obs)
        act_l.append(act)
        owner += [i] * len(act)
    return (np.concatenate(obs_l).astype(np.float32),
            np.concatenate(act_l).astype(np.float32),
            owner)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None,
                    help="demo_weights_skew*.pkl; uniform over demos if omitted")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--task", default="avoiding", choices=["avoiding", "pushing"])
    ap.add_argument("--history", type=int, default=1,
                    help="stack this many past frames into the observation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    files = sorted(TASK_DATA[args.task].glob("*.pkl"))
    obs, act, owner = load_pairs(files, history=args.history, task=args.task)
    owner = np.asarray(owner)
    print(f"demos={len(files)}  transitions={len(act)}  obs{obs.shape} act{act.shape}")

    # per-transition sampling probability, inherited from its demonstration
    if args.weights is not None:
        with open(args.weights, "rb") as fh:
            blob = pickle.load(fh)
        w_by_name = blob["weights"]
        per_demo = np.array([w_by_name.get(f.name, 0.0) for f in files], dtype=float)
        print(f"skew: target_kl={blob['target_kl']} achieved={blob['achieved_kl']:.4f}")
    else:
        per_demo = np.ones(len(files))
    per_demo = per_demo / per_demo.sum()
    # spread a demo's mass over its own transitions so long demos are not favoured
    counts = np.bincount(owner, minlength=len(files)).astype(float)
    p = per_demo[owner] / np.maximum(counts[owner], 1)
    p = p / p.sum()

    # normalize inputs/outputs; the frozen backbone stores these for inference
    o_mu, o_sd = obs.mean(0), obs.std(0) + 1e-6
    a_mu, a_sd = act.mean(0), act.std(0) + 1e-6
    obs_n = torch.tensor((obs - o_mu) / o_sd, device=args.device)
    act_n = torch.tensor((act - a_mu) / a_sd, device=args.device)

    model = VelocityMLP(obs_dim=obs.shape[1], hidden=args.hidden,
                        layers=args.layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        idx = rng.choice(len(act), size=args.batch, p=p)
        i = torch.tensor(idx, device=args.device)
        a1 = act_n[i]                                   # data endpoint
        a0 = torch.randn_like(a1)                       # noise endpoint
        t = torch.rand(len(i), 1, device=args.device)
        x_t = (1 - t) * a0 + t * a1                     # rectified-flow path
        target = a1 - a0                                # constant velocity
        loss = ((model(x_t, t, obs_n[i]) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f"  step {step:6d}  flow-matching loss {loss.item():.5f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "obs_mean": o_mu, "obs_std": o_sd,
        "act_mean": a_mu, "act_std": a_sd,
        "obs_dim": int(obs.shape[1]), "act_dim": 2, "history": args.history,
        "task": args.task, "hidden": args.hidden, "layers": args.layers,
        "weights_file": str(args.weights) if args.weights else None,
    }, args.out)
    print(f"saved backbone -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
