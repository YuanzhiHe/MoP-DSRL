#!/usr/bin/env python3
"""MoP vs MI vs DSRL on D3IL Avoiding, across a backbone mode-skew sweep.

The experiment this project has been trying to run. Three arms steer the SAME
frozen backbone through its noise input; they differ only in what, if anything,
regularizes the behavioural-mode distribution:

  dsrl  maximize E[r]                                    (lambda = 0)
  mop   maximize E[r] - lambda * KL(C_phi || C_ref)       anchor to the backbone's
                                                          own mode profile (ours)
  mi    maximize E[r] + lambda * H(C_phi)                 push toward uniform
                                                          (BMD-style intrinsic reward)

Swept over five backbones whose demonstration weighting gives
KL(demo || uniform) in {0, 0.05, 0.15, 0.30, 0.60}. That axis is the point: at
zero skew KL(p || uniform) = ln K - H(p), so mop and mi are provably the SAME
objective and must coincide -- the toy confirmed they agree to three decimals.
They can only separate as the backbone's profile departs from uniform, which is
why a null on stock LIBERO (single-solution tasks, C_ref a point mass) and on
stock D3IL (deliberately balanced demos) was structurally guaranteed.

Reward shift: only ``--n-rewarded`` of the 24 routes pay out, so a pure return
maximizer is SUPPOSED to collapse. The question is what each regularizer retains
while still solving the task.

Modes are the environment's ground-truth geometric labels, not a K-means proxy;
the mode histogram is therefore estimated from rollouts and its gradient taken
with the score-function estimator, as the reward's is.

    python scripts/run_d3il_headtohead.py --skews 0.00 0.30 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.envs_d3il import D3ILAvoidingEnv, N_MODES_AVOIDING  # noqa: E402
from scripts.eval_backbone_modes import load_backbone  # noqa: E402

BACKBONES = Path("/shared/scratch/SCWF00111/d3il_backbones")


def coarsen(mode_id: int, groups: int) -> int:
    """Map a 24-route id onto a coarser ground-truth partition.

    The route id factorises as g1*12 + g2*4 + g3 over the three obstacle rows
    (2 x 3 x 4). Keeping only the first row gives 2 groups, the first two gives
    6. Coarser partitions are still exact geometry, not a proxy, and they matter
    because the batch histogram is the estimator's weak point: with 24 modes and
    16 episodes per update most modes draw zero samples in a batch, so the
    score-function estimator can only push DOWN on modes it observes and never
    up on the ones it misses.
    """
    if mode_id < 0:
        return -1
    if groups >= 24:
        return mode_id
    g1, rest = divmod(mode_id, 12)
    g2 = rest // 4
    if groups == 2:
        return g1
    if groups == 6:
        return g1 * 3 + g2
    raise ValueError(f"unsupported mode grouping: {groups}")


class NoisePolicy(nn.Module):
    """mu_phi(z | o): the only trainable module. The backbone stays frozen."""

    def __init__(self, obs_dim: int, z_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.log_std = nn.Parameter(torch.zeros(z_dim))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)          # start at N(0, I) == the backbone

    def dist(self, o: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mu(self.body(o)), self.log_std.exp())


@torch.no_grad()
def decode(model, blob, obs_n: torch.Tensor, z: torch.Tensor, M: int) -> np.ndarray:
    """Frozen backbone: integrate the rectified flow from z to an action."""
    x, dt = z, 1.0 / M
    for k in range(M):
        t = torch.full((x.shape[0], 1), k * dt, device=x.device)
        x = x + dt * model(x, t, obs_n)
    return (x.cpu().numpy() * blob["act_std"] + blob["act_mean"])


def rollout(env, policy, model, blob, rewarded, device, M, greedy=False):
    """One episode. Returns (log_probs, reward, mode_id, success)."""
    obs = env.reset()
    logps, done = [], False
    while not done:
        o = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                         dtype=torch.float32, device=device)[None]
        d = policy.dist(o)
        z = d.mean if greedy else d.rsample()
        logps.append(d.log_prob(z.detach()).sum())
        a = decode(model, blob, o, z.detach(), M)[0]
        obs, r, done, info = env.step(a)
    earned = bool(info["success"]) and (info["mode_id"] in rewarded)
    return logps, float(earned), info["mode_id"], bool(info["success"])


def run_arm(arm, lam, skew, seed, args, C_ref):
    device = args.device
    model, blob = load_backbone(BACKBONES / f"avoiding_skew{skew}.pt", device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rewarded = set(rng.choice(N_MODES_AVOIDING, size=args.n_rewarded, replace=False).tolist())

    env = D3ILAvoidingEnv(seed=seed, history=blob.get("history", 1))
    policy = NoisePolicy(blob["obs_dim"], blob["act_dim"]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    K = args.mode_groups
    ref = torch.tensor(C_ref, dtype=torch.float32, device=device)
    # Running estimate of the policy's mode distribution. A single batch of 16
    # episodes cannot estimate a K-bin histogram, and an unobserved mode gets no
    # gradient at all, which biased the objective to only ever suppress visited
    # modes -- measured as MoP collapsing to a single route (H=0.000) at every
    # lambda. An EMA across iterations gives every mode a stable probability.
    C_ema = torch.full((K,), 1.0 / K, device=device)

    for it in range(args.iters):
        batch_lp, batch_r, batch_modes = [], [], []
        for _ in range(args.episodes_per_iter):
            lp, r, m, _ = rollout(env, policy, model, blob, rewarded, device, args.M)
            batch_lp.append(torch.stack(lp).sum())
            batch_r.append(r)
            batch_modes.append(m)
        lp = torch.stack(batch_lp)
        r = torch.tensor(batch_r, dtype=torch.float32, device=device)
        adv = r - r.mean()

        # Mode objective, credited PER EPISODE. The histogram C_phi is a batch
        # statistic, so its score-function gradient is
        #     grad = E_tau[ (dObjective/dC_phi)|_{mode(tau)} * grad log p(tau) ],
        # i.e. each trajectory is credited by the partial derivative AT ITS OWN
        # mode. Adding the scalar objective uniformly to every advantage instead
        # contributes exactly zero gradient -- a constant is absorbed as a
        # REINFORCE baseline since E[grad log p] = 0. That mistake made all three
        # arms run the same algorithm and produced pure seed noise; it is the
        # same failure class as the dead R_mode gradient on LIBERO, so the
        # per-episode coefficients below are the load-bearing part of this file.
        counts = torch.zeros(K, device=device)
        for m in batch_modes:
            mg = coarsen(m, K)
            if mg >= 0:
                counts[mg] += 1
        if counts.sum() > 0:
            C_ema = (1 - args.ema) * C_ema + args.ema * (counts / counts.sum())
        mode_coef = torch.zeros(len(batch_modes), device=device)
        if counts.sum() > 0 and arm != "dsrl":
            C_phi = C_ema
            for j, m0 in enumerate(batch_modes):
                m = coarsen(m0, K)
                if m < 0:
                    continue
                if arm == "mop":
                    # minimize KL(C_phi || C_ref): d/dp_m = log(p_m/q_m) + 1
                    mode_coef[j] = -(torch.log((C_phi[m] + 1e-8) / (ref[m] + 1e-8)) + 1.0)
                else:  # mi: maximize H(C_phi); dH/dp_m = -log(p_m) - 1
                    mode_coef[j] = -(torch.log(C_phi[m] + 1e-8) + 1.0)

        # Put the mode coefficient on the reward's scale before weighting. The
        # raw partials are log-ratios: log(p_m/q_m) reaches +-10 for a mode that
        # is rare under C_ref, while the binary reward's advantage lives in
        # [-1, 1]. At lambda=1 the mode term then dominated the objective by an
        # order of magnitude and RL simply stopped solving the task -- measured
        # mop success collapsing to 0.15-0.40 against the backbone's own 0.55.
        # Centring and scaling both signals makes lambda mean what it should:
        # the relative weight of the two objectives, not an accident of units.
        if mode_coef.abs().sum() > 0:
            mc = mode_coef - mode_coef.mean()
            sd = mc.std()
            if torch.isfinite(sd) and sd > 1e-6:
                mode_coef = mc / sd
        adv_sd = adv.std()
        adv_n = adv / adv_sd if torch.isfinite(adv_sd) and adv_sd > 1e-6 else adv
        shaped = adv_n + lam * mode_coef.detach()
        loss = -(shaped.detach() * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # --- evaluation ---
    modes, succ = [], []
    for _ in range(args.eval_episodes):
        _, _, m, s = rollout(env, policy, model, blob, rewarded, device, args.M)
        modes.append(m)
        succ.append(int(s))
    modes = np.asarray([coarsen(m, K) for m in modes])
    valid = modes >= 0
    hist = np.bincount(modes[valid], minlength=K).astype(float) if valid.any() else np.zeros(K)
    p = hist / max(hist.sum(), 1)
    nz = p > 0
    ref_np = np.asarray(C_ref)
    return dict(
        arm=arm, lam=lam, skew=skew, seed=seed,
        SR=float(np.mean(succ)),
        rewarded_SR=float(np.mean([m in rewarded for m in modes[valid]])) if valid.any() else 0.0,
        H_modes=float(-(p[nz] * np.log(p[nz])).sum()) if nz.any() else 0.0,
        mc_at_08=int((p >= 0.8 / K).sum()),
        n_modes=int(nz.sum()),
        tv_to_ref=float(0.5 * np.abs(p - ref_np).sum()),
        tv_to_uniform=float(0.5 * np.abs(p - 1.0 / K).sum()),
    )


def _ref_chunk(payload):
    """Worker: roll the frozen backbone out and count modes."""
    skew, n, seed, M, device = payload
    model, blob = load_backbone(BACKBONES / f"avoiding_skew{skew}.pt", device)
    env = D3ILAvoidingEnv(seed=seed, history=blob.get("history", 1))
    rng = np.random.default_rng(seed)
    counts = np.zeros(N_MODES_AVOIDING)
    for _ in range(n):
        obs = env.reset()
        done = False
        while not done:
            o = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                             dtype=torch.float32, device=device)[None]
            z = torch.tensor(rng.standard_normal((1, blob["act_dim"])),
                             dtype=torch.float32, device=device)
            obs, _, done, info = env.step(decode(model, blob, o, z, M)[0])
        if info["mode_id"] >= 0:
            counts[info["mode_id"]] += 1
    return counts


def _cell(payload):
    """Worker: one (arm, seed) cell of the sweep."""
    arm, lam, skew, seed, args, C_ref = payload
    return run_arm(arm, lam, skew, seed, args, C_ref)


def reference_profile(skew, args) -> np.ndarray:
    """C_ref: the frozen backbone's own mode distribution, measured by rollout.

    Estimated from many episodes because C_ref's own sampling error enters
    tv_to_ref for every arm: with 24 modes, a 200-episode histogram sits ~0.10 in
    total variation from the distribution it estimates, which is the size of the
    effect being measured.
    """
    n_w = max(1, args.workers)
    per = max(1, args.ref_episodes // n_w)
    payloads = [(skew, per, 12345 + i, args.M, args.device) for i in range(n_w)]
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        counts24 = sum(ex.map(_ref_chunk, payloads))
    K = args.mode_groups
    counts = np.zeros(K)
    for m in range(N_MODES_AVOIDING):
        counts[coarsen(m, K)] += counts24[m]
    return counts / max(counts.sum(), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skews", nargs="+", default=["0.00", "0.05", "0.15", "0.30", "0.60"])
    ap.add_argument("--arms", nargs="+", default=["dsrl", "mop", "mi"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--ref-episodes", type=int, default=150)
    ap.add_argument("--n-rewarded", type=int, default=4)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mode-groups", type=int, default=24, choices=[2, 6, 24],
                    help="ground-truth partition resolution (2, 6 or all 24 routes)")
    ap.add_argument("--ema", type=float, default=0.15,
                    help="EMA rate for the running mode histogram")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/_d3il/headtohead.json"))
    args = ap.parse_args()

    rows = []
    for skew in args.skews:
        C_ref = reference_profile(skew, args)
        nz = C_ref > 0
        print(f"=== skew {skew}: C_ref over {int(nz.sum())} modes, "
              f"H={-(C_ref[nz] * np.log(C_ref[nz])).sum():.3f} nats ===", flush=True)
        cells = [(arm, 0.0 if arm == "dsrl" else args.lam, skew, seed, args, C_ref)
                 for arm in args.arms for seed in range(args.seeds)]
        # One process per cell: the bottleneck is serial MuJoCo stepping (~38
        # env-steps/s per core), so cells scale almost linearly across cores.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_cell, cells):
                rows.append(res)
                print(f"  {res['arm']:5s} seed{res['seed']}  SR={res['SR']:.3f} "
                      f"H={res['H_modes']:.3f} mc={res['mc_at_08']:2d} "
                      f"tv_ref={res['tv_to_ref']:.3f} tv_unif={res['tv_to_uniform']:.3f}",
                      flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
