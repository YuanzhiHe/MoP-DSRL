#!/usr/bin/env python3
"""P0 experiment: does KL-to-reference actually preserve modes, vs DSRL and MI?

Rationale. On LIBERO the mode-preservation claim is untestable: every task has a
single solution, so the frozen backbone's reference mode histogram is a point
mass (measured C_ref entropy = 0.000 across 45 runs) and there is nothing to
collapse. BMD (ICML 2026) shows the effect appears only where the task admits
several genuine solutions -- DSRL keeps 2/2 modes on a 2-mode reach but 1/24 on
Kitchen. This script reproduces that regime in the smallest possible setting so
the estimator can be judged in hours on CPU rather than months on GPU.

Setup (mirrors BMD's 2D Gaussian-mixture benchmark).
  * Frozen backbone: a deterministic decoder over noise z in R^2. The quadrant of
    z selects one of 4 behavioural modes; within a mode the action is the mode
    centre plus small noise. Under z ~ N(0, I) the mode distribution is uniform,
    so the reference histogram C_ref = (1/4, 1/4, 1/4, 1/4) -- genuinely
    multimodal, unlike LIBERO.
  * The RL problem is DSRL's: the backbone is frozen and we learn only a
    Gaussian noise policy mu_phi(z) = N(m, diag(s^2)) over its input.
  * Reward shift: only a subset of modes is rewarded. A pure return maximizer is
    therefore *supposed* to collapse onto a rewarded mode -- the question is what
    each regularizer retains while still solving the task.

Arms.
  dsrl  : maximize E[r]                                   (lambda = 0)
  mop   : maximize E[r] - lambda * KL(C_phi || C_ref)      (ours: anchor to the
          pretrained policy's own mode distribution)
  mi    : maximize E[r] + lambda * H(C_phi)                (BMD-style analogue;
          with a single state the variational MI bound I(Z;S) reduces to the
          marginal mode entropy, so this is the faithful one-state reduction of
          their intrinsic reward -- noted as such, not claimed to be BMD itself)

Metrics (BMD's, so the numbers are directly comparable):
  SR      overall success -- reached any rewarded goal
  SR_M    mode-aggregated success, mean over modes of per-mode success
  mc@0.8  number of modes retained with probability >= 0.8 / K_expected
  H       entropy of the empirical mode histogram (nats; max ln 4 = 1.386)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

K_MODES = 4
MODE_CENTRES = torch.tensor(
    [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]], dtype=torch.float32
)
WITHIN_MODE_STD = 0.25
DECODER_TEMP = 0.5      # backbone peakiness
ASSIGN_TEMP = 2.0       # softmax temperature for the mode histogram
# Backbone mode bias. With a UNIFORM reference, KL(C_phi || C_ref) = ln K -
# H(C_phi) exactly, so anchoring to the reference and maximizing mode entropy
# are the SAME objective and the two arms are numerically indistinguishable
# (verified). The estimators only come apart when the pretrained policy's own
# mode distribution is skewed: MoP then preserves that profile, while the MI /
# entropy objective pushes toward uniform, i.e. away from the backbone. A skewed
# backbone is therefore the only setting in which our contribution is testable.
BACKBONE_BIAS = torch.tensor([1.2, 0.4, -0.4, -1.2], dtype=torch.float32)
GOAL_RADIUS = 1.0


def decode(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen backbone: a smooth multimodal decoder over the noise z.

    Mode membership is softmax over the four quadrant logits at a fairly peaked
    temperature, so deep inside a quadrant the action is essentially that mode's
    centre while remaining differentiable in z -- the property a real rectified-
    flow backbone has and that the regularizer's gradient path depends on.
    """
    logits = torch.stack([z[:, 0] + z[:, 1], -z[:, 0] + z[:, 1],
                          -z[:, 0] - z[:, 1], z[:, 0] - z[:, 1]], dim=1)
    w = torch.softmax(logits / DECODER_TEMP + BACKBONE_BIAS.to(z.device), dim=1)
    action = w @ MODE_CENTRES.to(z.device)
    return action, w.argmax(dim=1)


def rollout(m: torch.Tensor, log_s: torch.Tensor, n: int, gen: torch.Generator):
    """Sample n noise vectors from mu_phi and decode them."""
    eps = torch.randn(n, 2, generator=gen)
    z = m + log_s.exp() * eps
    action, mode = decode(z)
    action = action + WITHIN_MODE_STD * torch.randn(n, 2, generator=gen)
    return action, mode


def reward(action: torch.Tensor, rewarded: list[int]) -> torch.Tensor:
    """1 inside the radius of any rewarded goal, else 0."""
    d = torch.cdist(action, MODE_CENTRES.to(action.device)[rewarded])
    return (d.min(dim=1).values < GOAL_RADIUS).float()


def histogram(mode: torch.Tensor) -> torch.Tensor:
    return torch.bincount(mode, minlength=K_MODES).float() / mode.numel()


def soft_histogram(action: torch.Tensor) -> torch.Tensor:
    """Differentiable mode histogram: softmax over negative squared distance.

    This is the estimator the real method uses (mode_regularizer.soft_assign_
    distances), applied to the decoded action rather than to z, so the toy tests
    the actual mechanism rather than a proxy.
    """
    sq = ((action[:, None, :] - MODE_CENTRES.to(action.device)[None]) ** 2).sum(-1)
    T = ASSIGN_TEMP
    return torch.softmax(-sq / T, dim=1).mean(dim=0)


def train(arm: str, lam: float, rewarded: list[int], seed: int, steps: int, batch: int):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = torch.zeros(2, requires_grad=True)
    log_s = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([m, log_s], lr=3e-2)
    with torch.no_grad():
        z0 = torch.randn(20000, 2, generator=gen)
        a0, _ = decode(z0)
        C_ref = soft_histogram(a0)          # the backbone's OWN mode profile
    eps_sm = 1e-8

    for _ in range(steps):
        eps = torch.randn(batch, 2, generator=gen)
        z = m + log_s.exp() * eps
        action, _ = decode(z)
        action = action + WITHIN_MODE_STD * torch.randn(batch, 2, generator=gen)
        r = reward(action, rewarded)
        # The reward is a hard indicator, so it cannot be differentiated through
        # the decoder: estimate its gradient with REINFORCE (advantage-centred
        # to cut variance). The regularizers below ARE differentiable in z, so
        # they keep the low-variance reparameterized path.
        # z must be DETACHED here: it was built as m + s*eps, so log N(z; m, s)
        # with the dependency left in collapses to -eps^2/2 + const, whose
        # gradient wrt m is identically zero -- the score-function estimator
        # needs the sample treated as data.
        log_prob = torch.distributions.Normal(m, log_s.exp()).log_prob(z.detach()).sum(dim=-1)
        adv = r - r.mean()
        loss = -(adv.detach() * log_prob).mean()           # maximize return
        C_phi = soft_histogram(action)
        if arm == "mop" and lam > 0:
            kl = (C_phi * ((C_phi + eps_sm).log() - (C_ref + eps_sm).log())).sum()
            loss = loss + lam * kl
        elif arm == "mi" and lam > 0:
            ent = -(C_phi * (C_phi + eps_sm).log()).sum()
            loss = loss - lam * ent                        # maximize mode entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # --- evaluation ---
    with torch.no_grad():
        action, mode = rollout(m, log_s, 4000, gen)
        r = reward(action, rewarded)
        hist = histogram(mode)
        sr = r.mean().item()
        per_mode = []
        for k in range(K_MODES):
            sel = mode == k
            # success *of that mode*: does it land on its own goal?
            if sel.any():
                d = (action[sel] - MODE_CENTRES[k]).norm(dim=1)
                per_mode.append((d < GOAL_RADIUS).float().mean().item())
            else:
                per_mode.append(0.0)
        sr_m = float(np.mean(per_mode))
        mc = int(sum(1 for k in range(K_MODES) if hist[k].item() >= 0.8 / K_MODES))
        ent = float(-(hist * (hist + eps_sm).log()).sum())
        tv_ref = float(0.5 * (hist - C_ref).abs().sum())   # distance to backbone profile
        tv_unif = float(0.5 * (hist - torch.full_like(hist, 1.0 / K_MODES)).abs().sum())
    return dict(SR=sr, SR_M=sr_m, mc=mc, H=ent, TVref=tv_ref, TVunif=tv_unif,
                hist=[round(h, 3) for h in hist.tolist()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="runs/_toy/toy_mode_preservation.json")
    args = ap.parse_args()

    lambdas = [0.0, 0.3, 1.0, 3.0, 10.0]
    # G1: one rewarded mode (max pressure to collapse). G2: two rewarded modes.
    settings = {"G1_one_goal": [0], "G2_two_goals": [0, 2]}
    rows = []
    for setting, rewarded in settings.items():
        for arm in ("dsrl", "mop", "mi"):
            for lam in ([0.0] if arm == "dsrl" else lambdas[1:]):
                res = [train(arm, lam, rewarded, s, args.steps, args.batch)
                       for s in range(args.seeds)]
                agg = {k: (float(np.mean([r[k] for r in res])),
                           float(np.std([r[k] for r in res])))
                       for k in ("SR", "SR_M", "mc", "H", "TVref", "TVunif")}
                rows.append(dict(setting=setting, arm=arm, lam=lam,
                                 n_seeds=args.seeds, **{k: v for k, v in agg.items()}))
                print(f"{setting:14s} {arm:5s} lam={lam:<4} "
                      f"SR={agg['SR'][0]:.3f}+-{agg['SR'][1]:.3f}  "
                      f"SR_M={agg['SR_M'][0]:.3f}+-{agg['SR_M'][1]:.3f}  "
                      f"mc@0.8={agg['mc'][0]:.2f}/4  H={agg['H'][0]:.3f}  "
                      f"TV_to_backbone={agg['TVref'][0]:.3f}  TV_to_uniform={agg['TVunif'][0]:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
