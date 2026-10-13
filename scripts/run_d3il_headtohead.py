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
import traceback
import os
import sys

# Cap the BLAS thread pools BEFORE numpy and torch load. The sweep runs one
# worker process per allocated core, and each would otherwise inherit
# OMP_NUM_THREADS from the batch script, which sets it to the core count: 24
# workers of 24 threads each against a 192-core node, twice that where two jobs
# share one. Rollouts then crawl -- two sweeps spent thirty-six hours without
# finishing a reference estimate whose work is three minutes per worker. These
# variables are read when the libraries initialise their pools, so setting them
# inside the worker, after the import, does nothing. The pool supplies the
# parallelism; each worker takes one thread.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.envs_d3il import (D3ILAvoidingEnv, D3ILPushingEnv,  # noqa: E402
                               N_MODES_AVOIDING, N_MODES_PUSHING)
from scripts.eval_backbone_modes import load_backbone  # noqa: E402

BACKBONE_ROOT = Path("/shared/scratch/SCWF00111/d3il_backbones")
TASKS = {
    "avoiding": dict(dir=BACKBONE_ROOT, prefix="avoiding_skew",
                     env=lambda **kw: D3ILAvoidingEnv(**kw), n_modes=N_MODES_AVOIDING),
    "pushing": dict(dir=BACKBONE_ROOT / "pushing", prefix="pushing_skew",
                    env=lambda **kw: D3ILPushingEnv(**kw), n_modes=N_MODES_PUSHING),
}


def backbone_path(task: str, skew: str, bb_seed: int | None = None) -> Path:
    """Resolve a frozen backbone, optionally one of several trained per skew.

    One backbone per skew level confounds the level with that single draw from a
    noisy behaviour-cloning run: demonstration skew 0.15 produced a HIGHER
    measured reference skew (0.370) than skew 0.30 (0.302). With one checkpoint
    per level, every run seed shares its idiosyncrasies and a trend across levels
    has as many independent units as it has levels, not as it has runs.
    """
    t = TASKS[task]
    if bb_seed is not None:
        cand = t["dir"] / f"{t['prefix']}{skew}_bb{bb_seed}.pt"
        if cand.exists():
            return cand
    return t["dir"] / f"{t['prefix']}{skew}.pt"


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
    if groups >= 24 or groups == 4:
        # Pushing's four routes are already the partition; only Avoiding's 24
        # need coarsening, and only because a 16-episode batch cannot estimate a
        # 24-bin histogram.
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


class SkillPolicy(nn.Module):
    """mu_phi(z | o, k): the noise policy conditioned on a latent skill index.

    BMD's intrinsic reward is a variational lower bound on I(skill; behaviour),
    which needs a skill variable to condition on and a discriminator to recover
    it. Our ``mi`` arm maximizes the entropy of the GROUND-TRUTH mode histogram
    instead, which is the faithful one-state reduction of that bound but not the
    published mechanism -- it is handed labels the real method has to discover.
    This class plus ``SkillDiscriminator`` implement the mechanism itself, so the
    comparison is against BMD rather than against an idealisation of it.
    """

    def __init__(self, obs_dim: int, z_dim: int, n_skills: int, hidden: int = 128):
        super().__init__()
        self.n_skills = n_skills
        self.body = nn.Sequential(
            nn.Linear(obs_dim + n_skills, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.log_std = nn.Parameter(torch.zeros(z_dim))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)          # start at N(0, I) == the backbone

    def dist(self, o: torch.Tensor, k: int) -> torch.distributions.Normal:
        oh = torch.zeros(o.shape[0], self.n_skills, device=o.device)
        oh[:, k] = 1.0
        h = self.body(torch.cat([o, oh], dim=-1))
        return torch.distributions.Normal(self.mu(h), self.log_std.exp())


class SkillDiscriminator(nn.Module):
    """q_psi(k | s): recovers which skill produced a trajectory, from its end state."""

    def __init__(self, obs_dim: int, n_skills: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_skills),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


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


def run_bmd(lam, skew, seed, args, C_ref, bb_seed=None):
    """BMD-style arm: learned skills plus a variational mutual-information bound.

    A skill k ~ Uniform{0..K-1} is drawn per episode and the policy conditions on
    it. A discriminator q_psi(k | s_T) is trained to recover k from the final
    observation, and the intrinsic reward is the standard variational bound
        log q_psi(k | s_T) - log p(k) = log q_psi(k | s_T) + log K,
    which is maximized when distinct skills end in distinguishable states. The
    modes are therefore DISCOVERED, never supplied -- the ground-truth labels are
    used only to score the result, exactly as for the other arms.
    """
    device = args.device
    model, blob = load_backbone(backbone_path(args.task, skew, bb_seed), device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_all = TASKS[args.task]["n_modes"]
    rewarded = set(rng.choice(n_all, size=min(args.n_rewarded, n_all),
                              replace=False).tolist())
    env = TASKS[args.task]["env"](seed=seed, history=blob.get("history", 1))
    K = args.mode_groups

    policy = SkillPolicy(blob["obs_dim"], blob["act_dim"], K).to(device)
    disc = SkillDiscriminator(blob["obs_dim"], K).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

    def episode(k: int, greedy: bool = False):
        obs = env.reset()
        logps, done = [], False
        while not done:
            o = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                             dtype=torch.float32, device=device)[None]
            d = policy.dist(o, k)
            z = d.mean if greedy else d.rsample()
            logps.append(d.log_prob(z.detach()).sum())
            a = decode(model, blob, o, z.detach(), args.M)[0]
            obs, _, done, info = env.step(a)
        last = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                            dtype=torch.float32, device=device)
        earned = bool(info["success"]) and (info["mode_id"] in rewarded)
        return logps, float(earned), last, info["mode_id"], bool(info["success"])

    for _ in range(args.iters):
        lps, rs, lasts, ks = [], [], [], []
        for _ in range(args.episodes_per_iter):
            k = int(rng.integers(K))
            lp, r, last, _, _ = episode(k)
            lps.append(torch.stack(lp).sum()); rs.append(r)
            lasts.append(last); ks.append(k)
        S = torch.stack(lasts)
        kt = torch.tensor(ks, device=device)

        # discriminator step: recover the skill from where the episode ended
        logits = disc(S)
        loss_d = nn.functional.cross_entropy(logits, kt)
        opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

        # intrinsic reward = the variational bound, held fixed for the policy step
        with torch.no_grad():
            logq = nn.functional.log_softmax(disc(S), dim=-1)
            r_int = logq[torch.arange(len(ks), device=device), kt] + np.log(K)
        r = torch.tensor(rs, dtype=torch.float32, device=device)
        adv = r - r.mean()
        adv_sd = adv.std()
        adv_n = adv / adv_sd if torch.isfinite(adv_sd) and adv_sd > 1e-6 else adv
        ri = r_int - r_int.mean()
        ri_sd = ri.std()
        ri_n = ri / ri_sd if torch.isfinite(ri_sd) and ri_sd > 1e-6 else ri
        shaped = adv_n + lam * ri_n
        loss = -(shaped.detach() * torch.stack(lps)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    # Skill prior at evaluation. Cycling skills uniformly forces the mode
    # histogram toward uniform by construction, which penalises this arm on a
    # metric that measures distance to a non-uniform reference. The prior is a
    # free choice at deployment, so it is fitted: a held-out pass estimates which
    # mode each skill produces, and the evaluation then samples skills in the
    # proportions that reproduce C_ref. Setting --bmd-prior uniform restores the
    # unfitted protocol.
    if args.bmd_prior == "fitted":
        n_cal = max(K * 8, 40)
        hits = np.zeros((K, K))          # skill x mode
        for i in range(n_cal):
            k = int(i % K)
            _, _, _, m, _ = episode(k)
            mg = coarsen(m, K)
            if mg >= 0:
                hits[k, mg] += 1
        row = hits.sum(1, keepdims=True)
        cond = np.divide(hits, np.maximum(row, 1))        # p(mode | skill)
        # Least-squares weights over skills that best reproduce C_ref. When a
        # skill never reaches a labelled mode its row of `cond` is all zero, and
        # on Pushing enough rows degenerate that LAPACK's divide-and-conquer
        # driver fails to converge (LinAlgError, eleven hours into a sweep). Fall
        # back to the SVD-free pseudoinverse, then to the uniform prior: a worse
        # fit for this arm is recoverable, a dead job is not.
        A, b = np.nan_to_num(cond.T), np.asarray(C_ref, float)
        try:
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            try:
                w = np.linalg.pinv(A) @ b
            except np.linalg.LinAlgError:
                w = np.full(K, 1.0 / K)
        w = np.clip(np.nan_to_num(w), 0, None)
        prior = w / w.sum() if w.sum() > 1e-8 else np.full(K, 1.0 / K)
    else:
        prior = np.full(K, 1.0 / K)

    modes, succ = [], []
    for i in range(args.eval_episodes):
        k = int(rng.choice(K, p=prior)) if args.bmd_prior == "fitted" else int(i % K)
        _, _, _, m, sc = episode(k)
        modes.append(m); succ.append(int(sc))
    modes = np.asarray([coarsen(m, K) for m in modes])
    valid = modes >= 0
    hist = np.bincount(modes[valid], minlength=K).astype(float) if valid.any() else np.zeros(K)
    p = hist / max(hist.sum(), 1)
    nz = p > 0
    ref_np = np.asarray(C_ref)
    return dict(
        task=args.task, arm="bmd", lam=lam, skew=skew, seed=seed,
        bmd_prior=args.bmd_prior,
        # Stored so the fitted protocol can be audited after the fact: `skill_prior`
        # is the mixture evaluation drew from, `skill_modes` row k is the mode
        # distribution skill k produced during calibration. Their product is what
        # the fit predicts the evaluation histogram will be; the gap between that
        # prediction and the realised C_phi is the part fitting cannot reach.
        skill_prior=[float(x) for x in prior],
        skill_modes=([[float(x) for x in r] for r in cond]
                     if args.bmd_prior == "fitted" else None),
        bb_seed=bb_seed,
        SR=float(np.mean(succ)),
        rewarded_SR=float(np.mean([m in rewarded for m in modes[valid]])) if valid.any() else 0.0,
        H_modes=float(-(p[nz] * np.log(p[nz])).sum()) if nz.any() else 0.0,
        mc_at_08=int((p >= 0.8 / K).sum()),
        n_modes=int(nz.sum()),
        tv_to_ref=float(0.5 * np.abs(p - ref_np).sum()),
        tv_to_uniform=float(0.5 * np.abs(p - 1.0 / K).sum()),
        coef_absmax=0.0,
        collapsed=bool(int(nz.sum()) <= 1),
        C_phi=p.tolist(), C_ref=ref_np.tolist(),
    )


def run_arm(arm, lam, skew, seed, args, C_ref, bb_seed=None):
    if arm == "bmd":
        return run_bmd(lam, skew, seed, args, C_ref, bb_seed)
    device = args.device
    model, blob = load_backbone(backbone_path(args.task, skew, bb_seed), device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_all = TASKS[args.task]["n_modes"]
    rewarded = set(rng.choice(n_all, size=min(args.n_rewarded, n_all),
                              replace=False).tolist())

    env = TASKS[args.task]["env"](seed=seed, history=blob.get("history", 1))
    policy = NoisePolicy(blob["obs_dim"], blob["act_dim"]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    K = args.mode_groups
    # Smooth the reference. The anchoring coefficient is -log(p_m/q_m), which is
    # UNBOUNDED as q_m -> 0: at skew 0.60 the backbone covers only 18 of 24
    # routes, so any visit to a mode outside its support drew a penalty near -18
    # against a reward advantage in [-1, 1]. Mixing in a uniform floor bounds the
    # coefficient by log(K/eps). The entropy objective needs no such guard --
    # -log(p_m) is bounded by the visited support -- which is the asymmetry that
    # makes MoP the harder estimator of the two.
    # The anchoring TARGET may be smoothed, but scoring always uses the raw
    # C_ref. Scoring a smoothed arm against its own smoothed target would
    # flatter it; scoring it against C_ref when it was given 0.8*C_ref + 0.2*u
    # charges it for hitting the target it was handed. Keeping both explicit
    # makes the smoothing hyperparameter's cost visible instead of hidden.
    ref_target = (1 - args.ref_smooth) * np.asarray(C_ref) + args.ref_smooth / K
    ref = torch.tensor(ref_target, dtype=torch.float32, device=device)
    ref_np = np.asarray(C_ref)
    # Running estimate of the policy's mode distribution. A single batch of 16
    # episodes cannot estimate a K-bin histogram, and an unobserved mode gets no
    # gradient at all, which biased the objective to only ever suppress visited
    # modes -- measured as MoP collapsing to a single route (H=0.000) at every
    # lambda. An EMA across iterations gives every mode a stable probability.
    C_ema = torch.full((K,), 1.0 / K, device=device)
    coef_absmax = 0.0   # diagnostic: largest mode coefficient seen
    n_inert = 0         # batches where the mode term contributed no gradient

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
                    p_m, q_m = C_phi[m] + 1e-8, ref[m] + 1e-8
                    if args.divergence == "kl":
                        # d/dp_m KL(p || q) = log(p_m/q_m) + 1. UNBOUNDED as
                        # q_m -> 0: where the backbone never goes, a single visit
                        # is worth ~18 against a reward advantage in [-1, 1].
                        mode_coef[j] = -(torch.log(p_m / q_m) + 1.0)
                    elif args.divergence == "tv":
                        # d/dp_m (1/2)|p - q| = (1/2) sign(p_m - q_m), bounded by
                        # 1/2 everywhere including q_m = 0. No smoothing needed.
                        mode_coef[j] = -0.5 * torch.sign(p_m - q_m)
                    else:  # js
                        # d/dp_m JS(p, q) = (1/2) log(p_m/m_m) - 1/2 with
                        # m = (p+q)/2. Since m_m >= p_m/2 the log is at most
                        # log 2, so the q_m -> 0 direction -- the one that breaks
                        # KL -- is bounded; the p_m -> 0 direction is held off by
                        # the EMA floor.
                        mid = 0.5 * (p_m + q_m)
                        mode_coef[j] = -(0.5 * torch.log(p_m / mid) - 0.5)
                else:  # mi: maximize H(C_phi); dH/dp_m = -log(p_m) - 1
                    mode_coef[j] = -(torch.log(C_phi[m] + 1e-8) + 1.0)

        if args.coef_clip > 0:
            # A single extreme coefficient otherwise dominates the batch standard
            # deviation and squashes every other signal to nothing, so clipping
            # has to happen BEFORE standardisation, not after.
            mode_coef = mode_coef.clamp(-args.coef_clip, args.coef_clip)
        raw_coef_absmax = float(mode_coef.abs().max()) if len(mode_coef) else 0.0
        coef_absmax = max(coef_absmax, raw_coef_absmax)
        # Standardise, and ZERO the coefficient when it cannot be standardised.
        # Leaving it as the raw constant vector was a silent defect: a constant
        # added to every advantage contributes exactly zero policy gradient (it
        # is absorbed as a REINFORCE baseline), so the regularizer was inert
        # while appearing active. TV is most exposed, since its coefficient takes
        # only {-1/2, 0, +1/2} and a batch landing on one side of the reference
        # makes it constant; measured, that fires in 0.2-0.4% of batches against
        # KL's 0.0-0.1%. Zeroing makes the inert case explicit rather than hidden.
        mode_inert = True
        if mode_coef.abs().sum() > 0:
            mc = mode_coef - mode_coef.mean()
            sd = mc.std()
            if torch.isfinite(sd) and sd > 1e-6:
                mode_coef = mc / sd
                mode_inert = False
        if mode_inert:
            mode_coef = torch.zeros_like(mode_coef)
            n_inert += 1
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
        task=args.task, arm=arm, lam=lam, skew=skew, seed=seed,
        bb_seed=bb_seed,
        SR=float(np.mean(succ)),
        rewarded_SR=float(np.mean([m in rewarded for m in modes[valid]])) if valid.any() else 0.0,
        H_modes=float(-(p[nz] * np.log(p[nz])).sum()) if nz.any() else 0.0,
        mc_at_08=int((p >= 0.8 / K).sum()),
        n_modes=int(nz.sum()),
        tv_to_ref=float(0.5 * np.abs(p - ref_np).sum()),
        tv_to_uniform=float(0.5 * np.abs(p - 1.0 / K).sum()),
        coef_absmax=float(coef_absmax),
        frac_inert=float(n_inert / max(args.iters, 1)),
        # the evaluated histogram and its reference, so figures can show the
        # distributions themselves rather than only distances between them
        C_phi=p.tolist(),
        C_ref=ref_np.tolist(),
        collapsed=bool(int(nz.sum()) <= 1),
    )


def _ref_chunk(payload):
    """Worker: roll the frozen backbone out and count modes."""
    _single_threaded()
    skew, n, seed, M, device, task, bb_seed = payload
    model, blob = load_backbone(backbone_path(task, skew, bb_seed), device)
    env = TASKS[task]["env"](seed=seed, history=blob.get("history", 1))
    rng = np.random.default_rng(seed)
    counts = np.zeros(TASKS[task]["n_modes"])
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


def _single_threaded():
    """One torch thread per worker process.

    The BLAS pools are capped at import time at the top of this file, which is
    the part that mattered; this covers torch's own pool, which is set through
    an API rather than the environment.
    """
    torch.set_num_threads(1)


def _cell(payload):
    """Worker: one (arm, seed) cell of the sweep.

    An exception here propagates out of ``ProcessPoolExecutor.map`` and kills the
    whole sweep, discarding every cell the pool had finished but not yet yielded.
    One sweep died that way eleven hours in, so a failed cell is reported as a
    row and the remaining cells carry on.
    """
    _single_threaded()
    arm, lam, skew, seed, args, C_ref, bb_seed = payload
    try:
        return run_arm(arm, lam, skew, seed, args, C_ref, bb_seed)
    except Exception as exc:                             # noqa: BLE001
        traceback.print_exc()
        return {"arm": arm, "seed": seed, "skew": skew, "bb_seed": bb_seed,
                "failed": f"{type(exc).__name__}: {exc}",
                "SR": 0.0, "H_modes": 0.0, "mc_at_08": 0, "tv_to_ref": float("nan"),
                "tv_to_uniform": float("nan"), "coef_absmax": 0.0, "collapsed": True}


def reference_profile(skew, args, bb_seed=None) -> np.ndarray:
    """C_ref: the frozen backbone's own mode distribution, measured by rollout.

    Estimated from many episodes because C_ref's own sampling error enters
    tv_to_ref for every arm: with 24 modes, a 200-episode histogram sits ~0.10 in
    total variation from the distribution it estimates, which is the size of the
    effect being measured.
    """
    n_w = max(1, args.workers)
    per = max(1, args.ref_episodes // n_w)
    payloads = [(skew, per, 12345 + i, args.M, args.device, args.task, bb_seed)
                for i in range(n_w)]
    # This runs outside _cell, so an exception here takes the whole sweep down
    # rather than one cell; three Pushing jobs died this way after a day of work.
    # A chunk that raises contributes nothing and the estimate uses the rest.
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        futures = [ex.submit(_ref_chunk, pl) for pl in payloads]
        chunks, lost = [], 0
        for fu in futures:
            try:
                chunks.append(fu.result())
            except Exception:                            # noqa: BLE001
                traceback.print_exc()
                lost += 1
    if not chunks:
        raise RuntimeError(f"every reference chunk failed for skew {skew} "
                           f"backbone {bb_seed}")
    if lost:
        print(f"  reference estimate lost {lost}/{len(payloads)} chunks",
              flush=True)
    counts24 = sum(chunks)
    K = args.mode_groups
    counts = np.zeros(K)
    for m in range(TASKS[args.task]['n_modes']):
        counts[coarsen(m, K)] += counts24[m]
    return counts / max(counts.sum(), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="avoiding", choices=sorted(TASKS))
    ap.add_argument("--skews", nargs="+", default=["0.00", "0.05", "0.15", "0.30", "0.60"])
    ap.add_argument("--arms", nargs="+", default=["dsrl", "mop", "mi"],
                    help="dsrl | mop | mi | bmd (learned skills + variational MI)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bmd-prior", default="uniform", choices=["uniform", "fitted"],
                    help="skill prior the learned-skill arm uses at evaluation")
    ap.add_argument("--backbone-seeds", nargs="*", type=int, default=None,
                    help="train/evaluate across several backbones per skew, so the\n"
                         "skew level is not confounded with one BC draw")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first seed index; use a fresh block for an "
                         "independent replication rather than a re-run")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--ref-episodes", type=int, default=150)
    ap.add_argument("--n-rewarded", type=int, default=4)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mode-groups", type=int, default=24, choices=[2, 4, 6, 24],
                    help="ground-truth partition resolution (2, 6 or all 24 routes)")
    ap.add_argument("--divergence", default="kl", choices=["kl", "tv", "js"],
                    help="anchoring divergence for the mop arm; tv and js have\n"
                         "bounded coefficients where kl does not")
    ap.add_argument("--coef-clip", type=float, default=0.0,
                    help="clip |mode coefficient| before standardising (0 = off)")
    ap.add_argument("--ref-smooth", type=float, default=0.0,
                    help="mix this much uniform into C_ref, bounding log(p/q)")
    ap.add_argument("--ema", type=float, default=0.15,
                    help="EMA rate for the running mode histogram")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/_d3il/headtohead.json"))
    ap.add_argument("--resume", action="store_true",
                    help="keep the cells already in --out and run only the rest; "
                         "the per-cell write makes any killed sweep restartable")
    ap.add_argument("--resume-also", nargs="*", default=[], metavar="FILE",
                    help="count cells in these files as done without writing to "
                         "them. Lets a sweep that was split differently pick up "
                         "earlier work: two jobs writing one file clobber each "
                         "other, so each keeps its own output and reads the rest")
    args = ap.parse_args()

    rows = []
    done = set()
    if args.resume and args.out.exists():
        rows = [r for r in json.loads(args.out.read_text()) if not r.get("failed")]
        done = {(r["skew"], r.get("bb_seed"), r["arm"], r["seed"]) for r in rows}
        print(f"resume: {len(rows)} cells already in {args.out}", flush=True)
    for extra in args.resume_also:
        for f in sorted(Path().glob(extra)) or [Path(extra)]:
            if not f.exists():
                continue
            seen = [r for r in json.loads(f.read_text()) if not r.get("failed")]
            before = len(done)
            done |= {(r["skew"], r.get("bb_seed"), r["arm"], r["seed"]) for r in seen}
            print(f"resume-also: {len(done) - before} further cells from {f}",
                  flush=True)
    bb_seeds = args.backbone_seeds if args.backbone_seeds else [None]
    for skew in args.skews:
        for bb in bb_seeds:
            # Each backbone gets its OWN measured reference: C_ref is a property
            # of that checkpoint, not of the skew label it was trained under.
            todo = [(arm, seed) for seed in range(args.seed_offset,
                                                  args.seed_offset + args.seeds)
                    for arm in args.arms
                    if (skew, bb, arm, seed) not in done]
            if not todo:
                print(f"=== skew {skew} backbone {bb}: complete, skipped ===",
                      flush=True)
                continue
            C_ref = reference_profile(skew, args, bb)
            nz = C_ref > 0
            K = args.mode_groups
            print(f"=== skew {skew} backbone {bb}: C_ref over {int(nz.sum())} modes, "
                  f"H={-(C_ref[nz] * np.log(C_ref[nz])).sum():.3f} nats, "
                  f"KL||u={float(np.sum(C_ref[nz] * np.log(C_ref[nz] * K))):.3f} ===",
                  flush=True)
            cells = [(arm, 0.0 if arm == "dsrl" else args.lam, skew, seed,
                      args, C_ref, bb) for arm, seed in todo]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for res in ex.map(_cell, cells):
                    res["C_ref_kl_to_uniform"] = float(
                        np.sum(C_ref[nz] * np.log(C_ref[nz] * K)))
                    rows.append(res)
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(rows, indent=2))
                    print(f"  bb{bb} {res['arm']:5s} seed{res['seed']}  SR={res['SR']:.3f} "
                          f"H={res['H_modes']:.3f} mc={res['mc_at_08']:2d} "
                          f"tv_ref={res['tv_to_ref']:.3f} tv_unif={res['tv_to_uniform']:.3f} "
                          f"|coef|max={res['coef_absmax']:.1f} inert={res.get('frac_inert',0):.2f}"
                          f"{'  COLLAPSED' if res['collapsed'] else ''}",
                          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
