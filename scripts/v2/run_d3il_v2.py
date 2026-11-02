#!/usr/bin/env python3
"""Corrected D3IL sweep driver.

Separate from ``scripts/run_d3il_headtohead.py`` on purpose. The legacy driver
and this one disagree about what the update rule is, so their outputs are not
interchangeable and neither may count a cell as done for the other. This writes
to ``runs/_d3il_v2/`` and stamps every row with ESTIMATOR_VERSION.

What changed, and why
---------------------
*Conditional gradient.* The penalty acts on C[k] = P(mode = k | valid label).
The score-function coefficient for that quantity is (V/s)(g_mode - sum_k C[k]
g_k); see ``estimators.conditional_coefficient`` and the exact check in
``check_mathematics.py``. The legacy driver assigned g at the trajectory's own
mode, zero when unlabelled, then centred the batch, which is a different
quantity. C and s are carried as lagged EMA estimates, so the coefficient is a
plug-in approximation and not an unbiased estimate of the population gradient.

*One objective, one scale.* The legacy driver divided the reward advantage and
the mode coefficient by their own batch standard deviations before combining
them, which makes the effective penalty weight lam*sd(adv)/sd(coef), a quantity
that moves from batch to batch. No fixed objective corresponds to that update.
Here the advantage uses a leave-one-out baseline and neither term is rescaled,
so the update is the gradient of E[r] - lam*D up to the plug-in estimates. The
legacy rule is kept behind --standardise legacy as a diagnostic contrast and is
labelled a surrogate wherever it is reported. Weights do not transfer between
the two: --standardise fixed needs its own calibration.

*JS.* d/dp_k JS(p,q) = (1/2) log(p_k/m_k). The legacy expression carried an
extra -1/2. Applied only to labelled trajectories, that constant does not
cancel as a baseline.

*Rewarded success.* Membership is tested on the raw route id, the same label
the training reward uses, before any coarsening, and over all episodes rather
than only the labelled ones. Geometric success, completion, rewarded success
and the mode histogram are reported separately.

*Reference.* Raw per-bin counts, episode totals and lost chunks are stored, not
only the normalised histogram. The cache key covers the checkpoint hash and
every setting that changes what is being measured. An estimate that lost chunks
is recorded as incomplete and is not reused as if whole.
"""
from __future__ import annotations

import os

# Cap the BLAS pools before numpy and torch load; one process per cell.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import estimators as EST  # noqa: E402
from run_d3il_headtohead import (  # noqa: E402
    TASKS, NoisePolicy, SkillPolicy, SkillDiscriminator, backbone_path,
    coarsen, decode, load_backbone,
)

ESTIMATOR_VERSION = "v2.0"
_MP = mp.get_context("spawn")
OUT_ROOT = ROOT / "runs" / "_d3il_v2"


def _single_threaded():
    torch.set_num_threads(1)


def n_bins(args) -> int:
    """Bins the objective is defined over.

    The conditional objective has one bin per mode. The joint objective carries
    an extra bin for an episode that finished without a label, which makes it a
    different target: a policy can lower it by finishing more often as well as
    by redistributing between modes. Its reference is estimated over the same
    K+1 bins, never reused from the conditional one.
    """
    return args.mode_groups + (1 if args.objective == "joint" else 0)


def bin_of(raw_mode: int, args) -> int:
    """Bin a raw route id falls in, or -1 when the objective excludes it."""
    K = args.mode_groups
    if raw_mode >= 0:
        return coarsen(raw_mode, K)
    return K if args.objective == "joint" else -1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- bookkeeping
# Settings that change what a cell means. Anything here enters the cell key, so
# a run under different settings can never be counted as the same unit of work.
SEMANTIC = ("task", "objective", "standardise", "arm", "lam", "divergence",
            "mode_groups", "bmd_prior", "iters", "episodes_per_iter",
            "eval_episodes", "ref_episodes", "lr", "M", "n_rewarded", "ema",
            "s_ema", "s_floor", "ref_smooth", "coef_clip", "kl_eps")


def config_hash(args, arm: str, lam: float) -> str:
    d = {k: getattr(args, k, None) for k in SEMANTIC}
    d["arm"] = arm
    d["lam"] = lam
    d["estimator_version"] = ESTIMATOR_VERSION
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def cell_key(row: dict) -> tuple:
    """Identity of a unit of work. Includes the hash of every semantic setting."""
    return (row["task"], row["weighting"], row["backbone_seed"], row["arm"],
            row["rl_seed"], row["config_sha256"], row["checkpoint_sha256"],
            row.get("reference_sha256"))


# ---------------------------------------------------------------- rollouts
def rollout(env, policy, model, blob, rewarded, device, M, greedy=False):
    """One episode. The reward is earned on the RAW route id, never a coarsened one."""
    obs = env.reset()
    logps, done = [], False
    while not done:
        o = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                         dtype=torch.float32, device=device)[None]
        d = policy.dist(o)
        z = d.mean if greedy else d.rsample()
        logps.append(d.log_prob(z.detach()).sum())
        obs, r, done, info = env.step(decode(model, blob, o, z.detach(), M)[0])
    success = bool(info["success"])
    raw_mode = int(info["mode_id"])
    earned = success and (raw_mode in rewarded)
    return logps, float(earned), raw_mode, success


def evaluate(episode_fn, n_episodes, args, rewarded, n_raw):
    """Endpoints, each over its own denominator and none of them substituted.

    geometric success and rewarded success are over ALL episodes; the mode
    histogram is over the labelled ones, and is left undefined when there are
    none.
    """
    B = n_bins(args)
    raw_counts = np.zeros(n_raw, dtype=np.int64)
    coarse_counts = np.zeros(B, dtype=np.int64)
    n_succ = n_rewarded_succ = n_valid = 0
    ret = 0.0
    for i in range(n_episodes):
        raw_mode, success = episode_fn(i)
        if success:
            n_succ += 1
        if success and raw_mode in rewarded:
            n_rewarded_succ += 1
            ret += 1.0
        if raw_mode >= 0:
            n_valid += 1
            raw_counts[raw_mode] += 1
        b = bin_of(raw_mode, args)
        if b >= 0:
            coarse_counts[b] += 1
    out = dict(
        eval_episodes=int(n_episodes),
        valid_episodes=int(n_valid),
        completion_rate=float(n_valid / n_episodes),
        success_rate=float(n_succ / n_episodes),
        rewarded_success_rate=float(n_rewarded_succ / n_episodes),
        episode_return=float(ret / n_episodes),
        mode_counts_raw=raw_counts.tolist(),
        mode_counts=coarse_counts.tolist(),
    )
    if coarse_counts.sum() == 0:
        # No labelled episode means the conditional histogram has no estimate.
        # Recording zeros here would put a point at distance 1 from any
        # reference and quietly change the mean of every contrast.
        out.update(C_phi=None, fidelity_defined=False, H_modes=None,
                   n_modes=None, collapsed=None)
    else:
        p = coarse_counts / coarse_counts.sum()
        nz = p > 0
        out.update(C_phi=p.tolist(), fidelity_defined=True,
                   H_modes=float(-(p[nz] * np.log(p[nz])).sum()),
                   n_modes=int(nz.sum()), collapsed=bool(int(nz.sum()) <= 1))
    return out


def score(row: dict, C_ref: np.ndarray, kl_eps: float) -> dict:
    """Distances from the evaluated histogram to the reference it belongs to."""
    if not row["fidelity_defined"]:
        row.update({f"{n}_to_ref": None for n in ("tv", "js", "kl", "l2", "hellinger")})
        row["tv_to_uniform"] = None
        row["max_bin_error"] = None
        return row
    p = np.asarray(row["C_phi"], float)
    q = np.asarray(C_ref, float)
    K = p.size
    row["tv_to_ref"] = EST.tv(p, q)
    row["js_to_ref"] = EST.js(p, q)
    row["kl_to_ref"] = EST.kl(p, q, kl_eps)
    row["l2_to_ref"] = EST.l2(p, q)
    row["hellinger_to_ref"] = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    row["max_bin_error"] = float(np.max(np.abs(p - q)))
    row["tv_to_uniform"] = EST.tv(p, np.full(K, 1.0 / K))
    return row


# ------------------------------------------------------------------- arms
def run_arm(arm, lam, weighting, seed, args, ref, bb_seed=None):
    """One fine-tuning run. ``ref`` is the reference record, not just a histogram."""
    device = args.device
    ck = backbone_path(args.task, weighting, bb_seed)
    model, blob = load_backbone(ck, device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_raw = TASKS[args.task]["n_modes"]
    rewarded = set(rng.choice(n_raw, size=min(args.n_rewarded, n_raw),
                              replace=False).tolist())
    env = TASKS[args.task]["env"](seed=seed, history=blob.get("history", 1))
    policy = NoisePolicy(blob["obs_dim"], blob["act_dim"]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    B = n_bins(args)
    C_ref = np.asarray(ref["C_ref"], float)
    if C_ref.size != B:
        raise ValueError(f"reference has {C_ref.size} bins, objective needs {B}")
    ref_target = (1 - args.ref_smooth) * C_ref + args.ref_smooth / B

    # Lagged plug-in estimates. Both are read before the current batch updates
    # them, so a batch never conditions its own coefficient on itself.
    C_ema = np.full(B, 1.0 / B)
    s_ema = 1.0
    diag = dict(coef_absmax=0.0, coef_absmax_raw=0.0, n_clipped=0, n_inert=0,
                grad_norm_last=None, n_nonfinite=0, s_min=1.0,
                n_zero_reward_batches=0, n_batches=0)

    for _ in range(args.iters):
        batch_lp, batch_r, batch_raw = [], [], []
        for _ in range(args.episodes_per_iter):
            lp, r, raw_mode, _ = rollout(env, policy, model, blob, rewarded,
                                         device, args.M)
            batch_lp.append(torch.stack(lp).sum())
            batch_r.append(r)
            batch_raw.append(raw_mode)
        lp = torch.stack(batch_lp)
        r = np.asarray(batch_r, dtype=np.float64)
        n = len(r)
        diag["n_batches"] += 1
        if r.max() == r.min():
            diag["n_zero_reward_batches"] += 1

        if args.standardise == "legacy":
            adv_np = r - r.mean()
            sd = adv_np.std()
            adv_np = adv_np / sd if np.isfinite(sd) and sd > 1e-6 else adv_np
        else:
            # Leave-one-out baseline: an unbiased control variate that needs no
            # batch standard deviation, so the weight between the two terms is
            # whatever lam says it is.
            adv_np = r - (r.sum() - r) / max(n - 1, 1)

        bins = np.array([bin_of(m, args) for m in batch_raw])
        valid = bins >= 0
        modes = np.where(valid, bins, 0)

        coef = np.zeros(n)
        if arm != "dsrl":
            # read the lagged estimates, then update them with this batch
            C_hat, s_hat = C_ema.copy(), float(s_ema)
            diag["s_min"] = min(diag["s_min"], s_hat)
            if arm == "mop":
                g = EST.DIVERGENCES[args.divergence](C_hat, ref_target, args.kl_eps)
            else:                                  # mi: minimise -H(C)
                g = EST.entropy_grad(C_hat, args.kl_eps)
            coef = -EST.conditional_coefficient(valid, modes, g, C_hat, s_hat,
                                                args.s_floor)
            diag["coef_absmax_raw"] = max(diag["coef_absmax_raw"],
                                          float(np.abs(coef).max(initial=0.0)))
            if args.coef_clip > 0:
                before = coef.copy()
                coef = np.clip(coef, -args.coef_clip, args.coef_clip)
                diag["n_clipped"] += int((before != coef).sum())
            if args.standardise == "legacy":
                c = coef - coef.mean()
                sd = c.std()
                if np.isfinite(sd) and sd > 1e-6:
                    coef = c / sd
                else:
                    coef = np.zeros_like(coef)
                    diag["n_inert"] += 1
            diag["coef_absmax"] = max(diag["coef_absmax"],
                                      float(np.abs(coef).max(initial=0.0)))

        if valid.any():
            cnt = np.bincount(modes[valid], minlength=B).astype(float)
            C_ema = (1 - args.ema) * C_ema + args.ema * (cnt / cnt.sum())
        s_ema = (1 - args.s_ema) * s_ema + args.s_ema * float(valid.mean())

        shaped = torch.tensor(adv_np + lam * coef, dtype=torch.float32, device=device)
        if not torch.isfinite(shaped).all():
            diag["n_nonfinite"] += 1
            shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)
        loss = -(shaped * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
        diag["grad_norm_last"] = float(gn)
        opt.step()

    def ep(_i):
        _, _, raw_mode, success = rollout(env, policy, model, blob, rewarded,
                                          device, args.M)
        return raw_mode, success

    row = evaluate(ep, args.eval_episodes, args, rewarded, n_raw)
    row.update(diag)
    row["frac_inert"] = diag["n_inert"] / max(args.iters, 1)
    return row


def fit_skill_prior(A: np.ndarray, C_ref: np.ndarray):
    """Mixture weights over skills whose DEPLOYED conditional histogram matches C_ref.

    ``A[k, j] = P(valid and mode = j | skill = k)``, so a skill that rarely
    finishes contributes rarely. Under weights w the deployed conditional
    distribution is

        p_j(w) = sum_k w_k A[k, j] / sum_k w_k a_k,    a_k = sum_j A[k, j]

    and p(w) = C_ref is linear in w once the denominator is cleared:
    sum_k w_k (A[k, j] - C_ref[j] a_k) = 0 for every j. Minimising the norm of
    that residual over the simplex is a small constrained least squares, solved
    as such. The legacy code solved an unconstrained least squares against
    P(mode | skill, valid), clipped the negative entries and renormalised, which
    both ignores completion and leaves a solution that no longer minimises
    anything.
    """
    from scipy.optimize import minimize

    K = A.shape[0]
    a = A.sum(1)
    B = (A.T - np.outer(np.asarray(C_ref, float), a))      # modes x skills

    def obj(w):
        r = B @ w
        return float(r @ r)

    def jac(w):
        return 2.0 * (B.T @ (B @ w))

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)},)
    best, best_val = np.full(K, 1.0 / K), obj(np.full(K, 1.0 / K))
    starts = [np.full(K, 1.0 / K)] + [np.eye(K)[i] * 0.7 + 0.3 / K for i in range(K)]
    for w0 in starts:
        res = minimize(obj, w0 / w0.sum(), jac=jac, method="SLSQP",
                       bounds=[(0.0, 1.0)] * K, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success and res.fun < best_val:
            best, best_val = np.clip(res.x, 0, None), float(res.fun)
    s = best.sum()
    return (best / s if s > 1e-12 else np.full(K, 1.0 / K)), best_val


def run_bmd(lam, weighting, seed, args, ref, bb_seed=None):
    """Learned-skill arm: skills plus a variational mutual-information bound."""
    device = args.device
    model, blob = load_backbone(backbone_path(args.task, weighting, bb_seed), device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_raw = TASKS[args.task]["n_modes"]
    rewarded = set(rng.choice(n_raw, size=min(args.n_rewarded, n_raw),
                              replace=False).tolist())
    env = TASKS[args.task]["env"](seed=seed, history=blob.get("history", 1))
    K = args.mode_groups
    B = n_bins(args)
    policy = SkillPolicy(blob["obs_dim"], blob["act_dim"], K).to(device)
    disc = SkillDiscriminator(blob["obs_dim"], K).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

    def episode(k: int):
        obs = env.reset()
        logps, done = [], False
        while not done:
            o = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                             dtype=torch.float32, device=device)[None]
            d = policy.dist(o, k)
            z = d.rsample()
            logps.append(d.log_prob(z.detach()).sum())
            obs, _, done, info = env.step(decode(model, blob, o, z.detach(), args.M)[0])
        last = torch.tensor((obs - blob["obs_mean"]) / blob["obs_std"],
                            dtype=torch.float32, device=device)
        success = bool(info["success"])
        raw_mode = int(info["mode_id"])
        return logps, float(success and raw_mode in rewarded), last, raw_mode, success

    diag = dict(n_batches=0, n_zero_reward_batches=0, grad_norm_last=None,
                n_nonfinite=0, coef_absmax=0.0, coef_absmax_raw=0.0,
                n_clipped=0, n_inert=0, s_min=1.0)
    for _ in range(args.iters):
        lps, rs, lasts, ks = [], [], [], []
        for _ in range(args.episodes_per_iter):
            k = int(rng.integers(K))
            lp, r, last, _, _ = episode(k)
            lps.append(torch.stack(lp).sum()); rs.append(r)
            lasts.append(last); ks.append(k)
        S = torch.stack(lasts)
        kt = torch.tensor(ks, device=device)
        loss_d = nn.functional.cross_entropy(disc(S), kt)
        opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
        with torch.no_grad():
            logq = nn.functional.log_softmax(disc(S), dim=-1)
            r_int = (logq[torch.arange(len(ks), device=device), kt]
                     + np.log(K)).cpu().numpy().astype(np.float64)
        r = np.asarray(rs, dtype=np.float64)
        n = len(r)
        diag["n_batches"] += 1
        if r.max() == r.min():
            diag["n_zero_reward_batches"] += 1
        if args.standardise == "legacy":
            adv = r - r.mean()
            sd = adv.std(); adv = adv / sd if np.isfinite(sd) and sd > 1e-6 else adv
            ri = r_int - r_int.mean()
            sd = ri.std(); ri = ri / sd if np.isfinite(sd) and sd > 1e-6 else ri
        else:
            adv = r - (r.sum() - r) / max(n - 1, 1)
            ri = r_int - (r_int.sum() - r_int) / max(n - 1, 1)
        diag["coef_absmax"] = max(diag["coef_absmax"], float(np.abs(ri).max(initial=0.0)))
        shaped = torch.tensor(adv + lam * ri, dtype=torch.float32, device=device)
        if not torch.isfinite(shaped).all():
            diag["n_nonfinite"] += 1
            shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)
        loss = -(shaped * torch.stack(lps)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        diag["grad_norm_last"] = float(
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf")))
        opt.step()

    calib = None
    if args.bmd_prior == "fitted":
        # Calibration draws from its own stream and its own episodes, so the
        # weights are not fitted on the episodes they are then scored on.
        crng = np.random.default_rng(seed + 10_000_000)
        n_per = max(args.bmd_calib_per_skill, 8)
        A = np.zeros((K, B))
        tot = np.zeros(K)
        for k in range(K):
            for _ in range(n_per):
                _, _, _, raw_mode, _ = episode(k)
                tot[k] += 1
                b = bin_of(raw_mode, args)
                if b >= 0:
                    A[k, b] += 1
        A = A / tot[:, None]                      # P(valid and mode=j | skill=k)
        prior, resid = fit_skill_prior(A, np.asarray(ref["C_ref"], float))
        calib = dict(skill_mode_joint=A.tolist(), calib_episodes_per_skill=int(n_per),
                     fit_residual=float(resid), fitted_on_independent_episodes=True)
        _ = crng
    else:
        prior = np.full(K, 1.0 / K)

    erng = np.random.default_rng(seed + 20_000_000)

    def ep(i):
        k = int(erng.choice(K, p=prior)) if args.bmd_prior == "fitted" else int(i % K)
        _, _, _, raw_mode, success = episode(k)
        return raw_mode, success

    row = evaluate(ep, args.eval_episodes, args, rewarded, n_raw)
    row.update(diag)
    row["skill_prior"] = [float(x) for x in prior]
    row["bmd_calibration"] = calib
    row["frac_inert"] = 0.0
    return row


# -------------------------------------------------------------- reference
def _ref_chunk(payload):
    _single_threaded()
    weighting, n, seed, M, device, task, bb_seed = payload
    model, blob = load_backbone(backbone_path(task, weighting, bb_seed), device)
    env = TASKS[task]["env"](seed=seed, history=blob.get("history", 1))
    rng = np.random.default_rng(seed)
    counts = np.zeros(TASKS[task]["n_modes"], dtype=np.int64)
    n_valid = 0
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
            counts[int(info["mode_id"])] += 1
            n_valid += 1
    return counts, n, n_valid


def reference_record(weighting, args, bb_seed=None) -> dict:
    """Measure the frozen policy's own mode distribution, counts and all.

    The cache key covers the checkpoint hash and every setting that changes what
    is being measured, so an estimate can never be served to a configuration it
    was not measured under. An estimate that lost chunks is stored with
    ``complete: false`` and is not reused unless --allow-incomplete-reference
    says so, because a short estimate is still a biased one.
    """
    ck = backbone_path(args.task, weighting, bb_seed)
    ck_hash = sha256_file(ck)
    key = dict(checkpoint_sha256=ck_hash, task=args.task, K=args.mode_groups,
               M=args.M, ref_episodes=args.ref_episodes, chunk=args.ref_chunk,
               seed_base=args.ref_seed_base, estimator_version=ESTIMATOR_VERSION,
               objective=args.objective)
    kh = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    cache = OUT_ROOT / "_cref" / f"{kh}.json"
    if cache.exists():
        rec = json.loads(cache.read_text())
        if rec.get("complete") or args.allow_incomplete_reference:
            print(f"  reference read from {cache.name}", flush=True)
            return rec
        print(f"  cached reference {cache.name} is incomplete, remeasuring", flush=True)

    chunk = max(1, min(args.ref_chunk, args.ref_episodes))
    n_chunks = max(1, -(-args.ref_episodes // chunk))
    payloads = [(weighting, chunk, args.ref_seed_base + i, args.M, args.device,
                 args.task, bb_seed) for i in range(n_chunks)]
    n_w = max(1, args.ref_workers if args.ref_workers else args.workers)
    raw = np.zeros(TASKS[args.task]["n_modes"], dtype=np.int64)
    n_ep = n_valid = lost = 0
    t0 = time.time()
    if n_w == 1:
        for i, pl in enumerate(payloads, 1):
            c, ne, nv = _ref_chunk(pl)
            raw += c; n_ep += ne; n_valid += nv
            if i % 10 == 0 or i == len(payloads):
                print(f"  reference {n_ep}/{args.ref_episodes} episodes, "
                      f"{time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_w, mp_context=_MP) as ex:
            for fu in [ex.submit(_ref_chunk, pl) for pl in payloads]:
                try:
                    c, ne, nv = fu.result()
                    raw += c; n_ep += ne; n_valid += nv
                except Exception:                       # noqa: BLE001
                    traceback.print_exc()
                    lost += 1
    if n_valid == 0 and args.objective == "conditional":
        raise RuntimeError(f"reference for {args.task} {weighting} bb{bb_seed} "
                           f"produced no labelled episode")
    K = args.mode_groups
    B = n_bins(args)
    coarse = np.zeros(B, dtype=np.int64)
    for m in range(TASKS[args.task]["n_modes"]):
        coarse[coarsen(m, K)] += raw[m]
    if args.objective == "joint":
        coarse[K] = n_ep - n_valid
    rec = dict(
        task=args.task, weighting=weighting, backbone_seed=bb_seed,
        checkpoint_sha256=ck_hash, key=key, cache_key_sha256=kh,
        counts_raw=raw.tolist(), counts=coarse.tolist(),
        episodes=int(n_ep), valid_episodes=int(n_valid),
        requested_episodes=int(args.ref_episodes),
        chunks_lost=int(lost), chunks_total=len(payloads),
        complete=bool(lost == 0 and n_ep >= args.ref_episodes),
        C_ref=(coarse / coarse.sum()).tolist(),
        seconds=float(time.time() - t0),
    )
    rec["reference_sha256"] = hashlib.sha256(
        json.dumps(rec["counts"], sort_keys=True).encode()).hexdigest()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, indent=1))
    print(f"  reference cached to {cache.name}  "
          f"{n_valid}/{n_ep} labelled, {lost} chunks lost", flush=True)
    return rec


# ------------------------------------------------------------------- sweep
def _cell(payload):
    _single_threaded()
    arm, lam, weighting, seed, args, ref, bb_seed, attempt = payload
    base = dict(task=args.task, weighting=weighting, backbone_seed=bb_seed,
                arm=arm, rl_seed=seed, lam=lam, attempt=int(attempt),
                objective=args.objective, standardise=args.standardise,
                divergence=args.divergence if arm == "mop" else None,
                mode_groups=args.mode_groups,
                bmd_prior=args.bmd_prior if arm == "bmd" else None,
                iters=args.iters, episodes_per_iter=args.episodes_per_iter,
                eval_episodes=args.eval_episodes, ref_episodes=args.ref_episodes,
                lr=args.lr, M=args.M, n_rewarded=args.n_rewarded, ema=args.ema,
                s_ema=args.s_ema, s_floor=args.s_floor,
                ref_smooth=args.ref_smooth, coef_clip=args.coef_clip,
                kl_eps=args.kl_eps, optimiser="Adam",
                estimator_version=ESTIMATOR_VERSION,
                code_commit=args.code_commit,
                config_sha256=config_hash(args, arm, lam),
                checkpoint_sha256=ref["checkpoint_sha256"],
                reference_sha256=ref["reference_sha256"],
                reference_complete=ref["complete"],
                reference_episodes=ref["episodes"],
                reference_valid_episodes=ref["valid_episodes"],
                reference_counts=ref["counts"],
                C_ref=ref["C_ref"])
    try:
        if arm == "bmd":
            res = run_bmd(lam, weighting, seed, args, ref, bb_seed)
        else:
            res = run_arm(arm, lam, weighting, seed, args, ref, bb_seed)
        base.update(res)
        base["status"] = "completed"
        return score(base, np.asarray(ref["C_ref"], float), args.kl_eps)
    except Exception as exc:                            # noqa: BLE001
        traceback.print_exc()
        base.update(status="exception",
                    exception=f"{type(exc).__name__}: {exc}",
                    exception_traceback=traceback.format_exc()[-4000:],
                    fidelity_defined=False)
        return base


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="avoiding", choices=sorted(TASKS))
    ap.add_argument("--weightings", nargs="+", required=True,
                    help="demonstration weightings, the LAUNCH parameter")
    ap.add_argument("--arms", nargs="+", default=["dsrl", "mi", "mop", "bmd"])
    ap.add_argument("--backbone-seeds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--objective", default="conditional",
                    choices=["conditional", "joint"],
                    help="conditional: P(mode | valid label), the primary "
                         "endpoint. joint: a failure bin is carried as a mode, "
                         "which is a DIFFERENT target and needs its own "
                         "reference")
    ap.add_argument("--standardise", default="fixed", choices=["fixed", "legacy"],
                    help="fixed: leave-one-out baseline, neither term rescaled, "
                         "so lam weights a fixed objective. legacy: each term "
                         "divided by its own batch standard deviation, a "
                         "surrogate whose effective weight moves per batch")
    ap.add_argument("--divergence", default="tv", choices=["tv", "kl", "js", "l2"])
    ap.add_argument("--mode-groups", type=int, default=6, choices=[2, 4, 6, 24])
    ap.add_argument("--bmd-prior", default="uniform", choices=["uniform", "fitted"])
    ap.add_argument("--bmd-calib-per-skill", type=int, default=20)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--eval-episodes", type=int, default=500)
    ap.add_argument("--ref-episodes", type=int, default=900)
    ap.add_argument("--ref-chunk", type=int, default=8)
    ap.add_argument("--ref-seed-base", type=int, default=12345)
    ap.add_argument("--ref-workers", type=int, default=0)
    ap.add_argument("--allow-incomplete-reference", action="store_true")
    ap.add_argument("--n-rewarded", type=int, default=4)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ema", type=float, default=0.15,
                    help="EMA rate for the lagged mode histogram")
    ap.add_argument("--s-ema", type=float, default=0.15,
                    help="EMA rate for the lagged valid-label probability")
    ap.add_argument("--s-floor", type=float, default=0.05,
                    help="floor on the 1/s factor; fixed before any run")
    ap.add_argument("--coef-clip", type=float, default=0.0,
                    help="clip |coefficient|; 0 = off. Fixed before any run")
    ap.add_argument("--kl-eps", type=float, default=1e-8)
    ap.add_argument("--ref-smooth", type=float, default=0.0)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True,
                    help="JSONL; one record per cell, appended as it finishes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="a cell that has already failed this many times is "
                         "left unrun and reported as missing")
    ap.add_argument("--resume-also", nargs="*", default=[])
    ap.add_argument("--reference-only", action="store_true")
    return ap


def load_done(paths):
    """Completed cells, and how many times each cell has already been attempted.

    An exception never removes a row: the failed attempt stays in the file with
    its reason and traceback, and the retry is written beside it carrying the
    next attempt number. Counting attempts here is what keeps a cell that fails
    deterministically from being retried without limit.
    """
    done, attempts = set(), collections.Counter()
    for p in paths:
        for f in sorted(Path().glob(str(p))) or [Path(p)]:
            if not f.exists():
                continue
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                attempts[cell_key(r)] += 1
                if r.get("status") == "completed":
                    done.add(cell_key(r))
    return done, attempts


def main() -> int:
    import subprocess
    args = build_parser().parse_args()
    args.code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done, attempts = set(), collections.Counter()
    if args.resume:
        d, a = load_done([args.out])
        done |= d; attempts.update(a)
    d, a = load_done(args.resume_also)
    done |= d; attempts.update(a)
    if done or attempts:
        failed_before = sum(1 for k, v in attempts.items() if k not in done)
        print(f"resume: {len(done)} completed cells recognised, "
              f"{failed_before} cells have a failed attempt on record", flush=True)

    bbs = args.backbone_seeds if args.backbone_seeds else [None]
    fh = args.out.open("a")
    for weighting in args.weightings:
        for bb in bbs:
            ref = reference_record(weighting, args, bb)
            if args.reference_only:
                continue
            C = np.asarray(ref["C_ref"], float)
            nz = C > 0
            B = n_bins(args)
            print(f"=== {args.task} weighting {weighting} backbone {bb}: "
                  f"{int(nz.sum())}/{B} bins, "
                  f"TV to uniform {EST.tv(C, np.full(B, 1.0 / B)):.3f}, "
                  f"{ref['valid_episodes']}/{ref['episodes']} labelled ===",
                  flush=True)
            cells = []
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                for arm in args.arms:
                    lam = 0.0 if arm == "dsrl" else args.lam
                    probe = dict(task=args.task, weighting=weighting,
                                 backbone_seed=bb, arm=arm, rl_seed=seed,
                                 config_sha256=config_hash(args, arm, lam),
                                 checkpoint_sha256=ref["checkpoint_sha256"],
                                 reference_sha256=ref["reference_sha256"])
                    k = cell_key(probe)
                    if k in done:
                        continue
                    if attempts[k] > args.max_attempts:
                        print(f"  {arm} seed{seed}: {attempts[k]} attempts on "
                              f"record, at the limit, left unrun", flush=True)
                        continue
                    cells.append((arm, lam, weighting, seed, args, ref, bb,
                                  attempts[k] + 1))
            if not cells:
                print("  every cell already completed, skipped", flush=True)
                continue
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=_MP) as ex:
                for res in ex.map(_cell, cells):
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    if res["status"] == "completed":
                        print(f"  bb{bb} {res['arm']:5s} seed{res['rl_seed']} "
                              f"SR={res['success_rate']:.3f} "
                              f"rew={res['rewarded_success_rate']:.3f} "
                              f"comp={res['completion_rate']:.3f} "
                              f"tv_ref={res['tv_to_ref'] if res['tv_to_ref'] is None else round(res['tv_to_ref'],3)} "
                              f"|coef|={res['coef_absmax']:.2f} "
                              f"inert={res['frac_inert']:.2f}", flush=True)
                    else:
                        print(f"  bb{bb} {res['arm']:5s} seed{res['rl_seed']} "
                              f"EXCEPTION {res['exception']}", flush=True)
    fh.close()
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
