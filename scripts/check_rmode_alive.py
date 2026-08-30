#!/usr/bin/env python3
"""P0 gate: does R_mode actually carry gradient?

The LIBERO campaign (2026-07/08) produced r_mode == 0.000000 for every logged
update on libero_goal. Two numerical defects were responsible:

  1. vocab.py floored a near-constant dimension's MAD at an ABSOLUTE 1e-8, so
     1/mad ~ 1e8 for the (near-binary) gripper channel. Centroid separations
     reached ~1e17.
  2. mode_regularizer.soft_assign_distances used a FIXED temperature of 1.0, so
     softmax(-d^2/1.0) with d^2 ~ 1e17 saturated to an exact one-hot whose
     Jacobian is identically zero -- no gradient for any lambda.

This script reproduces that pathology on synthetic data (cheap, CPU, seconds),
then asserts the fixed code passes four gates. Run it before ANY GPU job.

    python scripts/check_rmode_alive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mop_dsrl.components.mode_regularizer import (  # noqa: E402
    adaptive_temperature,
    mode_kl_from_assignments,
    soft_assign_distances,
)
from mop_dsrl.components.vocab import (  # noqa: E402
    KMeansVocab,
    fit_robust_z_stats,
    robust_z_normalize,
)
from sklearn.cluster import KMeans  # noqa: E402

RNG = np.random.default_rng(0)
D_INFORMATIVE, D_CONSTANT, K, N = 12, 4, 8, 4000


def make_multimodal_actions() -> np.ndarray:
    """4 well-separated modes over informative dims + constant 'gripper' dims.

    The constant dims mimic the LIBERO gripper channel that triggered the bug:
    zero demonstration variance, hence MAD == 0 before flooring.
    """
    centers = RNG.normal(scale=3.0, size=(4, D_INFORMATIVE))
    which = RNG.integers(0, 4, size=N)
    informative = centers[which] + RNG.normal(scale=0.30, size=(N, D_INFORMATIVE))
    constant = np.ones((N, D_CONSTANT)) * 0.7  # exactly constant -> MAD = 0
    return np.concatenate([informative, constant], axis=1), which


def build_vocab(actions: np.ndarray) -> KMeansVocab:
    median, mad = fit_robust_z_stats(actions)
    normalized = robust_z_normalize(actions, median, mad)
    km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=42).fit(normalized)
    return KMeansVocab(K=K, median=median, mad=mad, centroids=km.cluster_centers_, seed=42)


def main() -> int:
    actions, _ = make_multimodal_actions()
    vocab = build_vocab(actions)
    cent = torch.tensor(vocab.centroids, dtype=torch.float32)

    failures: list[str] = []

    # --- Gate 1: normalized space is not blown up by the constant dims -------
    pair = torch.cdist(cent, cent).pow(2)
    iu = torch.triu_indices(K, K, offset=1)
    med_sep = pair[iu[0], iu[1]].median().item()
    print(f"[1] median pairwise centroid sq-dist = {med_sep:.3e}")
    print(f"    MAD: min={vocab.mad.min():.3e} max={vocab.mad.max():.3e} "
          f"ratio={vocab.mad.max() / vocab.mad.min():.1f}x")
    if not np.isfinite(med_sep) or med_sep > 1e6:
        failures.append(f"centroid separation {med_sep:.3e} still exploded")

    # --- Gate 2: adaptive temperature keeps the softmax unsaturated ----------
    T = adaptive_temperature(cent)
    normalized = torch.tensor(
        robust_z_normalize(actions[:256], vocab.median, vocab.mad), dtype=torch.float32
    )
    w = soft_assign_distances(normalized, cent, temperature=None)
    max_w = w.max(dim=-1).values
    print(f"[2] adaptive T = {float(T):.3e}; soft-assignment max weight: "
          f"mean={max_w.mean():.4f} (1.0 == fully saturated)")
    if max_w.mean().item() > 0.9999:
        failures.append("softmax still saturates to a one-hot (no gradient path)")

    # --- Gate 3: R_mode has NONZERO gradient wrt a policy parameter ----------
    # Stand-in for mu_phi: a learnable shift applied to sampled actions.
    n_on, B = 8, 16
    shift = torch.zeros(actions.shape[1], requires_grad=True)
    base = torch.tensor(actions[: B * n_on], dtype=torch.float32)
    med_t = torch.tensor(vocab.median, dtype=torch.float32)
    mad_t = torch.tensor(vocab.mad, dtype=torch.float32)
    normed = ((base + shift) - med_t) / mad_t
    soft = soft_assign_distances(normed, cent, temperature=None)
    C_ref = torch.full((B, K), 1.0 / K)
    r_mode = mode_kl_from_assignments(soft, C_ref, n_on=n_on, laplace_alpha=1.0 / (K * n_on))
    r_mode.backward()
    gnorm = shift.grad.norm().item()
    print(f"[3] R_mode = {r_mode.item():.6f}   ||dR_mode/dphi|| = {gnorm:.3e}")
    if not (gnorm > 1e-8):
        failures.append(f"R_mode gradient is {gnorm:.3e} -- STILL DEAD")

    # --- Gate 4: a multimodal backbone yields a non-degenerate C_ref ---------
    # Hard-assign samples drawn per 'observation'; entropy must exceed 0.
    ids = vocab.assign(actions[:2000])
    hist = np.bincount(ids, minlength=K).astype(np.float64)
    hist /= hist.sum()
    ent = float(-(hist * np.log(hist + 1e-12)).sum())
    print(f"[4] marginal cluster entropy = {ent:.3f} nats (max {np.log(K):.3f}); "
          f"clusters used = {(hist > 0.01).sum()}/{K}")
    if ent < 0.1:
        failures.append(f"cluster distribution is degenerate (entropy {ent:.3f})")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL GATES PASSED - R_mode carries gradient; safe to proceed to P1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
