#!/usr/bin/env python3
"""Verify the penalty gradients, numerically and against an exact model.

Two separate claims are checked.

1. Each ``*_grad`` returns the partial derivatives of the divergence it names.
   Checked by central differences on the divergence itself.

2. ``conditional_coefficient`` is the score-function coefficient for a penalty
   on the CONDITIONAL mode histogram. Checked exactly: over a trajectory space
   small enough to enumerate, the expectation E[coef * grad log mu] is compared
   with the true gradient of D(C(phi)), which is itself obtained by central
   differences on the exact C(phi). The old batch-centred coefficient is run
   through the same comparison so the difference is visible rather than argued.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimators as E

rng = np.random.default_rng(0)
FAIL = []


def report(name, got, want, tol):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    ok = err < tol
    if not ok:
        FAIL.append(name)
    print(f"  {name:<52} max|err|={err:.3e}  {'ok' if ok else 'FAIL'}")


# ---------------------------------------------------------------- derivatives
print("1. partial derivatives against central differences")
h = 1e-6
# The numerical guard is tested separately below. Here eps is zero, so this
# compares the formulas themselves, on histograms bounded away from the
# boundary where the logarithms are not differentiable.
for name in ("kl", "tv", "js", "l2"):
    worst = 0.0
    for trial in range(200):
        K = int(rng.integers(3, 8))
        p = rng.dirichlet(np.ones(K) * 3)
        q = rng.dirichlet(np.ones(K) * 3)
        if min(p.min(), q.min()) < 1e-2:
            continue
        if name == "tv":
            # the derivative of |p-q| does not exist where p_k == q_k; keep the
            # test away from that set rather than pretending it is smooth there
            if np.min(np.abs(p - q)) < 1e-3:
                continue
        eps = 0.0
        g = E.DIVERGENCES[name](p, q, eps)
        num = np.zeros(K)
        for k in range(K):
            pp, pm = p.copy(), p.copy()
            pp[k] += h; pm[k] -= h
            num[k] = (E.VALUES[name](pp, q, eps) - E.VALUES[name](pm, q, eps)) / (2 * h)
        worst = max(worst, float(np.max(np.abs(g - num))))
    report(f"{name}_grad vs central difference", [worst], [0.0], 2e-5)

# entropy, whose objective is -H
worst = 0.0
for trial in range(200):
    K = int(rng.integers(3, 8))
    p = rng.dirichlet(np.ones(K))
    g = E.entropy_grad(p)
    num = np.zeros(K)
    for k in range(K):
        pp, pm = p.copy(), p.copy()
        pp[k] += h; pm[k] -= h
        num[k] = (E.neg_entropy(pp) - E.neg_entropy(pm)) / (2 * h)
    worst = max(worst, float(np.max(np.abs(g - num))))
report("entropy_grad vs central difference", [worst], [0.0], 2e-5)

# what the numerical guard costs. The regularised KL value function has
# derivative log((p+eps)/(q+eps)) + p/(p+eps), so the implemented coefficient
# overstates it by eps/(p_k+eps): negligible at the histogram floor the EMA
# holds, and reported here rather than assumed.
print("  guard bias eps/(p+eps), eps=1e-8:",
      ", ".join(f"p={f:g} -> {1e-8/(f+1e-8):.1e}" for f in (1e-2, 1e-3, 1e-5)))

# the defect the old driver carried: a constant -1/2 added to the JS coefficient
p = rng.dirichlet(np.ones(5)); q = rng.dirichlet(np.ones(5))
legacy = 0.5 * np.log((p + 1e-8) / (0.5 * (p + q) + 1e-8)) - 0.5
print(f"  legacy JS coefficient differs from the derivative by a constant "
      f"{float(np.mean(legacy - E.js_grad(p, q))):+.3f}")

# ------------------------------------------------- exact score-function check
print("\n2. conditional coefficient against the exact gradient of D(C(phi))")


def exact_model(phi):
    """A trajectory space of six outcomes: (valid, mode) pairs plus a failure.

    mu_phi is a softmax over the six outcomes, so the whole expectation can be
    summed rather than sampled and there is no Monte Carlo error to hide behind.
    """
    logits = phi
    mu = np.exp(logits - logits.max())
    mu = mu / mu.sum()
    return mu


# outcome table: (valid, mode). Two outcomes are unlabelled.
VALID = np.array([1, 1, 1, 1, 0, 0])
MODE = np.array([0, 1, 2, 0, -1, -1])
K = 3


def C_of(mu):
    """Conditional mode histogram implied by the outcome distribution."""
    w = mu * VALID
    c = np.zeros(K)
    for i in range(len(mu)):
        if VALID[i]:
            c[MODE[i]] += mu[i]
    return c / w.sum(), float(w.sum())


def D_of(phi, q, name, eps):
    mu = exact_model(phi)
    C, _ = C_of(mu)
    return E.VALUES[name](C, q, eps)


def grad_log_mu(phi):
    """d/dphi_j log mu_i = delta_ij - mu_j."""
    mu = exact_model(phi)
    return np.eye(len(mu)) - mu[None, :]


for name in ("kl", "js", "l2", "tv"):
    worst_new = worst_old = 0.0
    for trial in range(60):
        phi = rng.normal(size=6)
        q = rng.dirichlet(np.ones(K) * 2)
        eps = 1e-8 if name in ("kl", "js") else 0.0
        mu = exact_model(phi)
        C, s = C_of(mu)
        if name == "tv" and np.min(np.abs(C - q)) < 1e-2:
            continue
        g = E.DIVERGENCES[name](C, q, eps)

        # true gradient of the divergence with respect to phi
        true = np.zeros(6)
        for j in range(6):
            pp, pm = phi.copy(), phi.copy()
            pp[j] += 1e-6; pm[j] -= 1e-6
            true[j] = (D_of(pp, q, name, eps) - D_of(pm, q, name, eps)) / 2e-6

        gl = grad_log_mu(phi)                      # outcome x parameter
        coef_new = E.conditional_coefficient(VALID, np.where(VALID == 1, MODE, 0),
                                             g, C, s, s_floor=0.0)
        est_new = (mu[:, None] * coef_new[:, None] * gl).sum(0)

        # the legacy coefficient: g at the trajectory's own mode, zero when
        # unlabelled, then the whole batch centred
        raw = np.where(VALID == 1, g[np.where(VALID == 1, MODE, 0)], 0.0)
        coef_old = raw - raw.mean()
        est_old = (mu[:, None] * coef_old[:, None] * gl).sum(0)

        worst_new = max(worst_new, float(np.max(np.abs(est_new - true))))
        worst_old = max(worst_old, float(np.max(np.abs(est_old - true))))
    report(f"{name}: conditional coefficient reproduces grad D", [worst_new], [0.0], 1e-5)
    print(f"  {'':<52} legacy coefficient is off by up to {worst_old:.3f}")

print("\nVERDICT:", "PASS" if not FAIL else f"FAIL {FAIL}")
sys.exit(0 if not FAIL else 1)
