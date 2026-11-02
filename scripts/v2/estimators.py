"""Gradient coefficients for the mode penalty, separated so they can be tested.

Nothing here touches torch or the environment: each function maps histograms to
the per-mode partial derivatives of a divergence, so a finite-difference check
can hold them against the divergence itself.

The penalty is a function of a BATCH STATISTIC, the mode histogram, so its
policy gradient is not the gradient of a per-trajectory reward. Writing
D(C) for the divergence between the policy's conditional mode distribution C
and the reference, and g_k = dD/dC[k],

    grad_phi D = E_tau[ coefficient(tau) * grad_phi log mu_phi(tau) ]

with the coefficient given by ``conditional_coefficient`` below. Adding the
scalar D to every trajectory instead contributes exactly zero, since a constant
is absorbed as a REINFORCE baseline.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def kl_grad(p: np.ndarray, q: np.ndarray, eps: float) -> np.ndarray:
    """d/dp_k KL(p || q) = log(p_k/q_k) + 1.

    Unbounded as q_k -> 0, which is why ``eps`` is explicit and recorded rather
    than hidden in the expression.
    """
    return np.log((p + eps) / (q + eps)) + 1.0


def tv_grad(p: np.ndarray, q: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """d/dp_k (1/2)|p - q|_1 = (1/2) sign(p_k - q_k). Bounded by 1/2."""
    return 0.5 * np.sign(p - q)


def js_grad(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """d/dp_k JS(p, q) = (1/2) log(p_k / m_k), m = (p + q)/2.

    The divergence is bounded by log 2. The derivative is not bounded globally:
    as p_k -> 0 with q_k > 0 the ratio p_k/m_k -> 0 and the log diverges. Only
    the q_k -> 0 direction, the one that breaks KL, is bounded, by (1/2) log 2.
    """
    m = 0.5 * (p + q)
    return 0.5 * np.log((p + eps) / (m + eps))


def l2_grad(p: np.ndarray, q: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """d/dp_k (1/2)||p - q||^2 = p_k - q_k."""
    return p - q


def entropy_grad(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """d/dp_k of the objective -H(p), which maximising entropy minimises.

    H(p) = -sum p log p, so dH/dp_k = -log(p_k) - 1 and d(-H)/dp_k =
    log(p_k) + 1. Returning the derivative of a quantity to MINIMISE keeps this
    interchangeable with the divergence gradients above.
    """
    return np.log(p + eps) + 1.0


DIVERGENCES = {"kl": kl_grad, "tv": tv_grad, "js": js_grad, "l2": l2_grad}


def kl(p, q, eps):
    p = np.asarray(p, float); q = np.asarray(q, float)
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def tv(p, q, eps=0.0):
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def js(p, q, eps=EPS):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + eps) / (m + eps)))
                 + 0.5 * np.sum(q * np.log((q + eps) / (m + eps))))


def l2(p, q, eps=0.0):
    d = np.asarray(p, float) - np.asarray(q, float)
    return float(0.5 * np.dot(d, d))


def neg_entropy(p, eps=EPS):
    p = np.asarray(p, float)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] + eps)))


VALUES = {"kl": kl, "tv": tv, "js": js, "l2": l2}


def conditional_coefficient(valid: np.ndarray, modes: np.ndarray,
                            g: np.ndarray, C: np.ndarray, s: float,
                            s_floor: float) -> np.ndarray:
    """Per-trajectory coefficient for a penalty on a CONDITIONAL histogram.

    The endpoint is C[k] = P(mode = k | the trajectory carries a valid label).
    Write V for the indicator that a trajectory is labelled and s = P(V = 1).
    Differentiating C[k] = E[V 1{mode=k}] / E[V] through the score function and
    collecting terms gives the coefficient

        (V / s) * ( g_mode - sum_k C[k] g_k )

    so an unlabelled trajectory carries exactly zero and a labelled one carries
    its own partial derivative measured against the C-weighted mean of all of
    them. Centring the batch instead, with zeros left in place of the unlabelled
    entries, subtracts (n_valid/n) * mean_valid(g), which is a different
    quantity and does not vanish in expectation.

    ``C`` and ``s`` are plug-in estimates carried between updates, so this is an
    approximation to the population coefficient, not an unbiased estimate of it.
    ``s_floor`` bounds the 1/s factor and is fixed before any run.
    """
    C = np.asarray(C, float)
    g = np.asarray(g, float)
    baseline = float(np.dot(C, g))
    s_eff = max(float(s), float(s_floor))
    out = np.zeros(len(modes), float)
    sel = valid.astype(bool)
    out[sel] = (g[modes[sel]] - baseline) / s_eff
    return out
