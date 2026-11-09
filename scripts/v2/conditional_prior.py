#!/usr/bin/env python3
"""Deployment prior over skills whose completion-conditional histogram matches.

The quantity a skill-conditioned policy is scored on is conditional on an
episode finishing, so the prior that reproduces a reference has to be fitted
against that conditional. Writing ``A[k, j] = P(valid and mode = j | skill k)``
and ``a_k = sum_j A[k, j]`` for the completion rate of skill ``k``, the
deployed conditional under weights ``w`` is

    p_j(w) = sum_k w_k A[k, j] / sum_k w_k a_k .

Clearing the denominator gives ``sum_k w_k (A[k, j] - C_ref[j] a_k) = 0``, and
minimising the norm of that residual over the simplex has a degenerate
solution: a skill that never finishes has ``a_k = 0`` and ``A[k, :] = 0``, so
its whole column is zero and any weight placed on it leaves the residual
untouched. Putting weight 1 there scores a residual of exactly zero while
deploying a policy that finishes nothing. A skill that finishes rarely scores
a residual of order ``a_k`` for the same reason, which rewards weight on the
skills that contribute least.

The fit here is over the conditional mixture instead. With
``B[k, :] = A[k, :] / a_k``, a probability vector for every skill that finishes
at all, and ``alpha_k`` the share of finished episodes coming from skill ``k``,

    p_j(alpha) = sum_k alpha_k B[k, j] ,     alpha on the simplex,

which is linear in ``alpha`` with no degenerate direction, since every ``B[k]``
sums to one. The deployment prior follows from ``w_k proportional to
alpha_k / a_k``, which is the change of variable ``alpha_k = w_k a_k / sum_m
w_m a_m`` read backwards.

A skill with no finished calibration episode has no ``B[k]`` to fit against and
no finite recovery from ``alpha_k`` to ``w_k``. Such a skill is reported as
inestimable rather than given a weight, because the calibration that would
settle it was never collected.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fit_conditional_prior", "deployed_conditional"]


def deployed_conditional(A: np.ndarray, w: np.ndarray):
    """The conditional histogram and completion a prior ``w`` actually deploys."""
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    joint = w @ A                                   # P(valid and mode = j)
    completion = float(joint.sum())
    if completion <= 0.0:
        return None, 0.0
    return joint / completion, completion


def fit_conditional_prior(A: np.ndarray, C_ref, min_episodes: int = 1,
                          counts: np.ndarray | None = None):
    """Fit the deployment prior whose deployed conditional matches ``C_ref``.

    ``counts[k]`` is how many calibration episodes skill ``k`` was measured on,
    used only to decide support; ``A`` carries the rates. A skill is estimable
    when it finished at least ``min_episodes`` calibration episodes.

    Returns a dict carrying the prior, the conditional mixture it came from,
    the residual, the completion the prior is predicted to deploy, and the
    skills that could not be estimated.
    """
    from scipy.optimize import minimize

    A = np.asarray(A, dtype=float)
    q = np.asarray(C_ref, dtype=float)
    K = A.shape[0]
    a = A.sum(1)

    # Support: a skill needs finished calibration episodes to carry a
    # conditional histogram at all. With counts given, the test is on episodes;
    # without, on whether the rate is positive.
    if counts is not None:
        finished = np.asarray(counts, dtype=float).sum(1) if np.ndim(counts) > 1 \
            else np.asarray(counts, dtype=float)
        support = finished >= float(min_episodes)
    else:
        support = a > 0.0
    idx = np.flatnonzero(support)
    inestimable = np.flatnonzero(~support).tolist()

    if idx.size == 0:
        # Nothing to fit against. The uniform prior is returned as a stated
        # fallback, not as a fit, and every skill is reported inestimable.
        w = np.full(K, 1.0 / K)
        p, comp = deployed_conditional(A, w)
        return dict(prior=w.tolist(), alpha=None, residual=float("nan"),
                    predicted_conditional=None if p is None else p.tolist(),
                    predicted_completion=comp, inestimable=list(range(K)),
                    support=support.tolist(), a=a.tolist(), fitted=False,
                    reason="no skill finished a calibration episode")

    B = A[idx] / a[idx][:, None]                    # supported skills x modes

    def obj(al):
        r = al @ B - q
        return float(r @ r)

    def jac(al):
        return 2.0 * B @ (al @ B - q)

    cons = ({"type": "eq", "fun": lambda al: al.sum() - 1.0,
             "jac": lambda al: np.ones_like(al)},)
    n = idx.size
    best, best_val = np.full(n, 1.0 / n), obj(np.full(n, 1.0 / n))
    # The problem is convex, so one start settles it; the vertices are kept as
    # a check that the solver agrees with itself.
    starts = [np.full(n, 1.0 / n)] + [np.eye(n)[i] * 0.7 + 0.3 / n
                                      for i in range(n)]
    for a0 in starts:
        res = minimize(obj, a0 / a0.sum(), jac=jac, method="SLSQP",
                       bounds=[(0.0, 1.0)] * n, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success and res.fun < best_val:
            best, best_val = np.clip(res.x, 0.0, None), float(res.fun)
    alpha = np.zeros(K)
    alpha[idx] = best / max(best.sum(), 1e-12)

    # Recover the deployment prior. Dividing by a_k is the inverse of the
    # change of variable; it is finite on the support by construction.
    w = np.zeros(K)
    w[idx] = alpha[idx] / a[idx]
    s = w.sum()
    w = w / s if s > 1e-12 else np.full(K, 1.0 / K)

    p, comp = deployed_conditional(A, w)
    return dict(prior=w.tolist(), alpha=alpha.tolist(), residual=best_val,
                predicted_conditional=None if p is None else p.tolist(),
                predicted_completion=comp, inestimable=inestimable,
                support=support.tolist(), a=a.tolist(), fitted=True,
                reason="")


def _tests() -> int:
    """The counterexamples the legacy objective scored as perfect fits."""
    rng = np.random.default_rng(0)
    ok = True

    def legacy(A, q):
        """The objective being replaced, for the comparison these tests make."""
        from scipy.optimize import minimize
        K = A.shape[0]
        a = A.sum(1)
        B = (A.T - np.outer(np.asarray(q, float), a))
        f = lambda w: float((B @ w) @ (B @ w))                  # noqa: E731
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        best, bv = np.full(K, 1.0 / K), f(np.full(K, 1.0 / K))
        for i in range(K):
            w0 = np.eye(K)[i] * 0.7 + 0.3 / K
            r = minimize(f, w0, method="SLSQP", bounds=[(0, 1)] * K,
                         constraints=cons, options={"maxiter": 500})
            if r.success and r.fun < bv:
                best, bv = np.clip(r.x, 0, None), float(r.fun)
        return best / best.sum(), bv

    print("=== 1. a skill that never finishes ===")
    # Skill 3 finishes nothing. Skills 0-2 carry a reachable mixture.
    A = np.array([[0.40, 0.10, 0.00, 0.00],
                  [0.05, 0.35, 0.05, 0.00],
                  [0.00, 0.05, 0.30, 0.10],
                  [0.00, 0.00, 0.00, 0.00]])
    q = np.array([0.30, 0.25, 0.25, 0.20])
    lw, lv = legacy(A, q)
    r = fit_conditional_prior(A, q)
    print(f"  legacy prior   {np.round(lw, 3)}  residual {lv:.3e}")
    print(f"  legacy deploys completion "
          f"{deployed_conditional(A, lw)[1]:.3f}")
    print(f"  fitted prior   {np.round(r['prior'], 3)}  "
          f"residual {r['residual']:.3e}")
    print(f"  fitted deploys completion {r['predicted_completion']:.3f}  "
          f"inestimable {r['inestimable']}")
    if lw[3] <= 0.5:
        print("  NOTE: legacy did not concentrate here on this instance")
    ok &= (r["prior"][3] == 0.0 and r["inestimable"] == [3]
           and r["predicted_completion"] > 0.0)

    print("\n=== 2. a skill that finishes very rarely ===")
    A2 = A.copy()
    A2[3] = np.array([0.001, 0.000, 0.000, 0.000])
    lw2, lv2 = legacy(A2, q)
    r2 = fit_conditional_prior(A2, q)
    print(f"  legacy prior   {np.round(lw2, 3)}  residual {lv2:.3e}  "
          f"deploys completion {deployed_conditional(A2, lw2)[1]:.3f}")
    print(f"  fitted prior   {np.round(r2['prior'], 3)}  "
          f"residual {r2['residual']:.3e}  deploys completion "
          f"{r2['predicted_completion']:.3f}")
    ok &= deployed_conditional(A2, r2["prior"])[1] >= \
        deployed_conditional(A2, lw2)[1]

    print("\n=== 3. a target that is exactly reachable ===")
    # Build A from known conditionals and completions, then ask for the
    # conditional a known alpha produces. The fit has to find it.
    Bk = np.array([[0.7, 0.2, 0.1, 0.0],
                   [0.1, 0.6, 0.2, 0.1],
                   [0.0, 0.1, 0.5, 0.4],
                   [0.25, 0.25, 0.25, 0.25]])
    av = np.array([0.6, 0.3, 0.45, 0.2])
    A3 = Bk * av[:, None]
    alpha_true = np.array([0.4, 0.1, 0.35, 0.15])
    q3 = alpha_true @ Bk
    r3 = fit_conditional_prior(A3, q3)
    p3 = np.asarray(r3["predicted_conditional"])
    err = float(np.abs(p3 - q3).max())
    print(f"  target     {np.round(q3, 4)}")
    print(f"  deployed   {np.round(p3, 4)}   max error {err:.2e}")
    print(f"  alpha      {np.round(r3['alpha'], 3)}  "
          f"(true {np.round(alpha_true, 3)})")
    print(f"  prior      {np.round(r3['prior'], 3)}  residual "
          f"{r3['residual']:.2e}")
    ok &= err < 1e-6

    print("\n=== 4. the recorded degeneracies ===")
    for name, prior in (("pushing bb2 seed901",
                         [1.0, 0.0, 1.2603496959481122e-18, 0.0]),
                        ("pushing bb0 seed903",
                         [8.686185395606774e-18, 0.0, 0.15, 0.85])):
        w = np.asarray(prior, float)
        print(f"  {name}: weight on {int((w > 1e-6).sum())} of 4 skills, "
              f"largest {w.max():.3f}")

    print("\n" + ("all checks passed" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_tests())
