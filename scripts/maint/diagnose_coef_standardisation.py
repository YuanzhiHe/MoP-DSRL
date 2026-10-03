#!/usr/bin/env python3
"""Two code-level objections to the bounded-anchoring claim, measured.

(a) Both the advantage and the mode coefficient are standardised to unit SD
    before being combined, so their relative scale is lambda regardless of the
    divergence. The "17.6 against a reward advantage in [-1, 1]" argument
    therefore does not describe the gradient that is actually applied. Whatever
    separates TV from KL has to be a property of the coefficient distribution's
    SHAPE after standardisation, not of its raw magnitude.

(b) When every episode in a batch falls on the same side of the reference, the
    centred coefficients are identically zero, the sd > 1e-6 guard skips the
    normalisation, and mode_coef stays a CONSTANT vector. A constant added to
    every advantage contributes exactly zero policy gradient. TV's coefficient
    takes only values in {-1/2, 0, +1/2}, so it hits that degenerate case far
    more often than KL does -- which is an alternative explanation for "TV never
    collapses": it may simply stop pushing when the policy is concentrated.

Replays the coefficient computation over sampled batches rather than rerunning
the environment, so both questions are answerable in seconds.
"""
from __future__ import annotations

import numpy as np

K, BATCH, TRIALS = 6, 16, 4000
C_REF = np.array([0.654, 0.213, 0.077, 0.044, 0.012, 0.000])   # measured, skew 0.60
EPS = 1e-8


def coefs(div: str, C_phi: np.ndarray, modes: np.ndarray) -> np.ndarray:
    p, q = C_phi[modes] + EPS, C_REF[modes] + EPS
    if div == "kl":
        return -(np.log(p / q) + 1.0)
    if div == "tv":
        return -0.5 * np.sign(p - q)
    m = 0.5 * (p + q)
    return -(0.5 * np.log(p / m) - 0.5)


def standardise(c: np.ndarray):
    """Mirror the driver exactly, including the branch that skips normalisation."""
    if np.abs(c).sum() == 0:
        return c, False
    mc = c - c.mean()
    sd = mc.std()
    if np.isfinite(sd) and sd > 1e-6:
        return mc / sd, True
    return c, False       # left as a constant vector -> zero policy gradient


def main() -> int:
    rng = np.random.default_rng(0)
    print(f"K={K} route groups, batch={BATCH} episodes, {TRIALS} sampled batches")
    print(f"C_ref = {C_REF}\n")
    print("Policy histograms span concentrated to uniform; modes drawn from each.\n")
    print(f"{'C_phi concentration':>22}  {'divergence':>10}  "
          f"{'inert batches':>14}  {'post-std |coef| mean':>21}  {'max':>7}")
    for conc, label in ((6.0, "concentrated"), (1.5, "moderate"), (0.0, "uniform")):
        if conc == 0.0:
            C_phi = np.full(K, 1.0 / K)
        else:
            C_phi = np.exp(-conc * np.arange(K) / K)
            C_phi /= C_phi.sum()
        for div in ("kl", "js", "tv"):
            inert, mags, mx = 0, [], 0.0
            for _ in range(TRIALS):
                modes = rng.choice(K, size=BATCH, p=C_phi)
                c = coefs(div, C_phi, modes)
                cs, normed = standardise(c)
                if not normed:
                    inert += 1
                else:
                    mags.append(np.abs(cs).mean())
                    mx = max(mx, float(np.abs(cs).max()))
            frac = inert / TRIALS
            mm = float(np.mean(mags)) if mags else float("nan")
            print(f"{label:>22}  {div.upper():>10}  {frac:>13.1%}  {mm:>21.3f}  {mx:>7.2f}")
        print()
    print("inert batches = the sd guard skipped normalisation, leaving a constant")
    print("vector, which contributes exactly zero policy gradient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
