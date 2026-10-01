# Pushing: the separation replicates, and one thing it did not test

Second task for the mode-preservation experiments, which until now rested
entirely on Avoiding. Pushing's behavioural mode is which box reaches which
target and in what order, so unlike Avoiding — whose mode is the arm's own
geometry — it is a property of where the *objects* end up. That makes it a
different test of the same claim rather than a re-parameterisation of it.

Setup: 4 routes (no coarsening needed), TV anchoring with no reference
smoothing, 12 seeds per cell, 60 iterations, 300 evaluation episodes, indexed by
reference skew measured from each frozen backbone. 137 of 144 runs cleared the
SR floor.

## MoP versus MI, fidelity gap, one-sided Welch

| KL(C_ref ‖ uniform) | MoP (TV) | MI | diff | d | p | n |
|---|---|---|---|---|---|---|
| 0.120 | +0.097 | −0.005 | +0.102 | +1.04 | 0.0112 | 11/12 |
| 0.327 | +0.234 | +0.037 | +0.197 | +1.93 | 0.0004 | 12/10 |
| 0.537 | +0.314 | +0.243 | +0.071 | +0.76 | 0.0390 | 12/12 |
| 0.770 | +0.500 | +0.171 | +0.329 | +3.03 | 0.0002 | 12/8 |

All four reject, and the effect again grows with reference skew. MI's gap is
negative or near zero at the two least-skewed references, i.e. it sits nearer
uniform than the policy it started from — the same signature it showed on
Avoiding.

## TV anchoring, away from the task it was chosen on

| KL(C_ref ‖ uniform) | max \|∂D/∂p_m\| | collapsed |
|---|---|---|
| 0.120 | 0.500 | 0 / 11 |
| 0.327 | 0.500 | 0 / 12 |
| 0.537 | 0.500 | 0 / 12 |
| 0.770 | 0.500 | 0 / 12 |

The coefficient sits exactly at the subgradient bound of 1/2 at every level, and
nothing collapsed in 47 runs. Raw KL on Avoiding reached 7.27 and collapsed 8 of
24. So the bounded divergence is not an artifact of the task it was derived on.

## What this sweep did not test

`--n-rewarded` defaults to 4 and Pushing has exactly 4 modes, so `min(4, 4)`
rewarded **every** route. DSRL therefore had no incentive to collapse, and the
data shows it: its mode entropy here is not the lowest of the three, where on
Avoiding it clearly was (0.30–0.58 against 1.12–1.58).

So the two claims have to be separated:

- **MoP and MI converge on different distributions** — supported here, and
  arguably more cleanly than on Avoiding, since it holds with no reward
  asymmetry to confound it.
- **A mode objective prevents reward-driven collapse** — *not* tested here. It
  needs the reward shift present.

Re-running with `--n-rewarded 1`, matching Avoiding's 4-of-24 proportion. The
sweep above stays as the no-asymmetry condition, which is worth reporting in its
own right: the two objectives diverge even when nothing is pushing the policy
off its modes.
