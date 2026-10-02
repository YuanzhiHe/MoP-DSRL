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

---

# With the reward shift present (`--n-rewarded 1`)

117 of 144 runs cleared the SR floor. Same configuration otherwise.

## MoP versus MI: holds

| KL(C_ref ‖ uniform) | MoP (TV) | MI | diff | d | p | n |
|---|---|---|---|---|---|---|
| 0.120 | +0.087 | +0.028 | +0.060 | +0.63 | 0.0914 | 9/11 |
| 0.327 | +0.235 | +0.005 | +0.229 | +1.67 | 0.0005 | 10/12 |
| 0.537 | +0.308 | +0.175 | +0.133 | +1.09 | 0.0273 | 9/8 |
| 0.770 | +0.457 | +0.268 | +0.189 | +1.24 | 0.0318 | 10/6 |

Three of four reject. The one that does not is the least-skewed reference, which
is where theory predicts the least separation.

## TV anchoring: holds

`max |∂D/∂p_m|` is exactly 0.500 at every level and 0 of 38 runs collapsed —
0 of 85 across both Pushing conditions.

## Collapse prevention: does NOT replicate

| KL(C_ref ‖ uniform) | DSRL | MoP | p (MoP vs DSRL) |
|---|---|---|---|
| 0.120 | 1.086 | 1.093 | 0.947 |
| 0.327 | 0.994 | 1.024 | 0.772 |
| 0.537 | 0.936 | 0.963 | 0.664 |
| 0.770 | 0.576 | 0.613 | 0.616 |

On Avoiding, unregularized DSRL fell to 0.30–0.58 nats against 1.12–1.58 for
either regularizer, every comparison below 10⁻⁴. Here the three arms are
indistinguishable.

The regularizers did not fail — nothing moved:

| | modes | backbone H | max H | DSRL final H | entropy it could lose | DSRL success |
|---|---|---|---|---|---|---|
| Avoiding | 6 | 2.65 | 1.79* | 0.41 | **2.24** | 0.82 |
| Pushing | 4 | 1.06 | 1.39 | 0.89 | **0.17** | 0.31 |

\* Avoiding's backbone entropy is measured over 24 routes before grouping, hence
above ln 6.

Pushing's backbone starts near the floor of its own entropy range, solves the
task 31% of the time, and pays out on 1 route in 4 — so the reward signal is thin
and no arm departs far from the backbone. There is neither diversity to destroy
nor performance to chase.

## What to claim

- **MoP and MI converge on different distributions, and the gap grows with
  reference skew.** Two tasks, with and without the reward shift, 7 of 8 levels
  significant. This is the robust result.
- **Bounded anchoring is stable.** TV's coefficient sits exactly at its ½ bound
  and never collapsed in 85 runs, against raw KL's 17.6 and 8-of-24 on Avoiding.
- **Mode regularization prevents reward-driven collapse *where there is collapse
  to prevent*.** Demonstrated on Avoiding; absent on Pushing for a reason that is
  measurable in advance. The precondition — the backbone needs entropy headroom
  and the task needs performance headroom — is a more useful statement than the
  unqualified claim would have been.
