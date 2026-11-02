# What the corrected implementation changes

Branch `corrected_v2`. The legacy driver is untouched and its outputs are
archived read-only; the corrected code lives in `scripts/v2/` and writes to
`runs/_d3il_v2/`.

## A. Conditional distribution and training gradient

The endpoint is `C[k] = P(mode = k | valid label)`. Writing `V` for the
indicator that a trajectory carries a label, `s = P(V = 1)` and
`g_k = dD/dC[k]`, the score-function coefficient is

    (V / s) * ( g_mode - sum_k C[k] g_k )

implemented in `estimators.conditional_coefficient`. An unlabelled trajectory
carries exactly zero; a labelled one carries its own partial derivative
measured against the C-weighted mean of all of them.

`check_mathematics.py` enumerates a six-outcome trajectory space, computes the
exact gradient of `D(C(phi))` by central differences, and compares. The
corrected coefficient reproduces it to 1e-8. The legacy batch-centred
coefficient misses it by up to 0.42 on KL, 0.18 on TV, 0.11 on L2 and 0.04 on
JS.

`C` and `s` are carried as EMA estimates that are **read before** the current
batch updates them, so a batch never conditions its own coefficient on itself.
Both are plug-in estimates, so the update is an approximation of the population
gradient and is described as one. The coefficient is detached before it
multiplies the log-probability.

A run whose evaluation yields no labelled episode has `fidelity_defined: false`
and `C_phi: null`. It is never recorded as zero.

Settings fixed in advance and written into every row: floor on `1/s` at 0.05,
coefficient clipping off, KL guard 1e-8.

The unconditional alternative is available as `--objective joint`, which adds a
failure bin, estimates its own reference over `K+1` bins and refuses a
reference measured for the conditional objective. It is a separately named
target, never a silent substitute.

## B. One objective, one scale

The legacy update divided the reward advantage and the mode coefficient by
their own batch standard deviations, making the effective weight
`lam * sd(adv) / sd(coef)`, which moves from batch to batch.

`--standardise fixed`, the default, uses a leave-one-out reward baseline and
rescales neither term, so the update is the gradient of `E[r] - lam*D` up to
the plug-in estimates. `--standardise legacy` keeps the old rule as a
diagnostic contrast and is called a surrogate wherever it is reported.

Weights do not transfer between the two rules. The corrected runs need their
own calibration; the weight the legacy sweep selected is not carried over.

## C. JS derivative

`d/dp_k JS(p,q) = (1/2) log(p_k / m_k)` with `m = (p+q)/2`. The legacy
expression carried an extra `-1/2`. Because the constant was applied only to
labelled trajectories, it did not cancel as a baseline, and the batch centring
that followed did not remove it either.

The divergence is bounded by `log 2`. The derivative is bounded only in the
`q_k -> 0` direction, by `(1/2) log 2`; as `p_k -> 0` with `q_k > 0` it
diverges. Both statements are in the docstring.

## D. Rewarded success

Membership is tested on the **raw** route id, the same label the training
reward uses, before any coarsening, and over **all** evaluation episodes.

Reported separately, each over its own denominator:

| field | denominator |
|---|---|
| `success_rate` | all evaluation episodes |
| `completion_rate` | all evaluation episodes |
| `rewarded_success_rate`, `episode_return` | all evaluation episodes |
| `C_phi`, `mode_counts` | labelled episodes, undefined when there are none |
| `mode_counts_raw` | labelled episodes, at full 24-route resolution |

## E. Reference and cache

One reference per frozen policy, measured once and handed to every arm in the
group; each row records the hash of the reference it was scored against.
Nothing picks a reference by majority after filtering.

Stored: per-bin counts at both resolutions, episodes attempted, episodes
labelled, chunks lost, chunks total, and a `complete` flag. The cache key is a
hash of the checkpoint's SHA-256 together with task, K, M, reference budget,
chunk size, seed base, objective and estimator version. An incomplete estimate
is marked and is remeasured rather than served, unless
`--allow-incomplete-reference` says otherwise.

## F. Results and resume

The cell key is `(task, weighting, backbone seed, arm, RL seed,
config_sha256, checkpoint_sha256, reference_sha256)` where `config_sha256`
covers every setting that changes what a cell means, including the estimator
version. A legacy cell can never satisfy a corrected one, and neither can a
cell run under a different weight, divergence, budget or reference.

Output is JSONL appended per cell. A cell that raises is written with its
status, exception and traceback and is never deleted; the retry is appended
beside it with the next attempt number; a cell at the attempt limit is reported
as missing.

## G. Bounded observations

Two of the ten Pushing observation channels are unwrapped orientation angles.
Their demonstration standard deviations are 313 and 693 radians, and along a
rollout they accumulate without bound: over 6,695 steps driven by the frozen
policy's own latent prior, channel 6 reached 6.1e11 normalised units, and more
than one step in a hundred exceeded 1e11. Every network that reads the
observation saturates there. The learned-skill arm feels it twice, because the
final observation is also its discriminator's input, so its intrinsic reward
reached 5.1e10 and two of the eight Pushing smoke cells ended in NaN.

Wrapping the angle is not available: the frozen decoder was trained on the
unwrapped value and its normaliser was fitted to it. `norm_obs` therefore
bounds the normalised observation at `--obs-clip`, 50 by default, in the
training rollouts, the evaluation rollouts and the reference measurement
alike, identically for every arm and both tasks. The largest value any other
channel reached in the same measurement was 11.8 on Pushing and 6.0 on
Avoiding, so the bound is four times outside a behaving rollout and ten orders
of magnitude inside the runaway. Every row records `obs_absmax`,
`obs_clipped_steps` and `obs_clipped_frac`, so a run that spent time against
the bound is visible rather than silently repaired.

A batch whose gradient norm is not finite is dropped and counted in
`n_nonfinite_grad` instead of being stepped on, in both arms, because one such
step leaves every weight NaN for the remainder of the run.

`obs_clip` is a semantic setting and enters both the configuration hash and the
reference cache key, and `ESTIMATOR_VERSION` moves to `v2.1`. Bounded and
unbounded results cannot be pooled or resumed into one another, and every
cached reference is remeasured.

## Verification

`check_mathematics.py` - derivatives against central differences, and the
conditional coefficient against the exact gradient. PASS.

`test_driver.py` - runs the real `run_arm` against the real environment:
decoder weights unchanged and carrying no gradient while the noise policy
updates; coefficients differing within a batch with unlabelled trajectories at
exactly zero; all-unlabelled batches, zero-probability reference bins,
all-zero-reward batches and the `1/s` floor all finite; every diagnostic
present; undefined fidelity scoring as undefined; the two standardisation rules
producing different runs; cell keys refusing to cross configurations. PASS.
