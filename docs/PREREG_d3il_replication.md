# Pre-registration: independent replication of the MoP/MI separation

Written and committed **before** the replication runs. The exploratory sweep it
follows is commit `e95f2f7`.

## Why this document exists

The exploratory sweep tested the separation on `tv_to_ref` and found nothing
(p 0.29–0.84 across five skews, effect sizes flipping sign). The separation
appeared only after switching to the fidelity gap, `tv_to_uniform - tv_to_ref`.
That switch has a principled justification — differencing the two distances
within a run cancels the per-seed variance that swamps either one alone — but it
was made **after** seeing the first statistic return nulls. Choosing a statistic
post hoc and reporting its p-value as confirmatory is exactly how an analysis
manufactures a result. So the claim is re-tested here on fresh seeds, with the
statistic, the skews, and the decision rule fixed in advance.

## Hypothesis

H1. At skews 0.15 and 0.30, the fidelity gap is greater for `mop` than for `mi`.

Directional, one-sided. The direction is not a guess: `mi` maximizes mode
entropy, whose optimum is the uniform histogram, so it should sit nearer uniform
than the backbone (negative gap). `mop` anchors to the backbone's own profile
(positive gap). The exploratory estimates were mop − mi = +0.260 at skew 0.15
and +0.153 at 0.30.

H0. The gap is equal for the two arms.

## Primary endpoint

`tv_to_uniform - tv_to_ref`, over the 6-group ground-truth mode partition,
measured on 1000 evaluation episodes per run.

**Primary test**: Welch one-sided t-test, `mop` vs `mi`, at skew 0.15 and 0.30.
Two tests, Holm-corrected. Confirmation requires **both** to reject at the
corrected level.

## Pre-specified secondary questions

These do not bear on whether H1 is confirmed.

- **S1.** At skew 0.00 the two arms must remain indistinguishable. This is not a
  hypothesis but an implementation check: KL(p ‖ uniform) = ln K − H(p) makes the
  objectives identical there, so a separation at skew 0 would indicate a bug, not
  a finding. Two-sided, reported with its effect size.
- **S2.** Why the curve broke at skew 0.60 (p=0.504 exploratory). Two candidates
  are distinguishable in the data: the failure filter left n=7 there (a power
  problem), or the backbone's narrow support inflates the anchoring coefficient
  even after smoothing (a mechanism problem). `coef_absmax` separates them.
- **S3.** Both arms beat unregularized `dsrl` on `H_modes`. This replicated
  robustly across the exploratory sweep and is expected to hold.

## Design

- Skews **0.00, 0.15, 0.30, 0.60**. 0.15 and 0.30 are confirmatory; 0.00 is the
  implementation check; 0.60 addresses S2.
- Arms `dsrl`, `mop`, `mi`; λ=1.0; `--mode-groups 6`, `--ema 0.15`,
  `--ref-smooth 0.20`; 120 iterations × 16 episodes; 1000 evaluation episodes;
  2000 reference episodes. Identical to the exploratory configuration.
- **Seeds 100–147** (48 per cell). Disjoint from the exploratory block (0–15),
  so this is a replication and not a re-analysis.
- Sample size: the exploratory effects were d=1.45 (skew 0.15) and d=0.89 (0.30).
  Powering for the smaller and halving it against the winner's curse, d=0.45
  needs ~78 per group at 80%; 48 seeds with the observed ~24% failure rate leave
  ~36, giving 80% power at d≈0.67. Underpowered for a halved effect and
  well-powered for the observed one; that limit is stated here rather than
  discovered afterwards.

## Exclusion rule

Runs with SR < 0.15 are dropped as failed optimizations, before any endpoint is
computed. Fixed in advance, applied identically to every arm. Justification:
unregularized `dsrl`, which has no mode objective, produced a 0.052 run in the
guard sweep, so ~a quarter of seeds fail to optimize for reasons unrelated to the
objective under test. The number dropped per cell is reported.

## What would falsify H1

Either confirmatory test failing to reject at the Holm-corrected level. A
non-significant replication will be reported as a failed replication — the
exploratory result would then be the more likely artifact, not the replication.
