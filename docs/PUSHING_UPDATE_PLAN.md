# What changes in the manuscript when the Pushing sweeps land

Every Pushing number in the manuscript comes from the single-frozen-policy runs
(`pushrw_*`, `bmdP_*`), pooled over runs. The multi-policy sweep (`pm_*`)
replaces that design, so each of these has to be recomputed rather than edited.
Written while the sweeps run, so nothing is missed afterwards.

## What already differs, on the two weightings whose sweep is complete

| | single policy, pooled | five policies, clustered |
|---|---|---|
| vs no mode term | +0.88 sd | +0.087 (p 0.043) |
| vs mode entropy | +0.85 sd | +0.011 (p 0.33) |
| vs learned skills | +0.56 sd | −0.019 (we trail) |
| our exclusion rate | 20.8% | 34.2% |
| unfiltered, vs learned skills | tie, 0.06 sd apart | −0.50 sd |

Two things changed at once and have to be separated before any of this is
written up: the design went from one frozen policy per weighting to five, and
the learned-skill arm was given the fitted skill prior, which on Avoiding is a
robustness check rather than the main protocol. `pu_*` supplies the uniform
prior under the new design and restores the comparison to the one Avoiding
reports.

## Lines to recompute

- 34, 116, 404, 754 — "three of the four on Pushing", the abstract, the
  introduction, the results opening and the conclusion.
- 418 — the pooled effect sizes 0.88, 0.56, 0.85.
- 458, 461 — the caption's run counts per cell and the claim about which arm
  leads where.
- 473-476 — Table 2, distance to the reference. The weighting labels are
  measured skew and will move with the new references.
- 506-509 — Table 3, task success. The new data has reference matching lowest
  at both complete weightings, which the current table does not show.
- 538 — exclusion rates, now 34.2% against 6.3%.
- 542-547 — the unfiltered comparison. The current text says the two methods
  converge; the new data has us behind by 0.50 standard deviations.
- 605-607 — Table 5, the three endpoints no arm optimises.
- 666 — the divergence comparison figure for Pushing.

## What the clustered analysis adds

Pushing gains the analysis Avoiding already has: one unit per frozen policy
rather than a pool over runs that share one. That answers the pseudoreplication
objection on both tasks instead of one, which is the reason the sweep was run.

## Still open

Whether the penalty weight is mis-set for Pushing. The rule for deciding that is
fixed in PREREG_pushing_lambda.md and reads only the collapse rate.
