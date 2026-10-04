# Does the result depend on coarsening the route partition?

The confirmatory tests were run on a 6-group partition of Avoiding's 24 routes,
because a 16-episode batch cannot estimate a 24-bin histogram. A reviewer's
objection: the grouping is an unreported analysis degree of freedom, every
reported quantity changes with it, and the 6-group map `g1*3 + g2` **discards the
third obstacle row entirely** — a policy collapsing from four row-3 options to
one registers as perfectly diverse.

So the confirmatory pair was re-run on the full 24-route partition, TV anchoring,
no reference smoothing, 20 seeds per cell, 1000 evaluation episodes.

## Result: it does not depend on the coarsening

Normalised fidelity gap, MoP vs MI, one-sided Welch, Holm across the family:

| skew | TV(C_ref‖u) | MoP | MI | diff | p | verdict |
|---|---|---|---|---|---|---|
| 0.30 | 0.394 | +0.266 | −0.072 | +0.338 | 0.0034 | reject |
| 0.15 | 0.430 | +0.444 | −0.181 | +0.625 | <0.0001 | reject |

Both reject, and the effect is **larger** on the full partition than on the
coarsening that was challenged (+0.625 against +0.240 at 6 groups). MI's
normalised gap is negative at both levels: on the full partition it lands
measurably nearer uniform than the policy it started from, which the 6-group map
had partly hidden by discarding the row it collapses within.

## Two caveats this run also surfaces

**Success rate is the price.** MoP 0.82 against MI 0.84–0.85 and DSRL 0.88–0.96.
Fidelity is bought with task performance, consistently, and the paper has to say
so rather than reporting fidelity alone.

**Attrition is configuration-dependent.** Here it is not significant
(MoP 7.5%, DSRL 2.5%, MI 0%, chi-square p = 0.164), against the strongly
differential 22.5% / 8.3% / 7.9% seen in the earlier KL-anchored runs. The
earlier figure appears to belong to KL anchoring rather than to mode
regularization in general, but the conditions differ in divergence, smoothing and
partition at once, so this is an observation and not yet an explanation.
