# Pre-registration: choosing the penalty weight on Pushing

Written and committed before any result from this sweep exists. Its purpose is
to fix the selection rule in advance, so the weight cannot be chosen by which
value happens to win the comparison.

## What prompted it

The penalty weight was set to 1.0 on Avoiding and carried over to Pushing
without being examined there. On the two Pushing weightings whose sweep is
complete, reference matching fails to optimise in 19 of 50 runs against 3 of 50
for the learned-skill baseline. Those failures are collapses: mean success rate
0.030, 1.74 of four mode groups visited, histogram entropy 0.463 nats against a
reference entropy of 1.229. Among runs that do optimise, the two methods are
level, 0.176 against 0.177 in total variation from the reference.

Pushing partitions behaviour into four groups and rewards one, against six
groups on Avoiding. The reward and the penalty therefore pull against each other
harder here, which is a reason to expect the weight to transfer badly, and a
reason the paper should state how it was chosen rather than leave it carried
over.

## The sweep

Weights 0.25 and 0.5, against the 1.0 already run. Weightings 0.00 and 0.10,
being the two whose sweep is complete; both, not the one where the method
currently trails. Arms: reference matching and mode entropy, the two that carry
this weight. Five frozen policies and five seeds per cell, as elsewhere.

## The rule, fixed now

The weight is chosen as **the largest of the three whose collapse rate over the
fifty runs of an arm stays at or below 0.15**, collapse being the
pre-registered success floor of 0.10 on Pushing. If no weight meets it, the one
with the lowest collapse rate is chosen. The rule is evaluated on the reference
matching arm and the same weight is then applied to mode entropy, so the two
are not tuned against each other.

Distance to the reference plays no part in the choice. Whatever the rule
selects is what the paper reports, including if it leaves the comparison
unchanged or worse.

## What gets reported either way

The collapse rates at all three weights, for both arms, and the comparison at
the selected weight. The result at 1.0 stays in the paper as the weight carried
over from Avoiding. If the selected weight does not improve the comparison, the
paper says that reference matching is less stable than the learned-skill
baseline on this task and reports the margin among runs that optimise.
