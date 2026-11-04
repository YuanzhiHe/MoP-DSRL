# Results, corrected implementation

Generated from the run files and the analysis outputs. No number in this document was entered by hand.

## Run accounting

| stage | planned | completed | missing | exceptions | retried | undefined fidelity | non-finite gradients |
|---|---|---|---|---|---|---|---|
| calibration | 216 | 216 | 0 | 0 | 0 | 11 | 0 |
| main | 800 | 800 | 0 | 0 | 0 | 54 | 0 |
| ablation/divergence | 200 | 200 | 0 | 0 | 0 | 10 | 0 |
| ablation/fitted prior | 50 | 50 | 0 | 0 | 0 | 0 | 0 |
| ablation/weight | 300 | 300 | 0 | 0 | 0 | 45 | 0 |

## Avoiding

Every run of the task, with no filtering. The episode reward is the rewarded-route indicator, so the mean return and the rewarded success rate are the same quantity and appear once.

| arm | TV to reference | success | rewarded success and return | completion | below floor | undefined fidelity |
|---|---|---|---|---|---|---|
| reference matching | 0.1755 | 0.7292 | 0.1509 | 0.7521 | 1 | 0 |
| mode entropy | 0.3686 | 0.6230 | 0.1140 | 0.6409 | 9 | 5 |
| no mode term | 0.3805 | 0.6813 | 0.2310 | 0.7063 | 3 | 0 |
| learned skill | 0.3787 | 0.7083 | 0.2279 | 0.7299 | 0 | 0 |

Contrasts, primary, every run. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1877 | [+0.1160, +0.2593] | 0.0000 | 18/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.2050 | [+0.1576, +0.2524] | 0.0000 | 20/20 |
| mop closer than bmd, TV | 20 | 19 | +0.2032 | [+0.1365, +0.2698] | 0.0000 | 17/20 |

Contrasts, sensitivity, success floor. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1820 | [+0.1142, +0.2497] | 0.0000 | 18/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.1984 | [+0.1520, +0.2448] | 0.0000 | 20/20 |
| mop closer than bmd, TV | 20 | 19 | +0.2067 | [+0.1426, +0.2707] | 0.0000 | 18/20 |

Redrawing the shared reference from its own stored counts, and rescoring both arms against each draw. Reported as closeness, in the same direction as the contrasts above.

| against | mean | 95% interval over draws |
|---|---|---|
| mode entropy | +0.1866 | [+0.1818, +0.1914] |
| no mode term | +0.2039 | [+0.2004, +0.2079] |
| learned skill | +0.2023 | [+0.1984, +0.2056] |

## Pushing

Every run of the task, with no filtering. The episode reward is the rewarded-route indicator, so the mean return and the rewarded success rate are the same quantity and appear once.

| arm | TV to reference | success | rewarded success and return | completion | below floor | undefined fidelity |
|---|---|---|---|---|---|---|
| reference matching | 0.2398 | 0.2176 | 0.0423 | 0.2240 | 29 | 13 |
| mode entropy | 0.3100 | 0.1528 | 0.0329 | 0.1579 | 50 | 20 |
| no mode term | 0.2686 | 0.2205 | 0.0453 | 0.2260 | 25 | 8 |
| learned skill | 0.1992 | 0.2523 | 0.0506 | 0.2594 | 20 | 8 |

Contrasts, primary, every run. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV (interval contains zero) | 20 | 19 | +0.0687 | [-0.0058, +0.1432] | 0.1030 | 12/20 |
| mop closer than dsrl, TV (interval contains zero) | 20 | 19 | +0.0411 | [-0.0122, +0.0944] | 0.1229 | 13/20 |
| mop closer than bmd, TV (interval contains zero) | 20 | 19 | -0.0349 | [-0.1045, +0.0347] | 0.8462 | 9/20 |

Contrasts, sensitivity, success floor. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1013 | [+0.0096, +0.1930] | 0.0321 | 15/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.0578 | [+0.0142, +0.1014] | 0.0180 | 13/20 |
| mop closer than bmd, TV (interval contains zero) | 20 | 19 | +0.0189 | [-0.0310, +0.0689] | 0.2188 | 13/20 |

Redrawing the shared reference from its own stored counts, and rescoring both arms against each draw. Reported as closeness, in the same direction as the contrasts above.

| against | mean | 95% interval over draws |
|---|---|---|
| mode entropy | +0.0669 | [+0.0590, +0.0746] |
| no mode term | +0.0395 | [+0.0327, +0.0471] |
| learned skill | -0.0350 | [-0.0416, -0.0276] |

## Advantage against measured reference skew

One point per frozen policy, the predictor being the total variation from that policy's own reference to the uniform histogram.

| task | comparison | n | slope | se | r | p |
|---|---|---|---|---|---|---|
| avoiding | over mode entropy | 20 | +0.7744 | 0.1361 | +0.802 | 0.0000 |
| avoiding | over no mode term | 20 | -0.2711 | 0.1364 | -0.424 | 0.0624 |
| avoiding | over learned skill | 20 | -0.2075 | 0.2061 | -0.231 | 0.3273 |
| pushing | over mode entropy | 20 | +0.4138 | 0.2229 | +0.401 | 0.0798 |
| pushing | over no mode term | 20 | -0.2238 | 0.1659 | -0.303 | 0.1940 |
| pushing | over learned skill | 20 | -0.1794 | 0.2233 | -0.186 | 0.4324 |

This trend is declared in the pre-registration and is the prediction the identity between reference matching and entropy maximisation under a uniform reference makes.

## Ablations

The tables are in ABLATIONS.md, written by the same run files. Sections present: avoiding/divergence, avoiding/divergence contrasts/fidelity, avoiding/divergence contrasts/geometric success, avoiding/skill prior, avoiding/skill prior contrast, pushing/divergence, pushing/divergence contrasts/fidelity, pushing/divergence contrasts/geometric success, pushing/skill prior, pushing/skill prior contrast, pushing/weight, pushing/weight contrasts/fidelity, pushing/weight contrasts/geometric success.

## What is settled and what is not

Complete, every planned cell present with no exception: calibration, main, ablation/divergence, ablation/fitted prior, ablation/weight.
On pushing the primary interval contains zero for: mop closer than mi, TV; mop closer than dsrl, TV; mop closer than bmd, TV. These comparisons do not distinguish the arms, and no equivalence is claimed, because no margin was set in advance.
