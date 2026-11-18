# Results, corrected implementation

Generated from the run files and the analysis outputs. No number in this document was entered by hand.

## Run accounting

| stage | planned | completed | missing | exceptions | retried | undefined fidelity | non-finite gradients |
|---|---|---|---|---|---|---|---|
| calibration | 216 | 216 | 0 | 0 | 0 | 7 | 0 |
| main | 800 | 800 | 0 | 0 | 0 | 47 | 0 |
| ablation/divergence | 200 | 200 | 0 | 0 | 0 | 6 | 0 |
| ablation/fitted prior | 50 | 50 | 0 | 0 | 0 | 2 | 0 |
| ablation/weight | 300 | 300 | 0 | 0 | 0 | 15 | 0 |
| frozen policy | 200 | 200 | 0 | 0 | 0 | 0 | 0 |

## Avoiding

Every run of the task, with no filtering. The episode reward is the rewarded-route indicator, so the mean return and the rewarded success rate are the same quantity and appear once.

| arm | TV to reference | success | rewarded success and return | completion | below floor | undefined fidelity |
|---|---|---|---|---|---|---|
| reference matching | 0.1873 | 0.7461 | 0.1622 | 0.7629 | 1 | 1 |
| mode entropy | 0.3694 | 0.6230 | 0.1140 | 0.6409 | 9 | 5 |
| no mode term | 0.3801 | 0.6813 | 0.2310 | 0.7063 | 3 | 0 |
| learned skill | 0.3789 | 0.7083 | 0.2279 | 0.7299 | 0 | 0 |

Contrasts, primary, every run. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1765 | [+0.1018, +0.2512] | 0.0000 | 18/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.1926 | [+0.1414, +0.2438] | 0.0000 | 20/20 |
| mop closer than bmd, TV | 20 | 19 | +0.1914 | [+0.1322, +0.2505] | 0.0000 | 19/20 |

Contrasts, sensitivity, success floor. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1674 | [+0.0949, +0.2399] | 0.0001 | 17/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.1824 | [+0.1320, +0.2328] | 0.0000 | 20/20 |
| mop closer than bmd, TV | 20 | 19 | +0.1914 | [+0.1322, +0.2505] | 0.0000 | 19/20 |

Redrawing the shared reference from its own stored counts, and rescoring both arms against each draw. Reported as closeness, in the same direction as the contrasts above.

| against | mean | 95% interval over draws |
|---|---|---|
| mode entropy | +0.1757 | [+0.1688, +0.1837] |
| no mode term | +0.1911 | [+0.1859, +0.1961] |
| learned skill | +0.1901 | [+0.1853, +0.1951] |

## Pushing

Every run of the task, with no filtering. The episode reward is the rewarded-route indicator, so the mean return and the rewarded success rate are the same quantity and appear once.

| arm | TV to reference | success | rewarded success and return | completion | below floor | undefined fidelity |
|---|---|---|---|---|---|---|
| reference matching | 0.1983 | 0.2006 | 0.0267 | 0.2059 | 34 | 10 |
| mode entropy | 0.2858 | 0.1384 | 0.0191 | 0.1429 | 52 | 26 |
| no mode term | 0.2457 | 0.2255 | 0.0369 | 0.2320 | 18 | 4 |
| learned skill | 0.1888 | 0.2560 | 0.0382 | 0.2626 | 21 | 1 |

Contrasts, primary, every run. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.0945 | [+0.0083, +0.1806] | 0.0499 | 13/20 |
| mop closer than dsrl, TV (interval contains zero) | 20 | 19 | +0.0494 | [-0.0045, +0.1033] | 0.0703 | 13/20 |
| mop closer than bmd, TV (interval contains zero) | 20 | 19 | -0.0109 | [-0.0614, +0.0396] | 0.6719 | 10/20 |

Contrasts, sensitivity, success floor. The direction was declared in advance and the family is Holm adjusted.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| mop closer than mi, TV (interval contains zero) | 18 | 17 | +0.0300 | [-0.0219, +0.0819] | 0.2388 | 9/18 |
| mop closer than dsrl, TV | 20 | 19 | +0.0889 | [+0.0452, +0.1325] | 0.0006 | 17/20 |
| mop closer than bmd, TV (interval contains zero) | 20 | 19 | +0.0129 | [-0.0283, +0.0541] | 0.2600 | 11/20 |

Redrawing the shared reference from its own stored counts, and rescoring both arms against each draw. Reported as closeness, in the same direction as the contrasts above.

| against | mean | 95% interval over draws |
|---|---|---|
| mode entropy | +0.0934 | [+0.0806, +0.1039] |
| no mode term | +0.0471 | [+0.0396, +0.0545] |
| learned skill | -0.0119 | [-0.0173, -0.0065] |

## Advantage against measured reference skew

One point per frozen policy, the predictor being the total variation from that policy's own reference to the uniform histogram.

| task | comparison | n | slope | se | r | p |
|---|---|---|---|---|---|---|
| avoiding | over mode entropy | 20 | +0.8114 | 0.1391 | +0.809 | 0.0000 |
| avoiding | over no mode term | 20 | -0.2412 | 0.1518 | -0.351 | 0.1296 |
| avoiding | over learned skill | 20 | -0.1952 | 0.1815 | -0.246 | 0.2965 |
| pushing | over mode entropy | 20 | +0.5157 | 0.2928 | +0.383 | 0.0952 |
| pushing | over no mode term | 20 | -0.3115 | 0.1842 | -0.370 | 0.1082 |
| pushing | over learned skill | 20 | -0.2638 | 0.1751 | -0.335 | 0.1493 |

This trend is declared in the pre-registration and is the prediction the identity between reference matching and entropy maximisation under a uniform reference makes.

## Ablations

The tables are in ABLATIONS.md, written by the same run files. Sections present: avoiding/divergence, avoiding/divergence contrasts/fidelity, avoiding/divergence contrasts/geometric success, avoiding/skill prior, avoiding/skill prior contrast, pushing/divergence, pushing/divergence contrasts/fidelity, pushing/divergence contrasts/geometric success, pushing/skill prior, pushing/skill prior contrast, pushing/weight, pushing/weight contrasts/fidelity, pushing/weight contrasts/geometric success.

## What is settled and what is not

Complete, every planned cell present with no exception: calibration, main, ablation/divergence, ablation/fitted prior, ablation/weight, frozen policy.
On pushing the primary interval contains zero for: mop closer than dsrl, TV; mop closer than bmd, TV. These comparisons do not distinguish the arms, and no equivalence is claimed, because no margin was set in advance.
