# Reanalysis of the legacy sweep

What this covers and what it cannot. The runs analysed here were produced by `scripts/run_d3il_headtohead.py`, whose update rule has the defects recorded in `docs/v2/FIXES.md`. Reanalysis repairs how the numbers are read; it leaves the training that produced them untouched. Every contrast below is a policy-level paired difference in closeness to the reference: seeds are averaged inside a frozen policy first, degrees of freedom follow the number of policies that contribute both arms, and the three primary contrasts in a family carry a Holm adjustment. A positive mean means the reference-matching arm sits closer to the reference.

The fine-tuned noise policies were never written to disk, so no endpoint can be recomputed from a checkpoint; what the legacy evaluation failed to record is gone until the runs are repeated.

## Avoiding

480 runs over 20 frozen policies (mop 120, mi 120, dsrl 120, bmd 120). The success floor of 0.15 used in the manuscript removes 10 runs and leaves 20 policies.

### Primary analysis: every run

| contrast | n | df | mean | sd | se | 95% CI | p | policies |
|---|---|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1506 | 0.1853 | 0.0414 | [+0.0639, +0.2374] | 0.0009 | 16/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.1991 | 0.0806 | 0.0180 | [+0.1614, +0.2369] | 0.0000 | 20/20 |
| mop closer than bmd, TV | 20 | 19 | +0.1548 | 0.1113 | 0.0249 | [+0.1027, +0.2070] | 0.0000 | 19/20 |

### Sensitivity: runs above the success floor

| contrast | n | df | mean | sd | se | 95% CI | p | policies |
|---|---|---|---|---|---|---|---|---|
| mop closer than mi, TV | 20 | 19 | +0.1372 | 0.1766 | 0.0395 | [+0.0545, +0.2199] | 0.0013 | 15/20 |
| mop closer than dsrl, TV | 20 | 19 | +0.2009 | 0.0825 | 0.0184 | [+0.1623, +0.2395] | 0.0000 | 20/20 |
| mop closer than bmd, TV | 20 | 19 | +0.1462 | 0.1054 | 0.0236 | [+0.0968, +0.1955] | 0.0000 | 19/20 |

### Other distances on the same cohort, every run

| contrast | n | df | mean | sd | se | 95% CI | p | policies |
|---|---|---|---|---|---|---|---|---|
| mop closer than mi, Hellinger | 20 | 19 | +0.1204 | 0.1641 | 0.0367 | [+0.0436, +0.1972] | 0.0020 | 16/20 |
| mop closer than dsrl, Hellinger | 20 | 19 | +0.1835 | 0.0827 | 0.0185 | [+0.1448, +0.2222] | 0.0000 | 20/20 |
| mop closer than bmd, Hellinger | 20 | 19 | +0.1383 | 0.1060 | 0.0237 | [+0.0887, +0.1879] | 0.0000 | 18/20 |
| mop closer than mi, largest bin error | 20 | 19 | +0.1342 | 0.1815 | 0.0406 | [+0.0492, +0.2191] | 0.0019 | 15/20 |
| mop closer than dsrl, largest bin error | 20 | 19 | +0.1945 | 0.0872 | 0.0195 | [+0.1537, +0.2354] | 0.0000 | 20/20 |
| mop closer than bmd, largest bin error | 20 | 19 | +0.1472 | 0.1090 | 0.0244 | [+0.0962, +0.1982] | 0.0000 | 19/20 |

### The shared reference's own sampling error

Counts rationalise for 480 of 480 runs. Rationalisation succeeds unevenly across arms, so the fixed-reference contrast is recomputed on exactly the runs that enter the resampling and shown beside it.

| contrast | policies | fixed reference | resampled mean | sd | resampled 95% |
|---|---|---|---|---|---|
| vs geometric mode entropy | 20 | +0.1506 | +0.1501 | 0.0023 | [+0.1457, +0.1543] |
| vs no mode term | 20 | +0.1991 | +0.1978 | 0.0019 | [+0.1943, +0.2016] |
| vs learned skills | 20 | +0.1548 | +0.1532 | 0.0018 | [+0.1500, +0.1569] |

### The fixed critical value the manuscript used

The manuscript multiplied every standard error by t(19) = 2.093 regardless of how many policies the contrast actually had.

| contrast | n | df | half-width as computed | half-width at t(19) | difference |
|---|---|---|---|---|---|
| vs geometric mode entropy | 20 | 19 | 0.0827 | 0.0827 | -0.0% |
| vs no mode term | 20 | 19 | 0.0386 | 0.0386 | -0.0% |
| vs learned skills | 20 | 19 | 0.0493 | 0.0493 | -0.0% |

## Pushing

353 runs over 18 frozen policies (mop 87, mi 87, dsrl 89, bmd 90). The success floor of 0.1 used in the manuscript removes 77 runs and leaves 18 policies.

### Primary analysis: every run

| contrast | n | df | mean | sd | se | 95% CI | p | policies |
|---|---|---|---|---|---|---|---|---|
| mop closer than mi, TV | 18 | 17 | +0.0565 | 0.1791 | 0.0422 | [-0.0325, +0.1456] | 0.1982 | 12/18 |
| mop closer than dsrl, TV | 18 | 17 | +0.0517 | 0.1268 | 0.0299 | [-0.0113, +0.1147] | 0.1526 | 10/18 |
| mop closer than bmd, TV | 18 | 17 | -0.0183 | 0.1158 | 0.0273 | [-0.0759, +0.0393] | 0.7437 | 10/18 |

### Sensitivity: runs above the success floor

| contrast | n | df | mean | sd | se | 95% CI | p | policies |
|---|---|---|---|---|---|---|---|---|
| mop closer than mi, TV | 17 | 16 | +0.0913 | 0.1529 | 0.0371 | [+0.0127, +0.1699] | 0.0255 | 13/17 |
| mop closer than dsrl, TV | 17 | 16 | +0.0707 | 0.1038 | 0.0252 | [+0.0173, +0.1241] | 0.0189 | 12/17 |
| mop closer than bmd, TV | 17 | 16 | +0.0324 | 0.0720 | 0.0175 | [-0.0046, +0.0694] | 0.0411 | 13/17 |

### Other distances on the same cohort, every run

| contrast | n | df | mean | sd | se | 95% CI | p | policies |
|---|---|---|---|---|---|---|---|---|
| mop closer than mi, Hellinger | 18 | 17 | +0.0311 | 0.2013 | 0.0475 | [-0.0690, +0.1312] | 0.5209 | 11/18 |
| mop closer than dsrl, Hellinger | 18 | 17 | +0.0337 | 0.1396 | 0.0329 | [-0.0357, +0.1032] | 0.4794 | 11/18 |
| mop closer than bmd, Hellinger | 18 | 17 | -0.0424 | 0.1368 | 0.0322 | [-0.1104, +0.0256] | 0.8970 | 6/18 |
| mop closer than mi, largest bin error | 18 | 17 | +0.0746 | 0.1951 | 0.0460 | [-0.0224, +0.1716] | 0.1232 | 12/18 |
| mop closer than dsrl, largest bin error | 18 | 17 | +0.0518 | 0.1134 | 0.0267 | [-0.0047, +0.1082] | 0.1046 | 10/18 |
| mop closer than bmd, largest bin error | 18 | 17 | -0.0176 | 0.1037 | 0.0244 | [-0.0692, +0.0340] | 0.7593 | 8/18 |

### The shared reference's own sampling error

Counts rationalise for 251 of 353 runs. Rationalisation succeeds unevenly across arms, so the fixed-reference contrast is recomputed on exactly the runs that enter the resampling and shown beside it.

| contrast | policies | fixed reference | resampled mean | sd | resampled 95% |
|---|---|---|---|---|---|
| vs geometric mode entropy | 18 | +0.0548 | +0.0526 | 0.0064 | [+0.0396, +0.0651] |
| vs no mode term | 18 | +0.0492 | +0.0474 | 0.0043 | [+0.0396, +0.0564] |
| vs learned skills | 5 | +0.1066 | +0.1041 | 0.0082 | [+0.0884, +0.1190] |

### The fixed critical value the manuscript used

The manuscript multiplied every standard error by t(19) = 2.093 regardless of how many policies the contrast actually had.

| contrast | n | df | half-width as computed | half-width at t(19) | difference |
|---|---|---|---|---|---|
| vs geometric mode entropy | 17 | 16 | 0.0786 | 0.0776 | -1.3% |
| vs no mode term | 17 | 16 | 0.0534 | 0.0527 | -1.3% |
| vs learned skills | 17 | 16 | 0.0370 | 0.0365 | -1.3% |

## What this changes for the manuscript

On Avoiding the three primary contrasts hold without the success floor and are larger without it than with it, and the Hellinger and largest-bin-error endpoints agree in direction.

On Pushing none of the three primary contrasts excludes zero once every run is counted, and the contrast against the learned-skill arm changes sign. The manuscript reports the floored numbers as its Pushing result; the floor, not the method, carries that comparison. A reference-matching gain on Pushing has to be established by the corrected runs, not asserted from these.

The shared reference contributes a spread of a few thousandths on both tasks, far below the contrasts themselves. The fixed critical value costs at most 1.3% of a half-width. Neither explains the Pushing result.
