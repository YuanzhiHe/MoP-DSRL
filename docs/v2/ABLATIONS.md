# Ablations, corrected implementation

### Avoiding, divergence at weighting 0.60

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| js | 25 | 0 | 0 | 0.2600 | 0.696 | 0.220 | 0.705 | 0.120 | 1.46 | 1.46 | 2.54 | 0 | 0 | 0.0000 | 0.000 |
| kl eps=1e-06 | 25 | 0 | 0 | 0.1109 | 0.817 | 0.224 | 0.829 | 0.000 | 4.48 | 4.48 | 3.45 | 0 | 0 | 0.0000 | 0.000 |
| kl eps=1e-08 smoothing=0.05 | 25 | 0 | 0 | 0.1321 | 0.792 | 0.166 | 0.805 | 0.000 | 5.46 | 5.46 | 5.96 | 0 | 0 | 0.0000 | 0.000 |
| l2 | 25 | 0 | 0 | 0.2476 | 0.737 | 0.236 | 0.752 | 0.000 | 0.62 | 0.62 | 3.84 | 0 | 0 | 0.0000 | 0.000 |
| tv | 25 | 0 | 0 | 0.1274 | 0.760 | 0.158 | 0.773 | 0.000 | 1.18 | 1.18 | 3.8 | 0 | 0 | 0.0000 | 0.000 |

At the common weight the coefficient that reaches the update differs by a factor of 9 across the variants, from 0.62 for l2 to 5.46 for kl eps=1e-08 smoothing=0.05. The weight was calibrated under total variation, so this comparison holds the weight fixed and lets the penalty strength vary with the derivative's own scale. A difference between two variants is therefore a difference between a divergence at its own scale and another at its own scale.

Total variation against each alternative, on fidelity, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js | 5 | 4 | +0.1325 | [+0.1223, +0.1428] | 0.0000 | 5/5 |
| tv over kl eps=1e-06 (interval contains zero) | 5 | 4 | -0.0165 | [-0.0970, +0.0639] | 0.7276 | 3/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | +0.0047 | [-0.0300, +0.0394] | 0.7276 | 4/5 |
| tv over l2 | 5 | 4 | +0.1202 | [+0.0444, +0.1960] | 0.0175 | 5/5 |

Total variation against each alternative, on geometric success, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | +0.0646 | [-0.0286, +0.1577] | 0.2534 | 4/5 |
| tv over kl eps=1e-06 (interval contains zero) | 5 | 4 | -0.0566 | [-0.1172, +0.0041] | 1.0000 | 0/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | -0.0313 | [-0.0712, +0.0086] | 1.0000 | 1/5 |
| tv over l2 (interval contains zero) | 5 | 4 | +0.0234 | [-0.0512, +0.0979] | 0.6499 | 3/5 |

### Avoiding, learned-skill prior at weighting 0.60

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior fitted | 25 | 0 | 0 | 0.2336 | 0.757 | 0.190 | 0.773 | 0.000 | 0 | 0.591 | 3.65 | 0 | 0 | 0.0000 | 0.000 |
| prior uniform | 25 | 0 | 0 | 0.2473 | 0.766 | 0.212 | 0.782 | 0.000 | 0 | 0.591 | 3.65 | 0 | 0 | 0.0000 | 0.000 |

Fitted against uniform, paired over frozen policies: n = 5, df = 4, mean +0.0138, 95% interval [-0.0126, +0.0401], p = 0.1101.

### Pushing, divergence at weighting 0.50

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| js | 25 | 0 | 4 | 0.0960 | 0.228 | 0.007 | 0.233 | 0.360 | 2.92 | 2.92 | 1.09 | 0 | 0 | 0.0720 | 0.000 |
| kl eps=1e-06 | 25 | 0 | 2 | 0.0755 | 0.276 | 0.010 | 0.282 | 0.120 | 15.7 | 15.7 | 3.23 | 0 | 0 | 0.0586 | 0.000 |
| kl eps=1e-08 smoothing=0.05 | 25 | 0 | 4 | 0.1000 | 0.200 | 0.004 | 0.206 | 0.360 | 9.04 | 9.04 | 1.61 | 0 | 0 | 0.0485 | 0.000 |
| l2 | 25 | 0 | 0 | 0.0877 | 0.281 | 0.007 | 0.286 | 0.160 | 0.867 | 0.867 | 1.66 | 0 | 0 | 0.0382 | 0.000 |
| tv | 25 | 0 | 4 | 0.1337 | 0.255 | 0.010 | 0.260 | 0.240 | 3.19 | 3.19 | 1.42 | 0 | 0 | 0.0535 | 0.000 |

At the common weight the coefficient that reaches the update differs by a factor of 18 across the variants, from 0.867 for l2 to 15.7 for kl eps=1e-06. The weight was calibrated under total variation, so this comparison holds the weight fixed and lets the penalty strength vary with the derivative's own scale. A difference between two variants is therefore a difference between a divergence at its own scale and another at its own scale.

Total variation against each alternative, on fidelity, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | -0.0364 | [-0.1733, +0.1004] | 1.0000 | 2/5 |
| tv over kl eps=1e-06 (interval contains zero) | 5 | 4 | -0.0572 | [-0.1814, +0.0669] | 1.0000 | 1/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | -0.0314 | [-0.1796, +0.1168] | 1.0000 | 2/5 |
| tv over l2 (interval contains zero) | 5 | 4 | -0.0463 | [-0.1826, +0.0899] | 1.0000 | 2/5 |

Total variation against each alternative, on geometric success, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | +0.0271 | [-0.1308, +0.1850] | 0.9884 | 4/5 |
| tv over kl eps=1e-06 (interval contains zero) | 5 | 4 | -0.0205 | [-0.1003, +0.0592] | 1.0000 | 2/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | +0.0555 | [-0.0901, +0.2011] | 0.6997 | 4/5 |
| tv over l2 (interval contains zero) | 5 | 4 | -0.0253 | [-0.0959, +0.0453] | 1.0000 | 2/5 |

### Pushing, learned-skill prior at weighting 0.50

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior fitted | 25 | 0 | 0 | 0.1556 | 0.309 | 0.012 | 0.314 | 0.160 | 0 | 4.83 | 19.1 | 0 | 0 | 0.0411 | 0.000 |
| prior uniform | 25 | 0 | 2 | 0.1101 | 0.292 | 0.011 | 0.298 | 0.160 | 0 | 3.86 | 33.6 | 0 | 0 | 0.0505 | 0.000 |

Fitted against uniform, paired over frozen policies: n = 5, df = 4, mean -0.0446, 95% interval [-0.1085, +0.0192], p = 0.9378.
