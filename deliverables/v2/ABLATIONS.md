# Ablations, corrected implementation

### Avoiding, divergence at weighting 0.60

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| js | 25 | 0 | 0 | 0.1927 | 0.738 | 0.227 | 0.749 | 0.000 | 1.46 | 1.46 | 2.87 | 0 | 0 | 0.0000 | 0.000 |
| kl eps=1e-08 | 25 | 0 | 0 | 0.1392 | 0.775 | 0.183 | 0.789 | 0.000 | 6.18 | 6.18 | 4.29 | 0 | 0 | 0.0000 | 0.000 |
| kl eps=1e-08 smoothing=0.05 | 25 | 0 | 0 | 0.1553 | 0.724 | 0.113 | 0.743 | 0.040 | 4.85 | 4.85 | 4.74 | 0 | 0 | 0.0000 | 0.000 |
| l2 | 25 | 0 | 0 | 0.1760 | 0.754 | 0.214 | 0.775 | 0.000 | 0.546 | 0.546 | 4.07 | 0 | 0 | 0.0000 | 0.000 |
| tv | 25 | 0 | 0 | 0.1327 | 0.802 | 0.190 | 0.814 | 0.000 | 1.17 | 1.17 | 4.78 | 0 | 0 | 0.0000 | 0.000 |

At the common weight the coefficient that reaches the update differs by a factor of 11 across the variants, from 0.546 for l2 to 6.18 for kl eps=1e-08. The weight was calibrated under total variation, so this comparison holds the weight fixed and lets the penalty strength vary with the derivative's own scale. A difference between two variants is therefore a difference between a divergence at its own scale and another at its own scale.

Total variation against each alternative, on fidelity, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | +0.0601 | [-0.0333, +0.1534] | 0.2973 | 4/5 |
| tv over kl eps=1e-08 (interval contains zero) | 5 | 4 | +0.0066 | [-0.0719, +0.0850] | 0.6214 | 3/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | +0.0227 | [-0.0951, +0.1405] | 0.6214 | 3/5 |
| tv over l2 (interval contains zero) | 5 | 4 | +0.0433 | [-0.0697, +0.1564] | 0.5208 | 3/5 |

Total variation against each alternative, on geometric success, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | +0.0634 | [-0.0735, +0.2003] | 0.2682 | 3/5 |
| tv over kl eps=1e-08 (interval contains zero) | 5 | 4 | +0.0270 | [-0.0140, +0.0681] | 0.2676 | 4/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | +0.0774 | [-0.0371, +0.1918] | 0.2676 | 5/5 |
| tv over l2 (interval contains zero) | 5 | 4 | +0.0477 | [-0.0637, +0.1590] | 0.2682 | 3/5 |

### Avoiding, learned-skill prior at weighting 0.60

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior fitted | 25 | 0 | 0 | 0.2274 | 0.761 | 0.202 | 0.778 | 0.000 | 0 | 0.591 | 3.65 | 0 | 0 | 0.0000 | 0.000 |
| prior uniform | 25 | 0 | 0 | 0.2442 | 0.766 | 0.212 | 0.782 | 0.000 | 0 | 0.591 | 3.65 | 0 | 0 | 0.0000 | 0.000 |

Fitted against uniform, paired over frozen policies: n = 5, df = 4, mean +0.0168, 95% interval [-0.0086, +0.0423], p = 0.0701.

### Pushing, divergence at weighting 0.50

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| js | 25 | 0 | 3 | 0.1034 | 0.246 | 0.002 | 0.252 | 0.240 | 2.5 | 2.5 | 0.553 | 0 | 0 | 0.0145 | 0.000 |
| kl eps=1e-08 | 25 | 0 | 1 | 0.0756 | 0.311 | 0.005 | 0.316 | 0.080 | 15 | 15 | 4.08 | 0 | 0 | 0.0154 | 0.000 |
| kl eps=1e-08 smoothing=0.05 | 25 | 0 | 2 | 0.1154 | 0.266 | 0.003 | 0.271 | 0.200 | 8.18 | 8.18 | 4.11 | 0 | 0 | 0.0248 | 0.000 |
| l2 | 25 | 0 | 0 | 0.1083 | 0.235 | 0.004 | 0.242 | 0.120 | 0.985 | 0.985 | 0.986 | 0 | 0 | 0.0166 | 0.000 |
| tv | 25 | 0 | 2 | 0.1191 | 0.238 | 0.004 | 0.243 | 0.320 | 4.76 | 4.76 | 5.56 | 0 | 0 | 0.0158 | 0.000 |

At the common weight the coefficient that reaches the update differs by a factor of 15 across the variants, from 0.985 for l2 to 15 for kl eps=1e-08. The weight was calibrated under total variation, so this comparison holds the weight fixed and lets the penalty strength vary with the derivative's own scale. A difference between two variants is therefore a difference between a divergence at its own scale and another at its own scale.

Total variation against each alternative, on fidelity, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | -0.0122 | [-0.1204, +0.0959] | 1.0000 | 3/5 |
| tv over kl eps=1e-08 (interval contains zero) | 5 | 4 | -0.0402 | [-0.1260, +0.0457] | 1.0000 | 1/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | -0.0010 | [-0.1334, +0.1314] | 1.0000 | 3/5 |
| tv over l2 (interval contains zero) | 5 | 4 | -0.0081 | [-0.1244, +0.1081] | 1.0000 | 2/5 |

Total variation against each alternative, on geometric success, paired over the five frozen policies.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| tv over js (interval contains zero) | 5 | 4 | -0.0085 | [-0.0971, +0.0801] | 1.0000 | 2/5 |
| tv over kl eps=1e-08 (interval contains zero) | 5 | 4 | -0.0732 | [-0.1728, +0.0264] | 1.0000 | 1/5 |
| tv over kl eps=1e-08 smoothing=0.05 (interval contains zero) | 5 | 4 | -0.0283 | [-0.1497, +0.0931] | 1.0000 | 3/5 |
| tv over l2 (interval contains zero) | 5 | 4 | +0.0027 | [-0.1262, +0.1315] | 1.0000 | 2/5 |

### Pushing, learned-skill prior at weighting 0.50

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior fitted | 25 | 0 | 2 | 0.1166 | 0.262 | 0.003 | 0.268 | 0.240 | 0 | 3.7 | 16.4 | 0 | 0 | 0.0143 | 0.000 |
| prior uniform | 25 | 0 | 1 | 0.0883 | 0.289 | 0.005 | 0.294 | 0.200 | 0 | 3.75 | 15.3 | 0 | 0 | 0.0177 | 0.000 |

Fitted against uniform, paired over frozen policies: n = 5, df = 4, mean -0.0361, 95% interval [-0.1213, +0.0490], p = 0.8480.

### Pushing, weight grid on the two lowest weightings

| variant | runs | exceptions | undefined fidelity | TV to reference | success | rewarded success and return | completion | below floor | coefficient, raw | coefficient, used | gradient norm | non-finite | non-finite gradients | steps at the obs bound | inert batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mi lam=0.25 | 50 | 0 | 2 | 0.2309 | 0.207 | 0.050 | 0.213 | 0.220 | 15.6 | 15.6 | 5.26 | 0 | 0 | 0.0160 | 0.000 |
| mi lam=0.50 | 100 | 0 | 14 | 0.3033 | 0.157 | 0.030 | 0.163 | 0.460 | 15.5 | 15.5 | 7.04 | 0 | 0 | 0.0213 | 0.000 |
| mi lam=1.00 | 50 | 0 | 4 | 0.2700 | 0.177 | 0.026 | 0.182 | 0.280 | 14.4 | 14.4 | 14.5 | 0 | 0 | 0.0129 | 0.000 |
| mop lam=0.25 | 50 | 0 | 0 | 0.2323 | 0.222 | 0.055 | 0.228 | 0.140 | 4.78 | 4.78 | 4.44 | 0 | 0 | 0.0087 | 0.000 |
| mop lam=0.50 | 100 | 0 | 8 | 0.2175 | 0.217 | 0.052 | 0.224 | 0.220 | 4.46 | 4.46 | 6.34 | 0 | 0 | 0.0148 | 0.000 |
| mop lam=1.00 | 50 | 0 | 1 | 0.1967 | 0.213 | 0.037 | 0.220 | 0.280 | 4.75 | 4.75 | 11.3 | 0 | 0 | 0.0072 | 0.000 |

Reference matching against mode entropy at each weight, on fidelity, paired over the frozen policies of the two lowest demonstration weightings.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| weight 0.25 (interval contains zero) | 10 | 9 | -0.0011 | [-0.0661, +0.0638] | 0.5155 | 6/10 |
| weight 0.50 | 10 | 9 | +0.0902 | [+0.0159, +0.1644] | 0.0339 | 8/10 |
| weight 1.00 (interval contains zero) | 10 | 9 | +0.0730 | [-0.0050, +0.1511] | 0.0633 | 7/10 |

Reference matching against mode entropy at each weight, on geometric success, paired over the frozen policies of the two lowest demonstration weightings.

| contrast | n | df | mean | 95% interval | p | policies favouring |
|---|---|---|---|---|---|---|
| weight 0.25 (interval contains zero) | 10 | 9 | +0.0149 | [-0.0135, +0.0433] | 0.1747 | 5/10 |
| weight 0.50 | 10 | 9 | +0.0598 | [+0.0242, +0.0954] | 0.0063 | 9/10 |
| weight 1.00 (interval contains zero) | 10 | 9 | +0.0366 | [-0.0196, +0.0928] | 0.1747 | 6/10 |

The weight the main matrix runs is 0.50, chosen before any main cell existed by a rule that reads only how often a weight drives geometric success below the task floor. That rule never reads fidelity, and this grid shows where the weight it chose sits relative to the separation between the arms. The main analysis stays at the chosen weight.
