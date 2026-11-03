# Advantage against reference skew

## Avoiding

| launch weighting | measured reference skew |
|---|---|
| 0.00 | 0.238 |
| 0.15 | 0.314 |
| 0.30 | 0.449 |
| 0.60 | 0.551 |

| comparison | n | slope | se | r | p | per-weighting gap |
|---|---|---|---|---|---|---|
| mop over mi | 20 | +0.7744 | 0.1361 | +0.802 | 0.0000 | 0.24:+0.071 0.31:+0.117 0.45:+0.248 0.55:+0.315 |
| mop over dsrl | 20 | -0.2711 | 0.1364 | -0.424 | 0.0624 | 0.24:+0.253 0.31:+0.245 0.45:+0.207 0.55:+0.116 |
| mop over bmd | 20 | -0.2075 | 0.2061 | -0.231 | 0.3273 | 0.24:+0.231 0.31:+0.226 0.45:+0.236 0.55:+0.120 |

The per-weighting column reads skew:gap, ordered by measured skew rather than by launch weighting.

## Pushing

| launch weighting | measured reference skew |
|---|---|
| 0.00 | 0.227 |
| 0.10 | 0.244 |
| 0.25 | 0.385 |
| 0.50 | 0.572 |

| comparison | n | slope | se | r | p | per-weighting gap |
|---|---|---|---|---|---|---|
| mop over mi | 20 | +0.4138 | 0.2229 | +0.401 | 0.0798 | 0.23:-0.014 0.24:-0.017 0.39:+0.137 0.57:+0.169 |
| mop over dsrl | 20 | -0.2238 | 0.1659 | -0.303 | 0.1940 | 0.23:+0.081 0.24:-0.027 0.39:+0.131 0.57:-0.020 |
| mop over bmd | 20 | -0.1794 | 0.2233 | -0.186 | 0.4324 | 0.23:-0.009 0.24:-0.066 0.39:-0.041 0.57:-0.023 |

The per-weighting column reads skew:gap, ordered by measured skew rather than by launch weighting.
