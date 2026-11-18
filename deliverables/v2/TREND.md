# Advantage against reference skew

## Avoiding

| launch weighting | measured reference skew |
|---|---|
| 0.00 | 0.240 |
| 0.15 | 0.317 |
| 0.30 | 0.455 |
| 0.60 | 0.555 |

| comparison | n | slope | se | r | p | per-weighting gap |
|---|---|---|---|---|---|---|
| mop over mi | 20 | +0.8114 | 0.1391 | +0.809 | 0.0000 | 0.24:+0.095 0.32:+0.095 0.46:+0.206 0.56:+0.311 |
| mop over dsrl | 20 | -0.2412 | 0.1518 | -0.351 | 0.1296 | 0.24:+0.275 0.32:+0.222 0.46:+0.163 0.56:+0.110 |
| mop over bmd | 20 | -0.1952 | 0.1815 | -0.246 | 0.2965 | 0.24:+0.258 0.32:+0.206 0.46:+0.190 0.56:+0.112 |

The per-weighting column reads skew:gap, ordered by measured skew rather than by launch weighting.

## Pushing

| launch weighting | measured reference skew |
|---|---|
| 0.00 | 0.242 |
| 0.10 | 0.244 |
| 0.25 | 0.417 |
| 0.50 | 0.537 |

| comparison | n | slope | se | r | p | per-weighting gap |
|---|---|---|---|---|---|---|
| mop over mi | 20 | +0.5157 | 0.2928 | +0.383 | 0.0952 | 0.24:+0.045 0.24:-0.003 0.42:+0.156 0.54:+0.180 |
| mop over dsrl | 20 | -0.3115 | 0.1842 | -0.370 | 0.1082 | 0.24:+0.120 0.24:+0.056 0.42:+0.028 0.54:-0.007 |
| mop over bmd | 20 | -0.2638 | 0.1751 | -0.335 | 0.1493 | 0.24:+0.077 0.24:-0.012 0.42:-0.080 0.54:-0.029 |

The per-weighting column reads skew:gap, ordered by measured skew rather than by launch weighting.
