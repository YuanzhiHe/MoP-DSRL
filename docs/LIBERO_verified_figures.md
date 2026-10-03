# LIBERO: what the repository actually records

Earlier notes and drafts carried "C_ref entropy 0.000 across 45 runs" and a total
of "2,542 GPU-hours". Neither survives checking against the repository, and both
were close to entering a shareable document. Verified figures, read from
`runs/*/*/metrics.csv` on 2026-09-21:

| quantity | claimed | verified |
|---|---|---|
| instrumented runs with an `r_mode` column | 45 | **11** |
| logged `r_mode` rows | — | **160,531** |
| runs with `r_mode` identically zero | "all" | **10 of 11**, all `libero_goal` (~131k rows) |
| exception | not mentioned | `tier1_main/libero_spatial_pi05_seed1`: non-zero in **91.8%** of 11,720 rows, max 1.755 |
| GPU-hours | 2,542 | **no record on disk** — do not cite |

There are 40 run directories and 13 `metrics.csv` files in total; some runs
predate the mode instrumentation and some checkpoints were pruned during a
scratch cleanup, which is the likeliest source of the inflated count.

## How to state the finding

Not "the regularizer is always inert". The correct claim is narrower and
stronger:

> On `libero_goal`, the mode term was identically zero in every instrumented run.
> The one `libero_spatial` run that was instrumented carried a non-zero term in
> 91.8% of its updates.

The exception is evidence *for* the argument, not against it: a single-solution
task suite gives a mode objective nothing to act on, while a suite whose tasks
admit real behavioural variation does not. Reporting only the null would have
made the weaker and less defensible claim.
