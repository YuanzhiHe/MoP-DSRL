# Pre-registered plan, corrected D3IL experiments

Fixed on 2026-11-02, before any corrected run existed. This is a new plan for a
new estimator, not a restatement of what the manuscript reports. Legacy results
do not transfer: the update rule, the penalty's scale and three reported
endpoints all changed, so no weight, threshold or effect size carries over.

## 1. Endpoints

**Primary.** Conditional behaviour fidelity, `D_TV(C_phi, C_ref)`, where
`C_phi[k] = P(mode = k | the episode carries a valid label)` measured over the
evaluation episodes and `C_ref` the frozen policy's own histogram measured
under the same partition and budget.

A run whose evaluation produces no labelled episode has **undefined** fidelity.
It is reported as undefined and counted, never filled with zero and never
dropped silently.

**Reported alongside, for every run, with no filtering.**

- geometric task success, over all evaluation episodes;
- completion rate, the fraction of episodes carrying a valid label;
- rewarded success and mean episode return, membership tested on the **raw**
  route id, over all evaluation episodes;
- optimisation failure, exception, and missing-cell counts;
- Hellinger distance and largest single-bin error, on the same cohort.

## 2. Unit of analysis

The frozen policy. Within one frozen policy an arm's RL seeds are averaged
first; contrasts are differences between arm means within that policy; the
sample size is the number of frozen policies contributing both arms.

Degrees of freedom are `n - 1` from that count, computed per contrast. No fixed
critical value is used.

## 3. Design

Two tasks. Four demonstration weightings each, the launch parameters:
Avoiding 0.00 / 0.15 / 0.30 / 0.60, Pushing 0.00 / 0.10 / 0.25 / 0.50.
Five frozen policies per weighting, five RL seeds per policy per arm, the same
five seeds for every arm within a policy. The main seed blocks are 700-704 on
Avoiding and 900-904 on Pushing. Smoke and calibration use disjoint blocks,
600-601 and 800-801 for smoke and 500-504 and 850-854 for calibration, so a
debugging or tuning run can never be counted into the main analysis.

Arms:

1. `dsrl` - noise-space policy gradient, no mode term;
2. `mi` - geometric mode entropy;
3. `mop` - reference matching, corrected, total variation;
4. `bmd` - learned-skill baseline with a variational mutual-information bound,
   labelled in every table as an adaptation that sees only the final
   observation.

Budgets, identical across arms within a task: 60 updates, 16 episodes per
update, Euler steps M = 10, Adam at 3e-3, evaluation 500 episodes on Avoiding
and 300 on Pushing, reference 900 and 600 episodes. Avoiding uses K = 6 with 4
of 24 routes rewarded; Pushing K = 4 with 1 of 4 rewarded.

Evaluation budgets stay at the legacy values so the corrected and legacy
cohorts remain comparable on budget. Any increase would be fixed before the
first main cell runs and applied to every arm of that task.

Full matrix: 2 tasks x 4 weightings x 5 policies x 5 seeds x 4 arms = 800 runs.

## 4. Fixed estimator settings

Declared here, before calibration, and recorded in every row.

| Setting | Value |
|---|---|
| objective | conditional |
| standardisation | `fixed`: leave-one-out reward baseline, neither term rescaled |
| EMA rate, mode histogram | 0.15, read before the current batch updates it |
| EMA rate, valid-label probability | 0.15, same lag |
| floor on 1/s | 0.05 |
| coefficient clipping | off |
| KL guard | 1e-8, used only by the KL ablation |
| reference smoothing | 0.0 in the main experiment |

The `legacy` standardisation is retained only as a diagnostic contrast and is
described as a surrogate wherever it appears, because no fixed
reward-minus-penalty objective corresponds to it.

## 5. Penalty weight

Calibration runs on **two frozen policies per task that the main analysis
excludes** - the Pushing checkpoints `bb` indices not used in the main five -
and on RL seeds disjoint from the main block. Candidates 0.25, 0.50, 1.00.

Rule, fixed now: on Pushing, if any candidate holds the fraction of runs below
the success floor at or under 0.15, take the **largest** such candidate; if
none does, take the candidate with the **lowest** such fraction. Ties at equal
fraction go to the larger weight. Avoiding uses the weight selected by the same
rule applied to its own calibration cohort.

The same grid and the same rule are applied to `mi` and to `bmd` on their own
weights, so the comparison methods get the tuning budget this method gets.

Selection never reads fidelity on the main cohort.

## 6. Statistics

Primary family, fixed now, four contrasts: `mop` against each of `dsrl`, `mi`,
`bmd`, on each task, on conditional fidelity, direction pre-declared as `mop`
closer to the reference. Holm adjustment across the family. Everything else is
exploratory and labelled as such.

Each contrast reports n, df, mean difference, SD, SE, a t-based 95% interval
and the adjusted p value.

The primary analysis uses **every** run, with undefined fidelity counted and
reported. The success-floor cohort - Avoiding 0.15, Pushing 0.10 - is a
labelled sensitivity analysis reported beside it with its own n.

Trend across skew uses one point per frozen policy. A trend over the four
weighting means is descriptive only.

The reference is shared by the arms within a policy, so the primary analysis
is conditional on it and a reference-resampling sensitivity analysis is
reported separately, drawn from the stored reference counts.

An interval containing zero is reported as not distinguishing the arms. No
equivalence or non-inferiority is claimed, because no margin was set in
advance.

## 7. Stopping

Runs are not stopped, extended or repeated on the basis of their results. A
cell that raises is recorded with its exception and retried at most once; both
attempts are reported. If compute runs short, whole arms are dropped and said
to be missing, and no legacy result is substituted.
