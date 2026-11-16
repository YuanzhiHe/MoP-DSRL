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
   observation. In the main matrix it deploys its skills under a **uniform**
   prior, which keeps it blind to the reference the proposed arm is matching.
   The prior fitted to the reference is a separate labelled comparison, run at
   each task's highest weighting, and is the stronger control of the two
   because it is handed the target: the mixture weights are solved on the
   simplex against `A[k, j] = P(valid and mode = j | skill = k)`, so a skill
   that rarely finishes contributes rarely, and the weights are fitted on
   episodes that the evaluation then discards.

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
| observation bound | 50 standard deviations of the demonstration data |

The observation bound applies to every arm, both tasks, the training
rollouts, the evaluation rollouts and the reference measurement alike, and
every row records how many of its steps met it. A batch whose gradient is not
finite is dropped and counted rather than stepped on.

The `legacy` standardisation is retained only as a diagnostic contrast and is
described as a surrogate wherever it appears, because no fixed
reward-minus-penalty objective corresponds to it.

## 5. Penalty weight

Calibration runs on frozen policies the main analysis excludes, and on RL
seeds disjoint from the main block. Candidates 0.25, 0.50, 1.00.

The calibration cohort is the **unnumbered checkpoint** of each weighting,
`<task>_skew<w>.pt`, which is a separately trained policy and not one of the
five `bb0`-`bb4` the main analysis uses; the hashes differ, so a calibration
cell and a main cell can never share a key. Four weightings per task, one
policy each, calibration seeds 500-502 on Avoiding and 850-852 on Pushing.

Rule, fixed now: on Pushing, if any candidate holds the fraction of runs below
the success floor at or under 0.15, take the **largest** such candidate; if
none does, take the candidate with the **lowest** such fraction. Ties at equal
fraction go to the larger weight. Avoiding uses the weight selected by the same
rule applied to its own calibration cohort.

The quantity the rule selects on is the fraction of calibration runs whose
geometric success falls below the task's floor - 0.10 on Pushing, 0.15 on
Avoiding - pooled over the four weightings and all calibration seeds of that
arm. It is a measure of how often the penalty destroys the task, and it reads
nothing about fidelity.

The same grid and the same rule are applied to `mi` and to `bmd` on their own
weights, so the comparison methods get the tuning budget this method gets. Each
arm may end on a different weight; the weight an arm runs at is recorded in
every row and enters its configuration hash.

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

## 8. Amendments

Changes to this document after it was fixed, with the reason and the date. No
amendment is made after the results it would affect have been seen.

**2026-11-02, section 5, calibration cohort.** The original text proposed using
`bb` indices outside the main five. The inventory shows exactly five numbered
checkpoints per weighting, so no such index exists. The cohort is now the
separately trained unnumbered checkpoint of each weighting. No calibration run
had been launched when this was changed.

**2026-11-02, section 3, the learned-skill prior.** The legacy sweep ran the
learned-skill arm under a uniform prior on Avoiding and a fitted one on
Pushing, so its two tasks were not comparing the same baseline. Both tasks now
run uniform in the main matrix and fitted as a separate labelled comparison. No
corrected cell of either kind had been launched when this was fixed.

**2026-11-02, section 4, bounded observations.** Channels 6 and 9 of the
Pushing observation are unwrapped orientation angles, with demonstration
standard deviations of 313 and 693 radians. Measured over 6,695 steps under the
frozen policy's own latent prior, channel 6 reaches 6.1e11 normalised units and
more than one step in a hundred exceeds 1e11, so every network reading that
observation saturates. In the learned-skill arm the final observation is also
the discriminator's input, which turned the intrinsic reward into a term of
order 1e10 and took two of the eight Pushing smoke cells to NaN. Wrapping the
angle is not available, because the frozen decoder was trained on the unwrapped
value. The normalised observation is therefore bounded at 50 standard
deviations, the same bound for every arm and both tasks. Over the same 6,695
Pushing steps and 1,825 Avoiding steps, the largest value any other channel
reached was 11.8, so the bound sits more than four times outside the range a
behaving rollout occupies and ten orders of magnitude below the runaway. The
estimator version moves to v2.1, which invalidates every cached reference and
every configuration hash, so no bounded and unbounded result can be pooled. The
smoke runs that exposed this are debugging data under section III of the brief
and do not enter any analysis; no main or calibration cell of the bounded code
had been launched when this was fixed.

The 216 calibration runs then measured how far this reaches. On Pushing the
largest normalised observation a run meets has a median across runs of 1.6e12,
and every one of the 108 runs spends steps against the bound, a median of 0.24
per cent of them. The runaway is the ordinary state of a Pushing run rather
than an edge case. On Avoiding the median is 3.0 and 103 of 108 runs never meet
the bound at all; the five that do are runs whose success has already
collapsed. The Avoiding calibration measured under v2.0 is therefore superseded
on evidence rather than on principle alone.

**2026-11-02, ablations, squared-error matching.** A squared-error histogram
objective, `D = (1/2) sum_j (p_j - q_j)^2` with derivative `p - q`, joins the
divergence comparison. It separates two explanations of any advantage the
proposed arm shows: matching a reference at all, against the particular choice
of total variation. Ablations are exploratory under section 6 and no ablation
cell had been launched when this was added.

**2026-11-12, section 3, the cloning budget of the frozen policy.** Each
backbone was cloned for 20000 steps on both tasks. Under the one-environment
rule the Pushing frozen policy completes a route in 3.6 per cent of episodes,
which leaves a 500-episode evaluation about eighteen labelled episodes and no
estimable route distribution; Avoiding completes about 70 per cent and is
unaffected. Measured on the frozen policy alone, with no arm and no contrast
in view, Pushing completion runs 0.036 at 20000 steps, 0.185 at 60000, 0.345
at 120000 and 0.350 at 240000, so it stops improving at 120000; stacking past
frames instead of raising the budget lowers it at every width tried. The
cloning loss keeps falling past that point on both tasks while completion does
not, so the loss is not what selects the budget. The Pushing backbones are
re-cloned at 120000 steps and the Avoiding backbones stay at 20000, the rule
being that each task is cloned until its frozen policy completes the task
often enough for its route distribution to be measured. The 20000-step Pushing
checkpoints, their frozen evaluations and their calibration are kept under a
`steps20000` name and enter nothing. This rule was written after the frozen
completion rates above had been seen and before any Pushing arm had been run
at the new budget; the Avoiding matrix and ablations were complete when it was
written and are not touched by it.

**2026-11-12, section 3, the observation and the environment rule, measured.**
The tangent observation and the one-environment rule were adopted together,
and a Pushing frozen success of 0.376 under the earlier rule became 0.042. The
two were separated by cloning five tangent backbones and rolling them out
under the current rule: 0.052 against 0.035 for sine and cosine over ten paired
cells each, a difference inside the evaluation's own noise. The fall is
therefore the environment rule, which the Avoiding frozen column shows
independently -- its first weighting reproduces the earlier numbers to the
digit and the rest were inflated by 15 to 20 per cent. The sine-cosine frame
holds the largest normalised observation at 2.1e3 where the tangent reaches
7.9e11.

### Amendment, 2026-11-14: the cohort the weight rule reads

The weight rule of section 5 was applied to Pushing on 2026-11-14 and returned
its fallback branch for all three arms, reporting 24 calibration runs per arm
where the stage produces 12. The stage named the calibration by a glob over
`v22_calibration_*.jsonl`, which also matches the three files the superseded
20000-step cloning cohort left in the same directory under a `_steps20000`
suffix. Pooling the two cohorts put runs from a frozen policy that completes
the task 0.036 of the time beside runs from one that completes it 0.345 of the
time, and the former cannot reach the floor at any weight, so the pooled
fraction stayed above tolerance whatever the current cohort showed.

The stage now names the six calibration files one by one, and the selection
script refuses a calibration in which one weighting and one backbone carry more
than one checkpoint. The decision recorded under the pooled files is kept as
`runs/_d3il_v2/v22_lambda_choice_mixedglob.json`; no main or ablation cell was
run against it, since the matrix was held before any of it started.

On the current cohort alone the rule takes its main branch on both tasks.
Pushing selects 1.00 for reference matching, 1.00 for the learned-skill
baseline and 0.50 for entropy maximisation; Avoiding's recorded decision is
unchanged, which the script verifies by refusing to redecide a task already
written.

### Amendment, 2026-11-16: the settings a staged run carries

The Pushing matrix completed on 2026-11-16 and its fidelity distance came out
identically zero for all four arms over all 400 main cells. The cause is that
`run_stage.py` named the task and left the task's own settings to the driver's
defaults, which describe Avoiding: six route groups, four rewarded routes, 500
evaluation and 900 reference episodes. Pushing has four routes and rewards one.
Six groups put all labelled episodes of a cell in one bin, so every reference
and every policy read as the point distribution and the distance between them
was zero by construction; four rewarded routes made the reward a different
objective from the one the matrix compares. The seed block was Avoiding's.

The whole staged Pushing cohort is therefore set aside under
`superseded_taskdefaults_*` and run again with the settings the launch plan
gives the task. The stage now builds them from that same table, and a gate
reads them back out of every run file. Avoiding is unaffected: its settings
are the defaults, and the stage now passes the same values explicitly, which
the gate confirms on all thirteen of its files.

Two gates located the run directory relative to their own source file. The job
wrapper runs a frozen copy of `scripts/` under `runs/_d3il_v2/_src/<job>`, so
under that wrapper both globbed a directory that does not exist, found nothing,
and passed. They now take the root from the environment, and a gate that finds
no files to inspect fails rather than passes.

### Amendment, 2026-11-16: a stage that dies leaves a file the checks accept

Three Pushing stages were submitted on 104 cores against the job wrapper's
fixed 48 GB request and were killed for memory inside their first minute. Each
had already opened its output file, so the four main files and the two weight
grid files sat at zero rows. The whole suite then passed 121 of 121 checks:
every check reads what a file contains, and a file containing nothing
contradicts nothing. The request is now made per core, which is how the stages
use memory, and a completeness check compares each file against the number of
runs its own stage plans. That check is kept out of the suite the launcher runs
before a cohort, where a file being written is short for an innocent reason,
and is asked for once the queue is empty.
