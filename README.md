# Reference Matching for Noise-Space Policy Fine-Tuning

Code, run records and analysis for the manuscript of the same name, submitted to
*Neurocomputing*. The manuscript is in [`paper/`](paper/).

## What the paper claims

Reinforcement learning over the latent noise input of a frozen generative policy
raises reward without touching the policy's weights, and changes which of several
valid behaviours that policy produces. Methods that try to preserve that
behaviour almost all maximise the entropy of a distribution over behavioural
modes. That rests on an assumption none of them state. For a reference
distribution `u` that is uniform over `K` modes,

```
D_KL(p || u) = ln K - H(p)
```

so matching a policy to its pretrained behaviour and maximising behavioural
entropy are one objective, and no benchmark whose pretrained policy is uniform
can separate them. Behaviour cloning does not give uniform policies: twenty
cloned D3IL policies span 0.027 to 1.296 nats from uniform, a range of sixteen
produced by the random seed alone.

The paper replaces the entropy term with a penalty on the total variation
between the fine-tuned policy's mode histogram and the frozen policy's own:

```
max_phi  E[r]  -  lambda * D_TV(C_phi, C_ref)
```

`C_ref` is measured once by rolling out the frozen policy. Modes come from task
geometry, using D3IL's own criteria, not from clustering actions. The penalty
carries no learned component and one scalar weight.

## Headline numbers

Each frozen policy is one unit of analysis; runs that share a frozen policy are
not independent.

| Comparison | *Avoiding* | *Pushing* |
|---|---|---|
| vs mode entropy | **+0.137** TV, 15/20 policies | **+0.095** TV, 12/15 |
| vs no mode term | **+0.201** TV, 20/20 | +0.072 TV, 11/16 |
| vs learned skills | +0.146 TV, 19/20 | +0.032, interval spans zero |

The margin over entropy maximisation widens with how far the pretrained
distribution sits from uniform, rising at 0.82 per unit of measured skew on
*Avoiding* (r = 0.96), from 0.040 at the least skewed reference to 0.284 at the
most. That is the trend the identity predicts. The margins over the other two
baselines fall over the same range.

38 independently trained frozen policies, 1,235 fine-tuning runs.

## Reproducing

The sweeps run on CPU. One `(weighting, frozen policy)` group is twenty cells of
1260 episodes each and takes several hours.

```bash
# one group: four arms x five seeds, against a reference measured once
python scripts/run_d3il_headtohead.py \
  --task avoiding --skews 0.00 --backbone-seeds 0 \
  --arms dsrl mop mi bmd --seeds 5 --seed-offset 700 \
  --lam 1.0 --mode-groups 6 --divergence tv --ema 0.15 --n-rewarded 1 \
  --iters 60 --episodes-per-iter 16 --eval-episodes 888 --ref-episodes 888 \
  --out runs/_d3il/mb_skew0.00.json
```

Arms are `dsrl` (no mode term), `mi` (mode entropy), `bmd` (learned skills) and
`mop` (reference matching). `--resume` keeps the cells already in the output
file; `--resume-also` counts cells in other files as done without writing to
them, which lets a sweep split across jobs pick up earlier work.

Analysis reads every run through one loader, which deduplicates on the settings
that are varied and scores each arm in a group against the one reference that
group's frozen policy defines:

```bash
python scripts/maint/paper_numbers.py          # the Avoiding tables and tests
python scripts/maint/pushing_clustered.py      # Pushing, clustered by policy
python scripts/maint/pushing_lambda.py         # the pre-registered weight rule
python scripts/make_figures_v2.py              # both figures, at the text width
python scripts/maint/verify_paper_numbers.py   # every number against the runs
```

## Where things are

| Path | Contents |
|---|---|
| `paper/` | manuscript, figures, highlights, cover letter |
| `runs/_d3il/` | one JSON per sweep, one record per fine-tuning run |
| `docs/PREREG_d3il_replication.md` | analysis plan, committed before the runs |
| `docs/PREREG_pushing_lambda.md` | the penalty-weight rule, committed before that sweep |
| `scripts/run_d3il_headtohead.py` | the sweep driver |
| `scripts/d3il_runs.py` | the shared loader |
| `scripts/maint/` | analysis, verification and diagnostics |
| `docs/SETUP_libero_pi05.md` | an earlier line of work on LIBERO, not in the paper |

Each record in `runs/_d3il/` carries the arm, the weighting, the frozen policy
seed, the run seed, the measured `C_ref` and `C_phi`, the success rate and the
distances, so every table can be rebuilt without rerunning anything.

## Data

D3IL and its demonstrations are public; see the citation in the manuscript. The
frozen policies are cloned from demonstration subsets reweighted toward a
geometric profile over modes, which `scripts/make_skewed_demos.py` constructs.
