# MoP-DSRL Experiment Driver

This directory contains the full pipeline for running every Tier-1, Tier-2,
ablation, baseline, and UT-5 cell on the SLURM cluster.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main SAC outer-loop training driver. CLI flags cover every method-spec hyperparameter and every toggle in §11. |
| `eval.py` *(N/A — built into train.py via `--final_eval_episodes`)* | — |
| `envs.py` | Environment adapters: `synthetic` (CPU smoke test) and `libero_*`. |
| `backbones.py` | Frozen-denoiser factory: `toy`, `toyflow`, `pi05`, `dpt`, `pi0`. |
| `build_vocab.py` | Build the K-means action-cluster vocabulary from a demo HDF5 dir or `.npy`. |
| `collect_results.py` | Aggregate every completed run into a tidy CSV. |
| `submit_all.py` | **Master experiment-matrix submitter.** Enumerates the full plan and submits one SLURM job per cell. |
| `submit_all.sh` | Convenience wrapper that activates the conda env and forwards args. |
| `slurm/env_setup.sh` | Sourced by every SLURM job — sets PYTHONPATH (LeRobot, LIBERO, RoboCasa), HF cache, MUJOCO_GL, thread caps. |
| `slurm/train_job.sh` | Generic SLURM wrapper that takes RUN_DIR + TRAIN_ARGS and runs `train.py`. |
| `run_unit_tests.sh` | Run the 17 CPU contract tests. |
| `slurm_train_libero.sh`, `slurm_eval_bc_sanity.sh` | Legacy single-job templates (kept for the README examples). |

## Quick start

```bash
# 0. Validate the package (17 CPU tests, ~6 s).
bash scripts/run_unit_tests.sh

# 1. Inspect the full experiment plan without submitting anything.
bash scripts/submit_all.sh --dry-run --tier all

# 2. Build the K-means vocab for libero_goal (one-time).
python scripts/build_vocab.py \
    --demos /shared/scratch/SCWF00111/Sam2RL/libero_datasets/libero_goal \
    --H 50 --d_a 7 --K 16 --seed 42 \
    --out vocabs/libero_goal_K16.pkl

# 3. Submit Tier-1 main only (LIBERO suites × 5 seeds × pi_0.5).
bash scripts/submit_all.sh --tier main --partition gpu_l40s --backbones pi05

# 4. Submit the full matrix once you're confident.
bash scripts/submit_all.sh --tier all --partition gpu_l40s

# 5. After jobs finish, aggregate results.
python scripts/collect_results.py
```

## Experiment matrix (default flags, theoretically optimal per method_spec §8)

| Group | Cells | Seeds | Total | Notes |
|---|---|---|---|---|
| `tier1_main` | 4 LIBERO suites × backbones | 5 | 20 (pi_0.5 only) | Primary results |
| `baseline_*` | BC, DSRL, V-GPS, X1, X2 on libero_goal | 5 | 25 | Comparison baselines |
| `robustness_*` | flip {10,30,50}%, sparse, OOD on libero_goal | 3 | 15 | Tier-2 |
| `ablation_*` | λ × 5, K × 2, M × 4, coupling, dual-ascent | 3 | 39 | All ablation axes |
| `ut5_bc_sanity` | 4 LIBERO suites BC | 1 | 4 | UT-5 gate |
| **default total** | | | **103** | |
| with `--include-robocasa` | + 2 RoboCasa suites in tier1_main | 5 | +10 | |
| with `--backbones pi05 dpt` | DP-T fallback | 5 | +20 | |

Every cell uses spec-§8 defaults: 5×10⁵ env steps, 256 batch, 10⁶ replay,
lr=3e-4 (actor/critic/alpha), τ=5e-3, γ=0.99, K=16, M=10 (pi_0.5)/4 (DP-T),
n_on=8, n_ref=50, n_obs=1000, 50 final eval episodes.

## Required environment paths

`scripts/slurm/env_setup.sh` defaults to the existing checkouts on this cluster:

```
LEROBOT_SRC = /shared/home1/c.c23089110/HPTD/third_party/lerobot_v044/src
LIBERO_SRC  = /shared/scratch/SCWF00111/Sam2RL/libero_repo/LIBERO
ROBOCASA_SRC= /shared/scratch/SCWF00111/Sam2RL/robocasa_repo/robocasa
HF_HOME     = /shared/scratch/SCWF00111/HPTD/hf_cache  (contains pi05_libero)
LIBERO_DATA_ROOT = /shared/scratch/SCWF00111/Sam2RL/libero_datasets
```

Override any of these by exporting the variable before sbatch.

## Cluster guidance

- `gpu_h100`  — 1 node, 1 H100 80 GB. Best for the largest pi_0.5 runs but
  has a per-user job limit; submit Tier-1 main here in batches.
- `gpu_l40s`  — 2 nodes × 8 L40S = 16 GPUs. Use for ablations / robustness /
  baselines; pi_0.5 in bf16 fits on 48 GB.
- 3-day max wall time across both partitions; the wrapper sets
  `--max_wall_seconds = SBATCH_TIME_SEC − 1800` so the driver writes a final
  checkpoint and eval row before SLURM hard-kills the job.

## Troubleshooting

- **`Requested node configuration is not available`** — your sbatch flags
  ask for resources the partition cannot satisfy (e.g. `--gres=gpu:1` on
  the `dev` partition). Submit to a GPU partition or drop `--gres`.
- **`No 'data' group` when building vocab** — the HDF5 file is not in
  LIBERO layout. Either supply a flat `.npy` of action chunks
  (`[N, H * d_a]` or `[N, H, d_a]`) or convert your demos.
- **`/var/spool/slurmd/.../env_setup.sh: No such file`** — the
  `train_job.sh` wrapper relies on `MOP_DSRL_ROOT` or `SLURM_SUBMIT_DIR` to
  locate `env_setup.sh`. Both are set automatically by `submit_all.py`.
