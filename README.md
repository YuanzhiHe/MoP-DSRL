# MoP-DSRL

Mode-Preserving Noise-Space Reinforcement Learning for Diffusion and Flow Robot Policies.

MoP-DSRL extends DSRL-style latent/noise-space RL with an explicit, observation-conditioned mode-preservation constraint. The frozen generative backbone (diffusion or rectified-flow) is never modified; an outer Gaussian policy `mu_phi(z|o)` learns to steer the noise input while a KL-based regularizer on action-cluster histograms keeps the policy's mode distribution close to the demonstration prior.

## Method overview

```
Observation o
     |
     v
mu_phi(z|o)  ---- z ----> Frozen backbone pi_theta ----> action chunk a
     |                     (DDIM or Euler ODE)
     |
     +--- R_mode constraint: KL(C_phi(o) || C_ref(o)) <= epsilon
```

The SAC outer loop optimizes:

    max_{phi} E[ r_outer ] - alpha * H(mu_phi) - lambda * R_mode(phi)

where `R_mode` is the per-observation KL between on-policy and reference action-cluster histograms (K-means vocabulary, K=16 by default). `lambda` is either fixed or tuned via dual ascent.

Three verifiable degeneration limits serve as contract tests:
- `lambda=0`: reduces to DSRL (no mode constraint)
- `lambda -> inf`: recovers the frozen backbone's cluster distribution
- `phi=phi_0` (mu_phi = N(0,I)): reproduces the BC baseline

## Repository structure

```
MoP-DSRL/
  pyproject.toml                        # Package metadata and dependencies
  README.md                             # This file
  scripts/
    run_unit_tests.sh                   # Run all CPU tests
    slurm_train_libero.sh               # SLURM template for training
    slurm_eval_bc_sanity.sh             # SLURM template for UT-5 evaluation
  mop_dsrl/
    __init__.py                         # Package exports: Config, FrozenDenoiser, SACTrainer
    config.py                           # Config dataclass with all hyperparameters
    denoiser.py                         # FrozenDenoiser ABC + ToyDenoiser (CPU stub)
    sac.py                              # SAC outer-loop trainer with mode regularizer
    components/
      policy.py                         # OuterGaussianPolicy mu_phi(z|o)
      critic.py                         # TwinQ ensemble + Polyak averaging
      mode_regularizer.py               # KL-histogram mode-coverage regularizer
      vocab.py                          # K-means action-cluster vocabulary builder
      reference_dist.py                 # C_ref computation from frozen backbone
      replay.py                         # FIFO replay buffer
      action_clip.py                    # L2 clip with flat-prior log-density correction
    tests/
      test_unit_contracts.py            # 6 CPU tests: DSRL equivalence, mode recovery, etc.
    flow_variant/
      __init__.py                       # Package exports for flow backbone
      euler_solver.py                   # M-step Euler ODE integrator
      rectified_flow_denoiser.py        # FrozenDenoiser subclass for flow-matching
      coupling_endpoint.py              # z_0 provider (BC mode / RL mode)
      velocity_field_pi0.py             # pi_0 velocity adapter + ToyVelocityField
      velocity_field_pi05.py            # pi_0.5 velocity adapter (primary backbone)
      velocity_field_dpt.py             # DP-T fallback adapter
      configs/
        libero_goal_pi0.yaml            # UT-5 eval config for pi_0
        libero_goal_pi05.yaml           # UT-5 eval config for pi_0.5
      eval/
        run_bc_sanity.py                # UT-5 BC-sanity evaluation CLI
        compare_to_diffusion_baseline.py  # Comparison table generator
        smoke_pi05_libero.py            # Quick 1-task smoke test for pi_0.5
      tests/
        test_unit_contracts_flow.py     # 11 CPU tests + 1 GPU/LIBERO gate test
```

## Prerequisites

### Hardware

| Tier | Minimum | Recommended |
|------|---------|-------------|
| CPU unit tests | Any x86_64 | -- |
| Tier-1 simulation (LIBERO/RoboCasa) | 1x A100/H100 80GB | 8x H100 SXM for parallel seeds |
| Tier-3 real robot | 1x RTX 4090 + ALOHA-2 | -- |

### Software

- Python 3.10 or 3.11
- CUDA 12.1+ (for GPU training/evaluation)
- MuJoCo 3.2+ (for LIBERO/RoboCasa simulation)

## Installation

### 1. Clone and install the package

```bash
git clone git@github.com:YuanzhiHe/MoP-DSRL.git
cd MoP-DSRL
pip install -e ".[dev]"
```

For training with experiment tracking:
```bash
pip install -e ".[all]"
```

### 2. Install simulation environments (for benchmarking)

**LIBERO** (primary benchmark):
```bash
pip install robosuite==1.4.1
pip install mujoco>=3.2
pip install libero  # or clone from https://github.com/Lifelong-Robot-Learning/LIBERO
```

**RoboCasa** (supplementary benchmark):
```bash
pip install robosuite>=1.5
pip install mujoco>=3.2
pip install robocasa  # or clone from https://github.com/robocasa/robocasa
```

### 3. Download backbone checkpoints

**pi_0.5 (primary, via HuggingFace Hub):**
```bash
# Automatic download at first use via LeRobot.
# Set LEROBOT_SRC if using a local lerobot checkout:
export LEROBOT_SRC=/path/to/lerobot/src
```

**pi_0 (alternative):**
```bash
# Download from Physical Intelligence's openpi release.
# Place checkpoint at a known path, e.g. checkpoints/pi0_libero/
```

**DP-T (Diffusion Policy - Transformer, fallback):**
```bash
# Download from the diffusion_policy release.
# Place checkpoint at e.g. checkpoints/dpt_libero_goal/
```

### 4. Verify installation

```bash
bash scripts/run_unit_tests.sh
```

All 17 CPU tests should pass without GPU, LIBERO, or any checkpoints.

## Quick start

### Run CPU unit tests

```bash
# From repo root:
python -m pytest mop_dsrl/tests/test_unit_contracts.py -q
python -m pytest mop_dsrl/flow_variant/tests/test_unit_contracts_flow.py -q -k "not gpu"
```

### Build the K-means vocabulary

```python
import numpy as np
from mop_dsrl.components.vocab import build_kmeans_vocab

# Load your demonstration action chunks: [N, H * d_a]
demo_actions = np.load("path/to/demo_actions.npy")

vocab = build_kmeans_vocab(demo_actions, K=16, seed=42)
# Save for reuse
import pickle
with open("vocab_K16.pkl", "wb") as f:
    pickle.dump(vocab, f)
```

### Initialize the SAC trainer (diffusion backbone)

```python
from mop_dsrl import Config, SACTrainer
from mop_dsrl.denoiser import ToyDenoiser  # replace with real backbone

cfg = Config(
    H=50,              # action chunk horizon
    d_a=7,             # per-step action dimension (LIBERO single-arm)
    K=16,              # K-means clusters
    lambda_=0.1,       # mode-preservation weight
    total_env_steps=500_000,
    device="cuda",
)

denoiser = ToyDenoiser(obs_dim=512, z_dim=cfg.z_dim, seed=0)
trainer = SACTrainer(cfg, denoiser=denoiser, obs_dim=512)
```

### Initialize with flow-matching backbone (pi_0.5)

```python
from mop_dsrl import Config, SACTrainer
from mop_dsrl.flow_variant import RectifiedFlowDenoiser

denoiser = RectifiedFlowDenoiser.from_pi05(
    pretrained_path="lerobot/pi05_base",
    H=50, d_a=7, M=10,
    device="cuda",
)
trainer = SACTrainer(cfg, denoiser=denoiser, obs_dim=512)
```

### Training loop skeleton

```python
import numpy as np
from mop_dsrl.components.vocab import build_kmeans_vocab
from mop_dsrl.components.reference_dist import compute_reference_histogram
from mop_dsrl.components.replay import ReplayBuffer
import torch

# 1. Build vocabulary from demonstrations
vocab = build_kmeans_vocab(demo_actions, K=cfg.K, seed=42)

# 2. Compute reference histogram C_ref
# (implement sample_actions_at for your backbone + observation set)
C_ref = compute_reference_histogram(sample_actions_at, n_obs=100, n_ref=50, vocab=vocab)

# 3. Create replay buffer
buffer = ReplayBuffer(cfg.replay_capacity, obs_dim=512, z_dim=cfg.z_dim, seed=cfg.seed)

# 4. Environment interaction loop
for step in range(cfg.total_env_steps):
    # Collect: sample z from mu_phi, denoise to action, step env, push to buffer
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    with torch.no_grad():
        z, _ = trainer._sample_z(obs_t)
        action = denoiser.denoise(obs_t, z).squeeze(0).cpu().numpy()

    obs_next, reward, done, info = env.step(action[:cfg.d_a])
    buffer.push(obs, z.squeeze(0).cpu().numpy(), reward, obs_next, done)
    obs = obs_next

    # Train after warmup
    if step >= cfg.warmup_env_steps and len(buffer) >= cfg.batch_size:
        batch = buffer.sample(cfg.batch_size)
        # Prepare C_ref for this batch (index into precomputed C_ref)
        C_ref_batch = torch.as_tensor(C_ref[:cfg.batch_size], dtype=torch.float32)
        info = trainer.update(batch, vocab=vocab, C_ref_batch=C_ref_batch)
```

## Running on HPC clusters (SLURM)

### Training

```bash
# Set required environment variables
export CHECKPOINT_DIR=/path/to/pi05_checkpoint
export SUITE=libero_goal
export SEED=0
export LAMBDA=0.1

# Submit job
sbatch scripts/slurm_train_libero.sh
```

### Multi-seed sweep

```bash
for SEED in 0 1 2 3 4; do
    export SEED
    sbatch scripts/slurm_train_libero.sh
done
```

### UT-5 BC-sanity evaluation

```bash
export DIFFUSION_REF_CSV=analysis/baselines/pi0_bc_libero_goal.csv
sbatch scripts/slurm_eval_bc_sanity.sh
```

### Environment variables reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CHECKPOINT_DIR` | Yes (training) | -- | Path to backbone checkpoint |
| `LEROBOT_SRC` | For pi_0.5 | -- | Path to lerobot source `src/` directory |
| `MUJOCO_GL` | For headless | -- | Set to `egl` for headless GPU rendering |
| `CUBLAS_WORKSPACE_CONFIG` | Recommended | -- | Set to `:16:8` for deterministic CUDA ops |
| `SUITE` | No | `libero_goal` | LIBERO suite name |
| `SEED` | No | `0` | Random seed |
| `LAMBDA` | No | `0.1` | Mode-preservation weight |
| `TOTAL_ENV_STEPS` | No | `500000` | Total environment steps |

## Experiment plan

### Tier-1: Simulation (LIBERO + RoboCasa)

Benchmarks: LIBERO-{Spatial, Object, Goal, Long}, RoboCasa-Kitchen{atomic, composite}

Backbones: pi_0.5 (flow-matching, primary), DP-T (diffusion, secondary)

Protocol: 5x10^5 env steps, 5 seeds, 50 eval episodes per task per seed

Baselines: BC-only frozen backbone, DSRL (lambda=0), V-GPS, DPPO, one preference-RL baseline

Key metrics:
- Success rate (mean +/- 95% bootstrap CI)
- cluster-recall@5 (mode retention)
- Sample efficiency (success at 1e5, 2.5e5, 5e5 steps)

### Tier-2: Robustness

- Preference label flip rates: {0, 10, 30, 50}%
- Sparse terminal reward
- OOD initial states

### Tier-3: Real robot

- Platform: ALOHA-2 bimanual
- Tasks: multimodal pick-place (left/right approach), 3-step kitchen subtask
- Budget: 4h online fine-tuning per task
- Key metric: success rate + dual-mode retention (each mode >= 20%)

### Ablations

- lambda in {0, 1e-2, 1e-1, 1, 10}
- K in {8, 16, 32}
- Denoising steps M in {2, 4, 8, 16}
- Coupling endpoint on/off
- Dual ascent on/off

### Statistical protocol

- 5 seeds (Tier-1 main) / 3 seeds (Tier-2, ablations)
- Two-sided bootstrap CI (B=10000)
- Holm-Bonferroni correction across baselines

## Unit test contracts

| Test | Contract | File |
|------|----------|------|
| UT-1 | `mu_phi` initializes to N(0,I) | `tests/test_unit_contracts.py` |
| UT-2 | lambda=inf recovers C_ref in TV distance | `tests/test_unit_contracts.py` |
| UT-3 | lambda=0 gives identical actor loss to DSRL | `tests/test_unit_contracts.py` |
| UT-4 | K-means vocabulary is deterministic given fixed seed | `tests/test_unit_contracts.py` |
| UT-5 | Flow BC matches diffusion BC within 2 pp | `flow_variant/tests/test_unit_contracts_flow.py` |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mop-dsrl2026,
  title     = {Mode-Preserving Noise-Space Reinforcement Learning for
               Diffusion and Flow Robot Policies},
  author    = {He, Yuanzhi},
  year      = {2026},
}
```

## License

MIT
