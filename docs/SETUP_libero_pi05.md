# Setup for the LIBERO and pi_0.5 experiments

This is the earlier line of work on LIBERO with a pi_0.5 backbone. None of it
appears in the manuscript, whose experiments are on D3IL and are described in
the top-level README. It is kept because the code it documents is still here.

---

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
| Tier-1 simulation (LIBERO / pi_0.5) | 1× RTX 5090 32 GB **or** 1× A100/H100 | 2× RTX 5090 / 8× H100 for parallel seeds |
| Tier-3 real robot | 1× RTX 4090 + ALOHA-2 | -- |

### Software

| Requirement | Notes |
|---|---|
| **Python 3.11** | pyproject requires `<3.12`; do *not* use system Python 3.12 |
| **CUDA 12.8+** runtime / driver ≥ 570 | required for RTX 5090 (Blackwell sm_120); 12.6 also works for older GPUs |
| **NVIDIA driver** | Driver 595+ for sm_120 wheels of PyTorch |
| **EGL / OSMesa** | Headless GL libs (`libegl1`, `libgl1-mesa-glx`, `libosmesa6-dev`) — needed by MuJoCo / robosuite |
| **git, curl, wget** | For source clones |
| **A HuggingFace account** | Required: accept the [PaliGemma gated repo](https://huggingface.co/google/paligemma-3b-mix-224) once for pi_0.5's tokenizer |

## Installation

This recipe was validated end-to-end on a **2× RTX 5090** server running Ubuntu
24.04 (driver 595.71.05, CUDA 13.2 runtime). It should work on any modern
Linux box with the prerequisites above. The whole flow takes ~30 minutes once
you have the prerequisites in place.

### Step 1 — Clone the experiments-pipeline branch

```bash
git clone -b experiments-pipeline git@github.com:YuanzhiHe/MoP-DSRL.git ~/MoP-DSRL
cd ~/MoP-DSRL
```

The `experiments-pipeline` branch contains the SLURM / local-multi-GPU
launchers (`scripts/setup_and_run.sh`, `scripts/run_local.py`,
`scripts/submit_all.py`) used below.

### Step 2 — Miniforge + a fresh Python 3.11 env

```bash
# Install miniforge if you don't have conda already.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
echo 'source $HOME/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc

# Pin Python 3.11 — pyproject forbids 3.12+ (numpy ABI compatibility).
conda create -n mop_dsrl python=3.11 pip -y
conda activate mop_dsrl
echo 'conda activate mop_dsrl' >> ~/.bashrc
```

### Step 3 — PyTorch (cu128 wheels for RTX 5090 / sm_120)

```bash
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.8" torchvision torchaudio

# Verify all GPUs are visible as sm_120 (or your card's compute capability)
python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f'  GPU{i}: {p.name} sm_{p.major}{p.minor}')
"
```

You must see `cuda available: True` and **`sm_120`** (or your card's value) for
every GPU. If a kernel error mentions "no kernel image is available", you have
the wrong CUDA wheel — re-install with the `cu128` index.

### Step 4 — Install MoP-DSRL + LIBERO + lerobot

We use `scripts/setup_and_run.sh` which handles five well-known pitfalls
(LIBERO's namespace-package layout, lerobot stomping torch back to a CPU
wheel, LIBERO's interactive-prompt-on-first-import, numpy version conflict,
LIBERO's poisoned `requirements.txt`).

```bash
bash scripts/setup_and_run.sh 4   # MoP-DSRL editable install + 17 CPU contract tests
bash scripts/setup_and_run.sh 5   # LIBERO clone (HTTP/1.1) + .pth shim + lerobot + restore torch
```

After Step 5 you should see `ALL OK` printed, listing matching versions of
numpy, torch (+cu128 suffix), libero.libero, lerobot, and `PI05Policy:
import OK`.

#### What `setup_and_run.sh 5` does (in case you want to do it manually)

1. Installs `robosuite==1.4.1`, `mujoco>=3.2.3`, and other LIBERO native deps
   from PyPI wheels (binary only — avoids hours-long source builds).
2. Clones LIBERO over **HTTP/1.1** with a big `http.postBuffer`. GitHub's
   HTTP/2 stream sometimes drops mid-pack on slow networks, leaving an empty
   `~/LIBERO`; the wrapper retries up to 5 times.
3. `pip install -e ~/LIBERO` to pick up LIBERO's declared deps **but** uses
   a manually written `.pth` shim
   (`$CONDA_PREFIX/lib/python3.11/site-packages/_libero_local.pth`) for the
   actual import path, because `LIBERO/libero/` has no `__init__.py` and PEP
   660 editable installs don't expose it.
4. Writes `~/.libero/config.yaml` **before** any `import libero.libero` call.
   Without this, LIBERO's `__init__` prompts the user interactively on first
   import (which hangs under non-TTY stdin).
5. Installs the runtime imports LIBERO actually needs but doesn't declare —
   `matplotlib`, `hydra-core`, `omegaconf`, `opencv-python`, `cloudpickle`,
   `gym==0.26.2` (LIBERO uses legacy `gym`, not `gymnasium`), `pyquaternion`,
   `termcolor`, etc. — all from binary wheels so a stray source build cannot
   block the installer for hours. **Do not** `pip install -r ~/LIBERO/requirements.txt`
   — that file pins `numpy==1.22.4`, `transformers==4.21.1`, and other ancient
   versions that brick `lerobot`.
6. Installs `lerobot[pi]` (or falls back to `lerobot` / GitHub source). lerobot
   inevitably downgrades torch to a CPU wheel — the wrapper detects this
   (`+cu` suffix missing) and force-reinstalls torch from the cu128 index
   pinned to whatever version lerobot pulled.

### Step 5 — Patch transformers for lerobot's pi_0.5

lerobot's pi_0.5 expects `transformers.models.siglip.check.check_whether_transformers_replace_is_installed_correctly`,
which only exists in the `fix/lerobot_openpi` branch of transformers (pinned
to 4.53.2/4.53.3 with custom Siglip modifications matching openpi).

```bash
pip uninstall -y transformers tokenizers
pip install --force-reinstall --no-deps \
    "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
pip install --force-reinstall --no-deps "tokenizers>=0.21,<0.22"

# Verify
python -c "
import transformers
from transformers.models.siglip.check import check_whether_transformers_replace_is_installed_correctly as chk
print('transformers:', transformers.__version__)
print('check passes:', chk())
from lerobot.policies.pi05 import PI05Policy
print('PI05Policy: import OK')
"
```

Expect `transformers: 4.53.2` (or `.3`), `check passes: True`, `PI05Policy: import OK`.

### Step 6 — HuggingFace login + accept the PaliGemma gated repo

pi_0.5 uses PaliGemma's tokenizer (`google/paligemma-3b-mix-224`), which is a
gated HuggingFace repo. One-time setup:

1. Create a read-scoped token at <https://huggingface.co/settings/tokens>.
2. On <https://huggingface.co/google/paligemma-3b-mix-224>, click **"Agree
   and access repository"** (instant approval).
3. Write the token to `~/.cache/huggingface/token` — but **do not** use
   `huggingface-cli login`; it hangs at the "Add token as git credential"
   step on some setups:

   ```bash
   mkdir -p ~/.cache/huggingface
   chmod 700 ~/.cache/huggingface
   printf '%s' 'YOUR_TOKEN_HERE' > ~/.cache/huggingface/token   # no trailing newline
   chmod 600 ~/.cache/huggingface/token
   wc -c ~/.cache/huggingface/token   # must equal token length, NOT length+1
   ```

4. Verify (should return your username in < 1 s):

   ```bash
   python -c "from huggingface_hub import whoami; print(whoami())"
   ```

#### IPv4 fallback — UK academic / hospital networks

On some networks, AAAA (IPv6) DNS records resolve but TCP connect to the v6
address hangs. `curl` falls back to v4 transparently; Python's `requests`
doesn't. Symptom: `whoami()` hangs forever while `curl https://huggingface.co
/api/whoami-v2` returns instantly.

The training scripts have an automatic IPv4-only patch (`scripts/_ipv4_patch.py`,
toggled by `MOP_DSRL_IPV4_ONLY=1` which is the default). If you hit this same
problem in your own scripts, just add `import scripts._ipv4_patch` as the
first line.

### Step 7 — Download pi_0.5 weights + LIBERO demo HDF5s

```bash
bash scripts/setup_and_run.sh 6
```

This downloads:
- `lerobot/pi05_libero` (~7 GB) to `~/.cache/huggingface/`
- LIBERO demo datasets for `libero_{goal,spatial,object,10}` (~5 GB total)
  to `~/LIBERO/libero/datasets/` — preferring the `yifengzhu-hf/LIBERO-datasets`
  HuggingFace mirror over LIBERO's gdown script (faster, no Google Drive rate
  limits).

### Step 8 — Build K-means action-cluster vocabularies

```bash
bash scripts/setup_and_run.sh 7   # builds vocabs/libero_{goal,spatial,object,10}_K16.pkl
```

### Step 9 — GPU smoke test

```bash
bash scripts/setup_and_run.sh 8 2>&1 | tee setup_logs/stage8.log
```

A successful smoke test prints something like:

```
Env=libero_goal obs_dim=190 action_dim=7
✓ Loaded state dict from model.safetensors
Trainer constructed; z_dim=350 ...
Computing C_ref on 30 held-out observations (n_ref=8, dict_path=True)…
Installed pi_0.5 R_mode override using 30 held-out dict obs
C_ref shape=(30, 16) entropy=...
step=100 wall=...s ...
[final-eval] sr=... mean_return=... n=3
Run complete: runs/_smoke_pi05_libero
```

The smoke run is intentionally tiny (1500 env steps) — the goal is to verify
that the full LIBERO ↔ pi_0.5 ↔ SAC ↔ R_mode pipeline is wired correctly.
Final success rate is meaningless at this scale; what matters is no traceback
and that `final_summary.json` exists.

## Running experiments

After `setup_and_run.sh 8` finishes successfully (smoke test produces
`runs/_smoke_pi05_libero/final_summary.json`), you can launch the real
experiments.

### Local multi-GPU (no SLURM)

`scripts/run_local.py` enumerates the experiment matrix and runs at most one
training job per GPU concurrently. Use this on workstations / single-node
servers (e.g. 2× RTX 5090):

```bash
# Main results only, 1 seed (≈4 days on 2× RTX 5090):
bash scripts/setup_and_run.sh 9_main_1seed

# 4 phases (tier1 → baselines → robustness → ablations), 3-seed main + 1-seed aux:
nohup bash scripts/setup_and_run.sh 9_phased > setup_logs/phased.log 2>&1 &

# Monitor:
bash scripts/setup_and_run.sh status   # snapshot of running + finished
tail -f setup_logs/main_1seed.log      # follow live

# Aggregate results into analysis/results_aggregated.csv:
bash scripts/setup_and_run.sh collect
```

### SLURM cluster

`scripts/submit_all.py` emits one sbatch job per cell of the experiment matrix:

```bash
bash scripts/submit_all.sh --dry-run --tier all          # inspect the plan
bash scripts/submit_all.sh --tier all --partition gpu_l40s   # submit
```

The wrapper `scripts/slurm/train_job.sh` sources `scripts/slurm/env_setup.sh`
which contains the canonical environment configuration (paths, MUJOCO_GL,
HF cache, IPv4 fallback, thread caps).

### Theoretically optimal hyperparameters (per `method_spec §8`)

Defaults baked into `scripts/submit_all.py`'s `OPTIMAL_DEFAULTS`:

| Parameter | Value | Note |
|---|---|---|
| `total_env_steps` | 500_000 | Tier-1 main |
| `warmup_env_steps` | 10_000 | |
| `replay_capacity` | 1_000_000 | |
| `batch_size` | 256 | |
| `actor_lr, critic_lr, alpha_lr` | 3e-4 | |
| `tau_polyak` | 5e-3 | Polyak averaging coefficient |
| `gamma` | 0.99 | |
| `H, d_a` | 50, 7 | LIBERO single-arm |
| `M` (Euler steps) | 10 (pi_0.5) / 4 (DP-T) | |
| `K` (K-means clusters) | 16 | |
| `n_on, n_ref, n_obs` | 8, 50, 1000 | |
| `lambda_` | **0.4** | 0.1 (spec) × `rmode_every`=4 compensation |
| `rmode_subsample` | 8 | held-out obs per R_mode evaluation |
| `rmode_every` | 4 | R_mode re-evaluated every K actor updates |
| `final_eval_episodes` | 50 | |

#### Why R_mode is subsampled

pi_0.5 forward is ~110 ms per chunk on a 5090. Computing R_mode at full batch
size (256 × 4 = 1024 forwards per actor update) would take **152 days per
run** — infeasible. We instead:

- Sample only `--rmode_subsample 8` held-out observations per R_mode
  evaluation (vs the full batch of 256).
- Recompute R_mode every `--rmode_every 4` actor updates; in-between, the
  most recent KL value is reused as a detached constant.

To keep the *integrated gradient pressure* on R_mode equal to the spec
λ=0.1, we scale λ by `rmode_every` → **λ=0.4**. The ablation sweep over
λ (`scripts/submit_all.py:plan_ablations`) is correspondingly scaled to
`{0, 4e-2, 4e-1, 4, 40}`, equivalent to the spec sweep
`{0, 1e-2, 1e-1, 1, 10}` in compensated units.

This is a standard stochastic gradient design choice (mini-batch + lazy
update); first-order optimization theory shows convergence to the same
fixed point as the full-batch / every-step baseline (Polyak-Ruppert
averaging). Empirically, the variance increase is absorbed by the 5×10⁵
training horizon.

### Expected wall-clock (2× RTX 5090, Option A defaults)

| Group | runs | hours / run | total GPU-hours | 2-GPU wall-clock |
|---|---|---|---|---|
| tier1_main (4 suites × 1 seed × pi_0.5) | 4 | 22 | 88 | ~1.8 d |
| baseline_bc (no SAC updates) | 1 | 11 | 11 | ~0.2 d |
| baseline_dsrl / vgps / x1 / x2 | 4 | 22 | 88 | ~1.8 d |
| **9_main_1seed total** | **9** | | **~187** | **~4 d** |
| ablations × 1 seed | 13 | 22 | 286 | ~6 d |
| robustness × 1 seed | 5 | 22 | 110 | ~2.3 d |
| **9_phased total** (3 seed main+baseline, 1 seed aux) | 60 | | ~1200 | ~25 d |

For 5 seeds (paper-grade Tier-1 main, full plan): multiply by 5/3 → ~42 days
of 2× RTX 5090 wall-clock for the full 103-run matrix.

### Resuming after interruption

`run_local.py` and `submit_all.py` both skip runs whose `run_dir` already
contains `final_summary.json`. Just relaunch the same command; finished runs
are silently passed over. Pass `--include-skipped` to force re-run.

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
