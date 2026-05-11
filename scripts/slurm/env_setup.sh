#!/usr/bin/env bash
# Common SLURM environment bootstrap for MoP-DSRL jobs.
# Source-only — never executed standalone.

set -euo pipefail

# ---- conda + python env -----------------------------------------------------
source /shared/apps/easybuild/x86_64/amd/zen4/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate /shared/home1/c.c23089110/.conda/envs/mop_dsrl

# ---- repo & path layout -----------------------------------------------------
export MOP_DSRL_ROOT="${MOP_DSRL_ROOT:-/shared/home1/c.c23089110/MoP-DSRL}"
cd "$MOP_DSRL_ROOT"

# Make the local `scripts/` package importable as `import scripts.*`.
export PYTHONPATH="${MOP_DSRL_ROOT}:${PYTHONPATH:-}"

# ---- third-party: lerobot + libero + robocasa -------------------------------
# These point at the existing checkouts on /shared/scratch.
export LEROBOT_SRC="${LEROBOT_SRC:-/shared/home1/c.c23089110/HPTD/third_party/lerobot_v044/src}"
export LIBERO_SRC="${LIBERO_SRC:-/shared/scratch/SCWF00111/Sam2RL/libero_repo/LIBERO}"
export ROBOCASA_SRC="${ROBOCASA_SRC:-/shared/scratch/SCWF00111/Sam2RL/robocasa_repo/robocasa}"

if [ -d "$LEROBOT_SRC" ]; then
    export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH}"
fi
if [ -d "$LIBERO_SRC" ]; then
    export PYTHONPATH="${LIBERO_SRC}:${PYTHONPATH}"
fi
if [ -d "$ROBOCASA_SRC" ]; then
    export PYTHONPATH="${ROBOCASA_SRC}:${PYTHONPATH}"
fi

# ---- HF cache (re-uses the SCWF00111/HPTD cache where pi05 weights live) ----
export HF_HOME="${HF_HOME:-/shared/scratch/SCWF00111/HPTD/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Use the public HF mirror when the main host times out (common from
# certain regions).  Override by exporting HF_ENDPOINT before sbatch /
# before run_local.py.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# If everything you need is already cached, set HF_HUB_OFFLINE=1 to skip
# the HEAD checks entirely.
# export HF_HUB_OFFLINE=1

# ---- demo / dataset roots ---------------------------------------------------
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/shared/scratch/SCWF00111/Sam2RL/libero_datasets}"

# ---- determinism + headless rendering --------------------------------------
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:16:8}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Cap thread counts — too many threads on shared nodes is destructive.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# ---- libero config (needed by libero on first import) -----------------------
mkdir -p "$HOME/.libero"
if [ -d "$LIBERO_SRC" ] && [ ! -f "$HOME/.libero/config.yaml" ]; then
    python -c "
import yaml, pathlib, sys
root = pathlib.Path('$LIBERO_SRC') / 'libero' / 'libero'
cfg = {
    'benchmark_root': str(root),
    'bddl_files':    str(root / 'bddl_files'),
    'init_states':   str(root / 'init_files'),
    'datasets':      str(root.parent / 'datasets'),
    'assets':        str(root / 'assets'),
}
out = pathlib.Path.home() / '.libero' / 'config.yaml'
out.write_text(yaml.safe_dump(cfg))
print(f'wrote {out}')
" || true
fi

echo "[env_setup] conda env: $CONDA_PREFIX"
echo "[env_setup] python:    $(which python)"
echo "[env_setup] LEROBOT_SRC=$LEROBOT_SRC"
echo "[env_setup] LIBERO_SRC=$LIBERO_SRC"
echo "[env_setup] HF_HOME=$HF_HOME"
echo "[env_setup] node:      $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi
