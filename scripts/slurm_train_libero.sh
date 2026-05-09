#!/usr/bin/env bash
#SBATCH --job-name=mop-dsrl-libero
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/libero_%j.out
#SBATCH --error=logs/libero_%j.err
#
# MoP-DSRL training on LIBERO with a single GPU.
#
# Before submitting:
#   1. Activate the conda/venv environment that has mop-dsrl installed.
#   2. Set CHECKPOINT_DIR to your pi_0 or DP-T checkpoint path.
#   3. Set LIBERO_ROOT to your LIBERO installation.
#   4. Adjust --partition and --gres to match your cluster.
#
# Example:
#   sbatch scripts/slurm_train_libero.sh
#
set -euo pipefail

# ---------- user configuration ----------
SUITE="${SUITE:-libero_goal}"                # libero_goal | libero_spatial | libero_object | libero_long
BACKBONE="${BACKBONE:-flow}"                 # diffusion | flow
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the backbone checkpoint path}"
SEED="${SEED:-0}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-500000}"
LAMBDA="${LAMBDA:-0.1}"
K="${K:-16}"
DEVICE="cuda:0"

# ---------- environment setup ----------
mkdir -p logs

# If using pi_0.5 via LeRobot, set LEROBOT_SRC to the lerobot/src directory
# export LEROBOT_SRC=/path/to/lerobot_v044/src

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=":16:8"

echo "========================================"
echo "MoP-DSRL Training"
echo "  Suite:        $SUITE"
echo "  Backbone:     $BACKBONE"
echo "  Checkpoint:   $CHECKPOINT_DIR"
echo "  Seed:         $SEED"
echo "  Env steps:    $TOTAL_ENV_STEPS"
echo "  Lambda:       $LAMBDA"
echo "  K:            $K"
echo "  Device:       $DEVICE"
echo "  SLURM Job:    ${SLURM_JOB_ID:-local}"
echo "========================================"

# ---------- run ----------
# This is the entry point you implement to wire the SACTrainer to the LIBERO env.
# The core modules are ready; you need to write a thin training script that:
#   1. Instantiates the environment (LIBERO suite)
#   2. Loads the frozen denoiser (RectifiedFlowDenoiser or ToyDenoiser)
#   3. Builds the K-means vocabulary from demonstration data
#   4. Computes C_ref from the frozen backbone
#   5. Runs the SACTrainer.update() loop with environment interaction
#
# Example placeholder (replace with your actual training script):
python -c "
from mop_dsrl import Config, SACTrainer
from mop_dsrl.flow_variant import RectifiedFlowDenoiser
print('MoP-DSRL package imports OK.')
print('Config z_dim:', Config(H=50, d_a=7).z_dim)
print('Implement your training loop in scripts/train.py')
"

echo "Job $SLURM_JOB_ID completed."
