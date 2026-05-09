#!/usr/bin/env bash
#SBATCH --job-name=mop-dsrl-ut5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ut5_%j.out
#SBATCH --error=logs/ut5_%j.err
#
# UT-5 BC-sanity evaluation: verifies the rectified-flow re-implementation
# matches the diffusion BC baseline within +/- 2 pp on LIBERO-Goal.
#
# Prerequisites:
#   - LIBERO installed and configured
#   - Backbone checkpoint available
#   - Diffusion BC reference CSV from baseline evaluation
#
# Usage:
#   DIFFUSION_REF_CSV=path/to/pi0_bc_libero_goal.csv \
#   sbatch scripts/slurm_eval_bc_sanity.sh
#
set -euo pipefail

mkdir -p logs analysis

CONFIG="${CONFIG:-mop_dsrl/flow_variant/configs/libero_goal_pi05.yaml}"
DIFFUSION_REF_CSV="${DIFFUSION_REF_CSV:?Set DIFFUSION_REF_CSV to the diffusion BC reference CSV}"
OUTPUT="${OUTPUT:-analysis/ut5_libero_goal.csv}"

export MUJOCO_GL=egl
export CUBLAS_WORKSPACE_CONFIG=":16:8"

echo "========================================"
echo "UT-5 BC-Sanity Evaluation"
echo "  Config:     $CONFIG"
echo "  Reference:  $DIFFUSION_REF_CSV"
echo "  Output:     $OUTPUT"
echo "  SLURM Job:  ${SLURM_JOB_ID:-local}"
echo "========================================"

python -m mop_dsrl.flow_variant.eval.run_bc_sanity \
    --config "$CONFIG" \
    --diffusion_reference_csv "$DIFFUSION_REF_CSV" \
    --output "$OUTPUT"

echo "UT-5 evaluation completed. Check analysis/ut5_verdict.json for results."
