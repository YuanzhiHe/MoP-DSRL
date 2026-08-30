#!/usr/bin/env bash
#SBATCH --job-name=mop-cpu
#SBATCH --partition=htc_genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cpu/%x_%j.out
#SBATCH --error=logs/cpu/%x_%j.out
#
# Generic CPU-partition wrapper. Never run this class of work on the cfl2 login
# node: it is shared, and its 192 cores exceed OpenBLAS's thread limit, which
# core-dumped scripts/build_vocab.py.
#
#   sbatch scripts/slurm/cpu_job.sh python scripts/toy_mode_preservation.py
set -euo pipefail

source /shared/apps/easybuild/x86_64/amd/zen4/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate /shared/home1/c.c23089110/.conda/envs/mop_dsrl

export MOP_DSRL_ROOT=/shared/home1/c.c23089110/MoP-DSRL
cd "$MOP_DSRL_ROOT"
export PYTHONPATH="$MOP_DSRL_ROOT:/shared/scratch/SCWF00111/Sam2RL/libero_repo/LIBERO"
# Match BLAS threads to the allocation, or OpenBLAS aborts on high-core nodes.
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs/cpu

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-local}"
echo "cmd: $*"
"$@"
echo "CPU job ${SLURM_JOB_ID:-local} done."
