#!/usr/bin/env bash
#SBATCH --job-name=mop-v2
#SBATCH --partition=htc_genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/cpu_v2/%x_%j.out
#SBATCH --error=logs/cpu_v2/%x_%j.out
#
# Wrapper for the corrected sweep. The provenance lines below are what made the
# legacy sweep reconstructable at all: the command line on a fixed line of the
# log is the only record of how a cell was launched, so it is printed before
# anything else can fail.
#
#   sbatch scripts/v2/cpu_job_v2.sh python scripts/v2/run_d3il_v2.py --task ...
set -euo pipefail

source /shared/apps/easybuild/x86_64/amd/zen4/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate /shared/home1/c.c23089110/.conda/envs/mop_dsrl

export MOP_DSRL_ROOT=/shared/home1/c.c23089110/MoP-DSRL
cd "$MOP_DSRL_ROOT"
export PYTHONPATH="$MOP_DSRL_ROOT"
mkdir -p logs/cpu_v2

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-local}"
echo "cmd: $*"
echo "commit: $(git rev-parse HEAD)"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --porcelain | wc -l) files"
# Freeze the code this job runs. Workers are spawned, so they re-import the
# driver from disk minutes or hours after launch; without a frozen copy, an
# edit to the working tree reaches those workers while the parent keeps the
# version it started with. The copy also records exactly what ran.
SRC_FREEZE="$MOP_DSRL_ROOT/runs/_d3il_v2/_src/${SLURM_JOB_ID:-local}"
mkdir -p "$SRC_FREEZE"
cp -r scripts "$SRC_FREEZE/"
echo "source frozen at: $SRC_FREEZE"

# Point the command at the frozen copy; data paths stay under MOP_DSRL_ROOT.
ARGS=()
for a in "$@"; do
  case "$a" in
    scripts/*) ARGS+=("$SRC_FREEZE/$a") ;;
    *)         ARGS+=("$a") ;;
  esac
done
export PYTHONPATH="$SRC_FREEZE:$MOP_DSRL_ROOT"

echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${ARGS[@]}"
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
