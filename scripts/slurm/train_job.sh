#!/usr/bin/env bash
#
# Generic MoP-DSRL training SLURM job.  This wrapper deliberately does NOT
# embed any #SBATCH directives so that the submitter (scripts/submit_all.py)
# is the single source of truth for partition, GPU resources, time, and
# memory.  Training arguments are passed in via the TRAIN_ARGS environment
# variable so this single wrapper serves every Tier-1, ablation, robustness,
# and baseline cell.
#
# Required env (set by submit_all.py before sbatch):
#   RUN_DIR     — absolute path for the run output directory
#   TRAIN_ARGS  — argument string passed verbatim to scripts/train.py
#
# Optional:
#   SBATCH_TIME_SEC — total SLURM time budget; the wrapper passes
#                     --max_wall_seconds=(SBATCH_TIME_SEC - 1800) so the
#                     training driver writes final artifacts before SLURM
#                     hard-kills the job.
set -euo pipefail

if [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RUN_DIR is required" >&2
    exit 2
fi
if [ -z "${TRAIN_ARGS:-}" ]; then
    echo "ERROR: TRAIN_ARGS is required" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" logs

# Locate env_setup.sh.  When SLURM runs us it stages the script under
# /var/spool/slurmd/job<id>/slurm_script, so BASH_SOURCE doesn't resolve
# the sibling files.  Prefer MOP_DSRL_ROOT (set in env_setup.sh itself
# but we can't yet rely on that) → SLURM_SUBMIT_DIR → BASH_SOURCE.
if [ -n "${MOP_DSRL_ROOT:-}" ] && [ -d "$MOP_DSRL_ROOT/scripts/slurm" ]; then
    SCRIPT_DIR="$MOP_DSRL_ROOT/scripts/slurm"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/scripts/slurm" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts/slurm"
elif [ -d "/shared/home1/c.c23089110/MoP-DSRL/scripts/slurm" ]; then
    SCRIPT_DIR="/shared/home1/c.c23089110/MoP-DSRL/scripts/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_setup.sh"

echo "========================================"
echo "MoP-DSRL Training Job"
echo "  SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "  Run dir:     $RUN_DIR"
echo "  Train args:  $TRAIN_ARGS"
echo "========================================"

# Stay under the SLURM time limit by 30 min so the script can write final
# checkpoints / eval rows even if the inner training hits the wall budget.
SBATCH_TIME_SEC="${SBATCH_TIME_SEC:-0}"
WALL_BUDGET_ARG=""
if [ -n "$SBATCH_TIME_SEC" ] && [ "$SBATCH_TIME_SEC" -gt 0 ]; then
    BUDGET=$((SBATCH_TIME_SEC - 1800))
    if [ "$BUDGET" -gt 0 ]; then
        WALL_BUDGET_ARG="--max_wall_seconds $BUDGET"
    fi
fi

# Run training; pass TRAIN_ARGS verbatim, then the wall-budget guardrail.
# train.py exits 0 both on true completion (writes final_summary.json) and on a
# clean wall-stop (writes a resumable checkpoint, NO final_summary). A real crash
# exits non-zero and `set -e` aborts here — we deliberately do NOT resubmit on a
# crash, only on a wall-stop.
# shellcheck disable=SC2086
python -u scripts/train.py \
    $TRAIN_ARGS \
    --run_dir "$RUN_DIR" \
    $WALL_BUDGET_ARG

# ---- self-resubmit chain: continue until the run truly completes -------------
# A single job cannot exceed the 3-day cluster wall limit, but a 250k-step run
# needs longer. When train.py stops at the wall budget without a final_summary,
# resubmit this same wrapper (same RUN_DIR + TRAIN_ARGS); --resume picks up the
# latest checkpoint. Guarded by a resubmit counter to prevent an infinite loop
# on a run that never makes progress.
if [ -f "$RUN_DIR/final_summary.json" ]; then
    echo "Run COMPLETE (final_summary.json present): $RUN_DIR"
else
    COUNT_FILE="$RUN_DIR/.resubmit_count"
    N=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
    N=$((N + 1))
    echo "$N" > "$COUNT_FILE"
    MAX_RESUBMIT="${MAX_RESUBMIT:-25}"
    if [ "$N" -gt "$MAX_RESUBMIT" ]; then
        echo "ERROR: hit MAX_RESUBMIT=$MAX_RESUBMIT for $RUN_DIR — stopping chain." >&2
        exit 1
    fi
    PART="${SLURM_JOB_PARTITION:-gpu_l40s}"
    CPUS="${SLURM_CPUS_PER_TASK:-8}"
    TLIMIT=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -oP 'TimeLimit=\K[0-9:-]+' | head -1)
    TLIMIT="${TLIMIT:-3-00:00:00}"
    JNAME="${SLURM_JOB_NAME:-mop_resume}"
    OUT="$MOP_DSRL_ROOT/logs/resubmit_${JNAME}_%j.out"
    echo "Run INCOMPLETE — resubmit #$N (partition=$PART time=$TLIMIT)"
    sbatch \
        --job-name="$JNAME" --partition="$PART" --nodes=1 --ntasks-per-node=1 \
        --cpus-per-task="$CPUS" --gres=gpu:1 --time="$TLIMIT" \
        --output="$OUT" --error="$OUT" \
        --export=ALL,RUN_DIR="$RUN_DIR",TRAIN_ARGS="$TRAIN_ARGS",SBATCH_TIME_SEC="$SBATCH_TIME_SEC" \
        "$MOP_DSRL_ROOT/scripts/slurm/train_job.sh"
fi

echo "Job ${SLURM_JOB_ID:-local} completed."
