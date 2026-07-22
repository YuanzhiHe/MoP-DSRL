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

# ---- wall-stop handling: let the scheduler resume, do NOT self-resubmit -------
# Earlier this wrapper resubmitted itself on a wall-stop. But when several runs
# hit the 3-day wall at once, their simultaneous resubmits exceed the QOS
# submit cap; the rejected sbatch made the wrapper exit non-zero → SLURM marked
# the job FAILED, and the freed slots got taken by fresh runs (priority
# inversion). Instead, exit cleanly here: falcon_scheduler.py (cron, every
# 20 min) is the single point that resumes checkpointed runs — it prioritizes
# in-progress runs over fresh ones and never over-submits past the QOS cap.
if [ -f "$RUN_DIR/final_summary.json" ]; then
    echo "Run COMPLETE (final_summary.json present): $RUN_DIR"
else
    echo "Run INCOMPLETE (wall-stop) — checkpoint saved; scheduler will resume $RUN_DIR"
fi

echo "Job ${SLURM_JOB_ID:-local} completed."
