#!/usr/bin/env bash
# Cron wrapper: top up Falcon GPU slots from the trimmed 30-run plan.
# Install:  (crontab -l 2>/dev/null; echo "*/20 * * * * /shared/home1/c.c23089110/MoP-DSRL/scripts/slurm/scheduler_cron.sh") | crontab -
# The self-resubmitting train_job.sh continues each run across the 3-day wall;
# this only STARTS new runs as slots free. Safe to run as often as you like.
set -uo pipefail
source /shared/apps/easybuild/x86_64/amd/zen4/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate /shared/home1/c.c23089110/.conda/envs/mop_dsrl
cd /shared/home1/c.c23089110/MoP-DSRL
mkdir -p logs
echo "===== $(date -Is) =====" >> logs/scheduler.log
python scripts/slurm/falcon_scheduler.py >> logs/scheduler.log 2>&1
