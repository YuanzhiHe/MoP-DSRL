#!/usr/bin/env bash
# Carry the four remaining Pushing groups to completion without a process pool.
#
# The pooled path stalls on Pushing at both stages and ten days of work has not
# found why, so every stage here runs with --workers 1: the reference is
# measured once per group and cached, then each cell is its own job. A cell is
# 1260 episodes at about eight seconds, so roughly three hours, and the queue
# runs several at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."

wait_for_jobs() {                      # $1 = name pattern, $2 = label
    local n
    while true; do
        n=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "$1" || true)
        [ "$n" -eq 0 ] && break
        echo "[$(date '+%m-%d %H:%M')] $2: $n still queued or running"
        sleep 600
    done
}

echo "[$(date '+%m-%d %H:%M')] waiting on the reference jobs"
wait_for_jobs '^ref' "references"

CREF=runs/_d3il/_cref
echo "[$(date '+%m-%d %H:%M')] cached references: $(ls $CREF 2>/dev/null | wc -l)"
ls $CREF 2>/dev/null || { echo "no references cached; stopping"; exit 1; }

COMMON="--task pushing --seeds 1 --lam 1.0 --mode-groups 4 --ema 0.15 \
--divergence tv --ref-smooth 0.0 --n-rewarded 1 --bmd-prior uniform --iters 60 \
--episodes-per-iter 16 --eval-episodes 300 --ref-episodes 600 --ref-workers 1 \
--workers 1 --resume"

n=0
for spec in 0.25:3:dsrl 0.25:3:mop 0.25:3:mi \
            0.50:1:mop 0.50:1:mi \
            0.50:3:dsrl 0.50:3:mop 0.50:3:mi 0.50:3:bmd \
            0.50:4:dsrl 0.50:4:mop 0.50:4:mi 0.50:4:bmd; do
    IFS=: read -r sk bb arm <<< "$spec"
    for sd in 900 901 902 903 904; do
        out="runs/_d3il/cell_${sk}_${bb}_${arm}_${sd}.json"
        [ -f "$out" ] && [ "$(python -c "import json;print(len(json.load(open('$out'))))" 2>/dev/null)" = "1" ] && continue
        sbatch --parsable -p htc_genoa -c 2 -t 8:00:00 -J c${sk}${bb}${arm:0:1}${sd: -1} \
            -o "logs/cpu/cell_${sk}_${bb}_${arm}_${sd}_%j.out" \
            scripts/slurm/cpu_job.sh python scripts/run_d3il_headtohead.py $COMMON \
            --arms "$arm" --skews "$sk" --backbone-seeds "$bb" --seed-offset "$sd" \
            --resume-also "runs/_d3il/pm_skew${sk}*.json" "runs/_d3il/pu_skew${sk}*.json" \
            --out "$out" >/dev/null 2>&1 && n=$((n+1))
        sleep 1
    done
done
echo "[$(date '+%m-%d %H:%M')] submitted $n per-cell jobs"

wait_for_jobs '^c0\.' "cells"
echo "[$(date '+%m-%d %H:%M')] all cell jobs finished"
python scripts/maint/sweep_status.py
python - <<'PY'
import sys, collections
sys.path.insert(0, 'scripts')
import d3il_runs as D
SK = ('0.00', '0.10', '0.25', '0.50')
r = D.load(['pm_skew*.json', 'cell_*.json'], floor=0.0)
c = collections.Counter((x['skew'], x.get('bb_seed'), x['arm']) for x in r)
miss = [(s, b) for s in SK for b in range(5)
        if any(c.get((s, b, a), 0) < 5 for a in ('dsrl', 'mop', 'mi'))]
print("multi-policy groups still short:", miss or "none")
u = D.load(['pu_skew*.json', 'cell_*.json'], floor=0.0)
cu = collections.Counter((x['skew'], x.get('bb_seed')) for x in u if x['arm'] == 'bmd')
mu = [(s, b) for s in SK for b in range(5) if cu.get((s, b), 0) < 5]
print("learned-skill groups still short:", mu or "none")
PY
