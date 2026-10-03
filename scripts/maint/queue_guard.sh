#!/usr/bin/env bash
# Submit one job only once this project is below a self-imposed job ceiling.
#
# The per-user QOS limit on htc_genoa is 10 submitted jobs, shared with whatever
# else the account is running. A sweep that fills all ten slots starves the
# account's other work, so this waits for room instead of taking it.
#
# Usage: queue_guard.sh <max_project_jobs> <sbatch args...>
set -uo pipefail
MAX=$1; shift
for _ in $(seq 1 300); do
    own=$(squeue -u "$USER" -h -o "%j" 2>/dev/null \
          | grep -cE "^(mb|bmdP|k24|bbvar|divf)" || true)
    total=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    if [ "$own" -lt "$MAX" ] && [ "$total" -lt 10 ]; then
        sbatch "$@" && exit 0
    fi
    sleep 300
done
echo "queue_guard: gave up waiting" >&2
exit 1
