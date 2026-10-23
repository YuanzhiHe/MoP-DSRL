#!/usr/bin/env bash
# Run a sweep at the width that deadlocks, with the per-cell work made tiny.
#
# The earlier smoke tests used two cells and passed against code that stalled at
# twenty: every worker blocked in futex, having inherited lock state from the
# threaded parent. A test that cannot reproduce the failure cannot clear a fix
# for it, so this one keeps the real shape -- four arms by five seeds, one pool
# of twenty -- and cuts only the episodes.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/_d3il/_smoke_width.json
rm -f "$OUT"
START=$(date +%s)

timeout 1800 python scripts/run_d3il_headtohead.py \
  --task avoiding --skews 0.00 --backbone-seeds 0 --arms dsrl mop mi bmd \
  --seeds 5 --seed-offset 7500 --lam 1.0 --mode-groups 6 --ema 0.15 \
  --divergence tv --n-rewarded 1 --bmd-prior fitted \
  --iters 2 --episodes-per-iter 4 --eval-episodes 10 --ref-episodes 24 \
  --workers 24 --out "$OUT" 2>&1 | grep -E "=== skew|^  bb|resume|Error" | tail -8
rc=$?
ELAPSED=$(( $(date +%s) - START ))

python - <<PY
import json, os
n = len(json.load(open("$OUT"))) if os.path.exists("$OUT") else 0
print(f"cells written: {n}/20   elapsed: ${ELAPSED}s   timeout rc=$rc")
print("VERDICT:", "PASS" if n == 20 else "FAIL (the pool stalled or died)")
raise SystemExit(0 if n == 20 else 1)
PY
