#!/usr/bin/env bash
# The reference estimate at the size the sweeps actually use.
#
# Twenty-four episodes returned in thirty seconds; six hundred never returned,
# and three rounds of jobs died waiting. Only a test at six hundred can clear a
# fix for that, so this runs the real entry point at the real reference size and
# the smallest everything else.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/_d3il/_smoke_ref600.json
rm -f "$OUT"
START=$(date +%s)
timeout 5400 python scripts/run_d3il_headtohead.py \
  --task pushing --skews 0.50 --backbone-seeds 3 --arms dsrl mop mi \
  --seeds 1 --seed-offset 7700 --lam 1.0 --mode-groups 4 --ema 0.15 \
  --divergence tv --n-rewarded 1 --bmd-prior uniform \
  --iters 2 --episodes-per-iter 4 --eval-episodes 10 --ref-episodes 600 --ref-workers 1 \
  --workers 24 --out "$OUT" 2>&1 | grep -E "=== skew|^  bb|reference estimate" | tail -6
EL=$(( $(date +%s) - START ))
python - <<PY
import json, os
n = len(json.load(open("$OUT"))) if os.path.exists("$OUT") else 0
print(f"cells: {n}/3   elapsed: ${EL}s")
print("VERDICT:", "PASS" if n == 3 else "FAIL")
raise SystemExit(0 if n == 3 else 1)
PY
