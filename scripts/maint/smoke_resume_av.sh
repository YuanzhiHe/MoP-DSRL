#!/usr/bin/env bash
# Smoke the resume + fitted-prior fallback end to end: a real JSON must appear,
# and a second pass over the same file must skip what the first wrote.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/_d3il/_smoke_av.json
rm -f "$OUT"
COMMON="--task avoiding --skews 0.00 --backbone-seeds 0 --arms bmd mop --seed-offset 7000 \
  --lam 1.0 --mode-groups 6 --ema 0.15 --divergence tv --n-rewarded 1 --bmd-prior fitted \
  --iters 2 --episodes-per-iter 4 --eval-episodes 12 --ref-episodes 12 --workers 4 --out $OUT"

echo "=== pass 1: 1 seed ==="
python scripts/run_d3il_headtohead.py $COMMON --seeds 1 --resume 2>&1 | tail -6
python -c "import json;d=json.load(open('$OUT'));print('pass1 rows:',len(d),'failed:',[r.get('failed') for r in d if r.get('failed')])"

echo "=== pass 1b: uniform prior must not touch the calibration matrix ==="
python scripts/run_d3il_headtohead.py ${COMMON/--bmd-prior fitted/--bmd-prior uniform} \
  --seeds 1 --seed-offset 7100 --out runs/_d3il/_smoke_av_unif.json 2>&1 | tail -3
python -c "
import json;d=json.load(open('runs/_d3il/_smoke_av_unif.json'))
print('uniform rows:',len(d),'failed:',[r.get('failed') for r in d if r.get('failed')])
print('skill_modes is None:',all(r.get('skill_modes') is None for r in d))"

echo "=== pass 2: 2 seeds, must skip the first ==="
python scripts/run_d3il_headtohead.py $COMMON --seeds 2 --resume 2>&1 | grep -E "resume:|skipped|bb0" | tail -6
python -c "
import json;d=json.load(open('$OUT'))
k=[(r['arm'],r['seed']) for r in d]
print('pass2 rows:',len(d));print('unique:',len(set(k))==len(k));print(sorted(k))
print('VERDICT:', 'PASS' if len(d)==4 and len(set(k))==4 else 'FAIL')"
