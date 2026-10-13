#!/usr/bin/env bash
# --resume-also must skip cells held in another file without writing to it.
set -uo pipefail
cd "$(dirname "$0")/../.."
A=runs/_d3il/_sa_src.json
B=runs/_d3il/_sa_out.json
rm -f "$A" "$B"
C="--task avoiding --skews 0.00 --backbone-seeds 0 --arms mop --seed-offset 7300 \
  --lam 1.0 --mode-groups 6 --ema 0.15 --divergence tv --n-rewarded 1 \
  --iters 2 --episodes-per-iter 4 --eval-episodes 12 --ref-episodes 12 --workers 4"

echo "=== build the source file: 2 seeds ==="
python scripts/run_d3il_headtohead.py $C --seeds 2 --out "$A" 2>&1 | tail -3
python -c "import json;d=json.load(open('$A'));print('source rows:',len(d),sorted((r['arm'],r['seed']) for r in d))"
cp "$A" /tmp/_sa_src_before.json

echo "=== run 3 seeds into a DIFFERENT file, reading the source ==="
python scripts/run_d3il_headtohead.py $C --seeds 3 --resume --resume-also "$A" --out "$B" 2>&1 \
  | grep -E "resume|bb0" | tail -5
python - <<'PY'
import json
src = json.load(open('runs/_d3il/_sa_src.json'))
out = json.load(open('runs/_d3il/_sa_out.json'))
before = json.load(open('/tmp/_sa_src_before.json'))
sk = {(r['arm'], r['seed']) for r in src}
ok = out and all((r['arm'], r['seed']) not in sk for r in out)
print('out rows:', len(out), sorted((r['arm'], r['seed']) for r in out))
print('no overlap with the source:', ok)
print('source left untouched:', src == before)
print('VERDICT:', 'PASS' if ok and len(out) == 1 and src == before else 'FAIL')
PY
