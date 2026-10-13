#!/usr/bin/env python3
"""The dedup key must separate every setting the sweeps vary.

Two runs differing only in a swept setting are not duplicates, but the key
compares a fixed tuple of fields, so a setting missing from it silently destroys
one arm of that sweep. It happened: the penalty-weight runs at 0.50 were dropped
as duplicates of the 0.25 runs because the key did not carry ``lam``. This
checks each swept setting by constructing a pair that differs only in it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

BASE = {"task": "pushing", "skew": "0.10", "bb_seed": 2, "arm": "mop",
        "seed": 901, "lam": 1.0, "bmd_prior": "uniform"}

VARIED = {"lam": 0.25, "bmd_prior": "fitted", "skew": "0.25",
          "bb_seed": 3, "arm": "mi", "seed": 902, "task": "avoiding"}

ok = True
print(f"{'field':<12}{'keys differ':>13}")
for field, other in VARIED.items():
    a = dict(BASE)
    b = dict(BASE, **{field: other})
    differ = D.row_key(a) != D.row_key(b)
    ok &= differ
    print(f"  {field:<10}{'yes' if differ else 'NO -- COLLIDES':>13}")

print(f"\nkey fields: {D.KEY}")
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
