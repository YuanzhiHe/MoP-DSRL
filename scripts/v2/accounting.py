#!/usr/bin/env python3
"""Run accounting for the legacy sweep, from the reconstructed manifest."""
import os
import collections
import json
from pathlib import Path

# A job runs a frozen copy of scripts/ under runs/_d3il_v2/_src/<job>,
# so this file's own location says nothing about where the run files
# are. The environment names the root; the fallback is a local run.
ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
M = [json.loads(l) for l in open(ROOT / "runs/_d3il_v2/manifest_legacy.jsonl")]

print(f"planned cells in the manifest: {len(M)}")
st = collections.Counter(r["status"] for r in M)
for k, v in st.most_common():
    print(f"  {k:<20} {v:5d}")

print("\n--- by task and arm (completed only) ---")
done = [r for r in M if r["status"] == "completed"]
tab = collections.Counter((r["task"], r["arm"]) for r in done)
for (t, a), n in sorted(tab.items()):
    print(f"  {str(t):<10} {a:<6} {n:5d}")

print("\n--- distinct frozen policies with at least one completed cell ---")
pols = collections.defaultdict(set)
for r in done:
    if r["backbone_seed"] is not None:
        pols[r["task"]].add((r["weighting"], r["backbone_seed"]))
for t, s in sorted(pols.items()):
    print(f"  {t:<10} {len(s):3d} policies  weightings={sorted({w for w,_ in s})}")

print("\n--- budgets actually used (completed cells) ---")
for f in ("iters", "episodes_per_iter", "eval_episodes", "ref_episodes", "lr",
          "euler_M", "mode_groups", "lam", "divergence", "ema", "ref_smooth",
          "coef_clip", "n_rewarded"):
    c = collections.Counter(r[f] for r in done)
    print(f"  {f:<20} " + ", ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: -x[1])[:6]))

print("\n--- jobs that ended abnormally ---")
bad = collections.Counter()
for r in M:
    for flag in ("job_traceback", "job_timeout", "job_oom"):
        if r.get(flag):
            bad[flag] += 1
print("  cells inside jobs flagged:", dict(bad) or "none")
logs_bad = {r["slurm_log"] for r in M if r.get("job_traceback") or r.get("job_timeout") or r.get("job_oom")}
print(f"  distinct jobs affected: {len(logs_bad)}")

print("\n--- duplicate rows for one cell identity ---")
dup = [r for r in M if (r.get("duplicate_rows") or 0) > 1]
print(f"  cells written more than once: {len(dup)}")

print("\n--- reconciliation with the manuscript's '1,235 fine-tuning runs' ---")
print(f"  completed cells, every file, every sweep : {len(done)}")
mainfiles = {"mb_skew0.00.json","mb_skew0.15.json","mb_skew0.30.json","mb_skew0.60.json",
             "bf_skew0.00.json","bf_skew0.15.json","bf_skew0.30.json","bf_skew0.60.json"}
amain = [r for r in done if r["source_file"] in mainfiles]
pmain = [r for r in done if (r["source_file"] or "").startswith(("pm_", "pu_"))]
lam  = [r for r in done if (r["source_file"] or "").startswith("pl_")]
print(f"  Avoiding main files                      : {len(amain)}")
print(f"  Pushing  main files                      : {len(pmain)}")
print(f"  penalty-weight sweep files               : {len(lam)}")
print(f"  those three together                     : {len(amain)+len(pmain)+len(lam)}")
