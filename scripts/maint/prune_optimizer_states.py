import os, sys, re
from pathlib import Path
R = Path("/shared/scratch/SCWF00111/HPTD/results")
DRY = os.environ.get("DRY", "1") == "1"
files = list(R.rglob("training_state/optimizer_state.safetensors"))
runs = {}
for f in files:
    # .../<run_root>/checkpoints/<step>/training_state/optimizer_state.safetensors
    try:
        ck = f.parents[2]                      # <step>
        root = f.parents[3]                    # checkpoints
        step = int(re.sub(r"\D", "", ck.name) or -1)
    except Exception:
        continue
    runs.setdefault(str(root), []).append((step, f))
keep, drop, dropped_bytes = [], [], 0
for root, items in runs.items():
    items.sort()
    keep.append(items[-1][1])                  # newest checkpoint's optimizer state
    for _, f in items[:-1]:
        drop.append(f); dropped_bytes += f.stat().st_size
print(f"runs={len(runs)}  total_opt_files={len(files)}  keep={len(keep)}  delete={len(drop)}")
print(f"reclaim={dropped_bytes/1099511627776:.2f} TB")
print("--- KEEPING (newest per run) ---")
for f in sorted(keep): print("  KEEP  ", f.relative_to(R))
if DRY:
    print("--- would delete (first 10) ---")
    for f in sorted(drop)[:10]: print("  DEL   ", f.relative_to(R))
    print(f"  ... and {max(0,len(drop)-10)} more")
else:
    man = R.parent / "deleted_optimizer_states.txt"
    with open(man, "w") as m:
        for f in sorted(drop):
            m.write(str(f) + "\n")
    for f in drop:
        f.unlink()
    print(f"DELETED {len(drop)} files; manifest -> {man}")
