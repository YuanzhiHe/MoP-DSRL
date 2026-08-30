#!/usr/bin/env python3
"""Prune superseded training artefacts under HPTD/results, keeping the newest.

Two layouts appear in that tree:

  pi05      <run>/checkpoints/<step>/training_state/optimizer_state.safetensors
  deepspeed <run>/checkpoint-<step>/global_step<step>/{*optim*, mp_rank_*_model_states.pt}

Both write one artefact per checkpoint interval. Optimizer/RNG/scheduler state is
only needed to RESUME training, so once a run is finished every copy but the last
is dead weight -- 1.37 TB of it in the pi05 tree alone. Intermediate *model*
weights are a judgement call and are handled by a separate --what selector.

Always dry-runs unless DRY=0. Every deletion is recorded to a manifest next to
the results tree so the removal is auditable after the fact.

    DRY=1 python scripts/maint/prune_checkpoints.py --what optim
    DRY=0 python scripts/maint/prune_checkpoints.py --what optim
    DRY=1 python scripts/maint/prune_checkpoints.py --what model
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

RESULTS = Path("/shared/scratch/SCWF00111/HPTD/results")
TB = 1099511627776


def step_of(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else -1


def collect(what: str) -> dict[str, list[tuple[int, Path]]]:
    """Map run-root -> [(step, file)] for the artefact class requested."""
    runs: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    # --- pi05 layout: .../checkpoints/<step>/training_state/... ---
    if what == "optim":
        for f in RESULTS.rglob("training_state/optimizer_state.safetensors"):
            runs[str(f.parents[3])].append((step_of(f.parents[2].name), f))

    # --- deepspeed layout: .../checkpoint-<step>/global_step<step>/... ---
    for ck in RESULTS.glob("*/*/checkpoint-*"):
        if not ck.is_dir():
            continue
        run_root = str(ck.parent)
        step = step_of(ck.name)
        for gs in ck.glob("global_step*"):
            if what == "optim":
                for f in gs.iterdir():
                    if f.is_file() and "optim" in f.name.lower():
                        runs[run_root + "::optim"].append((step, f))
            else:  # model weights
                for f in gs.glob("*model_states.pt"):
                    runs[run_root + "::model"].append((step, f))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["optim", "model"], default="optim")
    args = ap.parse_args()
    dry = os.environ.get("DRY", "1") == "1"

    runs = collect(args.what)
    keep: list[Path] = []
    drop: list[Path] = []
    for _, items in runs.items():
        items.sort(key=lambda t: t[0])
        newest = items[-1][0]
        for step, f in items:
            (keep if step == newest else drop).append(f)

    freed = sum(f.stat().st_size for f in drop if f.exists())
    print(f"what={args.what}  runs={len(runs)}  keep={len(keep)}  delete={len(drop)}")
    print(f"reclaim = {freed / TB:.2f} TB ({freed / 2**30:.0f} GB)")
    for f in sorted(keep)[:20]:
        print("  KEEP  ", f.relative_to(RESULTS))
    if len(keep) > 20:
        print(f"  ... and {len(keep) - 20} more kept")

    if dry:
        for f in sorted(drop)[:8]:
            print("  DEL   ", f.relative_to(RESULTS))
        print(f"  ... and {max(0, len(drop) - 8)} more would be deleted")
        print("\nDRY RUN -- nothing removed. Re-run with DRY=0 to apply.")
        return 0

    manifest = RESULTS.parent / f"deleted_{args.what}_checkpoints.txt"
    with open(manifest, "a") as m:
        for f in sorted(drop):
            m.write(f"{f}\n")
    n = 0
    for f in drop:
        try:
            f.unlink()
            n += 1
        except FileNotFoundError:
            pass
    print(f"DELETED {n} files, freed ~{freed / TB:.2f} TB; manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
