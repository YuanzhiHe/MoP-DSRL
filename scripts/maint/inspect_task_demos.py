#!/usr/bin/env python3
"""Report the stored fields of a D3IL task's demonstrations.

Deciding question for adding a task: can the ground-truth mode be recovered
offline? Avoiding's mode is pure robot geometry, so the stored Cartesian
trajectory was enough. Pushing and stacking define the mode by where OBJECTS end
up, so the demonstrations must carry object poses over time or the label cannot
be replayed without the simulator.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/shared/scratch/SCWF00111/d3il_data/data")


def describe(v, indent="    "):
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"{indent}{kk}: {getattr(vv, 'shape', type(vv).__name__)}")
    else:
        print(f"{indent}{getattr(v, 'shape', type(v).__name__)}")


def main() -> int:
    for task in sys.argv[1:]:
        d = ROOT / task
        files = sorted(d.rglob("*.pkl"))
        print(f"\n===== {task}: {len(files)} demonstrations under {d} =====")
        if not files:
            print("  none found")
            continue
        blob = pickle.load(open(files[0], "rb"))
        print(f"  top-level keys: {list(blob)}")
        for k, v in blob.items():
            print(f"  {k}:")
            describe(v)
        lens = []
        for f in files[:15]:
            b = pickle.load(open(f, "rb"))
            arr = None
            for group in b.values():
                if isinstance(group, dict):
                    for vv in group.values():
                        if hasattr(vv, "shape") and vv.ndim >= 1:
                            arr = vv
                            break
                if arr is not None:
                    break
            if arr is not None:
                lens.append(len(arr))
        if lens:
            print(f"  trajectory length over {len(lens)} demos: "
                  f"mean={np.mean(lens):.0f} min={min(lens)} max={max(lens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
