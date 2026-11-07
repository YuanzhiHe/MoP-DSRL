#!/usr/bin/env python3
"""Re-measure a stored reference with the current code and compare.

Every distance in the analysis is scored against a cached reference histogram,
and a cache carries no record of the code that produced it. This runs the
reference protocol again for the named policies, under the same seeds, chunk
size and episode count the cache key records, and prints the stored and
re-measured histograms side by side. A cache that the current code reproduces
is a cache the distances can rest on.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d3il_v2 as D  # noqa: E402

# A job runs from a frozen copy of the source, so the repository root
# has to come from the driver rather than from this file's location.
ROOT = D.ROOT

CREF = ROOT / "runs" / "_d3il_v2" / "_cref"


def stored(task, weighting, bb):
    """The cache entry the analysis actually used for this policy.

    Several entries can exist for one policy, from calibration runs that asked
    for a handful of episodes as well as from the main measurement. The
    analysis reads the longest, and the episode count differs by task, so the
    choice is made by taking the longest rather than by a fixed threshold.
    """
    best = (None, None)
    for f in sorted(CREF.glob("*.json")):
        r = json.loads(f.read_text())
        if (r.get("task") == task and r.get("weighting") == weighting
                and r.get("backbone_seed") == bb
                and (best[1] is None
                     or r.get("episodes", 0) > best[1].get("episodes", 0))):
            best = (f.name, r)
    return best


def remeasure(task, weighting, bb, key, workers):
    """The same protocol the cache key describes, run again now."""
    chunk, n_ep = key["chunk"], key["ref_episodes"]
    n_chunks = max(1, -(-n_ep // chunk))
    payloads = [(weighting, chunk, key["seed_base"] + i, key["M"], "cpu",
                 task, bb, key.get("obs_clip", 50.0))
                for i in range(n_chunks)]
    raw = np.zeros(D.TASKS[task]["n_modes"], dtype=np.int64)
    tot = valid = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=D._MP) as ex:
        for c, ne, nv in ex.map(D._ref_chunk, payloads):
            raw += c
            tot += ne
            valid += nv
    return raw, tot, valid


def main():
    task = sys.argv[1]
    workers = int(sys.argv[2])
    out = {}
    for spec in sys.argv[3:]:
        weighting, bb = spec.split(":")
        bb = int(bb)
        name, rec = stored(task, weighting, bb)
        if rec is None:
            print(f"{task} {weighting} bb{bb}: no stored reference")
            continue
        raw, tot, valid = remeasure(task, weighting, bb, rec["key"], workers)
        K = rec["key"]["K"]
        new = np.zeros(K, dtype=np.int64)
        for m in range(D.TASKS[task]["n_modes"]):
            new[D.coarsen(m, K)] += raw[m]
        old = np.asarray(rec["counts"], dtype=np.int64)
        p_old = old / old.sum()
        p_new = new / new.sum()
        tv = float(0.5 * np.abs(p_old - p_new).sum())
        print(f"{task} {weighting} bb{bb}  cache {name[:12]}")
        print(f"   stored    completion {rec['valid_episodes'] / rec['episodes']:.3f}"
              f"  C_ref {np.round(p_old, 3).tolist()}")
        print(f"   remeasure completion {valid / tot:.3f}"
              f"  C_ref {np.round(p_new, 3).tolist()}")
        print(f"   total variation between them {tv:.3f}", flush=True)
        out[spec] = dict(cache=name, stored_completion=rec["valid_episodes"] / rec["episodes"],
                         remeasured_completion=valid / tot,
                         stored_C_ref=p_old.tolist(), remeasured_C_ref=p_new.tolist(),
                         tv=tv)
    dest = ROOT / "runs" / "_d3il_v2" / f"reference_recheck_{task}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
