"""Shared loader for the D3IL sweep outputs.

Two sweeps can write the same cell. A level is split across jobs by frozen
policy to use more of the queue at once, and a job that times out before its
continuation takes over may redo cells the split job already produced. Counting
such a cell twice would weight one frozen policy above the others, so rows are
keyed and the first occurrence kept.

The key carries ``bmd_prior``: the learned-skill arm is run under both the
uniform and the fitted skill prior with everything else identical, so without it
the fitted runs would silently displace the uniform ones.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs" / "_d3il"

KEY = ("task", "skew", "bb_seed", "arm", "seed", "bmd_prior")


def row_key(r: dict) -> tuple:
    return tuple(r.get(k) for k in KEY)


def load(patterns, floor: float = 0.15, keep_failed: bool = False):
    """Rows from every file matching ``patterns``, deduplicated and filtered.

    ``floor`` drops runs whose success rate marks a failed optimisation, the
    exclusion rule the manuscript pre-registered. ``keep_failed`` keeps the
    placeholder rows written for cells that raised.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    seen, out, dupes = set(), [], 0
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat))):
            for r in json.load(open(f)):
                if not keep_failed and r.get("failed"):
                    continue
                k = row_key(r)
                if k in seen:
                    dupes += 1
                    continue
                seen.add(k)
                out.append(r)
    load.duplicates_dropped = dupes
    return [r for r in out if r["SR"] >= floor]


def unify_reference(rows, tol: float = 1e-6):
    """Score every arm in a group against the one reference the group shares.

    A group is (task, skew, frozen policy). Within one the reference is a
    property of the frozen policy, so all arms belong against the same estimate.
    The multi-policy sweeps already satisfy this: the driver measures C_ref once
    per group and hands the same array to every arm. The earlier Pushing runs do
    not, because the learned-skill arm ran as a separate job and estimated its
    own, leaving the arms at a level scored against references up to 0.072 apart
    in total variation.

    Where a group disagrees, the reference carried by the most rows wins, that
    being the one estimated from the rollouts three of the four arms shared, and
    every row in the group has tv_to_ref recomputed against it. Groups that
    already agree are returned untouched.
    """
    import numpy as np

    groups: dict = {}
    for r in rows:
        groups.setdefault((r.get("task"), r.get("skew"), r.get("bb_seed")), []).append(r)

    out, rescored = [], 0
    for members in groups.values():
        counts: dict = {}
        for r in members:
            k = tuple(np.round(np.asarray(r.get("C_ref", []), float), 6))
            counts.setdefault(k, []).append(r)
        if len(counts) <= 1:
            out.extend(members)
            continue
        common = np.asarray(max(counts, key=lambda k: len(counts[k])), float)
        for r in members:
            phi = np.asarray(r.get("C_phi", []), float)
            if phi.size != common.size:
                out.append(r)
                continue
            new = float(0.5 * np.abs(phi - common).sum())
            if abs(new - r["tv_to_ref"]) > tol:
                r = dict(r)
                r["tv_to_ref"] = new
                r["C_ref"] = [float(x) for x in common]
                rescored += 1
            out.append(r)
    unify_reference.rescored = rescored
    return out


def report(patterns, floor: float = 0.15) -> str:
    rows = load(patterns, floor)
    return (f"{len(rows)} rows after the success floor, "
            f"{load.duplicates_dropped} duplicate cells dropped")


if __name__ == "__main__":
    import sys
    pats = sys.argv[1:] or ["mb_skew*.json", "bf_skew*.json", "pm_skew*.json"]
    for p in pats:
        print(f"{p:22s} {report(p)}")
