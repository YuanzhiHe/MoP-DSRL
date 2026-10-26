#!/usr/bin/env python3
"""Emit the manuscript's table rows from the run files.

Transcribing means by hand is how stale numbers got into the paper twice, so the
rows are printed in the exact form the tables use: mean with its standard error
as a subscript, the lowest distance in bold.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3il_runs as D

ARMS = ("dsrl", "mi", "bmd", "mop")


def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum()) if q.size else np.nan


def cell(v, best):
    if not len(v):
        return "--"
    m, e = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
    s = f"${m:.3f}".replace("0.", ".", 1) + f"_{{{e:.3f}".replace("0.", ".", 1) + "}}$"
    return "$\\mathbf{" + s.strip("$").replace("_", "}_") + "$" if False else (
        f"$\\mathbf{{{m:.3f}}}_{{{e:.3f}}}$".replace("0.", ".") if best
        else f"${m:.3f}_{{{e:.3f}}}$".replace("0.", "."))


def emit(task, rows, label):
    print(f"\n% ---- {label} ----")
    skews = sorted({r["skew"] for r in rows},
                   key=lambda s: np.nanmean([tvu(r) for r in rows if r["skew"] == s]))
    for metric, key in (("distance", "tv_to_ref"), ("success", "SR")):
        print(f"% {metric}")
        for s in skews:
            sr = [r for r in rows if r["skew"] == s]
            sk = np.nanmean([tvu(r) for r in sr])
            vals = {a: [r[key] for r in sr if r["arm"] == a] for a in ARMS}
            means = {a: (np.mean(v) if v else np.nan) for a, v in vals.items()}
            best = (min(means, key=lambda a: means[a]) if metric == "distance"
                    else max(means, key=lambda a: means[a]))
            cells = " & ".join(cell(vals[a], a == best) for a in ARMS)
            n = " ".join(f"{len(vals[a])}" for a in ARMS)
            print(f"\\emph{{{task}}} & {sk:.2f} & {cells} \\\\   % n = {n}")


emit("Avoiding", D.unify_reference(D.load("mb_skew*.json", floor=0.15)), "Avoiding")

pm = [r for r in D.unify_reference(D.load("pm_skew*.json", floor=0.10))
      if r["arm"] != "bmd"]
pu = D.unify_reference(D.load("pu_skew*.json", floor=0.10))
emit("Pushing", pm + pu, "Pushing, learned-skill arm under the uniform prior")
