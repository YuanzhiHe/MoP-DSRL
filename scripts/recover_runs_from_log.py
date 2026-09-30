#!/usr/bin/env python3
"""Rebuild run records from a sweep's stdout when the job did not finish.

run_d3il_headtohead writes its JSON only after every cell completes, so a job
that hits its wall-clock limit loses everything. It does, however, print one line
per finished cell, and those lines carry every field the analysis needs except
the C_phi / C_ref arrays used for optional histogram figures. This parses them
back into the same record shape, so a timed-out sweep still yields its completed
cells instead of nothing.

    python scripts/recover_runs_from_log.py logs/cpu/ph-0.00_814665.out \
        --task pushing --skew 0.00 --out runs/_d3il/push_skew0.00.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LINE = re.compile(
    r"^\s+(?P<arm>dsrl|mop|mi)\s+seed(?P<seed>\d+)\s+"
    r"SR=(?P<SR>[-\d.]+)\s+H=(?P<H>[-\d.]+)\s+mc=\s*(?P<mc>\d+)\s+"
    r"tv_ref=(?P<tv_ref>[-\d.]+)\s+tv_unif=(?P<tv_unif>[-\d.]+)"
    r"(?:\s+\|coef\|max=(?P<coef>[-\d.]+))?(?P<collapsed>\s+COLLAPSED)?\s*$"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--skew", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows, seen = [], set()
    for log in args.logs:
        for line in log.read_text(errors="replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            key = (m["arm"], int(m["seed"]))
            if key in seen:          # a resumed job can repeat a cell
                continue
            seen.add(key)
            rows.append(dict(
                task=args.task, arm=m["arm"], skew=args.skew,
                seed=int(m["seed"]),
                lam=0.0 if m["arm"] == "dsrl" else args.lam,
                SR=float(m["SR"]), H_modes=float(m["H"]),
                mc_at_08=int(m["mc"]),
                tv_to_ref=float(m["tv_ref"]), tv_to_uniform=float(m["tv_unif"]),
                coef_absmax=float(m["coef"]) if m["coef"] else 0.0,
                collapsed=bool(m["collapsed"]),
            ))
    by_arm: dict[str, int] = {}
    for r in rows:
        by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
    print(f"recovered {len(rows)} cells from {len(args.logs)} log(s): {by_arm}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
