#!/usr/bin/env python3
"""Aggregate every completed run under ``runs/`` into a tidy CSV.

Walks ``runs/<group>/<run_name>/final_summary.json`` and emits one row per
finished run with: group, run_name, success_rate, mean_return, n_episodes,
wall_seconds, lambda_final.  Writes to ``analysis/results_aggregated.csv``.

Usage:
    python scripts/collect_results.py [--runs-root runs] [--out analysis/...csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="analysis/results_aggregated.csv")
    args = p.parse_args()

    rows = []
    root = Path(args.runs_root)
    if not root.is_dir():
        logger.error("runs root not found: %s", root); return 1

    for summary in sorted(root.glob("*/*/final_summary.json")):
        group = summary.parent.parent.name
        run_name = summary.parent.name
        try:
            with open(summary) as f:
                d = json.load(f)
        except Exception as e:
            logger.warning("Skip %s: %s", summary, e); continue
        ev = d.get("final_eval", {})
        rows.append({
            "group": group,
            "run_name": run_name,
            "run_dir": str(summary.parent),
            "success_rate": ev.get("success_rate", float("nan")),
            "mean_return":  ev.get("mean_return",  float("nan")),
            "mean_length":  ev.get("mean_length",  float("nan")),
            "n_episodes":   ev.get("n_episodes",   0),
            "wall_seconds": d.get("wall_seconds", 0),
            "lambda_final": d.get("lambda_final", float("nan")),
            "training_success_rate": d.get("training_success_rate", float("nan")),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "group", "run_name", "run_dir", "success_rate", "mean_return",
        "mean_length", "n_episodes", "wall_seconds", "lambda_final",
        "training_success_rate",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
