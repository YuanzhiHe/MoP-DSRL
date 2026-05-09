#!/usr/bin/env python3
"""Compare flow-matching BC results against the diffusion BC baseline.

Joins the locked diffusion pi_0 BC reference CSV with the flow-variant
UT-5 evaluation CSV and emits a Markdown table suitable for inclusion
in Section 6 of the paper.

Usage:
    cd Experiment/core_code
    python -m mop_dsrl.flow_variant.eval.compare_to_diffusion_baseline \
        --diffusion_csv Experiment/analysis/baselines/pi0_bc_libero_goal.csv \
        --flow_csv Experiment/analysis/flow_variant/ut5_libero_goal.csv \
        --output Experiment/analysis/flow_variant/bc_comparison_table.md
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_results(csv_path: str) -> dict[str, list[float]]:
    """Load per-(task, seed) success rates grouped by task."""
    task_rates: dict[str, list[float]] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sr = float(row["success_rate"])
            if not np.isnan(sr):
                task_rates[row["task"]].append(sr)
    return dict(task_rates)


def build_table(
    diff_rates: dict[str, list[float]],
    flow_rates: dict[str, list[float]],
) -> str:
    """Build a Markdown comparison table."""
    all_tasks = sorted(set(list(diff_rates.keys()) + list(flow_rates.keys())))

    lines = [
        "| Task | Diffusion pi_0 (BC) | Flow pi_0 (BC) | Delta (pp) |",
        "|------|--------------------:|---------------:|-----------:|",
    ]

    diff_all = []
    flow_all = []

    for task in all_tasks:
        d_vals = diff_rates.get(task, [])
        f_vals = flow_rates.get(task, [])

        d_mean = np.mean(d_vals) if d_vals else float("nan")
        f_mean = np.mean(f_vals) if f_vals else float("nan")

        if not np.isnan(d_mean) and not np.isnan(f_mean):
            delta = (f_mean - d_mean) * 100.0
            lines.append(
                f"| {task} | {d_mean:.3f} | {f_mean:.3f} | {delta:+.1f} |"
            )
            diff_all.extend(d_vals)
            flow_all.extend(f_vals)
        else:
            lines.append(f"| {task} | {'N/A' if np.isnan(d_mean) else f'{d_mean:.3f}'} "
                         f"| {'N/A' if np.isnan(f_mean) else f'{f_mean:.3f}'} | - |")

    # Overall
    if diff_all and flow_all:
        d_overall = np.mean(diff_all)
        f_overall = np.mean(flow_all)
        delta_overall = (f_overall - d_overall) * 100.0
        lines.append("|------|--------------------:|---------------:|-----------:|")
        lines.append(
            f"| **Overall** | **{d_overall:.3f}** | **{f_overall:.3f}** | **{delta_overall:+.1f}** |"
        )

    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compare diffusion and flow-matching BC results."
    )
    parser.add_argument("--diffusion_csv", required=True)
    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    diff_rates = load_results(args.diffusion_csv)
    flow_rates = load_results(args.flow_csv)

    table = build_table(diff_rates, flow_rates)

    header = "# BC Comparison: Diffusion vs Flow-Matching pi_0 on LIBERO-Goal\n\n"
    content = header + table + "\n"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(content)

    logger.info("Wrote %s", args.output)
    print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
