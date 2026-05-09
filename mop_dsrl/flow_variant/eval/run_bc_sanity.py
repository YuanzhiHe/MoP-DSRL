#!/usr/bin/env python3
"""UT-5 BC-sanity evaluation for the rectified-flow variant (method_spec S9).

Evaluates the rectified-flow re-implementation of pi_0 on LIBERO-Goal in
BC-only mode (z_0 ~ N(0, I), no RL outer policy). Compares mean success
rate against the locked diffusion pi_0 BC reference and emits a pass/fail
verdict.

Usage:
    cd Experiment/core_code
    python -m mop_dsrl.flow_variant.eval.run_bc_sanity \
        --config mop_dsrl/flow_variant/configs/libero_goal_pi0.yaml \
        --diffusion_reference_csv Experiment/analysis/baselines/pi0_bc_libero_goal.csv \
        --output Experiment/analysis/flow_variant/ut5_libero_goal.csv

Artifacts written:
    ut5_libero_goal.csv  — per-(task, seed) success rates and aggregates.
    ut5_verdict.json     — {S_diff, S_flow, delta_pp, pass}.
    ut5_summary.md       — short Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# LIBERO-Goal task names (10 tasks)
LIBERO_GOAL_TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "turn_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_green_light",
    "put_the_bowl_on_the_plate",
    "put_the_butter_at_the_back_of_the_stove",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_moka_pot_on_the_stove",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_diffusion_reference(csv_path: str) -> float:
    """Load the locked diffusion pi_0 BC reference success rate.

    Expects a CSV with at least columns: task, seed, success_rate.
    Returns the overall mean success rate across all tasks and seeds.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Diffusion reference CSV not found: {csv_path}. "
            f"Run Task 13 BC eval first to produce this file."
        )

    rates = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rates.append(float(row["success_rate"]))

    if not rates:
        raise ValueError(f"No data rows in reference CSV: {csv_path}")

    return float(np.mean(rates))


def run_evaluation(cfg: dict) -> list[dict]:
    """Run flow-matching BC evaluation on LIBERO-Goal.

    Returns a list of dicts with keys: task, seed, success_rate, n_episodes.
    """
    import torch

    # Reproducibility (PLAN.md Risk R4)
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg.get(
            "cublas_workspace_config", ":16:8"
        )

    from ..coupling_endpoint import sample_z0
    from ..rectified_flow_denoiser import RectifiedFlowDenoiser

    H = cfg["H"]
    d_a = cfg["d_a"]
    z_dim = H * d_a
    M = cfg["M"]
    backbone = cfg.get("backbone", "pi0")
    checkpoint_dir = cfg["checkpoint_dir"]
    seeds = cfg["seeds"]
    episodes_per_task = cfg["episodes_per_task"]
    max_steps = cfg.get("max_episode_steps", 300)

    # Build the frozen denoiser
    denoiser = RectifiedFlowDenoiser.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        backbone=backbone,
        H=H,
        d_a=d_a,
        M=M,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = []

    try:
        # Try to import LIBERO environment
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        benchmark = libero_benchmark.get_benchmark("libero_goal")()
        task_names = LIBERO_GOAL_TASKS

        for task_idx, task_name in enumerate(task_names):
            task = benchmark.get_task(task_idx)
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                successes = 0
                for ep in range(episodes_per_task):
                    env = OffScreenRenderEnv(
                        has_renderer=False,
                        has_offscreen_renderer=True,
                        render_camera="agentview",
                        **task.env_kwargs,
                    )
                    env.seed(seed * 1000 + ep)
                    obs = env.reset()

                    done = False
                    for step in range(max_steps):
                        o_tensor = torch.as_tensor(
                            obs["flat"],
                            dtype=torch.float32,
                            device=next(denoiser.parameters()).device
                            if list(denoiser.parameters())
                            else torch.device("cpu"),
                        ).unsqueeze(0)

                        z0, _ = sample_z0(o_tensor, z_dim, mu_phi=None)
                        with torch.no_grad():
                            a = denoiser.denoise(o_tensor, z0)
                        action = a.squeeze(0).cpu().numpy()

                        # Execute first action from chunk
                        obs, reward, done, info = env.step(action[:d_a])
                        if done:
                            break

                    successes += int(info.get("success", False))
                    env.close()

                sr = successes / episodes_per_task
                results.append({
                    "task": task_name,
                    "seed": seed,
                    "success_rate": sr,
                    "n_episodes": episodes_per_task,
                })
                logger.info(
                    "Task=%s seed=%d SR=%.3f (%d/%d)",
                    task_name, seed, sr, successes, episodes_per_task,
                )

    except ImportError:
        logger.warning(
            "LIBERO not installed. Running in dry-run mode. "
            "Install libero to run the actual UT-5 evaluation."
        )
        # Dry-run: produce placeholder results for structure validation
        for task_name in LIBERO_GOAL_TASKS:
            for seed in seeds:
                results.append({
                    "task": task_name,
                    "seed": seed,
                    "success_rate": float("nan"),
                    "n_episodes": episodes_per_task,
                })

    return results


def write_csv(results: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "seed", "success_rate", "n_episodes"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %s", output_path)


def write_verdict(
    S_diff: float,
    S_flow: float,
    gate_threshold: float,
    output_dir: str,
) -> dict:
    delta_pp = abs(S_flow - S_diff) * 100.0
    passed = delta_pp <= gate_threshold
    verdict = {
        "S_diff": round(S_diff, 6),
        "S_flow": round(S_flow, 6),
        "delta_pp": round(delta_pp, 2),
        "gate_threshold_pp": gate_threshold,
        "pass": passed,
    }
    path = Path(output_dir) / "ut5_verdict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2)
    logger.info("Wrote %s — pass=%s, delta=%.2f pp", path, passed, delta_pp)
    return verdict


def write_summary(verdict: dict, output_dir: str) -> None:
    path = Path(output_dir) / "ut5_summary.md"
    status = "PASS" if verdict["pass"] else "FAIL"
    md = f"""# UT-5 Flow-Matching BC Sanity — {status}

| Metric | Value |
|--------|-------|
| S_diff (diffusion pi_0 BC) | {verdict['S_diff']:.4f} |
| S_flow (rectified-flow pi_0 BC) | {verdict['S_flow']:.4f} |
| |S_flow - S_diff| | {verdict['delta_pp']:.2f} pp |
| Gate threshold | {verdict['gate_threshold_pp']:.1f} pp |
| **Verdict** | **{status}** |

## Protocol

- Backbone: rectified-flow Euler ODE, M=4 steps.
- Coupling endpoint: z_0 ~ N(0, I) (BC mode, no RL).
- Evaluation: LIBERO-Goal, 10 tasks x 5 seeds x 50 episodes = 2500 episodes.
- Reference: locked diffusion pi_0 BC baseline from Task 13.
- Gate predicate (method_spec S9): |S_flow - S_diff| <= 2.0 pp.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(md)
    logger.info("Wrote %s", path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="UT-5 BC-sanity evaluation for rectified-flow variant."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--diffusion_reference_csv",
        required=True,
        help="Path to the locked diffusion pi_0 BC reference CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output CSV (per-task-seed results).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.get("output_dir", str(Path(args.output).parent))

    # Load diffusion reference
    S_diff = load_diffusion_reference(args.diffusion_reference_csv)
    logger.info("Diffusion reference S_diff = %.4f", S_diff)

    # Run evaluation
    results = run_evaluation(cfg)

    # Compute flow success rate
    valid_rates = [r["success_rate"] for r in results if not np.isnan(r["success_rate"])]
    if valid_rates:
        S_flow = float(np.mean(valid_rates))
    else:
        S_flow = float("nan")
        logger.warning("No valid results (dry-run mode). Cannot compute gate verdict.")

    # Write artifacts
    write_csv(results, args.output)
    gate = cfg.get("gate_threshold_pp", 2.0)
    verdict = write_verdict(S_diff, S_flow, gate, output_dir)
    write_summary(verdict, output_dir)

    if not verdict["pass"]:
        logger.error(
            "UT-5 FAILED: delta=%.2f pp exceeds threshold %.1f pp",
            verdict["delta_pp"],
            gate,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
