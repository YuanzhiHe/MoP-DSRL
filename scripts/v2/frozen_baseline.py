#!/usr/bin/env python3
"""Evaluate each frozen policy under its own latent prior.

Every comparison in the main matrix runs between objectives, so the tables
report where a fine-tuned policy ends while leaving unstated where it started.
This measures that starting point on the same endpoints, the same evaluation
budget and the same rewarded route draw as the matrix, which puts the success
and return columns on a scale a reader can place.

The noise policy is initialised with a zeroed mean head and a unit log standard
deviation, so ``mu_phi(. | o)`` is the standard normal that the frozen decoder
was trained under. An untrained ``NoisePolicy`` is therefore the frozen policy
itself, and evaluating one reuses the driver's rollout, binning and scoring
without opening a second sampling path that could drift from it.

Rows carry ``arm = "frozen"`` and a weight of zero. They are a separate stage
and enter no contrast of the pre-registered family, which compares fine-tuning
objectives to one another.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d3il_v2 as D  # noqa: E402
import envfactory as EF  # noqa: E402


def _cell(payload):
    """One frozen policy under one evaluation seed."""
    D._single_threaded()
    weighting, seed, args, ref, bb = payload
    base = dict(task=args.task, weighting=weighting, backbone_seed=bb,
                arm="frozen", rl_seed=seed, lam=0.0, attempt=1,
                objective=args.objective, standardise=args.standardise,
                divergence=None, mode_groups=args.mode_groups, bmd_prior=None,
                iters=0, episodes_per_iter=0,
                eval_episodes=args.eval_episodes,
                ref_episodes=args.ref_episodes,
                lr=0.0, M=args.M, n_rewarded=args.n_rewarded, ema=args.ema,
                s_ema=args.s_ema, s_floor=args.s_floor,
                ref_smooth=args.ref_smooth, coef_clip=args.coef_clip,
                kl_eps=args.kl_eps, optimiser=None,
                estimator_version=D.ESTIMATOR_VERSION,
                # The frozen column is the common origin every arm is read
                # against, so it has to carry the same protocol stamp the arms
                # carry; a contrast across two protocols is not a contrast.
                env_protocol=EF.ENV_PROTOCOL, device=args.device,
                obs_version=D.TASKS[args.task].get("obs_version", "obs-v1"),
                code_commit=args.code_commit,
                config_sha256=D.config_hash(args, "frozen", 0.0),
                checkpoint_sha256=ref["checkpoint_sha256"],
                reference_sha256=ref["reference_sha256"],
                reference_complete=ref["complete"],
                reference_episodes=ref["episodes"],
                reference_valid_episodes=ref["valid_episodes"],
                reference_counts=ref["counts"],
                C_ref=ref["C_ref"])
    t0 = time.time()
    import resource
    try:
        ck = D.backbone_path(args.task, weighting, bb)
        model, blob = D.load_backbone(ck, args.device)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        n_raw = D.TASKS[args.task]["n_modes"]
        # The rewarded routes follow the evaluation seed exactly as they do in
        # every fine-tuned cell, so rewarded success and return are measured
        # against the same subset the arms were rewarded on.
        rewarded = set(rng.choice(n_raw, size=min(args.n_rewarded, n_raw),
                                  replace=False).tolist())
        env = EF.build(args.task, D.TASKS, seed, blob.get("history", 1))
        policy = D.NoisePolicy(blob["obs_dim"], blob["act_dim"]).to(args.device)
        diag = {}

        def ep(_i):
            with torch.no_grad():
                _, _, raw_mode, success = D.rollout(
                    env, policy, model, blob, rewarded, args.device,
                    args.M, args.obs_clip, diag)
            return raw_mode, success

        row = D.evaluate(ep, args.eval_episodes, args, rewarded, n_raw)
        row["obs_absmax"] = float(diag.get("obs_absmax", 0.0))
        row["obs_clipped_frac"] = (diag.get("obs_clipped_steps", 0)
                                   / max(diag.get("obs_steps", 0), 1))
        _ch = diag.get("obs_absmax_channels")
        row["obs_absmax_channels"] = (None if _ch is None
                                      else [float(x) for x in _ch])
        row["n_nonfinite"] = 0
        row["n_nonfinite_grad"] = 0
        row["frac_inert"] = 0.0
        base.update(row)
        base["status"] = "completed"
        out = D.score(base, np.asarray(ref["C_ref"], float), args.kl_eps)
    except Exception as exc:                            # noqa: BLE001
        traceback.print_exc()
        base.update(status="exception",
                    exception=f"{type(exc).__name__}: {exc}",
                    exception_traceback=traceback.format_exc()[-4000:],
                    fidelity_defined=False)
        out = base
    out["cell_seconds"] = round(time.time() - t0, 1)
    out["cell_peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    return out


def main() -> int:
    ap = D.build_parser()
    args = ap.parse_args()
    args.arms = ["frozen"]
    import subprocess
    args.code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=D.ROOT, capture_output=True,
        text=True).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume:
        done |= D.load_done([args.out])[0]
    done |= D.load_done(args.resume_also)[0]

    bbs = args.backbone_seeds if args.backbone_seeds else [None]
    refs = {}
    for weighting in args.weightings:
        for bb in bbs:
            refs[(weighting, bb)] = D.reference_record(weighting, args, bb)
    if args.reference_only:
        print("references only, no cell run")
        return 0

    cells = []
    for (weighting, bb), ref in refs.items():
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            probe = dict(task=args.task, weighting=weighting, backbone_seed=bb,
                         arm="frozen", rl_seed=seed,
                         config_sha256=D.config_hash(args, "frozen", 0.0),
                         checkpoint_sha256=ref["checkpoint_sha256"],
                         reference_sha256=ref["reference_sha256"])
            if D.cell_key(probe) in done:
                continue
            cells.append((weighting, seed, args, ref, bb))
    if not cells:
        print("every cell already completed, nothing to run", flush=True)
        return 0
    print(f"{len(cells)} frozen-policy evaluations over {len(refs)} "
          f"checkpoints, {args.workers} workers", flush=True)

    fh = args.out.open("a")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=D._MP,
                             max_tasks_per_child=1) as ex:
        for res in ex.map(_cell, cells):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            tag = (f"  w{res['weighting']} bb{res['backbone_seed']} "
                   f"frozen seed{res['rl_seed']}")
            if res["status"] == "completed":
                tv = res["tv_to_ref"]
                print(f"{tag} SR={res['success_rate']:.3f} "
                      f"rew={res['rewarded_success_rate']:.3f} "
                      f"comp={res['completion_rate']:.3f} "
                      f"TV={'n/a' if tv is None else f'{tv:.3f}'}", flush=True)
            else:
                print(f"{tag} {res['status']}: {res.get('exception')}",
                      flush=True)
    fh.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
