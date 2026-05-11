#!/usr/bin/env python3
"""Full training driver for MoP-DSRL.

Wires the SAC outer-loop trainer to a FrozenDenoiser and an environment,
runs the §5 update loop with the §4.3 mode regularizer, and writes:

  - ``<run_dir>/config.json``    — the resolved run config (incl. all hyperparams).
  - ``<run_dir>/metrics.csv``    — per-update training metrics.
  - ``<run_dir>/eval.csv``       — periodic evaluation rollouts.
  - ``<run_dir>/checkpoint.pt``  — final policy + critic state dicts.
  - ``<run_dir>/checkpoint_step{N}.pt`` — intermediate checkpoints at the
                                          requested ``--ckpt_interval``.

This driver is the single entry point used by every SLURM job in
``scripts/submit_all.py``; experiment-specific configurations are passed
through CLI flags so the same code path runs every Tier-1 main experiment,
every robustness condition, and every ablation cell.

Theoretically optimal hyperparameters for Tier-1 LIBERO main runs (per
method_spec §8) are baked in as defaults: 5e5 env steps, 256 batch, 1M
replay, lr=3e-4 across actor/critic/alpha, gamma=0.99, tau=5e-3, K=16,
M=10 (pi_0.5) / 4 (DP-T legacy), n_on=8, n_ref=50, n_obs=1000.
"""

from __future__ import annotations

# --- IPv4-only socket patch (MUST be first, before requests/urllib/hf_hub) ---
# On some networks (e.g. UK academic IPv6) Python's requests doesn't fall back
# to v4 the way curl does; AAAA lookups hang on TCP connect.  See
# scripts/_ipv4_patch.py.
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402

import argparse
import csv
import json
import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Allow `python scripts/train.py` from repo root or as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mop_dsrl import Config, SACTrainer
from mop_dsrl.components.replay import ReplayBuffer
from mop_dsrl.components.reference_dist import compute_reference_histogram
from mop_dsrl.components.vocab import build_kmeans_vocab, KMeansVocab

from scripts.backbones import make_backbone
from scripts.envs import make_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoP-DSRL training driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backbone & environment
    p.add_argument("--backbone", choices=["toy", "toyflow", "pi05", "dpt", "pi0"],
                   default="toy")
    p.add_argument("--env", default="synthetic",
                   help="synthetic | libero_goal | libero_spatial | libero_object | libero_long")
    p.add_argument("--task_id", type=int, default=0,
                   help="Task index inside the LIBERO suite (ignored for synthetic).")
    p.add_argument("--checkpoint_dir", default="",
                   help="Local checkpoint directory (DP-T) or HF Hub identifier (pi_0.5 fallback).")
    p.add_argument("--pretrained_path", default="",
                   help="HF Hub identifier or local dir for pi_0.5.")
    p.add_argument("--cache_dir", default="",
                   help="HF cache directory; empty = default.")
    p.add_argument("--local_files_only", action="store_true")

    # Action geometry & vocab
    p.add_argument("--H", type=int, default=50)
    p.add_argument("--d_a", type=int, default=7)
    p.add_argument("--M", type=int, default=10, help="Euler steps for flow backbones.")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--vocab_path", default="",
                   help="Pre-built vocab pickle. If empty, fits on synthetic actions.")

    # SAC core (method_spec §8 defaults)
    p.add_argument("--total_env_steps", type=int, default=500_000)
    p.add_argument("--warmup_env_steps", type=int, default=10_000)
    p.add_argument("--replay_capacity", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--alpha_lr", type=float, default=3e-4)
    p.add_argument("--tau_polyak", type=float, default=5e-3)
    p.add_argument("--gamma", type=float, default=0.99)

    # Mode regularizer (method_spec §4)
    p.add_argument("--lambda_", type=float, default=0.1, dest="lambda_")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--alpha_lambda", type=float, default=1e-3)
    p.add_argument("--n_on", type=int, default=8)
    p.add_argument("--n_ref", type=int, default=50)
    p.add_argument("--n_obs", type=int, default=1000)
    p.add_argument("--tune_lambda", action="store_true",
                   help="Enable dual ascent on lambda (method_spec §4.5).")

    # Toggles (method_spec §11)
    p.add_argument("--value_guidance", action="store_true")
    p.add_argument("--pref_reward", action="store_true")
    p.add_argument("--hierarchical", action="store_true")
    p.add_argument("--weight_level", action="store_true",
                   help="X1 counter-hypothesis: action-weight reweighting instead of cluster KL.")
    p.add_argument("--imitation_only", action="store_true",
                   help="X2 counter-hypothesis: BC regularizer instead of cluster KL.")
    p.add_argument("--no_coupling_endpoint", action="store_true",
                   help="Disable RL-mode coupling endpoint (z_0 ~ N(0, I)).")

    # Robustness perturbations
    p.add_argument("--pref_flip_rate", type=float, default=0.0,
                   help="Probability of flipping a preference label in pref-RL mode.")
    p.add_argument("--reward_mode", choices=["dense", "sparse"], default="dense",
                   help="`sparse` zeros all per-step rewards and only emits at done.")
    p.add_argument("--ood_init", action="store_true",
                   help="Sample initial states from a shifted distribution.")

    # Eval / I/O
    p.add_argument("--eval_interval", type=int, default=25_000)
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Episodes per eval pass during training.  Final eval uses --final_eval_episodes.")
    p.add_argument("--final_eval_episodes", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=100_000)
    p.add_argument("--log_interval", type=int, default=1000)

    # Engineering
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--obs_dim", type=int, default=32,
                   help="Observation feature dim (only used for synthetic env).")
    p.add_argument("--episode_steps", type=int, default=300)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--max_wall_seconds", type=int, default=0,
                   help="If >0, exit cleanly after approximately this many seconds.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_synthetic_vocab(z_dim: int, K: int, n: int, seed: int) -> KMeansVocab:
    """Fit a tiny K-means vocab on a synthetic action mixture (smoke-test path)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((max(2 * K, 8), z_dim)) * 3.0
    out = []
    n_per = max(n // centers.shape[0], 1)
    for c in centers:
        out.append(c + 0.6 * rng.standard_normal((n_per, z_dim)))
    chunks = np.concatenate(out, axis=0)
    return build_kmeans_vocab(chunks, K=K, n_init=5, max_iter=100, seed=seed)


def compute_C_ref(
    denoiser,
    obs_set: np.ndarray,
    z_dim: int,
    n_ref: int,
    vocab: KMeansVocab,
    device: str,
    obs_dicts: list | None = None,
) -> np.ndarray:
    """Sample n_ref actions per held-out obs under z ~ N(0, I) and assign to clusters.

    If ``obs_dicts`` is provided and the denoiser supports ``denoise_with_obs_dict``
    (i.e., pi_0.5 / pi_0 backbones that need image+language inputs), the dict path
    is taken — flat ``obs_set`` is then only used for length / device.
    """
    obs_t = torch.as_tensor(obs_set, dtype=torch.float32, device=device)

    use_dict_path = (
        obs_dicts is not None
        and hasattr(denoiser, "denoise_with_obs_dict")
        and hasattr(denoiser.velocity_field, "bind_obs_dict")
    )

    if use_dict_path:
        repeat = denoiser.velocity_field.repeat_batch

        def _sample_at(i: int, n: int) -> np.ndarray:
            batch_one = obs_dicts[i]
            batch_n = repeat(batch_one, n)
            with torch.no_grad():
                z = torch.randn(n, z_dim, device=device, dtype=torch.float32)
                a = denoiser.denoise_with_obs_dict(batch_n, z)
            return a.cpu().numpy().astype(np.float64)
    else:
        def _sample_at(i: int, n: int) -> np.ndarray:
            o_b = obs_t[i].unsqueeze(0).expand(n, -1).contiguous()
            with torch.no_grad():
                z = torch.randn(n, z_dim, device=device, dtype=o_b.dtype)
                a = denoiser.denoise(o_b, z)
            return a.cpu().numpy().astype(np.float64)

    return compute_reference_histogram(
        _sample_at, n_obs=obs_set.shape[0], n_ref=n_ref, vocab=vocab
    )


def _denoise(denoiser, obs_t: torch.Tensor, z: torch.Tensor,
             env=None, use_dict_path: bool = False) -> torch.Tensor:
    """Denoise wrapper that picks flat vs dict path based on backbone capability."""
    if use_dict_path and env is not None and hasattr(denoiser, "denoise_with_obs_dict"):
        batch = denoiser.velocity_field.prepare_libero_batch(
            env.last_obs_dict, env.task_instruction
        )
        if obs_t.shape[0] > 1:
            batch = denoiser.velocity_field.repeat_batch(batch, obs_t.shape[0])
        return denoiser.denoise_with_obs_dict(batch, z)
    return denoiser.denoise(obs_t, z)


def evaluate(
    trainer: SACTrainer,
    denoiser,
    env,
    n_episodes: int,
    d_a: int,
    device: str,
    deterministic: bool = False,
    use_dict_path: bool = False,
) -> dict:
    """Run ``n_episodes`` rollouts and return mean success / return / length."""
    successes = 0
    returns = []
    lengths = []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_return = 0.0
        ep_len = 0
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    mean, _ = trainer.policy.forward(obs_t)
                    z = mean
                else:
                    z, _ = trainer._sample_z(obs_t)
                a = _denoise(denoiser, obs_t, z, env=env, use_dict_path=use_dict_path)
            action_chunk = a.squeeze(0).cpu().numpy()
            first_action = action_chunk[:d_a]
            obs, reward, done, info = env.step(first_action)
            ep_return += float(reward)
            ep_len += 1
            if done:
                break
        successes += int(info.get("success", False))
        returns.append(ep_return)
        lengths.append(ep_len)
    return {
        "success_rate": successes / max(n_episodes, 1),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info("Run dir: %s", run_dir)
    logger.info("Args: %s", json.dumps(vars(args), indent=2, sort_keys=True))

    set_seed(args.seed)
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    z_dim = args.H * args.d_a

    # ---- Environment ----
    env_kwargs = {"seed": args.seed}
    if args.env == "synthetic":
        env_kwargs.update(
            obs_dim=args.obs_dim, action_dim=args.d_a, episode_steps=args.episode_steps
        )
    else:
        env_kwargs.update(task_id=args.task_id, max_steps=args.episode_steps)

    env = make_env(args.env, **env_kwargs)
    obs_dim = env.obs_dim
    logger.info("Env=%s obs_dim=%d action_dim=%d", args.env, obs_dim, env.action_dim)

    # ---- Backbone ----
    denoiser = make_backbone(
        name=args.backbone,
        H=args.H, d_a=args.d_a, M=args.M, device=args.device,
        obs_dim=obs_dim, seed=args.seed,
        checkpoint_dir=args.checkpoint_dir or None,
        pretrained_path=args.pretrained_path or None,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    for p in denoiser.parameters():
        p.requires_grad_(False)

    # ---- Config + trainer ----
    cfg = Config(
        H=args.H, d_a=args.d_a, K=args.K,
        n_on=args.n_on, n_ref=args.n_ref, n_obs=args.n_obs,
        lambda_=args.lambda_, epsilon=args.epsilon, alpha_lambda=args.alpha_lambda,
        replay_capacity=args.replay_capacity, batch_size=args.batch_size,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        tau_polyak=args.tau_polyak, gamma=args.gamma,
        warmup_env_steps=args.warmup_env_steps, total_env_steps=args.total_env_steps,
        eval_interval=args.eval_interval,
        backbone_family="flow" if args.backbone in ("toyflow", "pi05", "dpt") else "diffusion",
        value_guidance=args.value_guidance, tune_lambda=args.tune_lambda,
        pref_reward=args.pref_reward, hierarchical=args.hierarchical,
        weight_level=args.weight_level, imitation_only=args.imitation_only,
        seed=args.seed, device=args.device,
    )
    trainer = SACTrainer(cfg, denoiser=denoiser, obs_dim=obs_dim)
    logger.info("Trainer constructed; z_dim=%d clip_radius=%.3f target_entropy=%.3f",
                z_dim, cfg.clip_radius, cfg.effective_target_entropy)

    # ---- Vocab ----
    if args.vocab_path:
        with open(args.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        if vocab.K != args.K:
            raise ValueError(f"Vocab K={vocab.K} != requested K={args.K}")
        logger.info("Loaded vocab K=%d from %s", vocab.K, args.vocab_path)
    else:
        vocab = fit_synthetic_vocab(z_dim, args.K, n=4 * args.K * 64, seed=args.seed)
        logger.info("Fit synthetic vocab K=%d (no --vocab_path supplied)", args.K)

    # ---- Held-out observations + (for pi_0.5) their full obs dicts ----
    needs_obs_dicts = (
        args.backbone in ("pi05", "pi0")
        and hasattr(denoiser, "denoise_with_obs_dict")
        and args.env != "synthetic"
    )
    held_out_obs_list = []
    held_out_obs_dicts: list | None = [] if needs_obs_dicts else None
    for _ in range(args.n_obs):
        flat = env.reset()
        held_out_obs_list.append(flat)
        if needs_obs_dicts:
            try:
                batch = denoiser.velocity_field.prepare_libero_batch(
                    env.last_obs_dict, env.task_instruction
                )
                held_out_obs_dicts.append(batch)
            except Exception as e:
                logger.warning("prepare_libero_batch failed at #%d: %s", len(held_out_obs_dicts), e)
                # Fall back: disable dict path entirely
                needs_obs_dicts = False
                held_out_obs_dicts = None
                break
    held_out_obs = np.stack(held_out_obs_list, axis=0).astype(np.float32)
    obs0 = env.reset()
    logger.info("Computing C_ref on %d held-out observations (n_ref=%d, dict_path=%s)…",
                args.n_obs, args.n_ref, needs_obs_dicts)
    C_ref_full = compute_C_ref(
        denoiser, held_out_obs, z_dim, args.n_ref, vocab, args.device,
        obs_dicts=held_out_obs_dicts,
    )

    # --- pi_0.5 R_mode override: use aligned held-out (flat, dict) pairs ---
    # The SAC actor loss calls trainer._mode_regularizer(o, vocab, C_ref_batch) on the
    # replay batch's flat obs.  For pi_0.5 we can't denoise from a flat tensor, so we
    # ignore the replay obs and instead sample held-out dict obs for R_mode.  This is
    # methodologically valid: R_mode constrains the cluster histogram on a held-out
    # distribution, which is what C_ref was computed on anyway.
    if needs_obs_dicts:
        from mop_dsrl.components.mode_regularizer import (
            soft_assign_distances, mode_kl_from_assignments,
        )
        _held_flat_gpu = torch.as_tensor(
            held_out_obs, dtype=torch.float32, device=args.device
        )
        _C_ref_gpu = torch.as_tensor(
            C_ref_full, dtype=torch.float32, device=args.device
        )

        def _pi05_mode_regularizer(o, vocab_, C_ref_batch_):
            """Compute R_mode using held-out (flat, dict) pairs aligned by index."""
            B = o.shape[0]
            n_on = trainer.cfg.n_on
            idx = rng.integers(0, len(held_out_obs_dicts), size=B)
            # Build flat obs for policy.sample; one row per (held_i, repeated n_on)
            held_flat = _held_flat_gpu[idx]                        # [B, obs_dim]
            held_rep = held_flat.repeat_interleave(n_on, dim=0)    # [B*n_on, obs_dim]
            z_rep, _ = trainer._sample_z(held_rep)
            # Denoise via dict path, one obs-dict at a time (B forwards of n_on each).
            actions = []
            for i_local in range(B):
                batch_n = denoiser.velocity_field.repeat_batch(
                    held_out_obs_dicts[int(idx[i_local])], n_on
                )
                z_slice = z_rep[i_local * n_on:(i_local + 1) * n_on]
                actions.append(denoiser.denoise_with_obs_dict(batch_n, z_slice))
            a_rep = torch.cat(actions, dim=0)
            # Soft cluster assignment vs vocab + KL against C_ref(held_i)
            median = torch.as_tensor(vocab_.median, dtype=a_rep.dtype, device=args.device)
            mad = torch.as_tensor(vocab_.mad, dtype=a_rep.dtype, device=args.device)
            a_norm = (a_rep - median) / mad
            centroids = torch.as_tensor(vocab_.centroids, dtype=a_rep.dtype, device=args.device)
            soft = soft_assign_distances(a_norm, centroids, temperature=1.0)
            C_ref_held = _C_ref_gpu[idx]
            return mode_kl_from_assignments(
                soft, C_ref_held, n_on, trainer.cfg.laplace_alpha()
            )
        trainer._mode_regularizer = _pi05_mode_regularizer
        logger.info("Installed pi_0.5 R_mode override using %d held-out dict obs",
                    len(held_out_obs_dicts))
    logger.info("C_ref shape=%s entropy=%.3f", C_ref_full.shape,
                float(-(C_ref_full * np.log(C_ref_full + 1e-12)).sum(axis=-1).mean()))

    # ---- Replay buffer & loop ----
    buffer = ReplayBuffer(
        capacity=cfg.replay_capacity, obs_dim=obs_dim, z_dim=z_dim, seed=args.seed
    )

    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"
    metrics_f = open(metrics_path, "w", newline="")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow([
        "step", "wall_seconds", "ep_return", "ep_length", "success",
        "critic_loss", "actor_loss_q", "actor_loss_total",
        "r_mode", "alpha", "lambda",
    ])
    eval_f = open(eval_path, "w", newline="")
    eval_w = csv.writer(eval_f)
    eval_w.writerow(["step", "n_episodes", "success_rate", "mean_return", "mean_length"])

    t_start = time.time()
    obs = obs0
    ep_return = 0.0
    ep_len = 0
    n_eps = 0
    n_success = 0
    last_log = 0
    rng = np.random.default_rng(args.seed + 1)
    last_ckpt_step = 0

    for step in range(args.total_env_steps):
        # ---- collect transition ----
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        with torch.no_grad():
            if step < args.warmup_env_steps and not args.no_coupling_endpoint:
                # Bootstrap with z ~ N(0, I) before policy is useful.
                z = torch.randn(1, z_dim, device=args.device, dtype=obs_t.dtype)
            else:
                z, _ = trainer._sample_z(obs_t)
            a_chunk = _denoise(denoiser, obs_t, z, env=env, use_dict_path=needs_obs_dicts)
            action_chunk = a_chunk.squeeze(0).cpu().numpy()
            first_action = action_chunk[: args.d_a]

        obs_next, reward, done, info = env.step(first_action)
        if args.reward_mode == "sparse":
            r_eff = float(reward) if done else 0.0
        else:
            r_eff = float(reward)
        if args.pref_reward and args.pref_flip_rate > 0:
            if rng.uniform() < args.pref_flip_rate:
                r_eff = -r_eff

        buffer.push(obs, z.squeeze(0).cpu().numpy(), r_eff, obs_next, done)
        obs = obs_next
        ep_return += float(reward)
        ep_len += 1

        # ---- end-of-episode bookkeeping ----
        if done:
            n_eps += 1
            n_success += int(info.get("success", False))
            obs = env.reset()
            ep_return_done = ep_return
            ep_len_done = ep_len
            ep_return = 0.0
            ep_len = 0
        else:
            ep_return_done = float("nan")
            ep_len_done = float("nan")

        # ---- SAC updates ----
        info_update = None
        if step >= args.warmup_env_steps and len(buffer) >= cfg.batch_size:
            batch = buffer.sample(cfg.batch_size)
            # Indexing C_ref by random batch obs is approximate — we use the
            # held-out C_ref tiled to batch size; for the per-obs version we'd
            # need to recompute on collected obs, which is much costlier.
            B = cfg.batch_size
            idx = rng.integers(0, C_ref_full.shape[0], size=B)
            C_ref_batch = torch.as_tensor(
                C_ref_full[idx], dtype=torch.float32, device=args.device
            )
            info_update = trainer.update(batch, vocab=vocab, C_ref_batch=C_ref_batch)

        # ---- logging ----
        if (step - last_log) >= args.log_interval or done:
            wall = time.time() - t_start
            row = [
                step, f"{wall:.2f}",
                ep_return_done, ep_len_done,
                n_success / max(n_eps, 1),
            ]
            if info_update is not None:
                row += [
                    f"{info_update.critic_loss:.6f}",
                    f"{info_update.actor_loss_q:.6f}",
                    f"{info_update.actor_loss_total:.6f}",
                    f"{info_update.r_mode:.6f}",
                    f"{info_update.alpha:.6f}",
                    f"{info_update.lambda_:.6f}",
                ]
            else:
                row += ["", "", "", "", "", ""]
            metrics_w.writerow(row)
            metrics_f.flush()
            if (step - last_log) >= args.log_interval:
                last_log = step
                logger.info(
                    "step=%d wall=%.1fs eps=%d sr=%.3f r_mode=%s alpha=%s lambda=%s",
                    step, wall, n_eps, n_success / max(n_eps, 1),
                    f"{info_update.r_mode:.4f}" if info_update else "-",
                    f"{info_update.alpha:.4f}" if info_update else "-",
                    f"{info_update.lambda_:.4f}" if info_update else "-",
                )

        # ---- periodic eval ----
        if args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
            ev = evaluate(
                trainer, denoiser, env, args.eval_episodes, args.d_a, args.device,
                use_dict_path=needs_obs_dicts,
            )
            eval_w.writerow([
                step, ev["n_episodes"], ev["success_rate"],
                ev["mean_return"], ev["mean_length"],
            ])
            eval_f.flush()
            logger.info("[eval] step=%d sr=%.3f mean_return=%.3f n=%d",
                        step, ev["success_rate"], ev["mean_return"], ev["n_episodes"])

        # ---- checkpoint ----
        if args.ckpt_interval > 0 and step > 0 and step % args.ckpt_interval == 0:
            ckpt_path = run_dir / f"checkpoint_step{step}.pt"
            torch.save({
                "step": step,
                "policy": trainer.policy.state_dict(),
                "critic": trainer.critic.state_dict(),
                "log_alpha": trainer.log_alpha.detach().cpu(),
                "lambda": trainer.lambda_,
                "args": vars(args),
            }, ckpt_path)
            last_ckpt_step = step
            logger.info("[ckpt] step=%d wrote %s", step, ckpt_path)

        # ---- wall-clock budget ----
        if args.max_wall_seconds > 0 and time.time() - t_start > args.max_wall_seconds:
            logger.warning("Hit wall-clock budget (%ds), stopping at step %d",
                           args.max_wall_seconds, step)
            break

    # ---- final eval & checkpoint ----
    final_eval = evaluate(
        trainer, denoiser, env, args.final_eval_episodes, args.d_a, args.device,
        use_dict_path=needs_obs_dicts,
    )
    eval_w.writerow([
        "final", final_eval["n_episodes"], final_eval["success_rate"],
        final_eval["mean_return"], final_eval["mean_length"],
    ])
    eval_f.flush()
    logger.info("[final-eval] sr=%.3f mean_return=%.3f n=%d",
                final_eval["success_rate"], final_eval["mean_return"],
                final_eval["n_episodes"])

    final_ckpt = run_dir / "checkpoint.pt"
    torch.save({
        "step": "final",
        "policy": trainer.policy.state_dict(),
        "critic": trainer.critic.state_dict(),
        "log_alpha": trainer.log_alpha.detach().cpu(),
        "lambda": trainer.lambda_,
        "args": vars(args),
        "final_eval": final_eval,
    }, final_ckpt)
    with open(run_dir / "final_summary.json", "w") as f:
        json.dump(
            {
                "final_eval": final_eval,
                "wall_seconds": time.time() - t_start,
                "n_episodes": n_eps,
                "n_success": n_success,
                "training_success_rate": n_success / max(n_eps, 1),
                "vocab_K": vocab.K,
                "lambda_final": trainer.lambda_,
            },
            f, indent=2,
        )

    metrics_f.close()
    eval_f.close()
    env.close()
    logger.info("Run complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
