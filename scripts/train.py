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
    p.add_argument("--target_entropy", type=float, default=float("nan"),
                   help="SAC entropy target.  Default NaN = -dim(z) per method_spec §5; "
                        "for pi_0.5 (z_dim=350) the spec value is too aggressive and "
                        "causes alpha→0 policy collapse — use a less negative value "
                        "like -30 (= -z_dim/12).")
    p.add_argument("--clip_radius_sigma_factor", type=float, default=4.0,
                   help="L2 clip radius factor: r_z = factor * sqrt(z_dim).  "
                        "Spec default 4.0; reduce to 1.5 for pi_0.5 where the noise "
                        "input has limited dynamic range.")

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

    # R_mode performance (pi_0.5 path only — toy/toyflow ignore these)
    p.add_argument("--rmode_subsample", type=int, default=8,
                   help="For pi_0.5: number of held-out obs sampled per R_mode "
                        "evaluation (instead of full batch_size).  Cost is "
                        "O(rmode_subsample * n_on) pi_0.5 forwards per actor update.")
    p.add_argument("--rmode_every", type=int, default=4,
                   help="For pi_0.5: only evaluate R_mode every K actor updates.  "
                        "The most recent KL value is reused for the skipped updates.")

    # Action chunking — pi_0.5 paper executes ~10 actions per inference (it's a
    # chunk-predicting policy).  Executing only the first action of each H=50
    # chunk causes severe distribution shift vs the model's training regime,
    # which manifests as ~0% success rate even in pure BC mode.
    p.add_argument("--n_action_steps", type=int, default=10,
                   help="Number of actions from each action chunk to execute "
                        "before re-inferring.  pi_0.5 default = 10.")

    # Eval / I/O
    p.add_argument("--eval_interval", type=int, default=25_000)
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Episodes per eval pass during training.  Final eval uses --final_eval_episodes.")
    p.add_argument("--final_eval_episodes", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=100_000)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--min_c_ref_entropy", type=float, default=0.05,
                   help="Abort a lambda>0 run whose reference mode histogram is "
                        "this close to a point mass (nats). Guards against the "
                        "inert-R_mode failure seen across the LIBERO campaign.")
    p.add_argument("--allow_degenerate_c_ref", action="store_true",
                   help="Proceed despite a degenerate C_ref (measuring it on purpose).")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="If a checkpoint_step*.pt exists in --run_dir, load the "
                        "latest one and continue from its step (CSV rows are "
                        "appended after truncating to that step).  --no-resume "
                        "starts fresh and overwrites the run dir.")

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

    # pi_0.5 with KV cache at batch=50 needs >32GB on a 5090.  Chunk the per-obs
    # n_ref samples through the model in groups of `pi05_chunk`.
    pi05_chunk = int(os.environ.get("MOP_DSRL_PI05_BATCH", "4"))
    _i_seen = [0]

    if use_dict_path:
        repeat = denoiser.velocity_field.repeat_batch

        def _sample_at(i: int, n: int) -> np.ndarray:
            batch_one_cpu = obs_dicts[i]
            # Move to GPU on demand (obs_dicts are kept on CPU to save memory).
            batch_one = {
                k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                for k, v in batch_one_cpu.items()
            }
            outs = []
            for j in range(0, n, pi05_chunk):
                m = min(pi05_chunk, n - j)
                batch_m = repeat(batch_one, m)
                with torch.no_grad():
                    z = torch.randn(m, z_dim, device=device, dtype=torch.float32)
                    a = denoiser.denoise_with_obs_dict(batch_m, z)  # with_grad=False default
                outs.append(a.cpu().numpy().astype(np.float64))
                del batch_m, z, a
            # Empty cache every 16 obs so peak doesn't grow with the loop.
            _i_seen[0] += 1
            if _i_seen[0] % 16 == 0:
                torch.cuda.empty_cache()
            return np.concatenate(outs, axis=0)
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
             env=None, use_dict_path: bool = False,
             unnormalize: bool = True, d_a: int = 7) -> torch.Tensor:
    """Denoise wrapper that picks flat vs dict path based on backbone capability.

    For pi_0.5 the policy outputs *normalized* actions (mean/std).  When the
    caller is going to feed them to env.step (``unnormalize=True``, default),
    we run the action through lerobot's unnormalizer postprocessor so the
    result is in physical-action units LIBERO expects.

    The denoiser returns ``[B, H*d_a]`` flat — but lerobot's unnormalizer
    expects ``[..., d_a]`` (action dim in the last position).  We reshape to
    ``[B, H, d_a]``, post-process, then flatten back to ``[B, H*d_a]``.
    """
    if use_dict_path and env is not None and hasattr(denoiser, "denoise_with_obs_dict"):
        vf = denoiser.velocity_field
        # Use the OFFICIAL lerobot env_pre → policy_pre chain.  Verified by
        # diag_our_backbone_eval_policy.py: backbone+this chain gets 80–100%
        # on libero_goal task[0]; manually-built batch was 0%.
        if hasattr(vf, "prepare_libero_batch_from_vec") and hasattr(env, "_last_raw_vec_obs"):
            batch = vf.prepare_libero_batch_from_vec(
                env._last_raw_vec_obs, env.task_instruction,
            )
        else:
            batch = vf.prepare_libero_batch(
                env.last_obs_dict, env.task_instruction,
            )
        if obs_t.shape[0] > 1:
            batch = denoiser.velocity_field.repeat_batch(batch, obs_t.shape[0])
        a = denoiser.denoise_with_obs_dict(batch, z)  # [B, H*d_a]
        if unnormalize:
            try:
                post = denoiser.velocity_field._get_postprocessor()
                # Find PolicyAction class (cached after first lookup).
                PolicyAction = getattr(_denoise, "_PolicyAction", None)
                if PolicyAction is None:
                    from lerobot.processor import PolicyAction as _PA
                    PolicyAction = _PA
                    _denoise._PolicyAction = _PA
                B = a.shape[0]
                H = a.shape[1] // d_a
                a3 = a.view(B, H, d_a)
                pa = PolicyAction(a3)
                pa_out = post(pa)
                a3_post = pa_out if isinstance(pa_out, torch.Tensor) else pa_out.action
                a = a3_post.reshape(B, H * d_a)
            except Exception as e:
                import logging as _log
                _log.warning("postproc(action) failed: %s; using normalized action", e)
        return a
    return denoiser.denoise(obs_t, z)


def _fmt(x, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def evaluate(
    trainer: SACTrainer,
    denoiser,
    env,
    n_episodes: int,
    d_a: int,
    device: str,
    deterministic: bool = False,
    use_dict_path: bool = False,
    n_action_steps: int = 10,
    H: int = 50,
    vocab: KMeansVocab | None = None,
    C_ref_marginal: np.ndarray | None = None,
) -> dict:
    """Run ``n_episodes`` rollouts and return mean success / return / length.

    Executes ``n_action_steps`` actions from each predicted chunk before
    re-inferring — matches pi_0.5's training-time action-chunking regime.

    When ``vocab`` is given, also reports the behavioural-mode metrics used by
    the mode-preservation literature, which success rate alone cannot show (a
    policy that collapses onto one mode can still score 1.0):

      H_modes   entropy of the empirical mode histogram, nats (max ln K)
      mc_at_0.8 number of modes retained with mass >= 0.8/K   (BMD's mc@tau)
      SR_M      mode-aggregated success: mean success over the modes actually
                visited, so a collapsed policy cannot hide behind one good mode
      tv_to_ref total-variation distance to the frozen backbone's own mode
                profile -- the quantity KL-to-reference is meant to keep small,
                and what distinguishes it from an entropy/MI objective (which
                drives the histogram toward uniform instead)
    """
    successes = 0
    returns = []
    lengths = []
    info = {}
    episode_modes: list[int] = []   # dominant vocab cluster per episode
    episode_success: list[int] = []
    chunk_mode_counts = np.zeros(vocab.K, dtype=np.int64) if vocab is not None else None
    # Try to reach pi_0.5's internal policy.reset() for clearing the action queue.
    inner_policy = None
    try:
        inner_policy = denoiser.velocity_field._policy
    except Exception:
        pass
    for _ in range(n_episodes):
        obs = env.reset()
        if inner_policy is not None and hasattr(inner_policy, "reset"):
            inner_policy.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        ep_mode_counts = (np.zeros(vocab.K, dtype=np.int64)
                          if chunk_mode_counts is not None else None)
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    mean, _ = trainer.policy.forward(obs_t)
                    z = mean
                else:
                    z, _ = trainer._sample_z(obs_t)
                a = _denoise(denoiser, obs_t, z, env=env, use_dict_path=use_dict_path, d_a=d_a)
            chunk = a.squeeze(0).cpu().numpy().reshape(H, d_a)
            if chunk_mode_counts is not None:
                cid = int(vocab.assign(chunk.reshape(1, -1))[0])
                chunk_mode_counts[cid] += 1
                ep_mode_counts[cid] += 1
            for i in range(min(n_action_steps, H)):
                obs, reward, done, info = env.step(chunk[i])
                ep_return += float(reward)
                ep_len += 1
                if done:
                    break
        succ = int(info.get("success", False))
        successes += succ
        returns.append(ep_return)
        lengths.append(ep_len)
        if ep_mode_counts is not None and ep_mode_counts.sum() > 0:
            episode_modes.append(int(ep_mode_counts.argmax()))
            episode_success.append(succ)
    out = {
        "success_rate": successes / max(n_episodes, 1),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": n_episodes,
    }
    if chunk_mode_counts is not None and chunk_mode_counts.sum() > 0:
        K = vocab.K
        hist = chunk_mode_counts / chunk_mode_counts.sum()
        out["H_modes"] = float(-(hist * np.log(hist + 1e-12)).sum())
        out["H_modes_max"] = float(np.log(K))
        out["mc_at_0.8"] = int((hist >= 0.8 / K).sum())
        modes = np.asarray(episode_modes)
        succ = np.asarray(episode_success, dtype=float)
        per_mode = [succ[modes == k].mean() for k in range(K) if (modes == k).any()]
        out["SR_M"] = float(np.mean(per_mode)) if per_mode else 0.0
        out["n_modes_visited"] = int(len(per_mode))
        if C_ref_marginal is not None:
            ref = np.asarray(C_ref_marginal, dtype=float)
            ref = ref / max(ref.sum(), 1e-12)
            out["tv_to_ref"] = float(0.5 * np.abs(hist - ref).sum())
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    # Process-start timestamp for the SLURM wall-budget check. This MUST be
    # taken before the (slow) backbone load + held-out-obs + C_ref build, so the
    # --max_wall_seconds budget bounds the TOTAL job wall (startup + training).
    # Measuring from the post-C_ref training t_start instead would let a ~40-min
    # startup push the job past the SLURM hard limit → TIMEOUT kill (no graceful
    # checkpoint, no resubmit).
    proc_start = time.time()

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
        target_entropy=args.target_entropy,
        clip_radius_sigma_factor=args.clip_radius_sigma_factor,
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

    # Keep held-out obs dicts on CPU (each batch is ~1 MB of images); only move
    # to GPU just before pi_0.5 forward.  1000 obs × 1 MB ≈ 1 GB CPU RAM, vs
    # ~3 GB if kept on GPU (after dtype upcasting), which would compete with
    # the 15 GB of pi_0.5 weights + KV cache.
    def _to_cpu(batch: dict) -> dict:
        return {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in batch.items()}

    for i in range(args.n_obs):
        flat = env.reset()
        held_out_obs_list.append(flat)
        if needs_obs_dicts:
            try:
                vf = denoiser.velocity_field
                if hasattr(vf, "prepare_libero_batch_from_vec") and hasattr(env, "_last_raw_vec_obs"):
                    batch = vf.prepare_libero_batch_from_vec(
                        env._last_raw_vec_obs, env.task_instruction,
                    )
                else:
                    batch = vf.prepare_libero_batch(
                        env.last_obs_dict, env.task_instruction,
                    )
                # Move to CPU before stashing; pi_0.5 forward will copy back.
                held_out_obs_dicts.append(_to_cpu(batch))
            except Exception as e:
                logger.error(
                    "prepare_libero_batch failed at #%d/%d: %s.  pi_0.5 path needs dict obs; STOPPING.",
                    i, args.n_obs, e,
                )
                # Pi_0.5 flat-tensor denoise is BROKEN (bind_obs doesn't set the
                # prefix KV cache), so falling back to flat path would crash
                # later anyway.  Better to abort here with a clear message so
                # the operator can lower n_obs or free GPU memory.
                raise SystemExit(
                    f"pi_0.5 requires per-obs dict batches for R_mode; built only "
                    f"{i} of {args.n_obs}.  Reduce --n_obs (e.g. 200) or free GPU memory."
                )
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

        # Subsample R_mode evaluation: B_mode held-out obs × n_on z's per actor update.
        # Cost = B_mode * n_on * (pi_0.5 forward cost).  With defaults (8 * 4 = 32
        # forwards, ~110ms each on a 5090) ≈ 3.5s per R_mode evaluation.
        _rmode_state = {"step": 0, "last_value": torch.zeros((), device=args.device)}

        def _pi05_mode_regularizer(o, vocab_, C_ref_batch_):
            """Compute R_mode on a sub-sample of held-out (flat, dict) pairs.

            For pi_0.5, evaluating R_mode every actor update with full batch is
            prohibitively expensive (≈28s/update with B=64, n_on=4).  We
            sub-sample ``--rmode_subsample`` held-out obs per evaluation and
            only re-evaluate every ``--rmode_every`` updates; in-between, the
            most recent KL value is reused (still differentiable wrt phi via
            replay-batch coupling, so SAC actor still updates).
            """
            _rmode_state["step"] += 1
            B_mode = int(args.rmode_subsample)
            if _rmode_state["step"] % max(1, int(args.rmode_every)) != 1 \
               and _rmode_state["step"] != 1:
                return _rmode_state["last_value"]

            n_on = trainer.cfg.n_on
            idx = rng.integers(0, len(held_out_obs_dicts), size=B_mode)
            held_flat = _held_flat_gpu[idx]                        # [B_mode, obs_dim]
            held_rep = held_flat.repeat_interleave(n_on, dim=0)    # [B_mode*n_on, obs_dim]
            z_rep, _ = trainer._sample_z(held_rep)
            # Chunk through pi_0.5 to keep peak memory bounded (32GB at batch=8
            # × M=10 Euler is already ~24GB; halving keeps headroom for
            # SAC critic/policy graphs).
            actions = []
            chunk = int(os.environ.get("MOP_DSRL_PI05_BATCH", "4"))
            for i_local in range(B_mode):
                z_slice_full = z_rep[i_local * n_on:(i_local + 1) * n_on]
                # Move this obs dict to GPU (we stored on CPU to save memory).
                batch_cpu = held_out_obs_dicts[int(idx[i_local])]
                batch_gpu = {
                    k: (v.to(args.device, non_blocking=True) if hasattr(v, "to") else v)
                    for k, v in batch_cpu.items()
                }
                for j in range(0, n_on, chunk):
                    m = min(chunk, n_on - j)
                    batch_n = denoiser.velocity_field.repeat_batch(batch_gpu, m)
                    z_chunk = z_slice_full[j:j + m]
                    # with_grad=True → autograd through z; this is the gradient
                    # path that lets R_mode actually constrain the policy.
                    actions.append(
                        denoiser.denoise_with_obs_dict(batch_n, z_chunk, with_grad=True)
                    )
            a_rep = torch.cat(actions, dim=0)
            median = torch.as_tensor(vocab_.median, dtype=a_rep.dtype, device=args.device)
            mad = torch.as_tensor(vocab_.mad, dtype=a_rep.dtype, device=args.device)
            a_norm = (a_rep - median) / mad
            centroids = torch.as_tensor(vocab_.centroids, dtype=a_rep.dtype, device=args.device)
            soft = soft_assign_distances(a_norm, centroids, temperature=1.0)
            C_ref_held = _C_ref_gpu[idx]
            kl = mode_kl_from_assignments(
                soft, C_ref_held, n_on, trainer.cfg.laplace_alpha()
            )
            _rmode_state["last_value"] = kl.detach()
            return kl
        trainer._mode_regularizer = _pi05_mode_regularizer
        logger.info(
            "pi_0.5 R_mode: subsample=%d, every=%d (≈%d forwards per actor update)",
            args.rmode_subsample, args.rmode_every,
            args.rmode_subsample * trainer.cfg.n_on // max(1, args.rmode_every),
        )
    c_ref_entropy = float(
        -(C_ref_full * np.log(C_ref_full + 1e-12)).sum(axis=-1).mean()
    )
    logger.info("C_ref shape=%s entropy=%.3f (max %.3f)", C_ref_full.shape,
                c_ref_entropy, float(np.log(args.K)))
    # Gate. R_mode = KL(C_phi || C_ref) can only do work if the backbone is
    # actually multimodal at these observations. Across the July/August LIBERO
    # campaign C_ref entropy was 0.000 in 45 of 47 runs -- every reference
    # histogram was a point mass, so the regularizer had nothing to preserve and
    # r_mode stayed exactly 0 for ~18k updates per run while costing ~3x the
    # wall time. Refuse to burn GPU on that again: a lambda>0 run must start from
    # a non-degenerate reference. Override with --allow_degenerate_c_ref when
    # deliberately measuring the degenerate case.
    if args.lambda_ > 0 and c_ref_entropy < args.min_c_ref_entropy:
        msg = (
            f"C_ref entropy {c_ref_entropy:.4f} < {args.min_c_ref_entropy} nats: the "
            f"frozen backbone is effectively unimodal at these observations, so "
            f"R_mode is inert and lambda={args.lambda_} buys nothing but wall time. "
            f"Use a task/suite with genuinely multiple solutions, or pass "
            f"--allow_degenerate_c_ref to proceed anyway."
        )
        if args.allow_degenerate_c_ref:
            logger.warning("%s [proceeding: --allow_degenerate_c_ref]", msg)
        else:
            raise SystemExit(f"ABORT: {msg}")

    # ---- Replay buffer & loop ----
    buffer = ReplayBuffer(
        capacity=cfg.replay_capacity, obs_dim=obs_dim, z_dim=z_dim, seed=args.seed
    )

    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"

    # ---- resume: load latest checkpoint_step*.pt and continue ----
    start_step = 0
    if args.resume:
        ckpts = sorted(
            run_dir.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_step", "")),
        )
        if ckpts:
            ckpt_path = ckpts[-1]
            ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            start_step = int(ck["step"])
            trainer.policy.load_state_dict(ck["policy"])
            trainer.critic.load_state_dict(ck["critic"])
            # Older checkpoints lack target-critic / optimizer state; restore
            # what's there, fall back to sensible defaults otherwise.
            if "critic_target" in ck:
                trainer.critic_target.load_state_dict(ck["critic_target"])
            else:
                trainer.critic_target.load_state_dict(trainer.critic.state_dict())
            with torch.no_grad():
                trainer.log_alpha.copy_(ck["log_alpha"].to(args.device))
            if "lambda" in ck:
                trainer._lambda = float(ck["lambda"])  # lambda_ is a read-only property
            for opt_key, opt in (("actor_opt", trainer.actor_opt),
                                 ("critic_opt", trainer.critic_opt),
                                 ("alpha_opt", trainer.alpha_opt)):
                if opt_key in ck:
                    opt.load_state_dict(ck[opt_key])
            logger.info("[resume] loaded %s → continuing from step %d (opts=%s)",
                        ckpt_path.name, start_step, "restored" if "actor_opt" in ck else "fresh")

    # CSV files: on resume, keep only rows with step <= start_step, then append;
    # otherwise truncate and write a fresh header.
    def _truncate_csv(path: Path, header: list, keep_to_step: int) -> None:
        kept = []
        if path.exists():
            with open(path, newline="") as f:
                r = csv.reader(f)
                rows = list(r)
            for row in rows[1:]:  # skip old header
                if not row:
                    continue
                try:
                    s = int(row[0])
                except (ValueError, IndexError):
                    continue  # drop non-numeric step rows (e.g. "final")
                if s <= keep_to_step:
                    kept.append(row)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(kept)

    metrics_header = [
        "step", "wall_seconds", "ep_return", "ep_length", "success",
        "critic_loss", "actor_loss_q", "actor_loss_total",
        "r_mode", "alpha", "lambda",
    ]
    eval_header = ["step", "n_episodes", "success_rate", "mean_return", "mean_length",
                   "H_modes", "mc_at_0.8", "SR_M", "n_modes_visited", "tv_to_ref"]
    # Marginal of the reference histogram: the frozen backbone's overall mode
    # profile, which tv_to_ref measures fidelity to.
    C_ref_marginal = C_ref_full.mean(axis=0) if C_ref_full is not None else None
    if start_step > 0:
        _truncate_csv(metrics_path, metrics_header, start_step)
        _truncate_csv(eval_path, eval_header, start_step)
        metrics_f = open(metrics_path, "a", newline="")
        eval_f = open(eval_path, "a", newline="")
        metrics_w = csv.writer(metrics_f)
        eval_w = csv.writer(eval_f)
    else:
        metrics_f = open(metrics_path, "w", newline="")
        metrics_w = csv.writer(metrics_f)
        metrics_w.writerow(metrics_header)
        eval_f = open(eval_path, "w", newline="")
        eval_w = csv.writer(eval_f)
        eval_w.writerow(eval_header)

    t_start = time.time()
    obs = obs0
    ep_return = 0.0
    ep_len = 0
    n_eps = 0
    n_success = 0
    last_log = start_step
    rng = np.random.default_rng(args.seed + 1)
    last_ckpt_step = start_step
    wall_stopped = False

    for step in range(start_step, args.total_env_steps):
        # ---- collect transition (one inference → n_action_steps env actions) ----
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        with torch.no_grad():
            if step < args.warmup_env_steps and not args.no_coupling_endpoint:
                z = torch.randn(1, z_dim, device=args.device, dtype=obs_t.dtype)
            else:
                z, _ = trainer._sample_z(obs_t)
            a_chunk = _denoise(denoiser, obs_t, z, env=env, use_dict_path=needs_obs_dicts, d_a=args.d_a)
            chunk_2d = a_chunk.squeeze(0).cpu().numpy().reshape(args.H, args.d_a)

        # Execute up to n_action_steps actions from the chunk, accumulating reward.
        chunk_reward_sum = 0.0
        n_exec = 0
        done = False
        info = {}
        for i in range(min(args.n_action_steps, args.H)):
            obs_next, reward, done, info = env.step(chunk_2d[i])
            n_exec += 1
            chunk_reward_sum += float(reward)
            if done:
                break

        if args.reward_mode == "sparse":
            r_eff = chunk_reward_sum if done else 0.0
        else:
            r_eff = chunk_reward_sum
        if args.pref_reward and args.pref_flip_rate > 0:
            if rng.uniform() < args.pref_flip_rate:
                r_eff = -r_eff

        buffer.push(obs, z.squeeze(0).cpu().numpy(), r_eff, obs_next, done)
        obs = obs_next
        ep_return += chunk_reward_sum
        ep_len += n_exec

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
                n_action_steps=args.n_action_steps, H=args.H,
                vocab=vocab, C_ref_marginal=C_ref_marginal,
            )
            eval_w.writerow([
                step, ev["n_episodes"], ev["success_rate"],
                ev["mean_return"], ev["mean_length"],
                ev.get("H_modes", ""), ev.get("mc_at_0.8", ""), ev.get("SR_M", ""),
                ev.get("n_modes_visited", ""), ev.get("tv_to_ref", ""),
            ])
            eval_f.flush()
            logger.info("[eval] step=%d sr=%.3f mean_return=%.3f n=%d "
                        "H_modes=%s mc@0.8=%s SR_M=%s tv_to_ref=%s",
                        step, ev["success_rate"], ev["mean_return"], ev["n_episodes"],
                        _fmt(ev.get("H_modes")), ev.get("mc_at_0.8", "-"),
                        _fmt(ev.get("SR_M")), _fmt(ev.get("tv_to_ref")))

        # ---- checkpoint ----
        if args.ckpt_interval > 0 and step > 0 and step % args.ckpt_interval == 0:
            ckpt_path = run_dir / f"checkpoint_step{step}.pt"
            tmp_path = ckpt_path.with_suffix(".pt.tmp")
            torch.save({
                "step": step,
                "policy": trainer.policy.state_dict(),
                "critic": trainer.critic.state_dict(),
                "critic_target": trainer.critic_target.state_dict(),
                "log_alpha": trainer.log_alpha.detach().cpu(),
                "lambda": trainer.lambda_,
                "actor_opt": trainer.actor_opt.state_dict(),
                "critic_opt": trainer.critic_opt.state_dict(),
                "alpha_opt": trainer.alpha_opt.state_dict(),
                "args": vars(args),
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)  # atomic: never leave a half-written ckpt
            last_ckpt_step = step
            logger.info("[ckpt] step=%d wrote %s", step, ckpt_path)

        # ---- wall-clock budget (measured from process start, incl. startup) ----
        if args.max_wall_seconds > 0 and time.time() - proc_start > args.max_wall_seconds:
            logger.warning("Hit wall-clock budget (%ds since proc start), stopping at step %d",
                           args.max_wall_seconds, step)
            wall_stopped = True
            break

    # ---- wall-stop: save a resumable checkpoint and exit WITHOUT final_summary ----
    # A job killed by the SLURM wall limit must NOT write final_summary.json, or the
    # resume/skip logic (in run_local.py and the self-resubmitting SLURM wrapper)
    # would treat this partial run as complete. Save a checkpoint at the current step
    # so the next job continues from here, then exit cleanly.
    if wall_stopped:
        ckpt_path = run_dir / f"checkpoint_step{step}.pt"
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "step": step,
            "policy": trainer.policy.state_dict(),
            "critic": trainer.critic.state_dict(),
            "critic_target": trainer.critic_target.state_dict(),
            "log_alpha": trainer.log_alpha.detach().cpu(),
            "lambda": trainer.lambda_,
            "actor_opt": trainer.actor_opt.state_dict(),
            "critic_opt": trainer.critic_opt.state_dict(),
            "alpha_opt": trainer.alpha_opt.state_dict(),
            "args": vars(args),
        }, tmp_path)
        os.replace(tmp_path, ckpt_path)
        logger.warning("[wall-stop] saved resumable checkpoint at step %d → %s "
                       "(no final_summary; resubmit to continue)", step, ckpt_path)
        metrics_f.close()
        eval_f.close()
        env.close()
        return 0

    # ---- final eval & checkpoint (only reached on true completion) ----
    final_eval = evaluate(
        trainer, denoiser, env, args.final_eval_episodes, args.d_a, args.device,
        use_dict_path=needs_obs_dicts,
        n_action_steps=args.n_action_steps, H=args.H,
        vocab=vocab, C_ref_marginal=C_ref_marginal,
    )
    eval_w.writerow([
        "final", final_eval["n_episodes"], final_eval["success_rate"],
        final_eval["mean_return"], final_eval["mean_length"],
        final_eval.get("H_modes", ""), final_eval.get("mc_at_0.8", ""),
        final_eval.get("SR_M", ""), final_eval.get("n_modes_visited", ""),
        final_eval.get("tv_to_ref", ""),
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
