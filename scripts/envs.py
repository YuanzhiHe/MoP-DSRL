"""Environment adapters for the MoP-DSRL training driver.

Two environment paths are supported:

  ``synthetic``  — a deterministic CPU-only environment used for smoke tests
                   and unit-level validation.  No external deps.

  ``libero``     — a LIBERO benchmark suite via robosuite/mujoco.  Requires
                   ``libero`` to be importable and the appropriate demo /
                   checkpoint paths to be set via env vars.

Both adapters expose a uniform interface used by ``scripts/train.py``:

  env.reset() -> obs (np.ndarray, shape [obs_dim])
  env.step(action) -> (obs, reward, done, info)
  env.close()
  env.obs_dim
  env.action_dim       (per-step, equals d_a)

The synthetic env's reward is a smooth function of the executed action
chunk's first action vs. an obs-conditioned target, so SAC actually has
something to optimize during smoke tests.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic environment (smoke tests / non-GPU validation)
# ---------------------------------------------------------------------------


class SyntheticEnv:
    """Toy MDP used to validate the SAC update loop end-to-end on CPU.

    obs_t in R^{obs_dim} drifts under a fixed linear dynamics matrix; the
    reward is the negative L2 distance between the executed action's first
    step and an obs-conditioned target ``A @ obs_t``.  This gives the SAC
    actor a useful gradient signal so we can validate that loss curves and
    cluster histograms behave sensibly during smoke tests.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 7,
        episode_steps: int = 50,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_steps = episode_steps
        self._rng = np.random.default_rng(seed)
        self._A = 0.05 * self._rng.standard_normal((obs_dim, action_dim))
        self._W = 0.99 * np.eye(obs_dim) + 0.01 * self._rng.standard_normal(
            (obs_dim, obs_dim)
        )
        self._t = 0
        self._obs = self._rng.standard_normal(obs_dim).astype(np.float32)

    def reset(self) -> np.ndarray:
        self._t = 0
        self._obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        return self._obs.copy()

    def step(self, action: np.ndarray):
        target = self._obs @ self._A
        reward = float(-np.linalg.norm(action - target))
        # Linear dynamics with small action perturbation.
        proj = np.zeros(self.obs_dim, dtype=np.float32)
        proj[: self.action_dim] = action.astype(np.float32)
        self._obs = (self._W @ self._obs + 0.1 * proj).astype(np.float32)
        self._obs += 0.01 * self._rng.standard_normal(self.obs_dim).astype(np.float32)
        self._t += 1
        done = self._t >= self.episode_steps
        return self._obs.copy(), reward, bool(done), {"success": False}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# LIBERO environment adapter
# ---------------------------------------------------------------------------


def _ensure_libero_paths() -> None:
    """Add LIBERO source / config paths to sys.path/env if user has set them."""
    src = os.environ.get("LIBERO_SRC", "")
    if src and src not in sys.path and Path(src).is_dir():
        sys.path.insert(0, src)
    cfg = os.environ.get("LIBERO_CONFIG_PATH", "")
    if cfg:
        os.environ.setdefault("LIBERO_CONFIG_PATH", cfg)


class LiberoEnv:
    """Thin wrapper around a single-task LIBERO env.

    Provides a flat-vector observation by concatenating the proprioceptive
    state with a low-dim image embedding (default: just the proprioceptive
    state, since SAC over the noise-space policy operates on a feature
    vector and pi_0.5/DP-T handle pixels internally via their own encoders).
    """

    def __init__(
        self,
        suite: str,
        task_id: int,
        seed: int = 0,
        max_steps: int = 300,
        camera: str = "agentview",
    ):
        _ensure_libero_paths()
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        self.suite = suite
        self.task_id = int(task_id)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self._BenchmarkCls = libero_benchmark
        self._OffScreenRenderEnv = OffScreenRenderEnv

        bench = libero_benchmark.get_benchmark(suite)()
        task = bench.get_task(self.task_id)
        self._task = task
        self._env = OffScreenRenderEnv(
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera=camera,
            **task.env_kwargs,
        )
        self._env.seed(self.seed * 1000 + self.task_id)
        obs = self._env.reset()
        self._obs = self._flatten_obs(obs)
        self.obs_dim = int(self._obs.shape[0])
        self.action_dim = int(self._task.action_dim) if hasattr(self._task, "action_dim") else 7
        self._t = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_obs(obs: dict) -> np.ndarray:
        """Concatenate every numerical 1-D field in the obs dict.

        We deliberately drop image arrays here — SAC's noise-space policy
        operates on the same proprio-feature vector that the diffusion /
        flow backbones already condition on.  Image conditioning happens
        inside the velocity field, not the SAC policy.
        """
        parts = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray) and v.ndim == 1:
                parts.append(v.astype(np.float32))
        if not parts:
            raise RuntimeError("LIBERO obs dict had no 1-D numerical fields")
        return np.concatenate(parts, axis=0)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._t = 0
        obs = self._env.reset()
        self._obs = self._flatten_obs(obs)
        return self._obs.copy()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.action_dim:
            pad = np.zeros(self.action_dim - action.shape[0], dtype=np.float32)
            action = np.concatenate([action, pad], axis=0)
        action = action[: self.action_dim]
        obs, reward, done, info = self._env.step(action)
        self._obs = self._flatten_obs(obs)
        self._t += 1
        if self._t >= self.max_steps:
            done = True
        return self._obs.copy(), float(reward), bool(done), dict(info)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_env(name: str, **kwargs):
    """Construct an environment by name.

    Recognized values:
      ``synthetic``        — SyntheticEnv (CPU-only smoke tests).
      ``libero_goal``      — LIBERO-Goal suite.
      ``libero_spatial``   — LIBERO-Spatial.
      ``libero_object``    — LIBERO-Object.
      ``libero_long``      — LIBERO-10 (called "Long" in MoP-DSRL spec).
    """
    if name == "synthetic":
        return SyntheticEnv(**kwargs)
    if name.startswith("libero_"):
        # Map libero_long → libero_10 (the official LIBERO suite name).
        suite = "libero_10" if name == "libero_long" else name
        return LiberoEnv(suite=suite, **kwargs)
    raise ValueError(f"Unknown env name: {name!r}")
