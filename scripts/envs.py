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
        # Use lerobot's LIBERO factory — it sets control_mode='relative' which
        # is what pi_0.5 was trained on (stock OffScreenRenderEnv defaults to
        # absolute OSC → ~0% BC).  Verified by diag_lerobot_libero_env.py:
        # native pi_0.5 + this env path gets 100% on libero_goal task[0].
        from lerobot.envs.configs import LiberoEnv as LeRobotLiberoEnvCfg
        from lerobot.envs.factory import make_env

        self.suite = suite
        self.task_id = int(task_id)
        self.max_steps = int(max_steps)
        self.seed = int(seed)

        env_cfg = LeRobotLiberoEnvCfg(
            task=suite, task_ids=[self.task_id], control_mode="relative",
        )
        envs_by_suite = make_env(env_cfg, n_envs=1, use_async_envs=False)
        # envs_by_suite[<suite>][<task_id>] → gym SyncVectorEnv with 1 env
        suite_envs = envs_by_suite[list(envs_by_suite.keys())[0]]
        self._vec = suite_envs[list(suite_envs.keys())[0]]
        self._env_cfg = env_cfg

        # Get task instruction from LIBERO benchmark (lerobot wraps the same).
        bench = libero_benchmark.get_benchmark(suite)()
        task = bench.get_task(self.task_id)
        self._task = task
        instr = getattr(task, "language", None) or task.name.replace("_", " ")
        self.task_instruction = instr

        # Seed the underlying vec env's RNG.
        try:
            self._vec.reset(seed=self.seed * 1000 + self.task_id)
        except TypeError:
            self._vec.reset()

        obs_vec, _info = self._vec.reset()
        self._last_raw_obs = self._unbatch(obs_vec)
        self._obs = self._flatten_obs(self._last_raw_obs)
        self.obs_dim = int(self._obs.shape[0])
        self.action_dim = 7
        self._t = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _unbatch(d):
        """Recursively strip the leading [1, ...] batch dim and convert tensors → np.

        lerobot vec env emits nested dicts:
            {"pixels": {"image": [1,H,W,3], "image2": [1,H,W,3]},
             "robot_state": {"eef": {"pos": [1,3], "quat": [1,4], ...},
                             "gripper": {"qpos": [1,2], ...}, ...}}
        We flatten one level of batching everywhere.  Also synthesize
        flat aliases (`agentview_image`, `robot0_eef_pos`, ...) so legacy
        consumers of the LIBERO obs schema keep working.
        """
        import numpy as _np
        def _strip(v):
            if isinstance(v, dict):
                return {k: _strip(x) for k, x in v.items()}
            if hasattr(v, "cpu"):
                v = v.cpu().numpy()
            if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1:
                v = v[0]
            return v
        flat_nested = _strip(d)
        out = dict(flat_nested)
        # ---- aliases so old key names still resolve ----
        try:
            pixels = flat_nested.get("pixels", {}) or {}
            if "image" in pixels:
                out["agentview_image"] = pixels["image"]
            if "image2" in pixels:
                out["robot0_eye_in_hand_image"] = pixels["image2"]
            rs = flat_nested.get("robot_state", {}) or {}
            eef = rs.get("eef", {}) or {}
            if "pos" in eef:
                out["robot0_eef_pos"] = _np.asarray(eef["pos"]).astype(_np.float32)
            if "quat" in eef:
                out["robot0_eef_quat"] = _np.asarray(eef["quat"]).astype(_np.float32)
            grip = rs.get("gripper", {}) or {}
            if "qpos" in grip:
                out["robot0_gripper_qpos"] = _np.asarray(grip["qpos"]).astype(_np.float32)
        except Exception:
            pass
        return out

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
            if isinstance(v, np.ndarray):
                if v.ndim == 1:
                    parts.append(v.astype(np.float32))
                elif v.ndim == 0:  # scalar
                    parts.append(v.astype(np.float32).reshape(1))
                # Skip 3D images, 2D segmentation masks etc.
            elif isinstance(v, (int, float, np.integer, np.floating)):
                parts.append(np.array([v], dtype=np.float32))
        if not parts:
            # Fallback: flatten everything that's numeric, even multi-dim.
            for k in sorted(obs.keys()):
                v = obs[k]
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    parts.append(v.astype(np.float32).reshape(-1))
        if not parts:
            raise RuntimeError(
                "LIBERO obs dict had no numerical fields. Keys: "
                + ", ".join(sorted(obs.keys()))
            )
        return np.concatenate(parts, axis=0)

    # ------------------------------------------------------------------
    @property
    def last_obs_dict(self) -> dict:
        """Most recent raw LIBERO obs dict (incl. images) for backbone preprocessing."""
        return self._last_raw_obs

    def reset(self) -> np.ndarray:
        self._t = 0
        obs_vec, _info = self._vec.reset()
        self._last_raw_obs = self._unbatch(obs_vec)
        self._obs = self._flatten_obs(self._last_raw_obs)
        return self._obs.copy()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.action_dim:
            pad = np.zeros(self.action_dim - action.shape[0], dtype=np.float32)
            action = np.concatenate([action, pad], axis=0)
        action = action[: self.action_dim]
        # vec env expects [B, action_dim]
        action_b = action[None, :].astype(np.float32)
        out = self._vec.step(action_b)
        if len(out) == 5:
            obs_vec, reward, terminated, truncated, info = out
            done = bool(terminated[0] or truncated[0])
        else:
            obs_vec, reward, done, info = out
            done = bool(np.asarray(done).reshape(-1)[0])
        reward = float(np.asarray(reward).reshape(-1)[0])
        # info is sometimes a list[dict]; normalize to a single dict.
        if isinstance(info, list) and info:
            info_d = dict(info[0])
        elif isinstance(info, dict):
            info_d = {k: (v[0] if hasattr(v, "__getitem__") and not isinstance(v, str) else v)
                      for k, v in info.items()}
        else:
            info_d = {}
        # lerobot uses "is_success"; map to the conventional "success" key
        # that our evaluate() / train.py reads.
        if "success" not in info_d:
            if "is_success" in info_d:
                info_d["success"] = bool(info_d["is_success"])
            elif "task_success" in info_d:
                info_d["success"] = bool(info_d["task_success"])
        self._last_raw_obs = self._unbatch(obs_vec)
        self._obs = self._flatten_obs(self._last_raw_obs)
        self._t += 1
        if self._t >= self.max_steps:
            done = True
        return self._obs.copy(), reward, done, info_d

    def close(self) -> None:
        try:
            self._vec.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LIBERO obs dict → pi_0.5 batch adapter
# ---------------------------------------------------------------------------


def libero_obs_to_pi05_batch(
    raw_obs: dict,
    task_instruction: str,
    device: str = "cuda",
    image_size: int = 224,
) -> dict:
    """Convert a LIBERO observation dict into pi_0.5's expected batch format.

    pi_0.5 (LeRobot) expects keys:
      - observation.images.image_*   [B, C, H, W] float in [0, 1]
      - observation.state            [B, state_dim]
      - task / language tokens       (filled in by lerobot's preprocessor)

    LIBERO emits:
      - agentview_image    [H, W, 3] uint8
      - robot0_eye_in_hand_image  [H, W, 3] uint8 (if camera enabled)
      - robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos  proprio
    """
    import torch
    import torch.nn.functional as F

    def _to_chw_float(img: np.ndarray) -> torch.Tensor:
        # uint8 HWC → float CHW in [0, 1]
        t = torch.from_numpy(img.copy()).float() / 255.0
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        return t

    images = {}
    if "agentview_image" in raw_obs:
        img = _to_chw_float(raw_obs["agentview_image"])
        if img.shape[-1] != image_size or img.shape[-2] != image_size:
            img = F.interpolate(img.unsqueeze(0), size=(image_size, image_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        images["observation.images.image"] = img.unsqueeze(0).to(device)
    if "robot0_eye_in_hand_image" in raw_obs:
        img = _to_chw_float(raw_obs["robot0_eye_in_hand_image"])
        if img.shape[-1] != image_size or img.shape[-2] != image_size:
            img = F.interpolate(img.unsqueeze(0), size=(image_size, image_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        images["observation.images.wrist_image"] = img.unsqueeze(0).to(device)

    # Proprio state: concatenate eef pos / quat / gripper.
    state_parts = []
    for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if k in raw_obs:
            v = raw_obs[k]
            if isinstance(v, np.ndarray):
                state_parts.append(torch.from_numpy(v.copy()).float().flatten())
    if state_parts:
        state = torch.cat(state_parts, dim=0).unsqueeze(0).to(device)
    else:
        state = torch.zeros(1, 8, device=device)

    batch = dict(images)
    batch["observation.state"] = state
    batch["task"] = task_instruction
    return batch


def repeat_pi05_batch(batch: dict, n: int) -> dict:
    """Repeat every tensor in the batch n times along dim 0; broadcast strings."""
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.repeat_interleave(n, dim=0) if v.shape[0] == 1 else v.repeat(n, *([1] * (v.ndim - 1)))
        elif isinstance(v, str):
            out[k] = [v] * n
        else:
            out[k] = v
    return out


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
