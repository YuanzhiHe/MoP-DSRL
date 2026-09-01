"""D3IL environment adapter (Jia et al., "Towards Diverse Behavior Learning", ICLR 2024).

Why this exists. LIBERO cannot test the mode-preservation claim: every task has a
single solution, so the frozen backbone's reference histogram is a point mass
(C_ref entropy measured at 0.000 across 45 runs) and there is nothing to
preserve. D3IL is built for the opposite purpose -- each task admits many
distinct solutions, and crucially the environment computes the behavioural mode
from geometry, so mode labels are GROUND TRUTH rather than a K-means proxy over
action chunks. That removes the vocabulary from the measurement path entirely.

Avoiding: the robot must reach a goal line past two rows of obstacles; passing
each obstacle on the left or the right yields 2 x 3 x 4 = 24 distinct routes,
reported by the env as a 9-bit ``mode_encoding``.

Conventions this adapter normalises (see d3il/simulation/avoiding_sim.py:52-77):
  * The policy observation is 4-D: the current *desired* xy target concatenated
    with the env's 2-D observation, not the raw env observation.
  * The policy emits a 2-D DELTA which must be added to the current desired xy;
    the env expects a 7-D pose (xy, fixed z, fixed quaternion).
  * ``step`` returns ``info`` as a TUPLE ``(mode_encoding, success)``, whereas
    the rest of this codebase expects a dict with a "success" key.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

D3IL_ROOT = Path(os.environ.get("D3IL_ROOT", "/shared/home1/c.c23089110/d3il"))


def _load_avoiding_class():
    """Import ObstacleAvoidanceEnv from the D3IL checkout.

    Two obstacles. (1) D3IL's env module uses relative imports, so it must be
    loaded as part of its package rather than by file path. (2) Running a script
    out of this repo's scripts/ puts that directory on sys.path, where our own
    scripts/envs.py shadows D3IL's top-level `envs` package. So drop the
    shadowing entries for the duration of the import, and evict any `envs`
    already bound to the wrong module.
    """
    env_pkg_root = str(D3IL_ROOT / "environments" / "d3il")
    if not (Path(env_pkg_root) / "envs").is_dir():
        raise FileNotFoundError(f"D3IL envs package not found under {env_pkg_root}; set D3IL_ROOT")

    here = str(Path(__file__).resolve().parent)
    saved_path, saved_mods = list(sys.path), {}
    sys.path = [p for p in sys.path if p not in ("", ".", here)]
    sys.path.insert(0, env_pkg_root)
    sys.path.insert(1, str(D3IL_ROOT))
    for name in list(sys.modules):
        if name == "envs" or name.startswith("envs."):
            saved_mods[name] = sys.modules.pop(name)
    try:
        from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
        return ObstacleAvoidanceEnv
    finally:
        sys.path = saved_path
        for name in list(sys.modules):
            if name == "envs" or name.startswith("envs."):
                del sys.modules[name]
        sys.modules.update(saved_mods)


# 24 routes = 2 (first obstacle) x 3 (middle row) x 4 (final row)
N_MODES_AVOIDING = 24


class D3ILAvoidingEnv:
    """Adapter exposing D3IL's Avoiding task through this repo's env contract.

    Contract (scripts/envs.py): ``reset() -> obs``, ``step(a) -> (obs, r, done,
    info_dict)``, plus ``obs_dim`` / ``action_dim``. Adds ``mode_id`` and
    ``n_modes`` so the trainer can use ground-truth behavioural modes instead of
    a K-means vocabulary.
    """

    obs_dim = 4
    action_dim = 2

    def __init__(self, max_steps: int = 150, seed: int = 0, render: bool = False,
                 rewarded_modes: "set[int] | None" = None):
        """
        Args:
            rewarded_modes: if given, only these ground-truth mode ids earn
                reward on success. This is the reward shift that creates the
                collapse pressure the experiment measures: a pure return
                maximizer should abandon every unrewarded route, and the
                question is what each regularizer retains. ``None`` rewards any
                successful episode.
        """
        # Import the D3IL env by explicit path. A bare `import envs...` is
        # ambiguous: running a script out of scripts/ puts that directory on
        # sys.path[0], where this repo's own scripts/envs.py shadows D3IL's
        # `envs` package ("'envs' is not a package").
        ObstacleAvoidanceEnv = _load_avoiding_class()

        self._env = ObstacleAvoidanceEnv(render=render)
        self._env.start()
        self.max_steps = max_steps
        self.n_modes = N_MODES_AVOIDING
        self._rng = np.random.default_rng(seed)
        self._desired_xy = None
        self._fixed_z = None
        self._t = 0
        self._last_mode_encoding = np.zeros(9)
        self.rewarded_modes = rewarded_modes

    # --- contract ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        raw = self._env.reset()
        state = self._env.robot_state()
        self._desired_xy = np.asarray(state[:2], dtype=np.float64)
        self._fixed_z = np.asarray(state[2:], dtype=np.float64)
        self._t = 0
        self._last_mode_encoding = np.zeros(9)
        return self._obs(raw)

    def step(self, action: np.ndarray):
        # The policy emits a delta on the desired xy target, not an absolute pose.
        delta = np.asarray(action, dtype=np.float64).reshape(-1)[: self.action_dim]
        self._desired_xy = self._desired_xy + delta
        pose = np.concatenate([self._desired_xy, self._fixed_z, [0, 1, 0, 0]])
        raw, reward, done, info = self._env.step(pose)
        self._t += 1
        if self._t >= self.max_steps:
            done = True
        mode_encoding, success = self._unpack(info)
        self._last_mode_encoding = mode_encoding
        # D3IL is an imitation benchmark and returns reward=None; the RL signal
        # is terminal task success, optionally restricted to a subset of routes.
        if reward is None:
            earned = bool(success)
            if earned and self.rewarded_modes is not None:
                earned = self.mode_id in self.rewarded_modes
            reward = 1.0 if earned else 0.0
        return self._obs(raw), float(reward), bool(done), {
            "success": bool(success),
            "mode_encoding": mode_encoding,
            "mode_id": self.mode_id,
        }

    # --- mode bookkeeping -------------------------------------------------
    @property
    def mode_id(self) -> int:
        """Ground-truth behavioural mode in {0, ..., 23}, or -1 if undetermined.

        The 9-bit encoding is three one-hot groups (2, 3, 4) recording which side
        the arm passed each obstacle row. A partially-filled encoding means the
        episode has not yet committed to a route.
        """
        e = np.asarray(self._last_mode_encoding).reshape(-1)
        if e.size != 9:
            return -1
        g1, g2, g3 = e[:2], e[2:5], e[5:9]
        if g1.sum() != 1 or g2.sum() != 1 or g3.sum() != 1:
            return -1
        return int(g1.argmax() * 12 + g2.argmax() * 4 + g3.argmax())

    @staticmethod
    def _unpack(info) -> tuple[np.ndarray, bool]:
        """D3IL returns ``(mode_encoding, success)``; other envs return a dict."""
        if isinstance(info, tuple) and len(info) == 2:
            return np.asarray(info[0]).reshape(-1), bool(info[1])
        if isinstance(info, dict):
            return (np.asarray(info.get("mode_encoding", np.zeros(9))).reshape(-1),
                    bool(info.get("success", False)))
        return np.zeros(9), False

    def _obs(self, raw) -> np.ndarray:
        """Policy observation = desired xy target ++ env observation (4-D)."""
        return np.concatenate(
            [self._desired_xy, np.asarray(raw, dtype=np.float64).reshape(-1)]
        ).astype(np.float32)


def make_d3il_env(name: str, **kwargs):
    if name in ("d3il_avoiding", "avoiding"):
        return D3ILAvoidingEnv(**kwargs)
    raise ValueError(f"unknown D3IL env: {name}")
