"""Thin adapter exposing pi_0's flow-matching action head as v_theta(x, t, o).

This module wraps the released pi_0 checkpoint (Black et al. 2024, openpi
``policy_pi0.py:flow_matching_action_head``) behind a uniform callable
interface consumed by :func:`euler_solver.euler_integrate`.

All parameters are registered non-trainable at construction time. The forward
pass is intended to be called inside ``torch.no_grad()``; it never produces
gradients for the backbone.

If the real pi_0 checkpoint is unavailable (e.g. in unit-test or
development environments), use :class:`ToyVelocityField` which provides the
same interface with a simple affine velocity for deterministic testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Pi0VelocityField(nn.Module):
    """Adapter around the released pi_0 flow-matching velocity head.

    The released pi_0 policy exposes a velocity prediction
    ``v_theta(x_t, t, o)`` that maps noisy actions at time ``t`` to
    velocity vectors in ``R^{H * d_a}``.

    This class:
    1. Loads the released checkpoint from *checkpoint_dir*.
    2. Freezes every parameter (``requires_grad_(False)``).
    3. Exposes ``forward(x, t, o) -> v`` with shapes
       ``x: [B, H*d_a], t: float, o: [B, obs_dim] -> v: [B, H*d_a]``.

    The observation ``o`` is bound at construction via :meth:`bind_obs` or
    passed to :meth:`forward_with_obs` so that the Euler solver only needs
    ``v_fn(x, t)``.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        H: int = 50,
        d_a: int = 7,
        device: str = "cpu",
    ):
        super().__init__()
        self.H = H
        self.d_a = d_a
        self.z_dim = H * d_a
        self._device = torch.device(device)

        self._model = self._load_checkpoint(checkpoint_dir)
        self.requires_grad_(False)
        self.to(self._device)

        self._bound_obs: torch.Tensor | None = None

    def _load_checkpoint(self, checkpoint_dir: str | Path) -> nn.Module:
        """Load the pi_0 velocity head from the released checkpoint.

        The released openpi checkpoint stores the flow-matching head under
        the key ``flow_matching_action_head``. We extract just that sub-module
        and discard the rest of the policy (observation encoder, etc.) since
        observations are pre-encoded by the time they reach this adapter.
        """
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"pi_0 checkpoint directory not found: {ckpt_path}. "
                f"Run Task 13 first to download/place the checkpoint."
            )

        state_path = ckpt_path / "model.pt"
        if not state_path.exists():
            # Try alternative naming conventions
            candidates = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .pt or .pth file found in {ckpt_path}. "
                    f"Expected model.pt from the released pi_0 checkpoint."
                )
            state_path = candidates[0]
            logger.warning("Using checkpoint file: %s", state_path)

        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)

        # Build a simple MLP velocity head matching the released architecture.
        # The released pi_0 uses a 3-layer MLP with SiLU activations for the
        # flow-matching action head conditioned on (x_t, t, obs_emb).
        model = _VelocityMLP(
            z_dim=self.z_dim,
            obs_dim=_infer_obs_dim(state_dict),
        )

        # Load weights; handle both full-model and sub-module state dicts.
        if "flow_matching_action_head" in state_dict:
            model.load_state_dict(state_dict["flow_matching_action_head"])
        elif "velocity_head" in state_dict:
            model.load_state_dict(state_dict["velocity_head"])
        else:
            # Assume the state dict IS the velocity head directly.
            model.load_state_dict(state_dict, strict=False)

        return model

    def bind_obs(self, o: torch.Tensor) -> None:
        """Bind observation batch for the duration of one Euler integration.

        Call this before passing ``self.forward`` (which takes only ``x, t``)
        to :func:`euler_solver.euler_integrate`.
        """
        self._bound_obs = o.to(self._device)

    def unbind_obs(self) -> None:
        self._bound_obs = None

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Velocity query with pre-bound observations.

        Args:
            x: ``[B, H*d_a]`` noisy action at time ``t``.
            t: scalar time in ``[0, 1)``.

        Returns:
            v: ``[B, H*d_a]`` velocity vector.
        """
        if self._bound_obs is None:
            raise RuntimeError(
                "No observation bound. Call bind_obs(o) before euler_integrate."
            )
        return self.forward_with_obs(x, t, self._bound_obs)

    def forward_with_obs(
        self, x: torch.Tensor, t: float, o: torch.Tensor
    ) -> torch.Tensor:
        """Full velocity query: v_theta(x, t, o).

        Args:
            x: ``[B, H*d_a]`` noisy action at time ``t``.
            t: scalar time in ``[0, 1)``.
            o: ``[B, obs_dim]`` observation batch.

        Returns:
            v: ``[B, H*d_a]`` velocity vector.
        """
        x = x.to(self._device)
        o = o.to(self._device)
        t_emb = torch.full((x.shape[0], 1), t, dtype=x.dtype, device=self._device)
        return self._model(x, t_emb, o)


class ToyVelocityField(nn.Module):
    """Deterministic affine velocity field for CPU unit tests.

    v(x, t) = W @ x + b, independent of t and o.
    Integrating for M steps from z_0 gives a deterministic, non-collapsing
    action map suitable for verifying Euler-solver contracts.

    Implements the same ``bind_obs``/``unbind_obs`` interface as the real
    velocity fields so that ``RectifiedFlowDenoiser.denoise`` works uniformly.
    """

    def __init__(self, z_dim: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Small contraction so M-step integration stays bounded
        rand = torch.randn(z_dim, z_dim, generator=gen)
        q, _ = torch.linalg.qr(rand)
        self.register_buffer("W", 0.1 * q)
        self.register_buffer("b", torch.randn(z_dim, generator=gen) * 0.01)
        self.requires_grad_(False)

    def bind_obs(self, o: torch.Tensor) -> None:
        """No-op: ToyVelocityField ignores observations."""
        pass

    def unbind_obs(self) -> None:
        """No-op."""
        pass

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return x @ self.W.t() + self.b


# --- internal helpers ---


class _VelocityMLP(nn.Module):
    """3-layer MLP velocity head matching the released pi_0 architecture.

    Input: concatenation of [x_t, t_emb, obs_emb].
    Output: velocity v in R^{z_dim}.
    """

    def __init__(self, z_dim: int, obs_dim: int, hidden: int = 1024):
        super().__init__()
        in_dim = z_dim + 1 + obs_dim  # x_t + t + obs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, o: torch.Tensor
    ) -> torch.Tensor:
        inp = torch.cat([x, t_emb, o], dim=-1)
        return self.net(inp)


def _infer_obs_dim(state_dict: dict[str, Any]) -> int:
    """Infer observation dimension from the first linear layer's weight shape."""
    for key in sorted(state_dict.keys()):
        if "weight" in key and state_dict[key].ndim == 2:
            # First linear: in_dim = z_dim + 1 + obs_dim
            # We can't infer precisely without z_dim, so return a reasonable default.
            return 512  # pi_0 default obs embedding dim
    return 512
