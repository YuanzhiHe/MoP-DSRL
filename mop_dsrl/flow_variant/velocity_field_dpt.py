"""DP-T rectified-flow fallback adapter (PLAN.md Risk R1).

If the released pi_0 weights are unavailable, this module provides an
alternative velocity field derived from the DP-T (Diffusion Policy -
Transformer) checkpoint refitted as a rectified-flow model.

The DP-T backbone natively uses DDIM diffusion. This adapter wraps the
denoising prediction as a velocity field ``v_theta(x, t, o)`` using the
standard conversion:

    v(x, t) = (hat_x_0(x, t) - x) / (1 - t)

where ``hat_x_0`` is the clean-action prediction from the DP-T denoiser.

All parameters are registered non-trainable at construction time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DPTVelocityField(nn.Module):
    """Velocity field derived from the DP-T diffusion backbone.

    Converts the DDIM clean-prediction ``hat_x_0(x_t, t, o)`` into
    a flow-matching velocity using the identity
    ``v(x, t) = (hat_x_0 - x) / (1 - t)``.

    This is documented as the fallback for Risk R1 in PLAN.md: used when
    the pi_0 flow-matching checkpoint is unavailable.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        H: int = 16,
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
        """Load the DP-T denoiser checkpoint."""
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"DP-T checkpoint directory not found: {ckpt_path}. "
                f"Run Task 13 first to download/place the checkpoint."
            )

        state_path = ckpt_path / "model.pt"
        if not state_path.exists():
            candidates = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .pt or .pth file found in {ckpt_path}."
                )
            state_path = candidates[0]
            logger.warning("Using checkpoint file: %s", state_path)

        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)

        model = _CleanPredictionMLP(z_dim=self.z_dim, obs_dim=512)

        if "denoiser" in state_dict:
            model.load_state_dict(state_dict["denoiser"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        return model

    def bind_obs(self, o: torch.Tensor) -> None:
        """Bind observation batch for the duration of one Euler integration."""
        self._bound_obs = o.to(self._device)

    def unbind_obs(self) -> None:
        self._bound_obs = None

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Velocity query with pre-bound observations.

        Converts the DP-T clean prediction to a velocity:
        ``v(x, t) = (hat_x_0(x, t, o) - x) / (1 - t)``

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
        """Full velocity query converting diffusion prediction to flow velocity."""
        x = x.to(self._device)
        o = o.to(self._device)
        t_emb = torch.full((x.shape[0], 1), t, dtype=x.dtype, device=self._device)

        # DP-T predicts clean action: hat_x_0
        hat_x0 = self._model(x, t_emb, o)

        # Convert to velocity: v = (hat_x_0 - x) / (1 - t)
        # Clamp denominator to avoid division by zero at t=1.
        denom = max(1.0 - t, 1e-6)
        v = (hat_x0 - x) / denom
        return v


class _CleanPredictionMLP(nn.Module):
    """3-layer MLP clean-action predictor matching DP-T architecture."""

    def __init__(self, z_dim: int, obs_dim: int, hidden: int = 1024):
        super().__init__()
        in_dim = z_dim + 1 + obs_dim
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
