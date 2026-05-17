"""RectifiedFlowDenoiser: FrozenDenoiser subclass for flow-matching (method_spec S7).

Composes a velocity field adapter and the Euler ODE solver to produce the
same ``denoise(o, z) -> a`` interface that ``sac.py`` expects. The SAC
trainer is fully agnostic to whether the backbone is diffusion (DDIM) or
flow (Euler); it only calls ``denoiser.denoise(o, z)``.

Supported backbones:
  - ``pi05`` — pi_0.5 via LeRobot (primary, ``lerobot/pi05_base``).
  - ``dpt``  — DP-T diffusion-to-flow conversion (fallback for Risk R1).
  - ``toy``  — ToyVelocityField for CPU unit tests.

All parameters are registered non-trainable. The forward pass runs inside
``torch.no_grad()`` because pi_theta is frozen and R_mode gradients flow
only through mu_phi (the outer Gaussian policy), not through v_theta.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import torch

from ..denoiser import FrozenDenoiser
from .euler_solver import euler_integrate
from .velocity_field_dpt import DPTVelocityField
from .velocity_field_pi0 import ToyVelocityField


class RectifiedFlowDenoiser(FrozenDenoiser):
    """Frozen rectified-flow denoiser wrapping a velocity field + Euler solver.

    This subclass interprets the noise input ``z`` as the coupling endpoint
    ``z_0`` of the rectified-flow ODE, integrates from ``t=0`` to ``t=1``
    using ``M`` Euler steps, and returns the terminal state ``x(1) = a``.

    Args:
        velocity_field: a velocity field module with ``bind_obs`` /
            ``unbind_obs`` / ``forward(x, t)`` interface.
        M: number of Euler steps (default 10 for pi_0.5, 4 legacy).
    """

    def __init__(
        self,
        velocity_field: torch.nn.Module,
        M: int = 10,
    ):
        super().__init__()
        self.velocity_field = velocity_field
        self.M = M
        self.requires_grad_(False)

    @classmethod
    def from_pi05(
        cls,
        pretrained_path: str = "lerobot/pi05_base",
        H: int = 50,
        d_a: int = 7,
        M: int = 10,
        device: str = "cpu",
        local_files_only: bool = False,
        cache_dir: str = "",
        task_instruction: str = "",
    ) -> "RectifiedFlowDenoiser":
        """Construct from the pi_0.5 HuggingFace Hub checkpoint (primary backbone).

        Args:
            pretrained_path: HF Hub identifier or local path.
            H: action chunk horizon (50 for pi_0.5).
            d_a: per-step action dimension (7 for LIBERO single-arm).
            M: Euler steps (default 10 for pi_0.5).
            device: torch device string.
            local_files_only: skip HF download if True.
            cache_dir: HF cache directory.
            task_instruction: language instruction for the VLA encoder.

        Returns:
            A fully frozen ``RectifiedFlowDenoiser``.
        """
        from .velocity_field_pi05 import Pi05VelocityField

        vf = Pi05VelocityField(
            pretrained_path=pretrained_path,
            H=H,
            d_a=d_a,
            device=device,
            num_inference_steps=M,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            task_instruction=task_instruction,
        )
        return cls(velocity_field=vf, M=M)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        backbone: Literal["pi05", "dpt"] = "pi05",
        H: int = 50,
        d_a: int = 7,
        M: int = 10,
        device: str = "cpu",
        **kwargs,
    ) -> "RectifiedFlowDenoiser":
        """Construct from a checkpoint directory or HF Hub identifier.

        Args:
            checkpoint_dir: path or HF Hub identifier.
            backbone: ``"pi05"`` (primary) or ``"dpt"`` (fallback).
            H: action chunk horizon.
            d_a: per-step action dimension.
            M: Euler steps.
            device: torch device string.
            **kwargs: forwarded to the velocity field constructor.

        Returns:
            A fully frozen ``RectifiedFlowDenoiser``.
        """
        if backbone == "pi05":
            return cls.from_pi05(
                pretrained_path=str(checkpoint_dir),
                H=H, d_a=d_a, M=M, device=device, **kwargs,
            )
        elif backbone == "dpt":
            vf = DPTVelocityField(
                checkpoint_dir=checkpoint_dir, H=H, d_a=d_a, device=device
            )
            return cls(velocity_field=vf, M=M)
        else:
            raise ValueError(
                f"Unknown backbone: {backbone!r}; expected 'pi05' or 'dpt'"
            )

    @classmethod
    def from_toy(cls, z_dim: int, M: int = 4, seed: int = 0) -> "RectifiedFlowDenoiser":
        """Construct with a ToyVelocityField for CPU unit tests."""
        vf = ToyVelocityField(z_dim=z_dim, seed=seed)
        return cls(velocity_field=vf, M=M)

    def denoise(self, o: torch.Tensor, z: torch.Tensor,
                with_grad: bool = False) -> torch.Tensor:
        """Map (o, z_0) to action chunk a via M-step Euler integration.

        Args:
            o: ``[B, obs_dim]`` observation batch.
            z: ``[B, H*d_a]`` coupling endpoint ``z_0``.
            with_grad: if True, build the autograd graph so gradients can
                       flow from a back to z (used by R_mode for the actor
                       update).  Default False keeps the legacy behaviour
                       (no_grad), which is appropriate for data collection.

        Returns:
            a: ``[B, H*d_a]`` action chunk ``x(1)``.
        """
        self.velocity_field.bind_obs(o)
        try:
            ctx = torch.enable_grad() if with_grad else torch.no_grad()
            with ctx:
                a = euler_integrate(self.velocity_field, z, M=self.M)
        finally:
            self.velocity_field.unbind_obs()
        return a

    def denoise_with_obs_dict(
        self, batch: Dict[str, torch.Tensor], z: torch.Tensor,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Denoise using a fully-preprocessed LeRobot observation dict.

        This is the preferred entry point during LIBERO evaluation with
        pi_0.5, where the observation includes images and language tokens.

        Args:
            batch: preprocessed LeRobot observation dict.
            z: ``[B, H*d_a]`` coupling endpoint ``z_0``.
            with_grad: if True, build the autograd graph so gradients can
                       flow from a back to z (required for R_mode in the
                       SAC actor update).  Uses gradient checkpointing per
                       Euler step to keep peak activation memory at
                       ~1 step's worth instead of M.
                       Default False = no_grad path, which is correct for
                       data collection / C_ref / evaluation rollouts where
                       we only need values.

        Returns:
            a: ``[B, H*d_a]`` action chunk ``x(1)``.
        """
        if not hasattr(self.velocity_field, "bind_obs_dict"):
            raise TypeError(
                f"{type(self.velocity_field).__name__} does not support "
                f"bind_obs_dict; use denoise(o, z) instead."
            )
        self.velocity_field.bind_obs_dict(batch)
        try:
            if with_grad:
                # Gradient checkpointing across the M Euler steps: only the
                # CURRENT step's activations are kept; backward recomputes.
                # Without this, pi_0.5 (3.4B params) × M=10 steps overflows
                # any single-GPU memory budget.
                from torch.utils.checkpoint import checkpoint as _ckpt
                with torch.enable_grad():
                    dt = 1.0 / self.M
                    x = z
                    vf = self.velocity_field
                    for m in range(self.M):
                        t_m = float(m * dt)
                        def _step(x_in, t=t_m, vf=vf, dt=dt):
                            return x_in + dt * vf(x_in, t)
                        x = _ckpt(_step, x, use_reentrant=False)
                    a = x
            else:
                with torch.no_grad():
                    a = euler_integrate(self.velocity_field, z, M=self.M)
        finally:
            self.velocity_field.unbind_obs()
        return a
