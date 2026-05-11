"""Backbone factory for the MoP-DSRL training driver.

Construct a frozen denoiser by name:

  ``toy``     — ToyDenoiser (z -> a affine), deterministic CPU smoke test.
  ``toyflow`` — RectifiedFlowDenoiser.from_toy, exercises the flow code path
                without any real checkpoint.
  ``pi05``    — RectifiedFlowDenoiser.from_pi05 (lerobot/pi05_*), GPU.
  ``dpt``     — DP-T fallback via RectifiedFlowDenoiser(backbone="dpt"), GPU.
  ``pi0``     — Legacy pi_0 (deprecated, kept for completeness).

The ``toy`` and ``toyflow`` backbones can fully smoke-test the SAC training
loop on CPU without any checkpoint.  The real backbones require their
weights and lerobot to be importable.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _ensure_lerobot_src() -> None:
    src = os.environ.get("LEROBOT_SRC", "")
    if src and src not in sys.path and Path(src).is_dir():
        sys.path.insert(0, src)


def make_backbone(
    name: str,
    H: int,
    d_a: int,
    M: int,
    device: str,
    obs_dim: int,
    checkpoint_dir: str | None = None,
    pretrained_path: str | None = None,
    seed: int = 0,
    cache_dir: str = "",
    local_files_only: bool = False,
):
    """Construct a FrozenDenoiser instance by name (see module docstring)."""
    z_dim = H * d_a

    if name == "toy":
        from mop_dsrl.denoiser import ToyDenoiser
        denoiser = ToyDenoiser(obs_dim=obs_dim, z_dim=z_dim, seed=seed).to(device)
        return denoiser

    if name == "toyflow":
        from mop_dsrl.flow_variant import RectifiedFlowDenoiser
        return RectifiedFlowDenoiser.from_toy(z_dim=z_dim, M=M, seed=seed).to(device)

    if name == "pi05":
        _ensure_lerobot_src()
        from mop_dsrl.flow_variant import RectifiedFlowDenoiser
        path = pretrained_path or checkpoint_dir or "lerobot/pi05_libero"
        return RectifiedFlowDenoiser.from_pi05(
            pretrained_path=path,
            H=H, d_a=d_a, M=M, device=device,
            cache_dir=cache_dir, local_files_only=local_files_only,
        )

    if name == "dpt":
        from mop_dsrl.flow_variant import RectifiedFlowDenoiser
        if not checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for backbone=dpt")
        return RectifiedFlowDenoiser.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            backbone="dpt",
            H=H, d_a=d_a, M=M, device=device,
        )

    if name == "pi0":
        from mop_dsrl.flow_variant.velocity_field_pi0 import Pi0VelocityField
        from mop_dsrl.flow_variant import RectifiedFlowDenoiser
        if not checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for backbone=pi0")
        vf = Pi0VelocityField(
            checkpoint_dir=checkpoint_dir, H=H, d_a=d_a, device=device
        )
        return RectifiedFlowDenoiser(velocity_field=vf, M=M)

    raise ValueError(f"Unknown backbone: {name!r}")
