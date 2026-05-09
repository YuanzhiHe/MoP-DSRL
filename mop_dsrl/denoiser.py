"""Frozen-denoiser interface for the SAC outer loop.

The SAC trainer in `sac.py` only consumes the abstract base class
`FrozenDenoiser`. The Task 17 flow variant subclasses it with an Euler ODE
solver, and the Task 18 value-guidance arm subclasses it again to add the
Q-gradient hook. None of those extensions touch the SAC core.

`ToyDenoiser` is a deterministic affine map z -> a from R^{H*d_a} to itself.
It is used by the unit-test suite so the three contract tests run end-to-end
on CPU with no real backbone checkpoint.
"""

from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn as nn


class FrozenDenoiser(abc.ABC, nn.Module):
    """Abstract frozen denoiser pi_theta.

    Subclasses implement `denoise(o, z)`; subclasses MUST register all their
    parameters as non-trainable (`requires_grad_(False)`). The SAC core never
    backpropagates through `denoise`.
    """

    @abc.abstractmethod
    def denoise(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Map (o, z) to an action chunk a in R^{B, H*d_a}.

        Args:
            o: [B, obs_dim] observation batch.
            z: [B, H * d_a] noise input.

        Returns:
            a: [B, H * d_a] action chunk.
        """

    def numpy_denoise(self, o: np.ndarray, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o_t = torch.as_tensor(o, dtype=torch.float32)
            z_t = torch.as_tensor(z, dtype=torch.float32)
            if o_t.ndim == 1:
                o_t = o_t.unsqueeze(0)
                z_t = z_t.unsqueeze(0)
            a = self.denoise(o_t, z_t)
        return a.cpu().numpy()


class ToyDenoiser(FrozenDenoiser):
    """Deterministic affine z -> a denoiser used by unit tests.

    a = W @ z + b(o)

    where W is a fixed orthogonal matrix and b(o) is a fixed linear map of o.
    The map is bijective in z, so different z produce different a, which is
    enough to populate every K-means cluster non-degenerately when z ~ N(0, I).
    All parameters are frozen.
    """

    def __init__(self, obs_dim: int, z_dim: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # orthogonal W (z_dim x z_dim) for a non-collapsing map
        rand = torch.randn(z_dim, z_dim, generator=gen)
        q, _ = torch.linalg.qr(rand)
        self.register_buffer("W", q)
        self.register_buffer("B", torch.randn(obs_dim, z_dim, generator=gen) * 0.1)
        self.register_buffer("b0", torch.randn(z_dim, generator=gen) * 0.0)
        self.requires_grad_(False)

    def denoise(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # a = z @ W^T + o @ B + b0
        return z @ self.W.t() + o @ self.B + self.b0
