"""Coupling-endpoint provider for the rectified-flow variant (method_spec S7).

In BC mode (no outer policy), z_0 is drawn from N(0, I).
In RL mode, z_0 is drawn from mu_phi(. | o) via the OuterGaussianPolicy.

This module is a convenience wrapper that unifies the two paths so that
eval scripts and the SAC trainer can use the same sampling interface.
"""

from __future__ import annotations

import torch

from ..components.policy import OuterGaussianPolicy


def sample_z0(
    o: torch.Tensor,
    z_dim: int,
    mu_phi: OuterGaussianPolicy | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample coupling endpoints z_0 for a batch of observations.

    Args:
        o: ``[B, obs_dim]`` observation batch.
        z_dim: dimensionality of z_0 (= H * d_a).
        mu_phi: outer Gaussian policy. If ``None``, samples from ``N(0, I)``
            (BC mode).

    Returns:
        z0: ``[B, z_dim]`` coupling endpoints.
        log_prob: ``[B]`` log-densities. For BC mode, this is the standard
            normal log-density; for RL mode, the diagonal-Gaussian log-density
            from ``mu_phi``.
    """
    B = o.shape[0]
    device = o.device

    if mu_phi is None:
        # BC mode: z_0 ~ N(0, I)
        z0 = torch.randn(B, z_dim, device=device, dtype=o.dtype)
        # log p(z_0) under N(0, I)
        import math
        log_prob = -0.5 * (
            z0.pow(2).sum(dim=-1)
            + z_dim * math.log(2.0 * math.pi)
        )
        return z0, log_prob
    else:
        # RL mode: z_0 ~ mu_phi(. | o)
        return mu_phi.sample(o)


def sample_z0_n(
    o: torch.Tensor,
    z_dim: int,
    n: int,
    mu_phi: OuterGaussianPolicy | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n`` coupling endpoints per observation.

    Args:
        o: ``[B, obs_dim]`` observation batch.
        z_dim: dimensionality of z_0.
        n: number of samples per observation.
        mu_phi: outer policy or ``None`` for BC mode.

    Returns:
        z0: ``[B*n, z_dim]`` coupling endpoints.
        log_prob: ``[B*n]`` log-densities.
    """
    # Repeat each observation n times
    o_rep = o.repeat_interleave(n, dim=0)  # [B*n, obs_dim]
    return sample_z0(o_rep, z_dim, mu_phi)
