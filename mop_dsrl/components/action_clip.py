"""L2 clip with flat-prior log-density correction (method_spec §5).

The seed/coupling-endpoint z is L2-clipped to a fixed ball of radius
r_z = 4 * sqrt(H * d_a). Samples whose un-clipped norm exceeds r_z are
projected onto the boundary; their log-density is replaced by a flat-prior
constant equal to -log(volume of the ball) so the importance ratio stays
finite and the SAC actor loss does not see infinities.
"""

from __future__ import annotations

import math

import torch


def _log_ball_volume(d: int, r: float) -> float:
    # log V_d(r) = (d/2) log(pi) + d log(r) - lgamma(d/2 + 1)
    return 0.5 * d * math.log(math.pi) + d * math.log(r) - math.lgamma(d / 2.0 + 1.0)


def l2_clip_with_logprob_correction(
    z: torch.Tensor, log_prob: torch.Tensor, r: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip z onto the L2 ball of radius r and correct log_prob for clipped rows.

    Args:
        z: [B, d] sample.
        log_prob: [B] log density under mu_phi (Gaussian).
        r: clip radius.

    Returns:
        (z_clipped, log_prob_corrected) with shapes ([B, d], [B]).
    """
    if r <= 0:
        raise ValueError(f"radius must be positive, got {r}")
    norms = z.norm(dim=-1, keepdim=True)
    scale = torch.clamp(r / norms.clamp(min=1e-12), max=1.0)
    z_clipped = z * scale
    flat_log_prob = -_log_ball_volume(z.shape[-1], r)
    was_clipped = (norms.squeeze(-1) > r)
    log_prob_corrected = torch.where(
        was_clipped,
        torch.full_like(log_prob, flat_log_prob),
        log_prob,
    )
    return z_clipped, log_prob_corrected
