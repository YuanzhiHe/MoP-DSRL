"""Outer Gaussian noise-space policy mu_phi(z|o) (method_spec §3).

mu_phi(z | o) = N(z; m_phi(o), diag(sigma_phi(o)^2))

Init contract (UT-1): at construction, m_phi(o) == 0 and sigma_phi(o) == 1 for
every o, so mu_phi == N(0, I) at training step 0 and the un-fine-tuned method
exactly reproduces the frozen backbone.

The encoder f_phi is provided externally so that backbones with different
observation modalities (vision-language for pi_0, image stack for DP-T) plug
into the same policy class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _IdentityEncoder(nn.Module):
    """Fallback encoder used by unit tests; passes observations through."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return o


class OuterGaussianPolicy(nn.Module):
    """Diagonal-Gaussian policy mu_phi(z | o) over R^{H * d_a}.

    The mean head and log-std head are zero-initialized so that, at init,
    m_phi(o) = 0 and log_sigma_phi(o) = 0 (i.e., sigma = 1) for every o,
    independent of the encoder output. This holds exactly only for the
    final linear layers; encoder weights are arbitrary.
    """

    def __init__(
        self,
        z_dim: int,
        encoder: nn.Module | None = None,
        encoder_out_dim: int | None = None,
        log_std_min: float = -6.907755,  # log(1e-3)
        log_std_max: float = 0.0,  # log(1.0)
        obs_dim: int | None = None,
    ):
        super().__init__()
        if encoder is None:
            if obs_dim is None:
                raise ValueError("must provide encoder or obs_dim")
            encoder = _IdentityEncoder(obs_dim)
            encoder_out_dim = obs_dim
        if encoder_out_dim is None:
            raise ValueError("must provide encoder_out_dim when encoder is given")
        self.encoder = encoder
        self.z_dim = z_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mean_head = nn.Linear(encoder_out_dim, z_dim)
        self.log_std_head = nn.Linear(encoder_out_dim, z_dim)
        # Init contract (UT-1, method_spec §3): mu_phi(z|o) ≡ N(0, I) at step 0.
        # That requires post-bound mean = 0 and post-bound log_std = 0
        # (so sigma = 1). The mean head zero-init achieves the first; for the
        # log_std head we set the bias so that tanh-rescaled output equals
        # log_std_max (which equals 0 by default). The required pre-bound
        # value satisfies log_std_min + 0.5*(log_std_max - log_std_min)*(tanh(b)+1) = log_std_max,
        # i.e. tanh(b) = 1; we approximate with b = atanh(1 - 1e-4) ~ 4.95.
        import math as _math
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        with torch.no_grad():
            self.log_std_head.bias.fill_(0.5 * _math.log((2.0 - 1e-4) / 1e-4))

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(o)
        mean = self.mean_head(emb)
        log_std = self.log_std_head(emb)
        # tanh-bound log_std into [log_std_min, log_std_max] (method_spec §3).
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(log_std) + 1.0
        )
        return mean, log_std

    def sample(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample.

        Returns (z, log_prob) where log_prob is the diagonal-Gaussian log
        density. The caller is responsible for applying the §5 L2 clip and
        the flat-prior log-density correction; this module returns the
        unclipped sample.
        """
        mean, log_std = self.forward(o)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        # diagonal Gaussian log density
        log_prob = -0.5 * (
            ((z - mean) / std) ** 2 + 2.0 * log_std + torch.log(torch.tensor(2.0 * 3.141592653589793))
        )
        log_prob = log_prob.sum(dim=-1)
        return z, log_prob
