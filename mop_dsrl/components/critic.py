"""Twin-Q ensemble for the SAC outer loop (method_spec §5).

Each Q is a 4-layer MLP (o_emb, z) -> R; the observation embedding is supplied
by the outer policy encoder with the gradient stopped (see method_spec §5
"Networks"). Polyak averaging is implemented in-place by `polyak_update`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLPQ(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, o_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o_emb, z], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQEnsemble(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = _MLPQ(obs_dim, z_dim, hidden)
        self.q2 = _MLPQ(obs_dim, z_dim, hidden)

    def forward(self, o_emb: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(o_emb, z), self.q2(o_emb, z)


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)
