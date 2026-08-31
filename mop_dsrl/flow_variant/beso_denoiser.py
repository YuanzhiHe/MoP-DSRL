"""FrozenDenoiser wrapper around a D3IL BESO agent.

BESO (Reuss et al.) is D3IL's strongest published baseline on Avoiding
(0.950 SR / 0.856 behaviour entropy) and, unlike the DDPM agent in the same
repo, it already exposes exactly the interface noise-space RL needs:

    agent.sample_loop(sigmas, x_t, state, goal, sampler_type, extra_args)

``x_t`` is the *initial noise*, passed in rather than drawn internally, and the
'euler' / 'ddim' / 'heun' / 'dpm' / 'lms' samplers are deterministic ODE
integrators. So z -> a is a deterministic, differentiable map with no surgery on
the agent. (The DDPM agent draws ``torch.randn`` inside ``p_sample_loop`` and
injects fresh noise at every step, so it has no z -> a map at all.)

The backbone is frozen: parameters get ``requires_grad_(False)`` and are never
handed to an optimizer. Gradients still flow *through* it to the noise policy
when ``with_grad=True``, which is what the mode regularizer's actor-loss path
needs.
"""
from __future__ import annotations

import torch

from mop_dsrl.denoiser import FrozenDenoiser


class BesoDenoiser(FrozenDenoiser):
    """Wrap a trained BESO agent as a frozen z -> action-chunk decoder."""

    def __init__(
        self,
        agent,
        z_dim: int,
        n_sampling_steps: int = 10,
        sampler_type: str = "ddim",
        sigma_min: float = 0.001,
        sigma_max: float = 4.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.agent = agent
        self.z_dim = z_dim
        self.sampler_type = sampler_type
        self.n_sampling_steps = n_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._device = device
        # Freeze every parameter reachable from the agent.
        for attr in ("model", "network", "policy"):
            mod = getattr(agent, attr, None)
            if isinstance(mod, torch.nn.Module):
                mod.requires_grad_(False)
                mod.eval()

    def _sigmas(self) -> torch.Tensor:
        """Noise schedule for the ODE sampler (BESO's get_noise_schedule)."""
        fn = getattr(self.agent, "get_noise_schedule", None)
        if fn is not None:
            return fn(self.n_sampling_steps, "exponential").to(self._device)
        # Fallback: geometric spacing between sigma_max and sigma_min, then 0.
        ramp = torch.linspace(0, 1, self.n_sampling_steps, device=self._device)
        sig = self.sigma_max * (self.sigma_min / self.sigma_max) ** ramp
        return torch.cat([sig, sig.new_zeros(1)])

    def denoise(self, o: torch.Tensor, z: torch.Tensor,
                with_grad: bool = False) -> torch.Tensor:
        """Map (o, z) to an action chunk via BESO's deterministic ODE sampler.

        ``z`` is unit-scale noise (the RL policy's action space); BESO's sampler
        expects it at the schedule's starting noise level, hence the sigma_max
        scaling.
        """
        sigmas = self._sigmas()
        x_t = z.reshape(z.shape[0], -1) * sigmas[0]
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            a = self.agent.sample_loop(
                sigmas=sigmas,
                x_t=x_t,
                state=o,
                goal=None,
                sampler_type=self.sampler_type,
            )
        return a.reshape(a.shape[0], -1)
