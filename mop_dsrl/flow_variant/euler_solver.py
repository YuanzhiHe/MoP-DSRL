"""M-step Euler ODE integrator for rectified-flow denoising (method_spec S7).

Pure-function implementation: no global state, no autograd through the
velocity field (the caller wraps v_fn in torch.no_grad() because pi_theta is
frozen). Deterministic given (o, z_0, M).

Usage:
    a = euler_integrate(v_fn, z_0, M=4)  # a = x(1), the action chunk
"""

from __future__ import annotations

from typing import Callable

import torch


def euler_integrate(
    v_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x0: torch.Tensor,
    M: int = 4,
) -> torch.Tensor:
    """Integrate the ODE dx/dt = v(x, t) from t=0 to t=1 using M Euler steps.

    Args:
        v_fn: velocity field callable ``v_fn(x_m, t_m) -> v_m`` where
              ``x_m`` has shape ``[B, D]`` and ``t_m`` is a Python float
              in ``[0, 1)``.  The callable should NOT require gradients
              (caller is expected to wrap in ``torch.no_grad()``).
        x0:  initial state ``z_0`` of shape ``[B, D]`` (the coupling endpoint).
        M:   number of Euler steps (default 4 per method_spec S8).

    Returns:
        x_M: terminal state ``x(1)`` of shape ``[B, D]``, i.e. the action
             chunk ``a``.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    dt = 1.0 / M
    x = x0
    for m in range(M):
        t_m = m * dt
        v_m = v_fn(x, t_m)
        x = x + dt * v_m
    return x
