"""Flow-matching variant of MoP-DSRL (method_spec S7, Task 17).

Provides a rectified-flow FrozenDenoiser subclass that wraps the pi_0.5
velocity head (via LeRobot, ``lerobot/pi05_base``) behind an M-step Euler ODE
integrator. The coupling endpoint z_0 is drawn from N(0, I) in BC mode
and from mu_phi(. | o) in RL mode.

Primary backbone: pi_0.5 (Physical Intelligence, 2025).
Fallback backbone: DP-T diffusion-to-flow conversion (Risk R1).
Test stub: ToyVelocityField (CPU, no checkpoint).

Public surface:
- RectifiedFlowDenoiser (rectified_flow_denoiser.py)
- euler_integrate (euler_solver.py)
- Pi05VelocityField (velocity_field_pi05.py) — primary
- DPTVelocityField (velocity_field_dpt.py) — fallback
- ToyVelocityField (velocity_field_pi0.py) — CPU tests
- sample_z0, sample_z0_n (coupling_endpoint.py)
"""

from .coupling_endpoint import sample_z0, sample_z0_n
from .euler_solver import euler_integrate
from .rectified_flow_denoiser import RectifiedFlowDenoiser
from .velocity_field_dpt import DPTVelocityField
from .velocity_field_pi0 import ToyVelocityField

# Pi05VelocityField requires lerobot; lazy-import to keep CPU tests working.
try:
    from .velocity_field_pi05 import Pi05VelocityField
except ImportError:
    Pi05VelocityField = None  # type: ignore[assignment,misc]

__all__ = [
    "RectifiedFlowDenoiser",
    "euler_integrate",
    "Pi05VelocityField",
    "DPTVelocityField",
    "ToyVelocityField",
    "sample_z0",
    "sample_z0_n",
]
