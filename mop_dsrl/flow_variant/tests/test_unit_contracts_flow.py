"""Unit tests for the flow-matching variant (Task 17).

Tests:
  1. Euler determinism: same (o, z_0) produces bit-identical a.
  2. Euler M=1: single step equals x_0 + v(x_0, 0).
  3. Shape correctness with ToyVelocityField.
  4. RectifiedFlowDenoiser conforms to FrozenDenoiser interface.
  5. All parameters are frozen (requires_grad=False).
  6. Coupling endpoint BC mode produces N(0, I) samples.
  7. (UT-5 gate) marked with pytest.mark.gpu — skipped on CPU CI.

Run with:
    cd Experiment/core_code
    python -m pytest mop_dsrl/flow_variant/tests/test_unit_contracts_flow.py -q

CPU only (no GPU, no LIBERO) except for test_ut5_gate.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

# allow running from Experiment/core_code or repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from mop_dsrl.flow_variant.coupling_endpoint import sample_z0, sample_z0_n
from mop_dsrl.flow_variant.euler_solver import euler_integrate
from mop_dsrl.flow_variant.rectified_flow_denoiser import RectifiedFlowDenoiser
from mop_dsrl.flow_variant.velocity_field_pi0 import ToyVelocityField


# --- fixtures ---


def _toy_vf(z_dim: int = 12, seed: int = 0) -> ToyVelocityField:
    return ToyVelocityField(z_dim=z_dim, seed=seed)


def _toy_denoiser(z_dim: int = 12, M: int = 4, seed: int = 0) -> RectifiedFlowDenoiser:
    return RectifiedFlowDenoiser.from_toy(z_dim=z_dim, M=M, seed=seed)


# --- test 1: Euler determinism ---


def test_euler_determinism():
    """Same (z_0) produces identical a across two calls."""
    z_dim = 12
    vf = _toy_vf(z_dim)

    z0 = torch.randn(4, z_dim)

    with torch.no_grad():
        a1 = euler_integrate(vf, z0.clone(), M=4)
        a2 = euler_integrate(vf, z0.clone(), M=4)

    assert torch.equal(a1, a2), "Euler integration must be deterministic"


# --- test 2: Euler M=1 gives x_0 + v(x_0, 0) ---


def test_euler_single_step():
    """With M=1, Euler gives x_0 + 1.0 * v(x_0, 0)."""
    z_dim = 8
    vf = _toy_vf(z_dim, seed=7)

    z0 = torch.randn(3, z_dim)

    with torch.no_grad():
        a = euler_integrate(vf, z0.clone(), M=1)
        v0 = vf(z0, 0.0)
        expected = z0 + v0

    assert torch.allclose(a, expected, atol=1e-6), (
        f"M=1 Euler should give x_0 + v(x_0, 0); max diff={torch.abs(a - expected).max():.2e}"
    )


# --- test 3: shape correctness ---


def test_euler_shape():
    """Output shape matches input shape [B, z_dim]."""
    z_dim = 20
    B = 5
    vf = _toy_vf(z_dim)

    z0 = torch.randn(B, z_dim)
    with torch.no_grad():
        a = euler_integrate(vf, z0, M=4)

    assert a.shape == (B, z_dim), f"Expected shape ({B}, {z_dim}), got {a.shape}"


# --- test 4: RectifiedFlowDenoiser interface ---


def test_denoiser_interface():
    """RectifiedFlowDenoiser produces correct output shape from denoise(o, z)."""
    z_dim = 12
    obs_dim = 8
    B = 4

    denoiser = _toy_denoiser(z_dim, M=4)

    o = torch.randn(B, obs_dim)
    z = torch.randn(B, z_dim)

    a = denoiser.denoise(o, z)

    assert a.shape == (B, z_dim), f"Expected ({B}, {z_dim}), got {a.shape}"
    assert a.dtype == torch.float32


def test_denoiser_numpy_interface():
    """numpy_denoise works via the inherited FrozenDenoiser method."""
    z_dim = 12
    obs_dim = 8

    denoiser = _toy_denoiser(z_dim, M=4)

    o = np.random.randn(obs_dim).astype(np.float32)
    z = np.random.randn(z_dim).astype(np.float32)

    a = denoiser.numpy_denoise(o, z)

    assert a.shape == (1, z_dim), f"Expected (1, {z_dim}), got {a.shape}"
    assert a.dtype == np.float32


# --- test 5: all parameters frozen ---


def test_all_parameters_frozen():
    """Every parameter in RectifiedFlowDenoiser has requires_grad=False."""
    denoiser = _toy_denoiser(z_dim=12, M=4)

    for name, param in denoiser.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name} has requires_grad=True; all must be frozen"
        )


# --- test 6: coupling endpoint BC mode ---


def test_coupling_endpoint_bc_mode():
    """BC mode (mu_phi=None) samples from N(0, I) with correct log_prob."""
    torch.manual_seed(42)
    z_dim = 16
    B = 1024

    o = torch.randn(B, 8)
    z0, log_prob = sample_z0(o, z_dim, mu_phi=None)

    assert z0.shape == (B, z_dim)
    assert log_prob.shape == (B,)

    # Check statistics: mean ~ 0, std ~ 1
    assert z0.mean().abs() < 0.1, f"BC z_0 mean = {z0.mean():.4f}, expected ~0"
    assert (z0.std() - 1.0).abs() < 0.1, f"BC z_0 std = {z0.std():.4f}, expected ~1"

    # Verify log_prob against manual N(0,I) computation
    expected_lp = -0.5 * (z0.pow(2).sum(dim=-1) + z_dim * math.log(2.0 * math.pi))
    assert torch.allclose(log_prob, expected_lp, atol=1e-5), (
        f"log_prob mismatch; max diff={torch.abs(log_prob - expected_lp).max():.2e}"
    )


def test_coupling_endpoint_n_samples():
    """sample_z0_n produces B*n samples."""
    torch.manual_seed(0)
    B, n, z_dim = 4, 10, 8
    o = torch.randn(B, 5)

    z0, log_prob = sample_z0_n(o, z_dim, n, mu_phi=None)

    assert z0.shape == (B * n, z_dim)
    assert log_prob.shape == (B * n,)


# --- test 7: Euler solver rejects M < 1 ---


def test_euler_rejects_invalid_M():
    """euler_integrate raises ValueError for M < 1."""
    vf = _toy_vf(z_dim=4)
    z0 = torch.randn(2, 4)

    with pytest.raises(ValueError, match="M must be >= 1"):
        euler_integrate(vf, z0, M=0)


# --- test 8: denoiser determinism through full path ---


def test_denoiser_determinism():
    """Same (o, z) yields identical a through the full denoiser path."""
    z_dim = 12
    obs_dim = 6
    denoiser = _toy_denoiser(z_dim, M=4, seed=3)

    o = torch.randn(3, obs_dim)
    z = torch.randn(3, z_dim)

    a1 = denoiser.denoise(o, z.clone())
    a2 = denoiser.denoise(o, z.clone())

    assert torch.equal(a1, a2), "Full denoiser path must be deterministic"


# --- test 9: M-step composition (multi-step vs fewer larger steps) ---


def test_euler_multi_step_refines():
    """More Euler steps should give a different (generally more accurate) result
    than fewer steps. This is a structural test, not an accuracy test."""
    z_dim = 12
    vf = _toy_vf(z_dim, seed=5)

    z0 = torch.randn(2, z_dim)

    with torch.no_grad():
        a_m1 = euler_integrate(vf, z0.clone(), M=1)
        a_m4 = euler_integrate(vf, z0.clone(), M=4)
        a_m16 = euler_integrate(vf, z0.clone(), M=16)

    # M=1 and M=4 should generally differ (unless velocity is constant)
    diff_1_4 = torch.abs(a_m1 - a_m4).max().item()
    diff_4_16 = torch.abs(a_m4 - a_m16).max().item()

    # With ToyVelocityField (affine, non-constant), different M values
    # will produce different trajectories.
    assert diff_1_4 > 1e-6, (
        f"M=1 and M=4 should differ for non-constant velocity; diff={diff_1_4:.2e}"
    )


# --- UT-5 gate (GPU only, requires LIBERO) ---


def _has_libero() -> bool:
    try:
        import libero  # noqa: F401
        return True
    except ImportError:
        return False


# The reference CSV and checkpoint paths are relative to Experiment/core_code
# which is _PKG_ROOT.
_DIFF_REF_CSV = os.path.join(
    _PKG_ROOT, "..", "..", "Experiment", "analysis", "baselines", "pi0_bc_libero_goal.csv"
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_libero()
    or not os.path.exists(_DIFF_REF_CSV),
    reason="UT-5 requires GPU + LIBERO + diffusion BC reference CSV"
)
def test_ut5_gate():
    """UT-5: BC-only flow success rate within +-2 pp of diffusion reference.

    This test is skipped on CPU CI. It requires:
    - A GPU with CUDA
    - LIBERO installed and configured
    - The pi_0 checkpoint at Experiment/datasets/checkpoints/pi0_libero/
    - The diffusion BC reference at Experiment/analysis/baselines/pi0_bc_libero_goal.csv

    Run explicitly with:
        pytest -m gpu mop_dsrl/flow_variant/tests/test_unit_contracts_flow.py::test_ut5_gate
    """
    import json
    import subprocess

    result = subprocess.run(
        [
            sys.executable, "-m", "mop_dsrl.flow_variant.eval.run_bc_sanity",
            "--config", "mop_dsrl/flow_variant/configs/libero_goal_pi0.yaml",
            "--diffusion_reference_csv", "Experiment/analysis/baselines/pi0_bc_libero_goal.csv",
            "--output", "Experiment/analysis/flow_variant/ut5_libero_goal.csv",
        ],
        capture_output=True,
        text=True,
        cwd=_PKG_ROOT,
    )

    verdict_path = os.path.join(
        _PKG_ROOT, "Experiment", "analysis", "flow_variant", "ut5_verdict.json"
    )
    assert os.path.exists(verdict_path), (
        f"Verdict file not created. stdout: {result.stdout[:500]}, "
        f"stderr: {result.stderr[:500]}"
    )

    with open(verdict_path) as f:
        verdict = json.load(f)

    assert verdict["pass"], (
        f"UT-5 FAILED: S_diff={verdict['S_diff']:.4f}, "
        f"S_flow={verdict['S_flow']:.4f}, delta={verdict['delta_pp']:.2f} pp"
    )


if __name__ == "__main__":
    test_euler_determinism()
    test_euler_single_step()
    test_euler_shape()
    test_denoiser_interface()
    test_denoiser_numpy_interface()
    test_all_parameters_frozen()
    test_coupling_endpoint_bc_mode()
    test_coupling_endpoint_n_samples()
    test_euler_rejects_invalid_M()
    test_denoiser_determinism()
    test_euler_multi_step_refines()
    print("ALL CPU TESTS PASSED")
