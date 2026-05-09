# flow_variant — Rectified-Flow Backbone for MoP-DSRL

Task 17 (`experiment_flow_extension`). Implements method_spec S7.

## Module layout

```
flow_variant/
  __init__.py                          # Package exports
  euler_solver.py                      # M-step Euler ODE integrator
  velocity_field_pi0.py                # pi_0 velocity adapter + ToyVelocityField
  velocity_field_dpt.py                # DP-T fallback adapter (Risk R1)
  rectified_flow_denoiser.py           # FrozenDenoiser subclass
  coupling_endpoint.py                 # z_0 provider (BC mode / RL mode)
  configs/
    libero_goal_pi0.yaml               # Default UT-5 eval config
  eval/
    run_bc_sanity.py                   # UT-5 CLI
    compare_to_diffusion_baseline.py   # Comparison table generator
  tests/
    test_unit_contracts_flow.py        # 11 tests (10 CPU + 1 GPU/LIBERO)
```

## Quick start

Run CPU tests (no GPU, no LIBERO, no checkpoints):

```bash
cd Experiment/core_code
python -m pytest mop_dsrl/flow_variant/tests/test_unit_contracts_flow.py -q
```

Run UT-5 BC-sanity gate (requires GPU + LIBERO + checkpoints):

```bash
cd Experiment/core_code
python -m mop_dsrl.flow_variant.eval.run_bc_sanity \
    --config mop_dsrl/flow_variant/configs/libero_goal_pi0.yaml \
    --diffusion_reference_csv Experiment/analysis/baselines/pi0_bc_libero_goal.csv \
    --output Experiment/analysis/flow_variant/ut5_libero_goal.csv
```

## Integration with SAC

The SAC trainer consumes `FrozenDenoiser.denoise(o, z)`. To use the flow
variant, pass `RectifiedFlowDenoiser` instead of the diffusion denoiser:

```python
from mop_dsrl.flow_variant import RectifiedFlowDenoiser

denoiser = RectifiedFlowDenoiser.from_checkpoint(
    checkpoint_dir="Experiment/datasets/checkpoints/pi0_libero",
    backbone="pi0", H=50, d_a=7, M=4, device="cuda",
)
trainer = SACTrainer(config, denoiser=denoiser, obs_dim=obs_dim)
```

The rest of the SAC loop is unchanged. The coupling endpoint `z_0` is
sampled by `mu_phi` (the outer Gaussian policy) identically to the
diffusion variant's noise seed.
