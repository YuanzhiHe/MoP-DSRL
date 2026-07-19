#!/usr/bin/env bash
# Reproducible recipe for the MoP-DSRL conda env on the Falcon (SuperComputing
# Wales) cluster. Run once on a login node to (re)build the env that
# scripts/slurm/env_setup.sh activates. Validated 2026-07-19: smoke_falcon.sh
# reached eval success_rate=1.0 on an L40S node.
#
# The env matches the workstation's behaviour: the pi_0.5 gate requires the
# openpi "transformers replace" (the transformers `fix/lerobot_openpi` branch,
# which self-versions as 4.53.3 and ships transformers/models/siglip/check.py).
# lerobot is the plain PyPI 0.4.4 wheel; our code uses its factory API
# (make_env_pre_post_processors), not the LiberoProcessorStep class directly.
set -euo pipefail

source /shared/apps/easybuild/x86_64/amd/zen4/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate /shared/home1/c.c23089110/.conda/envs/mop_dsrl

# Base already provides torch 2.11.0+cu130 (runs on H100 sm_90 / L40S sm_89).
python -c "import torch; assert torch.version.cuda, 'need CUDA torch'; print('torch', torch.__version__)"

# --- general deps (do NOT let these drag torch to CPU) ---
pip install h5py draccus huggingface_hub datasets \
    deepdiff safetensors accelerate einops termcolor jsonlines \
    imageio imageio-ffmpeg av opencv-python-headless gymnasium wandb zarr \
    omegaconf hydra-core packaging rerun-sdk matplotlib bddl easydict thop

# --- torchvision matched to torch 2.11 (nms op ABI must match) ---
pip install "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 --no-deps

# --- LIBERO sim backend: robosuite 1.4.0 needs mujoco 2.3.x (mj_fullM ABI) ---
pip install "robosuite==1.4.0" "mujoco==2.3.2" "gym==0.26.2" diffusers

# --- lerobot 0.4.4 (PyPI wheel; factory API used by our code) ---
pip install "lerobot==0.4.4" --no-deps

# --- CRITICAL: transformers from the openpi branch (provides siglip.check gate) ---
pip install --no-deps --force-reinstall \
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
pip install "tokenizers>=0.21,<0.22"   # transformers 4.53.3 pins <0.22

# --- validate the pi_0.5 gate + our import chain ---
export PYTHONPATH="/shared/home1/c.c23089110/MoP-DSRL:/shared/scratch/SCWF00111/Sam2RL/libero_repo/LIBERO"
export HF_HOME=/shared/scratch/SCWF00111/HPTD/hf_cache MUJOCO_GL=egl
python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(); print('pi05 gate: OK')"
python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; from lerobot.envs.factory import make_env_pre_post_processors; print('imports: OK')"
echo "Falcon env build complete."
