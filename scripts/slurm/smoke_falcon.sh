#!/usr/bin/env bash
#SBATCH --job-name=mop_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_l40s
#SBATCH --time=01:30:00
#SBATCH --output=/shared/home1/c.c23089110/MoP-DSRL/logs/falcon_smoke_%j.out
#SBATCH --error=/shared/home1/c.c23089110/MoP-DSRL/logs/falcon_smoke_%j.out
#
# End-to-end Falcon validation: build K-means vocabs + tiny pi_0.5 LIBERO
# train/eval to confirm CUDA + EGL rendering + resume all work on a compute node.
set +e
export PYTHONUNBUFFERED=1

MOP_DSRL_ROOT=/shared/home1/c.c23089110/MoP-DSRL
source "$MOP_DSRL_ROOT/scripts/slurm/env_setup.sh"
cd "$MOP_DSRL_ROOT"

echo "==== NODE $(hostname) ===="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('cuda avail:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# ---- 1. Build vocabs (skip existing) ----
# NOTE: the config.yaml 'datasets' path is stale/missing on Falcon; the real
# LIBERO demos live under Sam2RL/libero_datasets (goal/spatial/object only —
# libero_10/long demos are absent here, so its vocab is copied from the
# workstation separately).
mkdir -p "$MOP_DSRL_ROOT/vocabs"
DATA_ROOT="${LIBERO_DEMO_ROOT:-/shared/scratch/SCWF00111/Sam2RL/libero_datasets}"
echo "==== LIBERO demos root: $DATA_ROOT ===="
ls "$DATA_ROOT" 2>/dev/null | head
for SUITE in libero_goal libero_spatial libero_object; do
    OUT="$MOP_DSRL_ROOT/vocabs/${SUITE}_K16.pkl"
    [ -f "$OUT" ] && { echo "vocab exists: $OUT"; continue; }
    DEMO="$DATA_ROOT/$SUITE"
    [ -d "$DEMO" ] || { echo "MISSING demos: $DEMO — skip"; continue; }
    echo "==== build vocab $SUITE ===="
    python scripts/build_vocab.py --demos "$DEMO" --H 50 --d_a 7 --K 16 --seed 42 --out "$OUT" \
        && echo "OK $OUT" || echo "FAIL vocab $SUITE"
done
ls -la "$MOP_DSRL_ROOT/vocabs/"

# ---- 2. Tiny smoke train (validates CUDA + EGL + train + eval + ckpt) ----
export CUDA_VISIBLE_DEVICES=0
rm -rf "$MOP_DSRL_ROOT/runs/_falcon_smoke"
echo "==== SMOKE TRAIN (1500 steps) ===="
python -u scripts/train.py \
    --backbone pi05 --env libero_goal --task_id 0 \
    --H 50 --d_a 7 --K 16 --M 10 \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --vocab_path "$MOP_DSRL_ROOT/vocabs/libero_goal_K16.pkl" \
    --total_env_steps 1500 --warmup_env_steps 300 \
    --batch_size 64 --replay_capacity 4096 \
    --n_obs 30 --n_ref 8 --n_on 4 \
    --eval_interval 0 --final_eval_episodes 3 \
    --ckpt_interval 1000 --log_interval 100 \
    --episode_steps 200 --device cuda \
    --lambda_ 1.6 --target_entropy -30.0 --clip_radius_sigma_factor 1.5 \
    --run_dir "$MOP_DSRL_ROOT/runs/_falcon_smoke"

echo "==== RESULT ===="
if [ -f "$MOP_DSRL_ROOT/runs/_falcon_smoke/final_summary.json" ]; then
    echo "SMOKE_OK"
    cat "$MOP_DSRL_ROOT/runs/_falcon_smoke/final_summary.json"
else
    echo "SMOKE_FAIL — no final_summary.json"
fi
