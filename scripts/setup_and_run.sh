#!/usr/bin/env bash
# =============================================================================
# MoP-DSRL — end-to-end setup + experiment driver for a 2× RTX 5090 server.
#
# This is a single self-contained script.  Run individual stages with their
# number, or `all` to run them in order:
#
#     bash scripts/setup_and_run.sh 1     # check system
#     bash scripts/setup_and_run.sh 2     # install conda (if missing)
#     bash scripts/setup_and_run.sh 3     # create venv + install PyTorch
#     bash scripts/setup_and_run.sh 4     # install MoP-DSRL + run unit tests
#     bash scripts/setup_and_run.sh 5     # install LIBERO + lerobot + EGL (idempotent)
#     bash scripts/setup_and_run.sh 6     # download pi_0.5 weights + LIBERO demos
#     bash scripts/setup_and_run.sh 7     # build K-means vocabularies
#     bash scripts/setup_and_run.sh 8     # GPU smoke test (pi_0.5 + LIBERO)
#     bash scripts/setup_and_run.sh 9a    # launch ablations (1 seed)
#     bash scripts/setup_and_run.sh 9b    # launch robustness (1 seed)
#     bash scripts/setup_and_run.sh 9c    # launch tier1 main (3 seeds)
#     bash scripts/setup_and_run.sh 9d    # launch baselines (3 seeds)
#     bash scripts/setup_and_run.sh 9e    # launch everything (5/3/1 default)
#     bash scripts/setup_and_run.sh status   # show running + completed runs
#     bash scripts/setup_and_run.sh collect  # aggregate results
#     bash scripts/setup_and_run.sh all       # 1..8 sequentially (NO 9)
#
# Hardware target:
#   * 2× NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32 GB each)
#   * Driver 595.71.05 / CUDA 13.2 (already installed)
#   * Headless server, EGL rendering required for LIBERO/mujoco
#
# Idempotent: rerunning a stage either skips already-done work or wipes its own
# scratch dir before re-doing it.  Per-stage logs are written to setup_logs/.
# =============================================================================
set -uo pipefail

# ---- paths ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/setup_logs"
mkdir -p "$LOG_DIR" "$REPO_ROOT/runs" "$REPO_ROOT/logs" "$REPO_ROOT/vocabs"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mop_dsrl}"
# IMPORTANT: pyproject requires Python <3.12, so we pin 3.11.
PY_VERSION="${PY_VERSION:-3.11}"

LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"
PI05_HF_ID="${PI05_HF_ID:-lerobot/pi05_libero}"

# ---- helpers ----------------------------------------------------------------
log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m[ok]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; }

stage_header() {
    local name="$1"
    printf "\n\033[1;36m========================================================================\033[0m\n"
    printf "\033[1;36m Stage %s\033[0m\n" "$name"
    printf "\033[1;36m========================================================================\033[0m\n"
}

# Source conda (after it's installed).  Exposes `conda` as a shell function.
activate_conda() {
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        err "Conda not found. Run stage 2 first."
        return 1
    fi
}

ensure_env_active() {
    activate_conda || return 1
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
        err "Env $CONDA_ENV_NAME missing. Run stage 3 first."
        return 1
    }
    ok "Active env: $CONDA_PREFIX  ($(python --version 2>&1))"
}

# =============================================================================
# Stage 1: System info
# =============================================================================
stage_1() {
    stage_header "1: system info"
    log "host: $(hostname)"
    log "kernel: $(uname -r)"
    log "user: $USER"
    log "cwd: $PWD"
    nvidia-smi -L
    echo
    nvidia-smi | head -25
    echo
    log "system python: $(command -v python3 || true) ($(python3 --version 2>&1))"
    log "system conda:  $(command -v conda || echo 'none')"
    log "free disk in repo: $(df -h "$REPO_ROOT" | tail -1)"
    log "free RAM: $(free -h | awk '/^Mem:/ {print $7" available of "$2}')"
    log "egl libs present: $(ldconfig -p | grep -ciE 'libEGL\\.so|libGL\\.so' || true)"
}

# =============================================================================
# Stage 2: Install miniforge (only if missing).
# =============================================================================
stage_2() {
    stage_header "2: install miniforge"
    if [ -x "$CONDA_ROOT/bin/conda" ]; then
        ok "miniforge already at $CONDA_ROOT (skipping)"
        return 0
    fi
    log "Downloading miniforge → $CONDA_ROOT"
    cd /tmp
    wget -nv -O Miniforge3.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash Miniforge3.sh -b -p "$CONDA_ROOT"
    rm -f Miniforge3.sh
    cd "$REPO_ROOT"
    # Make conda available on next shells.
    if ! grep -q "miniforge3/etc/profile.d/conda.sh" "$HOME/.bashrc" 2>/dev/null; then
        echo "source $CONDA_ROOT/etc/profile.d/conda.sh" >> "$HOME/.bashrc"
    fi
    ok "miniforge installed.  Run 'source ~/.bashrc' in any new shell."
}

# =============================================================================
# Stage 3: Create conda env + install PyTorch (cu128 for Blackwell sm_120).
# =============================================================================
stage_3() {
    stage_header "3: conda env + PyTorch"
    activate_conda || return 1

    if conda env list | grep -qE "^\\s*$CONDA_ENV_NAME\\s"; then
        ok "env $CONDA_ENV_NAME already exists (skipping create)"
    else
        log "Creating env $CONDA_ENV_NAME with Python $PY_VERSION"
        conda create -n "$CONDA_ENV_NAME" "python=$PY_VERSION" pip -y
    fi

    conda activate "$CONDA_ENV_NAME"
    pip install --upgrade pip wheel setuptools

    # PyTorch — Blackwell sm_120 needs CUDA 12.8+ wheels.  cu128 is what
    # supports RTX 5090; older cu118/cu121 wheels will throw
    # "no kernel image is available for execution on the device".
    log "Installing PyTorch (cu128) — this can take a few minutes"
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7" "torchvision" "torchaudio" 2>&1 | tail -5 || {
        err "cu128 wheel install failed.  Trying nightly index…"
        pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
            torch torchvision torchaudio 2>&1 | tail -10
    }

    # Smoke test: both GPUs visible, matmul on each works.
    log "GPU sanity check"
    python - <<'PY'
import torch
print("torch:", torch.__version__, " built with cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(),
      " device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  sm_{p.major}{p.minor}  mem={p.total_memory/1e9:.1f} GB")
    a = torch.randn(2048, 2048, device=f"cuda:{i}", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=f"cuda:{i}", dtype=torch.bfloat16)
    c = (a @ b).float()
    print(f"  GPU{i} matmul OK: {c.shape} {c.dtype} max={c.max().item():.2f}")
PY
}

# =============================================================================
# Stage 4: Install MoP-DSRL + run CPU contract tests (17 expected to pass).
# =============================================================================
stage_4() {
    stage_header "4: install MoP-DSRL + unit tests"
    ensure_env_active || return 1

    log "pip install -e .[dev,rl] + h5py + einops"
    pip install -e ".[dev,rl]" 2>&1 | tail -5
    pip install h5py einops 2>&1 | tail -3

    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    log "Running CPU unit tests…"
    if python -m pytest \
        mop_dsrl/tests/test_unit_contracts.py \
        mop_dsrl/flow_variant/tests/test_unit_contracts_flow.py \
        -v -k "not gpu" --timeout=180 2>&1 | tail -30; then
        ok "Unit tests passed."
    else
        err "Unit tests failed.  See the tail above; do not proceed."
        return 1
    fi
}

# =============================================================================
# Stage 5: LIBERO + lerobot + EGL + libero config.
# =============================================================================
# Idempotent.  Handles all the gotchas we hit on the 2× RTX 5090 server:
#   * `git clone` over HTTP/2 dies mid-pack — use HTTP/1.1 + big postBuffer + retries.
#   * `LIBERO/libero/` has no __init__.py (namespace pkg) → `pip install -e` says
#     "Successfully installed libero" but `import libero` fails.  Use a `.pth` shim.
#   * `lerobot` install downgrades torch from cu128 to a CPU wheel — force-restore.
#   * LIBERO's `libero/libero/__init__.py` prompts interactively on first import
#     ("Do you want to specify a custom path for the dataset folder? (Y/N):").
#     Pre-write `~/.libero/config.yaml` BEFORE the first `import libero.libero`.
# =============================================================================
stage_5() {
    stage_header "5: LIBERO + lerobot + EGL"
    ensure_env_active || return 1

    log "EGL/GL libs check (no install attempt — needs root)"
    ldconfig -p | grep -iE "libEGL\.so|libGL\.so" | head -3 || {
        err "libEGL not found.  Ask sysadmin: apt install libegl1 libgl1-mesa-glx libosmesa6-dev"
        err "Continuing — install may still succeed if EGL is in a non-standard path."
    }

    log "Installing robosuite + mujoco + h5py (LIBERO native deps)"
    pip install "robosuite==1.4.1" "mujoco>=3.2.3" h5py easydict thop bddl future 2>&1 | tail -5

    # --- LIBERO clone (HTTP/2 stream cancel is common; fall back to HTTP/1.1) ---
    if [ -z "$(ls -A "$LIBERO_DIR" 2>/dev/null)" ]; then
        log "Cloning LIBERO → $LIBERO_DIR  (HTTP/1.1 + retries)"
        rm -rf "$LIBERO_DIR"
        git config --global http.postBuffer 524288000
        git config --global http.version HTTP/1.1
        local tries=0 max_tries=5 success=0
        while [ $tries -lt $max_tries ]; do
            tries=$((tries + 1))
            log "clone attempt $tries/$max_tries"
            if git clone --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"; then
                success=1; break
            fi
            rm -rf "$LIBERO_DIR"
            sleep $((tries * 5))
        done
        [ $success -eq 0 ] && { err "LIBERO clone failed after $max_tries tries — STOP"; return 1; }
        ok "LIBERO cloned"
    else
        ok "LIBERO already at $LIBERO_DIR (skip clone)"
    fi

    # --- LIBERO install (mostly for its declared deps).  We rely on the .pth
    # shim below for actual importability since LIBERO/libero has no __init__.py
    # and PEP 660 editable installs don't expose it. ---
    log "pip install -e $LIBERO_DIR  (gets LIBERO's declared deps; import via .pth shim)"
    pip install -e "$LIBERO_DIR" 2>&1 | tail -5 || pip install "$LIBERO_DIR" 2>&1 | tail -5 || true

    # --- .pth shim: add LIBERO_DIR to sys.path so `import libero` works ---
    local SP
    SP=$(python -c "import site; print(site.getsitepackages()[0])")
    local PTH_FILE="$SP/_libero_local.pth"
    echo "$LIBERO_DIR" > "$PTH_FILE"
    ok "Wrote $PTH_FILE → $LIBERO_DIR  (makes 'import libero' work)"

    # --- Pre-write ~/.libero/config.yaml BEFORE the first libero.libero import ---
    log "Writing ~/.libero/config.yaml (must precede first 'import libero.libero')"
    LIBERO_ROOT="$LIBERO_DIR/libero/libero" python - <<'PY'
import os, yaml, pathlib
root = pathlib.Path(os.environ["LIBERO_ROOT"])
cfg = {
    'benchmark_root': str(root),
    'bddl_files':    str(root / 'bddl_files'),
    'init_states':   str(root / 'init_files'),
    'datasets':      str(root.parent / 'datasets'),  # LIBERO/libero/datasets
    'assets':        str(root / 'assets'),
}
out = pathlib.Path.home() / '.libero' / 'config.yaml'
out.parent.mkdir(exist_ok=True)
out.write_text(yaml.safe_dump(cfg))
print('wrote', out)
for k, v in cfg.items():
    print(f"  {k}: {v}  ({'exists' if pathlib.Path(v).exists() else 'MISSING (download in stage 6)'})")
PY

    # --- lerobot (latest pyPI; extras vary across versions, try several) ---
    log "Installing lerobot — will likely downgrade torch to CPU wheel; we restore right after"
    if pip install "lerobot[pi]" transformers accelerate 2>&1 | tail -5; then
        ok "lerobot[pi] installed"
    elif pip install "lerobot[pi05]" transformers accelerate 2>&1 | tail -5; then
        ok "lerobot[pi05] installed"
    elif pip install "lerobot" transformers accelerate 2>&1 | tail -5; then
        ok "lerobot (no extras) installed"
    else
        err "PyPI lerobot install failed; trying github source"
        pip install "git+https://github.com/huggingface/lerobot.git" transformers accelerate 2>&1 | tail -5
    fi

    # --- Force-restore torch cu128 (lerobot stomped it to CPU wheel) ---
    local cur_ver cur_cuda
    cur_ver=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
    cur_cuda=$(python -c "import torch; print(torch.version.cuda or 'CPU')" 2>/dev/null || echo "CPU")
    log "After lerobot: torch=$cur_ver  cuda=$cur_cuda"
    if [[ "$cur_ver" != *"+cu"* ]]; then
        local base_ver="${cur_ver%%+*}"
        log "Force-reinstalling torch==$base_ver from cu128 index (keeps lerobot's version pin)"
        pip install --force-reinstall --no-deps \
            --index-url https://download.pytorch.org/whl/cu128 \
            "torch==$base_ver" 2>&1 | tail -5
        pip install --force-reinstall --no-deps \
            --index-url https://download.pytorch.org/whl/cu128 \
            torchvision torchaudio 2>&1 | tail -5 || true
    else
        ok "torch already has +cu suffix — no restore needed"
    fi

    # --- Reinstall mop-dsrl so the relaxed numpy<… pin applies ---
    log "Reinstalling mop-dsrl in editable mode (numpy>=1.26, no upper cap)"
    pip install -e ".[dev,rl]" 2>&1 | tail -5

    # --- Final sanity (must NOT prompt; use </dev/null to be safe) ---
    log "Full sanity check"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    python - </dev/null <<'PY'
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy, torch
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "  cuda:", torch.version.cuda, "  avail:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU not available after torch restore — STOP"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  sm_{p.major}{p.minor}")
import libero, libero.libero
from libero.libero import benchmark
b = benchmark.get_benchmark('libero_goal')()
print("libero.libero at:", libero.libero.__file__)
print("libero_goal n_tasks:", b.n_tasks)
for i in range(min(3, b.n_tasks)):
    print(f"  task[{i}]: {b.get_task(i).name}")
import lerobot
print("lerobot:", lerobot.__version__)
from lerobot.policies.pi05 import PI05Policy
print("PI05Policy: import OK")
print("ALL OK")
PY
    ok "Stage 5 complete.  Continue with stage 6."
}

# =============================================================================
# Stage 6: Download pi_0.5 LIBERO weights + LIBERO demo HDF5s.
# =============================================================================
stage_6() {
    stage_header "6: download pi_0.5 + LIBERO demos"
    ensure_env_active || return 1
    : "${HF_HOME:=$HOME/.cache/huggingface}"
    export HF_HOME

    log "pi_0.5 weights → HF cache at $HF_HOME (~7 GB)"
    python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="${PI05_HF_ID}", repo_type="model")
print("pi_0.5 cached at:", path)
PY

    log "LIBERO demo HDF5s (~5 GB total across 4 suites)"
    python - <<'PY'
import os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
try:
    from libero.libero.libero_utils import download_libero_datasets
except Exception:
    from libero.libero import download_libero  # older layouts
    download_libero_datasets = download_libero.download_datasets
import yaml, pathlib
cfg = yaml.safe_load((pathlib.Path.home()/'.libero/config.yaml').read_text())
print("dataset root:", cfg['datasets'])
for s in ('libero_goal','libero_spatial','libero_object','libero_10'):
    print("downloading", s); sys.stdout.flush()
    try:
        download_libero_datasets([s])
    except Exception as e:
        print(f"  ! {s} failed: {e}")
PY

    log "Listing dataset folder"
    python - <<'PY'
import yaml, pathlib, os
cfg = yaml.safe_load((pathlib.Path.home()/'.libero/config.yaml').read_text())
root = pathlib.Path(cfg['datasets'])
print("datasets at:", root)
for sub in sorted(root.glob('libero_*')):
    files = list(sub.glob('*.hdf5'))
    print(f"  {sub.name}: {len(files)} hdf5 files")
PY
    ok "Weights + demos in place."
}

# =============================================================================
# Stage 7: Build K-means vocabularies for every LIBERO suite (K=16).
# =============================================================================
stage_7() {
    stage_header "7: build K-means vocabularies"
    ensure_env_active || return 1
    mkdir -p "$REPO_ROOT/vocabs"

    DATA_ROOT=$(python -c "import yaml, pathlib; print(yaml.safe_load((pathlib.Path.home()/'.libero/config.yaml').read_text())['datasets'])")
    log "LIBERO data root: $DATA_ROOT"

    for SUITE in libero_goal libero_spatial libero_object libero_10; do
        OUT="$REPO_ROOT/vocabs/${SUITE}_K16.pkl"
        if [ -f "$OUT" ]; then
            ok "vocab exists: $OUT (skipping)"
            continue
        fi
        DEMO_DIR="$DATA_ROOT/$SUITE"
        if [ ! -d "$DEMO_DIR" ]; then
            err "missing demo dir: $DEMO_DIR — skip (re-run stage 6)"
            continue
        fi
        log "Building vocab for $SUITE → $OUT"
        python scripts/build_vocab.py \
            --demos "$DEMO_DIR" \
            --H 50 --d_a 7 --K 16 --seed 42 \
            --out "$OUT"
    done
    ls -la "$REPO_ROOT/vocabs/"
}

# =============================================================================
# Stage 8: GPU smoke test — short real pi_0.5 + LIBERO-Goal training.
# =============================================================================
stage_8() {
    stage_header "8: GPU smoke test (real pi_0.5 + LIBERO)"
    ensure_env_active || return 1

    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUBLAS_WORKSPACE_CONFIG=":16:8"
    export CUDA_VISIBLE_DEVICES=0
    rm -rf "$REPO_ROOT/runs/_smoke_pi05_libero"

    python -u scripts/train.py \
        --backbone pi05 --env libero_goal --task_id 0 \
        --H 50 --d_a 7 --K 16 --M 10 \
        --pretrained_path "$PI05_HF_ID" \
        --vocab_path "$REPO_ROOT/vocabs/libero_goal_K16.pkl" \
        --total_env_steps 1500 --warmup_env_steps 300 \
        --batch_size 64 --replay_capacity 4096 \
        --n_obs 30 --n_ref 8 --n_on 4 \
        --eval_interval 0 --final_eval_episodes 3 \
        --ckpt_interval 0 --log_interval 100 \
        --episode_steps 200 \
        --device cuda \
        --run_dir "$REPO_ROOT/runs/_smoke_pi05_libero" 2>&1 | tee "$LOG_DIR/stage_8_smoke.log"

    if [ -f "$REPO_ROOT/runs/_smoke_pi05_libero/final_summary.json" ]; then
        ok "GPU smoke OK — see runs/_smoke_pi05_libero/"
        python -c "import json; print(json.dumps(json.load(open('runs/_smoke_pi05_libero/final_summary.json')), indent=2))"
    else
        err "Smoke test did not produce final_summary.json"
        tail -40 "$LOG_DIR/stage_8_smoke.log"
        return 1
    fi
}

# =============================================================================
# Stage 9 — experiment launchers.  Each uses scripts/run_local.py with --n-gpus 2.
# Each call is wrapped in nohup so the launcher survives ssh disconnects.
# Use `bash scripts/setup_and_run.sh status` to monitor.
# =============================================================================
_launch() {
    # _launch <stage_tag> <run_local args…>
    local tag="$1"; shift
    ensure_env_active || return 1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUBLAS_WORKSPACE_CONFIG=":16:8"
    mkdir -p "$LOG_DIR"
    log "dry-run for $tag"
    python scripts/run_local.py --dry-run --n-gpus 2 "$@" | tail -25
    log "Launching $tag in the background.  Log: $LOG_DIR/$tag.log"
    nohup python -u scripts/run_local.py --n-gpus 2 "$@" \
        >"$LOG_DIR/$tag.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$LOG_DIR/$tag.pid"
    ok "$tag launched PID=$pid"
    log "Tail with:   tail -f $LOG_DIR/$tag.log"
    log "Stop with:   kill -INT \$(cat $LOG_DIR/$tag.pid)"
}

stage_9a() { _launch "ablations_seed0"   --tier ablations  --seeds-aux 0; }
stage_9b() { _launch "robustness_seed0"  --tier robustness --seeds-aux 0; }
stage_9c() { _launch "tier1_main_3seed"  --tier main       --seeds-main 0 1 2; }
stage_9d() { _launch "baselines_3seed"   --tier baselines  --seeds-main 0 1 2; }
stage_9e() { _launch "all_full"          --tier all; }
stage_9_ut5() { _launch "ut5_bc_sanity"  --tier ut5; }

# Sequential 9c → 9d → 9b → 9a under a single nohup-able launcher.
# Main results first (tier1 main + baselines), then robustness and ablations.
# Each phase runs run_local.py in the FOREGROUND so they don't compete for GPUs.
# Designed for `nohup bash scripts/setup_and_run.sh 9_phased &` — survives ssh disconnects.
stage_9_phased() {
    stage_header "9_phased: tier1 → baselines → robustness → ablations (sequential)"
    ensure_env_active || return 1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUBLAS_WORKSPACE_CONFIG=":16:8"
    mkdir -p "$LOG_DIR"

    local started=$(date -Is)
    log "Phased run started at $started — will sequence 9c → 9d → 9b → 9a"
    log "Each phase logs to $LOG_DIR/phase_*.log; this dispatcher logs to dispatch_*.log."

    _phase() {
        local tag="$1"; shift
        log "─── PHASE: $tag ───  (args: $*)"
        local t0=$(date +%s)
        # Run run_local.py in the foreground so the next phase only starts after it finishes.
        python -u scripts/run_local.py --n-gpus 2 "$@" \
            > "$LOG_DIR/phase_${tag}.log" 2>&1
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        if [ $rc -eq 0 ]; then
            ok "phase $tag completed in $((dt/3600))h$(( (dt%3600)/60 ))m"
        else
            err "phase $tag exited non-zero ($rc). Continuing to next phase anyway."
        fi
    }

    _phase tier1_main_3seed  --tier main       --seeds-main 0 1 2
    _phase baselines_3seed   --tier baselines  --seeds-main 0 1 2
    _phase robustness_seed0  --tier robustness --seeds-aux 0
    _phase ablations_seed0   --tier ablations  --seeds-aux 0

    ok "All 4 phases done.  Started: $started   Finished: $(date -Is)"
    log "Aggregate with:  bash scripts/setup_and_run.sh collect"
}

# Minimal "main results only, 1 seed" phased run: 9c + 9d at seed 0 only.
# 27 runs → 9 runs (4 tier1 + 5 baselines).  Used for first-pass results.
# Wall clock on 2× RTX 5090 ≈ 4 days.
stage_9_main_1seed() {
    stage_header "9_main_1seed: tier1 main (4 runs) + baselines (5 runs), seed 0 only"
    ensure_env_active || return 1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUBLAS_WORKSPACE_CONFIG=":16:8"
    mkdir -p "$LOG_DIR"

    local started=$(date -Is)
    log "Started at $started — sequencing tier1_main(seed0) → baselines(seed0)"

    _phase() {
        local tag="$1"; shift
        log "─── PHASE: $tag ───  (args: $*)"
        local t0=$(date +%s)
        python -u scripts/run_local.py --n-gpus 2 "$@" \
            > "$LOG_DIR/phase_${tag}.log" 2>&1
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        if [ $rc -eq 0 ]; then
            ok "phase $tag completed in $((dt/3600))h$(( (dt%3600)/60 ))m"
        else
            err "phase $tag exited non-zero ($rc). Continuing anyway."
        fi
    }

    _phase tier1_main_1seed  --tier main       --seeds-main 0
    _phase baselines_1seed   --tier baselines  --seeds-main 0

    ok "Both phases done.  Started: $started   Finished: $(date -Is)"
    log "Aggregate with:  bash scripts/setup_and_run.sh collect"
}

# =============================================================================
# Status + collect
# =============================================================================
stage_status() {
    stage_header "status"
    ensure_env_active || return 1
    echo "---- background launchers ----"
    for pidfile in "$LOG_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        local tag pid
        tag=$(basename "$pidfile" .pid)
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            printf "  %-20s PID %-8s RUNNING  (log: %s)\n" "$tag" "$pid" "$LOG_DIR/$tag.log"
        else
            printf "  %-20s PID %-8s exited\n" "$tag" "$pid"
        fi
    done
    echo
    echo "---- python train.py processes ----"
    pgrep -af "scripts/train.py" || echo "  (none)"
    echo
    echo "---- GPU usage ----"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo
    echo "---- finished runs ----"
    find "$REPO_ROOT/runs" -mindepth 3 -name final_summary.json | wc -l \
        | awk '{print "  "$1" runs with final_summary.json"}'
    echo
    echo "---- latest manifest ----"
    ls -1t "$REPO_ROOT/submissions/local_manifest_"*.tsv 2>/dev/null | head -3
}

stage_collect() {
    stage_header "collect results"
    ensure_env_active || return 1
    python scripts/collect_results.py \
        --runs-root "$REPO_ROOT/runs" \
        --out "$REPO_ROOT/analysis/results_aggregated.csv"
    log "CSV at $REPO_ROOT/analysis/results_aggregated.csv"
    head -10 "$REPO_ROOT/analysis/results_aggregated.csv" || true
}

# =============================================================================
# Dispatcher
# =============================================================================
usage() {
    sed -n '4,40p' "$0"
}

main() {
    if [ $# -lt 1 ]; then
        usage
        return 0
    fi
    case "$1" in
        1)        stage_1 ;;
        2)        stage_2 ;;
        3)        stage_3 ;;
        4)        stage_4 ;;
        5)        stage_5 ;;
        6)        stage_6 ;;
        7)        stage_7 ;;
        8)        stage_8 ;;
        9a)       stage_9a ;;
        9b)       stage_9b ;;
        9c)       stage_9c ;;
        9d)       stage_9d ;;
        9e)       stage_9e ;;
        9_phased|phased) stage_9_phased ;;
        9_main_1seed|main_1seed) stage_9_main_1seed ;;
        9_ut5|ut5) stage_9_ut5 ;;
        status)   stage_status ;;
        collect)  stage_collect ;;
        all)
            stage_1 && stage_2 && stage_3 && stage_4 \
                && stage_5 && stage_6 && stage_7 && stage_8
            ;;
        *)
            err "Unknown stage: $1"
            usage
            return 1
            ;;
    esac
}

main "$@" 2>&1 | tee -a "$LOG_DIR/dispatch_$(date +%Y%m%dT%H%M%S).log"
