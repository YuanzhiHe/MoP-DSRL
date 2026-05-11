#!/usr/bin/env bash
# Convenience wrapper around scripts/submit_all.py.
#
# Activates the mop_dsrl conda env and forwards every argument verbatim.
#
# Examples:
#   bash scripts/submit_all.sh --dry-run
#   bash scripts/submit_all.sh --tier all --partition gpu_l40s
#   bash scripts/submit_all.sh --tier main --partition gpu_h100 --backbones pi05
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

source /shared/apps/easybuild/x86_64/amd/zen4/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate /shared/home1/c.c23089110/.conda/envs/mop_dsrl

exec python scripts/submit_all.py "$@"
