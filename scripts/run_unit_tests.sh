#!/usr/bin/env bash
# Run all CPU unit tests (no GPU, no LIBERO, no checkpoints required).
# Usage: bash scripts/run_unit_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "=== Running core unit tests ==="
python -m pytest mop_dsrl/tests/test_unit_contracts.py -q -v

echo ""
echo "=== Running flow-variant unit tests ==="
python -m pytest mop_dsrl/flow_variant/tests/test_unit_contracts_flow.py -q -v -k "not gpu"

echo ""
echo "ALL TESTS PASSED"
