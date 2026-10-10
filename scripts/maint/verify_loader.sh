#!/usr/bin/env bash
# The shared loader must not change any number the paper reports. Regenerate the
# figures and recompute the headline statistics, then compare against the files
# built before the change.
set -uo pipefail
cd "$(dirname "$0")/../.."
echo "=== dedup report ==="
python scripts/d3il_runs.py
echo "=== paper_numbers ==="
python scripts/maint/paper_numbers.py 2>&1 | tail -25
echo "rc=$?"
echo "=== figures ==="
python scripts/make_figures_v2.py 2>&1 | tail -5
echo "rc=$?"
