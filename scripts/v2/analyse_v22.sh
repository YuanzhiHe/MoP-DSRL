#!/usr/bin/env bash
# Every table, figure and document the paper reads, rebuilt from the v2.2 runs
# in one pass. The order is the order of the dependencies: the floor and the
# observation range are inputs to the report, the report and the trend are
# inputs to the figures, and the deliverables copy what the others wrote.
#
#   sbatch --cpus-per-task=32 --time=06:00:00 \
#       --job-name=v22-analyse scripts/v2/cpu_job_v2.sh \
#       bash scripts/v2/analyse_v22.sh
set -euo pipefail
R="${MOP_DSRL_ROOT:?}"
D="$R/runs/_d3il_v2"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$R"

step() { echo; echo "=== $* ==="; }

# The outputs carry the cohort's prefix for the same reason the run files do:
# the earlier cohort's files sit in this directory and a bare name would be
# read by a later table without anything saying which cohort it came from.
step sampling floor
python "$S/sampling_floor.py" --out "$D/v22_sampling_floor.json"
step observation range
python "$S/check_observations.py" --out "$D/v22_observation_range.json"
step main report
python "$S/report_v2.py"
step trend
python "$S/trend_v2.py"
step ablations
python "$S/ablations.py"
step frozen
python "$S/frozen_report.py"
step figures
python "$S/make_figures_corrected.py"
step deliverables
python "$S/deliverables.py"
echo
echo "ANALYSIS: PASS"
