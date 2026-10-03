#!/usr/bin/env bash
# Block until this user's experiment jobs drain, then print what landed.
# Analysis and the follow-up review are driven from the session, not from here.
set -uo pipefail
while squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qE "bb-multi|bmdP-|k24-|mb-|divf-"; do
    sleep 300
done
echo "=== all experiment jobs drained ==="
squeue -u "$USER" -h -o "%.9i %.11j %.8T" 2>/dev/null | head -5
echo
echo "=== result files ==="
ls -la runs/_d3il/*.json 2>/dev/null | tail -20
echo
echo "=== backbones ==="
ls /shared/scratch/SCWF00111/d3il_backbones/avoiding_skew*_bb*.pt 2>/dev/null | wc -l
