#!/usr/bin/env bash
# Compile the manuscript with Tectonic.
#
# Tectonic rather than the conda texlive env: that env ships the binaries with no
# TeX tree at all (no article.cls, no hyperref.sty), so pdflatex and xetex there
# cannot compile anything. Tectonic carries its own bundle, cached on scratch
# because home has a hard quota.
set -uo pipefail
cd "$(dirname "$0")/../../paper"
export PATH="$HOME/bin:$PATH"
export TECTONIC_CACHE_DIR=/shared/scratch/SCWF00111/tectonic_cache
command -v tectonic >/dev/null || { echo "no tectonic on PATH"; exit 1; }

tectonic -X compile main.tex --keep-logs --keep-intermediates > build.log 2>&1
rc=$?
echo "tectonic rc=$rc"
if [ $rc -ne 0 ]; then grep -E "^error|^!|error:" build.log | head -20; exit 1; fi

echo "=== overfull hboxes ==="
grep -c "Overfull \\\\hbox" main.log || true
grep "Overfull \\\\hbox" main.log | head -10
echo "=== undefined citations/refs ==="
grep -E "Citation .* undefined|Reference .* undefined" main.log | head -10
echo "=== pages ==="
grep -oE "\([0-9]+ pages" main.log | tail -1
ls -l main.pdf
