#!/usr/bin/env bash
# Compile the manuscript with pdfLaTeX.
#
# TeX Live lives on scratch because home has a hard quota a TeX tree exhausts;
# the conda texlive env is binaries with no TeX tree and compiles nothing.
set -uo pipefail
cd "$(dirname "$0")/../../paper"
export PATH="/shared/scratch/SCWF00111/texlive/2025/bin/x86_64-linux:$PATH"
command -v pdflatex >/dev/null || { echo "no pdflatex on PATH"; exit 1; }
pdflatex --version | head -1

rm -f main.aux main.bbl main.blg main.out main.toc main.log

run() {
    pdflatex -interaction=nonstopmode -file-line-error main.tex > "$1.log" 2>&1
    local rc=$?
    echo "$1 rc=$rc"
    if [ $rc -ne 0 ]; then
        grep -E "^[^ ]+\.(tex|cls|sty):[0-9]+:|^! " "$1.log" | head -20
        return 1
    fi
}

run p1 || exit 1
bibtex main > b.log 2>&1; echo "bibtex rc=$?"
grep -iE "error|warning--" b.log | head -10
run p2 || exit 1
run p3 || exit 1

echo "=== overfull hboxes ==="
grep -c "Overfull \\\\hbox" main.log || true
grep "Overfull \\\\hbox" main.log | head -10
echo "=== undefined citations/refs ==="
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" main.log | head -10
echo "=== pages ==="
grep -oE "\([0-9]+ pages" main.log | tail -1
ls -l main.pdf

echo "=== highlights ==="
pdflatex -interaction=nonstopmode highlights.tex > h.log 2>&1 && echo "highlights rc=0" \
  || grep -E "^! " h.log | head -5
