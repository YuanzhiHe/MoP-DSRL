#!/usr/bin/env bash
# Typeset the paper on this cluster.
#
# There is no TeX on the login node's PATH. The only complete distribution
# here is the 2018 TeX Live under the older cluster's tree, whose elsarticle
# predates the \affiliation{organization=...} syntax this paper uses, so a
# current class sits in TEXMFHOME and takes precedence over the 2018 one.
# The conda environment named texlive holds pdfTeX 2026 but no macro tree at
# all: no latex.ltx, no document class, no format file. It cannot typeset.
#
#   ./build.sh          three passes, report errors
#   ./build.sh clean    remove the intermediate files
set -euo pipefail
cd "$(dirname "$0")"

export PATH=/shared/apps/hawk/tools/texlive/2018/bin/x86_64-linux:$PATH
export TEXMFHOME=$HOME/texmf

if [ "${1:-}" = clean ]; then
  rm -f main.aux main.log main.out main.spl main.blg
  echo "cleaned"; exit 0
fi

command -v pdflatex >/dev/null || { echo "pdflatex not found"; exit 1; }
[ -n "$(kpsewhich elsarticle.cls)" ] || {
  echo "elsarticle.cls not found; it belongs in $TEXMFHOME/tex/latex/elsarticle"
  exit 1; }

for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode main.tex > "build-pass$pass.log" 2>&1 || true
done

if grep -qE '^! ' build-pass3.log; then
  echo "FAILED"; grep -E '^! ' -A3 build-pass3.log | head -20; exit 1
fi
bad=$(grep -cE 'Reference .* undefined|Citation .* undefined' build-pass3.log || true)
echo "pdflatex: $(pdflatex --version | head -1)"
echo "class:    $(kpsewhich elsarticle.cls)"
grep -E 'Output written' build-pass3.log
echo "undefined references or citations: $bad"
rm -f build-pass1.log build-pass2.log build-pass3.log
