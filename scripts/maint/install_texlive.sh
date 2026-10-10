#!/usr/bin/env bash
# Install TeX Live to scratch so the manuscript builds with pdfLaTeX.
#
# Springer's sn-jnl.cls is written for pdfLaTeX; under XeTeX it needs three
# preamble patches for pdfTeX primitives it expects to exist, which is not
# something a submitted source file should carry. scratch rather than home
# because home has a hard quota that a TeX tree would exhaust.
set -uo pipefail
ROOT=/shared/scratch/SCWF00111/texlive
mkdir -p "$ROOT/tmp" && cd "$ROOT/tmp"

echo "=== fetch installer ==="
timeout 600 curl -sL https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
  -o install-tl.tar.gz || { echo "download failed"; exit 1; }
tar xzf install-tl.tar.gz
DIR=$(find . -maxdepth 1 -name "install-tl-*" -type d | head -1)
echo "installer: $DIR"

cat > profile.txt <<PROF
selected_scheme scheme-full
TEXDIR $ROOT/2025
TEXMFLOCAL $ROOT/texmf-local
TEXMFSYSVAR $ROOT/2025/texmf-var
TEXMFSYSCONFIG $ROOT/2025/texmf-config
TEXMFVAR ~/.texlive/texmf-var
TEXMFCONFIG ~/.texlive/texmf-config
TEXMFHOME ~/texmf
option_doc 0
option_src 0
PROF

echo "=== install (scheme-full, no docs/src) ==="
"$DIR/install-tl" -profile profile.txt -no-interaction 2>&1 | tail -30
echo "install rc=$?"

BIN=$(find "$ROOT/2025/bin" -maxdepth 1 -type d -name "*linux*" | head -1)
echo "=== bin: $BIN ==="
"$BIN/pdflatex" --version | head -2
"$BIN/bibtex"   --version | head -1
