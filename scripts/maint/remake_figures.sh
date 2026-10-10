#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
python scripts/make_figures_v2.py 2>&1 | tail -20
echo "rc=$?"
python - <<'PY'
from pypdf import PdfReader
import pathlib
for f in sorted(pathlib.Path("paper/figures").glob("*.pdf")):
    b = PdfReader(str(f)).pages[0].mediabox
    w = float(b.width)
    print(f"{f.name}: {w:.1f} x {float(b.height):.1f} pt -> scale at 372pt textwidth = {372/w:.4f}")
PY
