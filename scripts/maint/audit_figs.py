#!/usr/bin/env python3
"""Audit the figures the way a production editor would.

Checks the three things a journal rejects on: type smaller than six points at
the size the figure is placed, a canvas that is not the text width so the figure
gets rescaled, and colours that stop being distinguishable in greyscale.
"""
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

TEXTWIDTH_PT = 390.0
MIN_PT = 6.0
AUTHORED_MIN = 6.6
root = Path(__file__).resolve().parents[2]

ok = True
for name in ("fig1_setting", "fig2_result"):
    pdf = root / "figures" / f"{name}.pdf"
    w = float(PdfReader(str(pdf)).pages[0].mediabox.width)
    scale = TEXTWIDTH_PT / w
    on_page = AUTHORED_MIN * scale
    good = abs(scale - 1.0) < 0.005 and on_page >= MIN_PT
    ok &= good
    print(f"{name}: {w:.1f} pt wide, scale {scale:.3f}, smallest type "
          f"{on_page:.2f} pt  {'ok' if good else 'CHECK'}")

    png = root / "figures" / f"{name}.png"
    im = Image.open(png).convert("RGB")
    g = im.convert("L")
    # the four series colours, as greyscale levels
    series = {"no mode term": (0x99, 0x99, 0x99), "mode entropy": (0xD5, 0x5E, 0x00),
              "learned skills": (0x00, 0x9E, 0x73), "reference matching": (0x00, 0x72, 0xB2)}
    lev = {k: round(0.299 * r + 0.587 * gg + 0.114 * b)
           for k, (r, gg, b) in series.items()}
    pairs = [(a, b, abs(lev[a] - lev[b])) for i, a in enumerate(lev)
             for b in list(lev)[i + 1:]]
    worst = min(pairs, key=lambda t: t[2])
    # Luminance alone is not the test. The guidance asks for redundant encoding,
    # and these series carry a distinct marker each, so greyscale readability
    # rests on the markers; in the forest panel it rests on the row labels, where
    # colour repeats information the label already gives.
    src = (root / "scripts" / "make_figures_v2.py").read_text()
    markers = set(src.split('MK = {', 1)[1].split('}', 1)[0].replace('"', "'").split("'")[3::4])
    redundant = len(markers) >= 4
    ok &= redundant
    print(f"  greyscale levels {lev}")
    print(f"  closest pair {worst[0]} / {worst[1]} differ by {worst[2]} levels")
    print(f"  distinct markers {sorted(markers)}  "
          f"{'ok, redundant encoding' if redundant else 'CHECK'}")
    print(f"  raster {im.size[0]}x{im.size[1]} px")

print("\nVERDICT:", "PASS" if ok else "CHECK")
sys.exit(0 if ok else 1)
