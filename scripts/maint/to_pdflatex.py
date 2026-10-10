#!/usr/bin/env python3
"""Drop the XeTeX compatibility patches from the manuscript preamble.

sn-jnl.cls is written for pdfLaTeX and uses pdfTeX primitives directly. Building
it with Tectonic's XeTeX engine needed three patches: hyperref forced onto the
xetex driver, a no-op \\headerps@out, and a stand-in \\pdf@box with a replacement
\\doiurl. None of them belong in a file submitted to the journal, and under
pdfLaTeX none of them are needed.
"""
import pathlib, sys

P = pathlib.Path(__file__).resolve().parents[2] / "paper" / "main.tex"
s = P.read_text()

HEADER = """% Tectonic builds with XeTeX; hyperref otherwise selects a PostScript driver
% that sn-jnl's \\burl support cannot use.
\\PassOptionsToPackage{xetex}{hyperref}
"""
PATCH = """
\\makeatletter
% sn-jnl pulls in hyperref's PostScript header emission, which the XeTeX engine
% used to build this document cannot execute. Neutralising it affects only the
% PDF-mark side channel, not the typeset output.
\\providecommand{\\headerps@out}[1]{}
% The bibliography style formats DOIs through a pdfTeX-specific box primitive,
% and defines \\doiurl only once the class has loaded, so the replacement waits.
\\providecommand{\\pdf@box}{\\relax}
\\AtBeginDocument{\\providecommand{\\doiurl}[1]{}%
  \\renewcommand{\\doiurl}[1]{\\href{https://doi.org/#1}{https://doi.org/#1}}}
\\makeatother
"""
missing = [n for n, t in (("header", HEADER), ("patch block", PATCH)) if t not in s]
if missing:
    print("already converted or changed:", ", ".join(missing))
    sys.exit(1)
P.write_text(s.replace(HEADER, "", 1).replace(PATCH, "", 1))
print("removed the xetex header and the three-patch block")
