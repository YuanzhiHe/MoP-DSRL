#!/usr/bin/env python3
"""The README quotes the paper's numbers; check they still agree.

The README had drifted six months out of date and described a different method
from the one the manuscript presents, which matters because the paper's data
availability statement sends readers here. Every figure it quotes is checked
against main.tex, and the claims about the method against what the driver does.
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
readme = (root / "README.md").read_text()
tex = (root / "paper" / "main.tex").read_text()

ok = True
print(f"{'quantity':<34}{'README':>10}{'main.tex':>12}  match")
for label, pat_r, pat_t in (
        ("vs entropy, Avoiding", r"\*\*\+0\.137\*\*", r"\$0\.137\$"),
        ("vs entropy, Pushing", r"\*\*\+0\.095\*\*", r"\$0\.095\$"),
        ("vs no mode term, Avoiding", r"\*\*\+0\.201\*\*", r"\$0\.201\$"),
        ("vs no mode term, Pushing", r"\+0\.072", r"\$0\.072\$"),
        ("vs learned skills, Avoiding", r"\+0\.146", r"\$0\.146\$"),
        ("vs learned skills, Pushing", r"\+0\.032", r"\$0\.032\$"),
        ("slope per unit skew", r"0\.82 per unit", r"\$0\.82\$ per unit"),
        ("margin at the most skewed", r"0\.284", r"\$0\.284\$"),
        ("frozen policies", r"38 independently", r"38\s*\n?independently"),
        ("fine-tuning runs", r"1,235", r"1,235"),
        ("skew range", r"0\.027 to 1\.296", r"\$0\.027\$ to \$1\.296\$"),
):
    a, b = bool(re.search(pat_r, readme)), bool(re.search(pat_t, tex))
    ok &= a and b
    print(f"  {label:<32}{'yes' if a else 'MISSING':>10}"
          f"{'yes' if b else 'MISSING':>12}  {'ok' if a and b else 'CHECK'}")

print("\nmethod description")
driver = (root / "scripts" / "run_d3il_headtohead.py").read_text()
for label, needle, where in (
        ("total variation is the default", 'default="tv"', driver),
        ("modes come from geometry", "coarsen", driver),
        ("no K-means in the driver", "KMeans", driver),
):
    present = needle in where
    want = label != "no K-means in the driver"
    ok &= present == want
    print(f"  {label:<32}{'present' if present else 'absent':>10}"
          f"  {'ok' if present == want else 'CHECK'}")

print("\nVERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
