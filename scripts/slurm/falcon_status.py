#!/usr/bin/env python3
"""One-shot monitoring dashboard for the trimmed Falcon MoP-DSRL campaign.

Shows, for all 30 planned runs: done / running / pending / stalled, latest
training step, latest eval success-rate, resubmit count, and flags recent job
failures + scheduler health. Run any time:

    python scripts/slurm/falcon_status.py
"""
from __future__ import annotations
import subprocess, sys, os, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from scripts.slurm.falcon_scheduler import build_plan, RUNS_ROOT  # noqa: E402


def last_line(p: Path) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buf = b""
            step = min(4096, size)
            while step and b"\n" not in buf.strip(b"\n"):
                f.seek(max(0, size - step))
                buf = f.read(step) + buf
                if step >= size:
                    break
                step *= 2
        lines = [l for l in buf.decode(errors="replace").splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def squeue_map() -> dict:
    out = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%T"],
        capture_output=True, text=True).stdout
    m = {}
    for line in out.splitlines():
        if "|" in line:
            name, state = line.split("|", 1)
            m[name] = state
    return m


def recent_failures() -> list[str]:
    out = subprocess.run(
        ["sacct", "-u", os.environ["USER"], "--starttime", "now-2days", "-X", "-n",
         "-o", "JobName%40,State"], capture_output=True, text=True).stdout
    bad = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("mop-") and parts[1] in (
                "FAILED", "TIMEOUT", "OUT_OF_ME+", "NODE_FAIL", "CANCELLED"):
            bad.append(f"{parts[0]}  {parts[1]}")
    return bad


def main() -> int:
    plan = build_plan()
    sq = squeue_map()
    now = time.time()

    done = run = pend = stall = 0
    rows = []
    for r in plan:
        rd = r.run_dir(RUNS_ROOT)
        fs = rd / "final_summary.json"
        nm = r.job_name
        # progress
        ev = last_line(rd / "eval.csv")
        step = ""
        ml = last_line(rd / "metrics.csv")
        if ml and ml.split(",")[0].isdigit():
            step = ml.split(",")[0]
        evsr = ev.split(",")[2] if ev.count(",") >= 2 and not ev.startswith("step") else ""
        nres = ""
        rc = rd / ".resubmit_count"
        if rc.exists():
            nres = rc.read_text().strip()
        # state
        if fs.exists():
            done += 1; state = "DONE"
        elif nm in sq:
            state = sq[nm][:4]
            if state.startswith("R"): run += 1
            else: pend += 1
        else:
            # started (has ckpt) but no job + not done => stalled
            ck = list(rd.glob("checkpoint_step*.pt"))
            if ck and (now - max(c.stat().st_mtime for c in ck)) > 2400:
                stall += 1; state = "STALL"
            elif ck:
                pend += 1; state = "resub?"
            else:
                pend += 1; state = "queue"
        rows.append((f"{r.group}/{r.name}"[:40], state, step, evsr, nres))

    print(f"\n=== MoP-DSRL Falcon campaign — {done}/{len(plan)} done | "
          f"{run} running | {pend} pending | {stall} STALLED ===\n")
    print(f"  {'run':40s} {'state':7s} {'step':>8s} {'eval_sr':>8s} {'resub':>6s}")
    for name, state, step, evsr, nres in rows:
        print(f"  {name:40s} {state:7s} {step:>8s} {evsr:>8s} {nres:>6s}")

    fails = recent_failures()
    if fails:
        print(f"\n  ⚠ recent failed/timeout jobs (2d): ")
        for f in fails[:12]:
            print(f"     {f}")

    slog = RUNS_ROOT.parent / "logs" / "scheduler.log"
    if slog.exists():
        print(f"\n  scheduler last: {last_line(slog)[:100]}")
    if stall:
        print(f"\n  ⚠ {stall} STALLED run(s) — chain broke (node fail?). "
              f"Next cron tick (≤20 min) restarts them from checkpoint.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
