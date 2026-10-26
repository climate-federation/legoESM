"""Sample how many cores a benchmark process actually keeps busy.

Every earlier CPU-occupancy number on these lanes came from a shell one-liner
that read /proc/<pid>/stat with the wrong field offsets. After `sed 's/^.*) //'`
strips `pid (comm) `, utime and stime are at INDEX 11 and 12 of what remains
(stat fields 14 and 15); the shell probes used 13 or 14 placeholders and so read
cutime/cstime, which are the CPU of reaped children and are identically zero for
a threads-only JAX process. Both reviewers caught it independently. Hence this
file: the parse is one function with a test, not a blank count in a heredoc.

Usage:
    sample_cpu_cores.py --pattern run_cpu_mpi_scaling.py --out cores.jsonl \
        --interval 5 --duration 1800

It follows every matching process (matching on /proc/<pid>/cmdline, excluding
its own pid and any srun/slurmstepd wrapper whose command line also contains the
pattern), keeps a PER-PID baseline, and writes one JSON line per sample. The
window that matters is selected afterwards from the benchmark's own
"[phase] timing-start/-end" markers -- this tool does not guess.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")


def parse_cpu_ticks(stat_line: str) -> tuple[int, int]:
    """utime, stime in clock ticks from one /proc/<pid>/stat line.

    The comm field is parenthesised and may itself contain spaces and
    parentheses, so split on the LAST ')' rather than the first.
    """
    rest = stat_line[stat_line.rindex(")") + 1:].split()
    # rest[0] is state (field 3), so field N is rest[N - 3].
    return int(rest[11]), int(rest[12])


def cores_busy(prev_ticks: int, now_ticks: int, dt_seconds: float) -> float:
    """Mean cores busy over the interval. Ticks in, cores out -- no unit mixing:
    the shell version divided milliseconds by seconds and printed 450.0 for a
    process using 4.5 cores."""
    if dt_seconds <= 0:
        raise ValueError("dt must be positive")
    return (now_ticks - prev_ticks) / CLK_TCK / dt_seconds


def _matching_pids(pattern: str, self_pid: int) -> list[int]:
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue
        if pattern not in cmd:
            continue
        # srun and slurmstepd carry the script name in their own command line;
        # counting them mixes two processes' counters into one delta.
        if not comm.startswith("python"):
            continue
        out.append(pid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--per-thread", action="store_true",
                    help="also record per-thread busy fractions (cap vs partition)")
    args = ap.parse_args()

    prev: dict[int, tuple[float, int]] = {}
    prev_tids: dict[int, dict[str, int]] = {}
    end = time.time() + args.duration
    n = 0
    with open(args.out, "w") as fh:
        while time.time() < end:
            for pid in _matching_pids(args.pattern, os.getpid()):
                try:
                    line = Path(f"/proc/{pid}/stat").read_text()
                    nthreads = len(os.listdir(f"/proc/{pid}/task"))
                except OSError:
                    continue
                ut, st = parse_cpu_ticks(line)
                now, ticks = time.time(), ut + st
                # Per-thread busy counts separate a fixed worker-pool cap from a
                # decomposition limit: exactly K recurring busy threads means a cap,
                # a broad tile-shaped set means the work is being split that way.
                tid_busy = None
                if args.per_thread:
                    tid_busy = {}
                    for tid in os.listdir(f"/proc/{pid}/task"):
                        try:
                            tl = Path(f"/proc/{pid}/task/{tid}/stat").read_text()
                        except OSError:
                            continue
                        a, b = parse_cpu_ticks(tl)
                        tid_busy[tid] = a + b
                if pid in prev:
                    t0, c0 = prev[pid]
                    if now - t0 > 0:
                        rec = {"epoch": now, "pid": pid, "threads": nthreads,
                               "cores_busy": round(cores_busy(c0, ticks, now - t0), 3)}
                        if tid_busy is not None:
                            prior = prev_tids.get(pid, {})
                            deltas = sorted(
                                (v - prior.get(k, v)) / CLK_TCK / (now - t0)
                                for k, v in tid_busy.items())
                            rec["tid_busy_top20"] = [round(x, 3) for x in deltas[-20:]]
                            rec["tid_busy_over_10pct"] = sum(1 for x in deltas if x > 0.1)
                            prev_tids[pid] = tid_busy
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n += 1
                prev[pid] = (now, ticks)
            time.sleep(args.interval)
    print(f"[sampler] {n} samples -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
