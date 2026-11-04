"""Aggregate bandwidth of concurrent STREAM triads over their common window.

usage: stream_sync.py <log> <n_elements_millions> <n_copies> <n_reps>
log lines: "rank=<i> ... REP <k> <t0> <t1>" (stream_triad.c with a start
time, CLOCK_MONOTONIC seconds).  Common window = [latest first-start,
earliest last-end] over all copies.  Each rep moves 3*n*8 bytes; a rep
straddling a window edge is prorated by its overlap (uniform rate within a
rep assumed).  Lower bound: only reps wholly inside the window.  Also the
sum over copies of each copy's median rep bandwidth.
"""
import collections
import re
import statistics
import sys


def aggregate(lines, mel: int, ncop: int, nrep: int) -> dict:
    B = 3 * mel * 1e6 * 8
    reps = collections.defaultdict(dict)
    for line in lines:
        m = re.search(r"rank=(\d+) .*REP (\d+) ([\d.]+) ([\d.]+)", line)
        if m:
            r, k = int(m.group(1)), int(m.group(2))
            assert k not in reps[r], f"duplicate rep {r} {k}"
            reps[r][k] = (float(m.group(3)), float(m.group(4)))
    assert sorted(reps) == list(range(ncop)), f"copies {sorted(reps)}"
    for r, d in reps.items():
        assert sorted(d) == list(range(nrep)), f"copy {r} reps {sorted(d)}"
        iv = [d[k] for k in range(nrep)]
        assert all(t1 > t0 for t0, t1 in iv), f"copy {r} non-positive rep"
        assert all(a[1] <= b[0] for a, b in zip(iv, iv[1:])), f"copy {r} reps overlap"
    reps = {r: [d[k] for k in range(nrep)] for r, d in reps.items()}
    firsts = [r[0][0] for r in reps.values()]
    lo, hi = max(firsts), min(r[-1][1] for r in reps.values())
    assert hi > lo, f"no common window: {lo} {hi}"
    every = [iv for r in reps.values() for iv in r]
    moved = sum(B * max(0.0, min(t1, hi) - max(t0, lo)) / (t1 - t0) for t0, t1 in every)
    inside = sum(B for t0, t1 in every if t0 >= lo and t1 <= hi)
    med = sum(statistics.median(B / (t1 - t0) for t0, t1 in r) for r in reps.values())
    return dict(window_s=hi - lo, start_spread_s=max(firsts) - min(firsts),
                prorated_GBs=moved / (hi - lo) / 1e9,
                contained_GBs=inside / (hi - lo) / 1e9, median_sum_GBs=med / 1e9)


if __name__ == "__main__":
    log, mel, ncop, nrep = sys.argv[1], *map(int, sys.argv[2:5])
    a = aggregate(open(log), mel, ncop, nrep)
    print(f"copies {ncop} reps/copy {nrep} start spread {a['start_spread_s']*1e3:.1f} ms "
          f"window {a['window_s']:.2f} s")
    print(f"common-window aggregate {a['prorated_GBs']:.1f} GB/s; contained-reps-only "
          f"{a['contained_GBs']:.1f} GB/s; sum of per-copy medians {a['median_sum_GBs']:.1f} GB/s")
