"""Sum a jax.profiler perfetto trace by op name -- where a step's wall time
goes on one process.  ``python summarize_perfetto_trace.py <dir-or-file>``
(the newest ``perfetto_trace.json.gz`` under a directory).  Prints the
top ops by total duration per thread ('XLA Ops' is the device-side
executable; collectives show as *permute/all-reduce/all-gather thunks).

Durations are INCLUSIVE and summed over concurrent worker threads, so
the printed shares are CPU-time shares, not wall-time shares -- use them
to rank op classes, not to divide a step's wall time."""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict


def main(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.json.gz"),
                                 recursive=True), key=os.path.getmtime)
        if not files:
            raise SystemExit(f"no perfetto trace under {path}")
        path = files[-1]
    with gzip.open(path, "rt") as fh:
        ev = json.load(fh)
    events = ev["traceEvents"] if isinstance(ev, dict) else ev
    names = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            names[(e["pid"], e["tid"])] = e["args"]["name"]
    tot = defaultdict(float)
    per_thread = defaultdict(float)
    cnt = defaultdict(int)
    for e in events:
        if e.get("ph") != "X":
            continue
        th = names.get((e["pid"], e["tid"]), f"{e['pid']}/{e['tid']}")
        key = (th, e["name"])
        tot[key] += e["dur"]
        cnt[key] += 1
        per_thread[th] += e["dur"]
    print(f"trace {path}: {len(events)} events; {len(per_thread)} threads")
    # category view: op-name prefix (the .NNNN instance suffix stripped)
    # summed over EVERY XLA/Eigen worker thread -- where the device time
    # goes by op class (fusion / ppermute / all-reduce / while / copy ...)
    import re
    cat = defaultdict(float)
    ccount = defaultdict(int)
    for (th, name), dur in tot.items():
        if not (th.startswith("tf_XLA") or th.startswith("XLA")):
            continue
        base = re.sub(r"[._]\d+$", "", name)
        base = re.sub(r"[._]\d+$", "", base)
        cat[base] += dur
        ccount[base] += cnt[(th, name)]
    xla_total = sum(cat.values())
    print(f"== XLA worker threads, by op class (CPU-time, inclusive, summed "
          f"over threads): {xla_total / 1e6:.3f} s ==")
    for name, dur in sorted(cat.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {dur / 1e6:8.3f} s {100 * dur / max(xla_total, 1e-9):5.1f}%"
              f"  x{ccount[name]:7d}  {name[:80]}")
    for th, t in sorted(per_thread.items(), key=lambda kv: -kv[1])[:6]:
        print(f"== thread {th!r}: {t / 1e6:.3f} s total ==")
        rows = [(k[1], v, cnt[k]) for k, v in tot.items() if k[0] == th]
        for name, dur, n in sorted(rows, key=lambda r: -r[1])[:25]:
            print(f"  {dur / 1e6:8.3f} s  x{n:6d}  {name[:90]}")


if __name__ == "__main__":
    main(sys.argv[1])
