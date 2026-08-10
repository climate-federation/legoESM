#!/usr/bin/env python
"""Per-thunk device-timeline attribution from jax.profiler chrome traces.

Replaces the nsys route for the MPAS wait-vs-payload question: nsys
silently omitted ncclDevKernel rows on this lane (campaign 2026-08-07),
while the XLA profiler records what the runtime itself executed. Input =
the ``trace.json.gz`` files ``bench_mpas_spmd_scaling.py --profile-dir``
writes for ranks 0-3 (one node; shared clock).

Per rank, on the busiest GPU device track:
  * total span vs summed event durations -> GAP share (launch/idle),
  * duration histogram by op class (nccl* collectives vs fusions),
  * per-occurrence collective (start, end) lists in trace order.
Cross-rank (ranks on ONE node only):
  * per-occurrence collective START spread across ranks — the skew
    discriminator (GLM consult 2026-08-10): spread ~= duration - wire
    means arrival skew IS the collective's cost; spread << duration
    means the transport itself is slow.
  * calibration: matched collective END times must nearly coincide
    (participants finish together); a large END spread means the
    cross-rank clock alignment is invalid and the spread numbers must
    not be quoted.

USE
  python analyze_jax_trace_gaps.py --trace-root <dir>  # holding rank0..rank3
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path


def load_trace(rank_dir: Path) -> list[dict]:
    hits = sorted(rank_dir.rglob("*.trace.json.gz"))
    if not hits:
        raise SystemExit(f"no trace.json.gz under {rank_dir}")
    with gzip.open(hits[-1], "rt") as f:
        return json.load(f)["traceEvents"]


def device_events(events: list[dict]) -> list[dict]:
    """Complete ('X') events on the busiest GPU device track."""
    # pid -> name via process_name metadata
    pid_name = {e.get("pid"): e.get("args", {}).get("name", "")
                for e in events if e.get("name") == "process_name"}
    gpu_pids = {p for p, n in pid_name.items()
                if "GPU" in n or "gpu" in n}
    xs = [e for e in events
          if e.get("ph") == "X" and e.get("pid") in gpu_pids
          and e.get("dur", 0) > 0]
    if not xs:
        # fallback: any pid whose events look like kernels
        xs = [e for e in events if e.get("ph") == "X"
              and e.get("dur", 0) > 0]
    # busiest pid/tid pair by total duration
    from collections import defaultdict
    tot = defaultdict(float)
    for e in xs:
        tot[(e["pid"], e.get("tid"))] += e["dur"]
    if not tot:
        raise SystemExit("no complete events in trace")
    # keep ALL tids of the busiest pid (kernels spread across streams)
    busiest_pid = max(tot, key=tot.get)[0]
    return [e for e in xs if e["pid"] == busiest_pid]


def is_collective(name: str) -> bool:
    n = name.lower()
    return ("nccl" in n or "collective-permute" in n
            or "all-reduce" in n or "allreduce" in n)


def _union_ms(evs: list[dict]) -> float:
    """Total device-BUSY time as the union of event intervals — events
    on different streams overlap, so a plain duration sum overstates
    busy (the CPU smoke showed busy 33 s inside a 1.5 s span)."""
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def per_rank_summary(evs: list[dict]) -> dict:
    evs = sorted(evs, key=lambda e: e["ts"])
    t0, t1 = evs[0]["ts"], max(e["ts"] + e["dur"] for e in evs)
    span = t1 - t0
    busy = _union_ms(evs)
    coll = [e for e in evs if is_collective(e["name"])]
    comp = [e for e in evs if not is_collective(e["name"])]
    return {
        "span_ms": span / 1e3,
        "busy_ms": busy / 1e3,
        "gap_share": 1.0 - min(busy / span, 1.0),
        "n_events": len(evs),
        "n_collectives": len(coll),
        "coll_ms": sum(e["dur"] for e in coll) / 1e3,
        "comp_ms": sum(e["dur"] for e in comp) / 1e3,
        "coll_dur_us_median": (statistics.median(e["dur"] for e in coll)
                               if coll else None),
        "collectives": [(e["name"], e["ts"], e["ts"] + e["dur"])
                        for e in coll],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", required=True, type=Path)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ranks = sorted(d for d in args.trace_root.iterdir()
                   if d.is_dir() and d.name.startswith("rank"))
    if not ranks:
        raise SystemExit(f"no rank*/ dirs under {args.trace_root}")

    summaries = {}
    for rd in ranks:
        s = per_rank_summary(device_events(load_trace(rd)))
        summaries[rd.name] = s
        print(f"{rd.name}: span {s['span_ms']:.1f} ms, busy "
              f"{s['busy_ms']:.1f} ms (gap {100 * s['gap_share']:.0f}%), "
              f"collectives {s['n_collectives']} x median "
              f"{s['coll_dur_us_median'] and round(s['coll_dur_us_median'])} us "
              f"= {s['coll_ms']:.1f} ms, compute {s['comp_ms']:.1f} ms")

    # Cross-rank matching by TIME OVERLAP (participation differs per rank
    # — a coloured round runs a kernel only on its participants, so
    # occurrence-index matching is invalid; job 26854741 showed 88 vs
    # 136 events across one node's ranks).  Two collectives on two ranks
    # that overlap in wall-clock (shared node clock) and overlap by
    # >50% of the shorter one are treated as the same logical exchange.
    rank_names = sorted(summaries)
    if len(rank_names) < 2:
        print("cross-rank: <2 ranks — spread not computed")
    else:
        base = summaries[rank_names[0]]["collectives"]
        spreads, end_spreads, durs, pairs = [], [], [], 0
        for other in rank_names[1:]:
            oc = summaries[other]["collectives"]
            j = 0
            for (nm, s0, e0) in base:
                while j < len(oc) and oc[j][2] < s0:
                    j += 1
                if j >= len(oc):
                    break
                nm1, s1, e1 = oc[j]
                ov = min(e0, e1) - max(s0, s1)
                if ov <= 0 or ov < 0.5 * min(e0 - s0, e1 - s1):
                    continue
                spreads.append(abs(s1 - s0))
                end_spreads.append(abs(e1 - e0))
                durs.append(statistics.median([e0 - s0, e1 - s1]))
                pairs += 1
        if not pairs:
            print("cross-rank: no overlapping collective pairs matched")
        else:
            cal = statistics.median(end_spreads)
            md = statistics.median(durs)
            print(f"cross-rank ({len(rank_names)} ranks, {pairs} "
                  f"overlap-matched pairs):")
            print(f"  median START spread : {statistics.median(spreads):8.1f} us")
            print(f"  p90    START spread : {sorted(spreads)[int(0.9 * len(spreads))]:8.1f} us")
            print(f"  median duration     : {md:8.1f} us")
            print(f"  median END spread   : {cal:8.1f} us  (calibration — "
                  f"must be << duration for the spread to be quotable)")
            if cal > 0.5 * md:
                print("  CALIBRATION FAILED: end spread not small — clock "
                      "alignment invalid, DO NOT quote the start spread")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items()
                           if kk != "collectives"}
                       for k, v in summaries.items()}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
