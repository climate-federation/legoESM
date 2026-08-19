#!/usr/bin/env python
"""Payload or wait? Compare ONE halo collective from BOTH of its partners.

THE QUESTION.  The MPAS GPU lane spends 5.94 of 9.72 ms per step inside
``ncclDevKernel_SendRecv`` (s9/np64), median 216 us against a ~30 us wire
latency.  Two mechanisms explain that and they imply opposite engineering:
cutting the ROUND COUNT pays well if the time is idle waiting, and pays only
the per-round latency if the time is real transfer.

WHY A SINGLE-RANK TRACE CANNOT ANSWER IT.  Grouped by schedule slot, those
durations reproduce to CV <= 1.5 % across steps.  That refutes RANDOM jitter
-- and nothing else.  A consistently late peer is exactly as reproducible.
Wait is an inter-rank quantity; one timeline cannot see it.

THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives at
the collective and ends when the transfer completes.  So for ONE collective
seen from BOTH partners:

  * both durations ~equal  -> the time is transfer.  PAYLOAD-bound.
  * one long, one short    -> the long one arrived early and idled.  WAIT-
                              bound, and (long - short) is the skew.

HOW COLLECTIVES ARE MATCHED, and why not by overlap.  The obvious method --
pair kernels whose [start, end] intervals overlap -- does not work here and
its own control proves it: all 64 ranks issue their round-r collective at
about the same time, so rank 0's kernel overlaps a NON-partner's just as
readily as its true partner's.  Overlap is necessary, never sufficient.

This uses the SCHEDULE as ground truth instead, which makes the pairing
exact.  ``spmd_schedule_cost(..., round_profile_for_device=d)`` gives the
rounds device ``d`` participates in, IN THE ORDER IT ISSUES THEM.  If A's
profile shows it exchanges with B at global round r, and B's profile shows
the mirror-image entry at the same r, then A's k-th halo fill issues that
collective at position ``k*len(A_profile) + index_of_r_in_A`` and B's at
``k*len(B_profile) + index_of_r_in_B``.  Same collective, by construction.

Three things are then checked rather than assumed:
  * SYMMETRY -- B's profile must name A as its partner at the same round.
    A one-sided entry means the schedule was misread.
  * FILL COUNT -- both ranks must have executed the same number of halo
    fills (``n_calls / len(profile)``).  Ranks step in lockstep, so a
    mismatch means the traces are not of the same run or one is truncated.
  * DIVISIBILITY -- each rank's call count must be a whole number of fills.

NON-PARTNER CONTROL.  Passing a pair that does not exchange (e.g. rank 0 and
a same-node rank that is not its neighbour) finds no shared round and is
reported as such.  That is the correct outcome and it is what shows the
matching is schedule-driven rather than coincidence -- an overlap-based
method would happily have "matched" that pair.

SAME NODE ONLY.  Timestamps come from independent nsys processes and share a
timebase only within one machine (``CLOCK_MONOTONIC`` is per-machine and
undisciplined across nodes; inter-node drift dwarfs a 216 us signal).  The
capture co-locates the profiled ranks and records the rank->node map; this
script requires that map and REFUSES a cross-node pair.

No verdict string is printed.  The interpretation belongs in the analysis
once the controls pass, not baked into the tool where it gets echoed back as
evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import scaling_metadata  # noqa: E402

_SENDRECV_LIKE = "ncclDevKernel_SendRecv%"


def load_kernels(sqlite_path: Path, kernel_like: str = _SENDRECV_LIKE):
    """``(start_ns, end_ns)`` per collective, in issue order.

    Raises on an EMPTY result: a name that matches nothing would otherwise
    make every downstream count zero and read as "no skew".
    """
    if not sqlite_path.is_file():
        raise SystemExit(f"missing sqlite: {sqlite_path}")
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName = s.id "
            "WHERE s.value LIKE ? ORDER BY k.start", (kernel_like,)).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(
            f"{sqlite_path.name}: no kernel matches {kernel_like!r}. The NCCL "
            f"kernel may be named differently in this build — check with "
            f"SELECT DISTINCT value FROM StringIds WHERE value LIKE '%nccl%'.")
    out = []
    for start, end in rows:
        if start is None or end is None or end < start:
            raise SystemExit(
                f"{sqlite_path.name}: bad kernel row start={start} end={end}")
        out.append((int(start), int(end)))
    return out


def read_rank_nodes(path: Path):
    """Parse the capture's ``_rank_nodes.tsv`` into ``{rank: node}``."""
    if not path.is_file():
        raise SystemExit(
            f"missing {path}. It is written by _nsys_rank_wrapper.sh and is "
            f"what proves the profiled ranks shared a node; without it a "
            f"cross-node pair cannot be refused and its timestamps would be "
            f"incomparable.")
    nodes = {}
    for line in path.read_text().splitlines():
        fields = dict(f.split("=", 1) for f in line.strip().split("\t") if "=" in f)
        if "rank" in fields and "node" in fields:
            nodes[int(fields["rank"])] = fields["node"]
    if not nodes:
        raise SystemExit(f"{path}: no rank=/node= records parsed")
    return nodes


def shared_round(profile_a, profile_b, rank_a: int, rank_b: int):
    """Global round where A and B exchange, plus each one's issue index.

    Returns ``(round, index_in_a, index_in_b)`` or ``None`` when the two do
    not exchange at all (the non-partner control).  Enforces SYMMETRY: B must
    name A at the same round, otherwise the schedule has been misread and any
    pairing built on it would be wrong.
    """
    a_hits = [(i, e) for i, e in enumerate(profile_a) if e["partner"] == rank_b]
    if not a_hits:
        return None
    if len(a_hits) > 1:
        raise SystemExit(
            f"rank {rank_a} exchanges with rank {rank_b} in {len(a_hits)} "
            f"rounds; this pairing is not unique and the script would have to "
            f"guess which collective is which.")
    idx_a, entry = a_hits[0]
    b_hits = [i for i, e in enumerate(profile_b)
              if e["partner"] == rank_a and e["round"] == entry["round"]]
    if not b_hits:
        raise SystemExit(
            f"asymmetric schedule: rank {rank_a} names {rank_b} at round "
            f"{entry['round']}, but {rank_b}'s profile has no mirror entry. "
            f"The schedule has been misread; refusing to pair.")
    return entry["round"], idx_a, b_hits[0]


def compare(kernels_a, kernels_b, len_a, len_b, idx_a, idx_b, drop_fills):
    """One record per halo fill, comparing the SAME collective on both ranks."""
    if len(kernels_a) % len_a or len(kernels_b) % len_b:
        raise SystemExit(
            f"call counts are not whole fills: rank A has {len(kernels_a)} "
            f"calls over {len_a} rounds/fill, rank B {len(kernels_b)} over "
            f"{len_b}. The traces and the schedule disagree about the run.")
    fills_a, fills_b = len(kernels_a) // len_a, len(kernels_b) // len_b
    if fills_a != fills_b:
        raise SystemExit(
            f"rank A executed {fills_a} halo fills, rank B {fills_b}. Ranks "
            f"step in lockstep, so these are not the same run (or one trace "
            f"is truncated).")
    recs = []
    for k in range(drop_fills, fills_a):
        sa, ea = kernels_a[k * len_a + idx_a]
        sb, eb = kernels_b[k * len_b + idx_b]
        dur_a, dur_b = ea - sa, eb - sb
        recs.append({
            "fill": k,
            "a_dur_us": dur_a / 1e3,
            "b_dur_us": dur_b / 1e3,
            # >0: A started later, i.e. B was waiting for A.
            "start_delta_us": (sa - sb) / 1e3,
            # The idle time the EARLIER arriver spent waiting, which is the
            # quantity a round-count cut would actually remove.
            "wait_estimate_us": abs(dur_a - dur_b) / 1e3,
            "ratio_long_short": (max(dur_a, dur_b) / min(dur_a, dur_b)
                                 if min(dur_a, dur_b) > 0 else None),
        })
    return recs


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqlite-dir", required=True, type=Path,
                   help="Capture dir holding rank_N.sqlite + _rank_nodes.tsv")
    p.add_argument("--pairs", required=True,
                   help="Comma-separated rank pairs, e.g. 0:1,0:2,0:3 "
                        "(include a non-partner as the control).")
    p.add_argument("--subdivision", type=int, required=True)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--partition-method", default="sfc")
    p.add_argument("--lloyd", type=int, default=0)
    p.add_argument("--reorder-target", type=int, default=None,
                   help="The run's --reorder-for, when it differs from the "
                        "device count. A different ownership is a different "
                        "schedule, so getting this wrong misaligns everything.")
    p.add_argument("--drop-fills", type=int, default=3,
                   help="Leading halo fills to discard as warmup/compile.")
    p.add_argument("--out", default="results/a1/nsys_rank_skew.json")
    args = p.parse_args()

    pairs = []
    for item in (s.strip() for s in args.pairs.split(",")):
        if not item:
            continue
        a, _, b = item.partition(":")
        if not b:
            raise SystemExit(f"bad pair {item!r}; expected 'A:B'")
        pairs.append((int(a), int(b)))
    if not pairs:
        raise SystemExit("--pairs selected nothing")

    nodes = read_rank_nodes(args.sqlite_dir / "_rank_nodes.tsv")
    for a, b in pairs:
        for r in (a, b):
            if r not in nodes:
                raise SystemExit(f"rank {r} absent from the rank->node map")
        if nodes[a] != nodes[b]:
            raise SystemExit(
                f"pair {a}:{b} spans nodes ({nodes[a]} vs {nodes[b]}). "
                f"Timestamps from different machines share no timebase, so "
                f"any skew computed from them would be drift, not physics.")

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

    print(f"[mesh] L{args.subdivision} lloyd={args.lloyd} …", flush=True)
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
                               lloyd_iterations=args.lloyd)
    profiles, kernels = {}, {}
    for rank in sorted({r for pr in pairs for r in pr}):
        cost = spmd_schedule_cost(mesh, args.n_devices,
                                  method=args.partition_method,
                                  reorder_target=args.reorder_target,
                                  round_profile_for_device=rank)
        profiles[rank] = cost["round_profile"]
        kernels[rank] = load_kernels(args.sqlite_dir / f"rank_{rank}.sqlite")
        print(f"[rank {rank}] node={nodes[rank]} "
              f"{len(kernels[rank])} collectives, "
              f"{len(profiles[rank])} rounds/fill", flush=True)

    out_pairs = {}
    for a, b in pairs:
        key = f"{a}:{b}"
        shared = shared_round(profiles[a], profiles[b], a, b)
        if shared is None:
            print(f"\n[{key}] NOT PARTNERS — no shared round. This is the "
                  f"control: a schedule-driven pairing finds nothing here, "
                  f"where an overlap-based one would have matched.", flush=True)
            out_pairs[key] = {"partners": False}
            continue
        rnd, idx_a, idx_b = shared
        recs = compare(kernels[a], kernels[b], len(profiles[a]),
                       len(profiles[b]), idx_a, idx_b, args.drop_fills)
        if not recs:
            raise SystemExit(
                f"{key}: no fills left after dropping {args.drop_fills}")
        med_a = st.median(r["a_dur_us"] for r in recs)
        med_b = st.median(r["b_dur_us"] for r in recs)
        med_wait = st.median(r["wait_estimate_us"] for r in recs)
        med_delta = st.median(r["start_delta_us"] for r in recs)
        payload = profiles[a][idx_a]
        out_pairs[key] = {
            "partners": True, "round": rnd, "n_fills": len(recs),
            "halo_cells": payload["halo_cells"],
            "halo_edges": payload["halo_edges"],
            "median_a_us": round(med_a, 2), "median_b_us": round(med_b, 2),
            "median_wait_estimate_us": round(med_wait, 2),
            "median_start_delta_us": round(med_delta, 2),
            "wait_fraction_of_longer": round(
                med_wait / max(med_a, med_b), 4) if max(med_a, med_b) else None,
            "fills": recs,
        }
        print(f"\n[{key}] round {rnd}: cells={payload['halo_cells']} "
              f"edges={payload['halo_edges']}, {len(recs)} fills")
        print(f"   rank {a} median {med_a:8.1f} us")
        print(f"   rank {b} median {med_b:8.1f} us")
        print(f"   |difference| {med_wait:8.1f} us  "
              f"= {100 * med_wait / max(med_a, med_b):.1f}% of the longer")
        print(f"   start delta  {med_delta:+8.1f} us "
              f"({'A later' if med_delta > 0 else 'B later'})")

    payload_out = {
        "pairs": out_pairs,
        "rank_nodes": {str(k): v for k, v in nodes.items()},
        "metadata": scaling_metadata(
            grid="voronoi", component="atmosphere",
            resolution=f"L{args.subdivision}", n_levels=0, precision="n/a",
            decomposition="cell_partition", solver_variant="n/a",
            scaling_kind="halo-skew-attribution", transport="nccl",
            extra={"n_devices": args.n_devices,
                   "partition_method": args.partition_method,
                   "lloyd_iterations": args.lloyd,
                   "reorder_target": args.reorder_target,
                   "drop_fills": args.drop_fills,
                   "sqlite_dir": str(args.sqlite_dir)}),
    }
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload_out, f, indent=2)
    print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
