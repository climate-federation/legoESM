#!/usr/bin/env python
"""Is a halo collective's time PAYLOAD or WAIT? Decide it from an nsys trace.

The MPAS GPU lane spends most of its step inside ``ncclDevKernel_SendRecv``
(s9/np64: 5.94 of 9.72 ms on rank 0), at a median far above the clean wire
estimate.  Two mechanisms explain that, and they imply OPPOSITE engineering:

* **payload-bound** — the time is real bytes on the wire.  Cutting the ROUND
  COUNT then saves only the per-round latency (~30 us x rounds removed), not
  the transfer, because the same bytes still have to move.  That is the LOW
  end of a round-depth partitioner's value.
* **wait-bound** — the collective blocks until the slowest peer arrives, so
  the duration is mostly idle.  Cutting rounds then saves much more, but so
  would fixing the imbalance, and a partitioner that ignores it may buy
  nothing.

WHAT THIS SCRIPT CAN AND CANNOT DECIDE -- read this before quoting it.

It reports two things from an EXISTING trace, with no GPU job:

1. **Reproducibility.** Durations grouped by position in the repeating
   schedule, as within-position CV against the overall CV.
2. **Payload correlation.** Each position paired with the payload the
   production schedule says that round moves (``spmd_schedule_cost(...,
   round_profile_for_device=)``), as a Pearson r.

Neither SETTLES payload-vs-wait, and an earlier version of this docstring
wrongly claimed both did:

* Low within-position CV refutes only RANDOM jitter.  A consistently late
  peer, repeatable stream serialisation, or a fixed topology path produces
  stable, position-dependent WAIT that is just as reproducible.
* A high correlation is ASSOCIATIVE, not causal.  Payload and the identity
  (hence workload) of the round's partner are both functions of schedule
  position, so they are confounded by construction.

What settles it needs measurements this trace does not contain, both GPU
jobs, and both small:

* **Cross-rank arrival skew** -- a MULTI-RANK capture, timing each rank's
  entry into the same collective.  Wait is an inter-rank quantity and a
  single-rank trace cannot see it.  (``cube_skew_j26526100`` is an example
  of the 4-rank shape needed.)
* **A payload A/B at fixed schedule** -- change the packed width (tracer
  count or precision) while holding mesh, device count and partition fixed.
  Payload-bound time moves with width; wait does not.

Use this script to size the term, to check reproducibility, and to prune --
not to award a mechanism.

ALIGNMENT, the thing that silently corrupts this comparison: an nsys capture
is per RANK, and a rank issues a collective only in the colour classes that
touch it.  At s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill while the
graph's ``max_degree`` is 11.  So the k-th observed call is the k-th round
CONTAINING THAT DEVICE.  ``round_profile_for_device`` returns exactly that
subset, in order.  The check that this is the RIGHT subset is that the
detected period divides the trace length; when it does not, the schedule and
the trace describe different runs and the script REFUSES rather than pairing
rounds with unrelated collectives.

Run (no GPU needed; the mesh build dominates at subdiv>=9):
  python scripts/bench/analyze_nsys_halo_rounds.py \\
      --sqlite /scratch/.../s9np64_rank0.sqlite \\
      --subdivision 9 --n-devices 64 --partition-method sfc --lloyd 0 \\
      --out results/a1/nsys_halo_rounds_s9np64.json
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

# Measured Levante fabric constants (campaign jobs 26677438/39): IB latency
# is flat with communicator size but bandwidth HALVES past 2 nodes.
_IB_LATENCY_US = 29.7
_IB_BANDWIDTH_GBS = 12.1

_SENDRECV_LIKE = "ncclDevKernel_SendRecv%"


def sendrecv_durations_us(sqlite_path: str, kernel_like: str = _SENDRECV_LIKE):
    """Durations (us) of every matching collective kernel, in launch order."""
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
            f"{sqlite_path}: no kernel matching {kernel_like!r}. Wrong trace, "
            f"or nsys captured no CUDA kernels — either way there is nothing "
            f"to measure.")
    return [(e - s) / 1e3 for s, e in rows]


def _within_cv(durations, period, min_cycles):
    cvs = []
    for pos in range(period):
        group = durations[pos::period]
        if len(group) >= min_cycles and st.mean(group) > 0:
            cvs.append(st.pstdev(group) / st.mean(group))
    return st.mean(cvs) if cvs else None


def detect_period(durations, device_rounds, min_cycles: int = 3,
                  max_fills: int = 8, tolerance: float = 1.25):
    """Schedule period, chosen among STRUCTURALLY POSSIBLE candidates only.

    Returns ``(period, within_cv, overall_cv, candidates)``.

    A free scan over every period is INVALID here, and the first version of
    this script was: within-position CV falls mechanically as the period
    grows, because each position then holds fewer samples.  Scanning 2..64
    over a 288-call trace duly picked 64 -- the largest allowed -- and would
    have picked whatever ceiling it was given.

    The schedule constrains it instead.  One halo fill costs the device
    ``device_rounds`` collectives, and a step is a whole number of fills, so
    the period must be ``device_rounds * k``.  Among those few candidates,
    take the SMALLEST whose within-position CV is within *tolerance* of the
    best: a larger multiple always fits at least as well (k and 2k induce
    nested groupings), so preferring the smallest is what keeps the answer
    from inflating.
    """
    overall = st.pstdev(durations) / st.mean(durations)
    candidates = []
    for k in range(1, max_fills + 1):
        period = device_rounds * k
        if len(durations) // period < min_cycles:
            break
        cv = _within_cv(durations, period, min_cycles)
        if cv is not None:
            candidates.append({"fills": k, "period": period,
                               "within_cv": round(cv, 4)})
    if not candidates:
        raise SystemExit(
            f"cannot detect a period: {len(durations)} calls is under "
            f"{min_cycles} cycles of even one fill ({device_rounds} rounds). "
            f"Capture more steps.")
    best_cv = min(c["within_cv"] for c in candidates)
    chosen = next(c for c in candidates
                  if c["within_cv"] <= best_cv * tolerance)
    return chosen["period"], chosen["within_cv"], overall, candidates


def pearson(xs, ys):
    import math

    if len(xs) != len(ys):
        # zip() would silently truncate and return a number for mismatched
        # series -- a plausible-looking wrong correlation.
        raise ValueError(
            f"pearson: series differ in length ({len(xs)} vs {len(ys)})")
    if not all(math.isfinite(v) for v in (*xs, *ys)):
        raise ValueError("pearson: non-finite value in input")
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    if sx == 0 or sy == 0:
        # A constant series has no correlation to report; saying 0.0 would
        # read as "payload does not explain it" when the truth is "payload
        # does not vary here, so this trace cannot answer the question".
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)


def payload_values(halo_cells, halo_edges, cell_width, nlev):
    """Values one device puts on the wire for a round.

    Read off the production exchange (``_ppermute_halo_fill``): it sends
    ``concatenate(cell_pack[sc].ravel(), u_shard[se].ravel())``, i.e.
    ``halo_cells * cell_width + halo_edges * nlev`` values, where the row
    counts are the round's PADDED extents -- the schedule pads every pair to
    the round maximum and ppermute moves that whole buffer, so a pair's own
    send count does not set its cost.

    The two terms carry DIFFERENT widths (the packed cell record vs one
    edge field over levels), so ``halo_cells + halo_edges`` is NOT the
    payload and correlating against it mis-weights the mixture.
    """
    return halo_cells * cell_width + halo_edges * nlev


def modeled_us(values, bytes_per_value, latency_us, bandwidth_gbs):
    """Latency + bytes/bandwidth for one round, as a REFERENCE not a truth."""
    return latency_us + (values * bytes_per_value) / (bandwidth_gbs * 1e9) * 1e6


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqlite", required=True, help="nsys-exported .sqlite")
    p.add_argument("--subdivision", type=int, required=True)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--partition-method", default="sfc")
    p.add_argument("--lloyd", type=int, default=0,
                   help="Must match the mesh the traced run used.")
    p.add_argument("--reorder-target", type=int, default=None,
                   help="Device count the traced run's mesh was REORDERED "
                        "for, when it differs from --n-devices (the scaling "
                        "bench's --reorder-for). Ownership built for one "
                        "count and run at another is a DIFFERENT partition "
                        "with a different round count, so getting this wrong "
                        "misaligns every pairing. Default: same as "
                        "--n-devices.")
    p.add_argument("--device", type=int, default=0,
                   help="Rank whose trace this is (default 0).")
    p.add_argument("--nlev", type=int, default=26)
    p.add_argument("--n-tracers", type=int, default=0,
                   help="Tracers the traced run carried. Sets the packed "
                        "cell width via the production layout "
                        "(_pack_cell_state: T | p_s | phis | tracers), i.e. "
                        "W = nlev*(1+n_tracers)+2. Read it off the run's "
                        "manifest; guessing changes the payload and so "
                        "changes the correlation.")
    p.add_argument("--cell-pack-width", type=int, default=None,
                   help="Override W directly instead of deriving it from "
                        "--n-tracers. W re-weights the cell term against the "
                        "edge term, so it MOVES the correlation -- it is not "
                        "a free parameter. The report gives the cells-only "
                        "correlation alongside as a sensitivity.")
    p.add_argument("--bytes-per-value", type=int, default=4)
    p.add_argument("--comm-latency-us", type=float, default=_IB_LATENCY_US)
    p.add_argument("--comm-bandwidth-gbs", type=float,
                   default=_IB_BANDWIDTH_GBS)
    p.add_argument("--drop-first-cycles", type=int, default=3,
                   help="Cycles to discard as warmup/compile.")
    p.add_argument("--kernel-like", default=_SENDRECV_LIKE)
    p.add_argument("--out", default="results/a1/nsys_halo_rounds.json")
    args = p.parse_args()

    if args.lloyd < 0:
        raise SystemExit(f"--lloyd must be >= 0, got {args.lloyd}")
    if args.subdivision < 0:
        raise SystemExit(
            f"--subdivision must be >= 0, got {args.subdivision}")

    # Production layout (_pack_cell_state): T (nlev) | p_s (1) | phis (1) |
    # tracers (nlev*n_q).  Defaulting W to nlev, as an earlier version did,
    # understates every cell term by 2 and drops tracers entirely.
    cell_width = (args.nlev * (1 + args.n_tracers) + 2
                  if args.cell_pack_width is None else args.cell_pack_width)
    durations = sendrecv_durations_us(args.sqlite, args.kernel_like)
    print(f"[trace] {len(durations)} collectives in {args.sqlite}", flush=True)

    # The SCHEDULE must be built first: it supplies the only structurally
    # valid period candidates (see detect_period).
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

    print(f"[mesh] building L{args.subdivision} lloyd={args.lloyd} …",
          flush=True)
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
                               lloyd_iterations=args.lloyd)
    cost = spmd_schedule_cost(mesh, args.n_devices,
                              method=args.partition_method,
                              reorder_target=args.reorder_target,
                              round_profile_for_device=args.device)
    profile = cost["round_profile"]
    print(f"[sched] n_rounds={cost['n_rounds']} "
          f"max_degree={cost['max_degree']} "
          f"device {args.device} participates in {len(profile)}", flush=True)
    if not profile:
        raise SystemExit(
            f"device {args.device} participates in no round; nothing to pair.")

    period, within_cv, overall_cv, candidates = detect_period(
        durations, len(profile))
    fills_per_cycle = period // len(profile)
    print(f"[align] period={period} = {len(profile)} rounds x "
          f"{fills_per_cycle} fill(s); within-position CV {within_cv:.3f} "
          f"vs overall {overall_cv:.3f}", flush=True)
    print(f"[align] candidates: {candidates}", flush=True)
    # THE REAL CONSISTENCY CHECK. An earlier version guarded on
    # `period % len(profile) == 0`, which is VACUOUS: the period is
    # constructed as len(profile) * k, so it can never fail. The check with
    # teeth is that the trace length is a whole number of periods -- the
    # capture covers complete steps, so if the schedule's round count
    # described this run, its period would divide the call count.
    #
    # This is what caught the s9/np64 case: the trace holds 288 calls (12
    # steps x 24), but the schedule scored here gives device 0 seven rounds,
    # and no multiple of 7 divides 288. Reporting a correlation from that
    # pairing would have been a confident wrong answer (it read r = +0.155,
    # i.e. "not payload-bound") built on rounds paired with unrelated
    # collectives.
    if len(durations) % period:
        raise SystemExit(
            f"REFUSING to pair: the trace has {len(durations)} collectives, "
            f"which is not a whole number of periods of {period} "
            f"({len(durations) / period:.2f}). The schedule scored here "
            f"(device {args.device} in {len(profile)} of {cost['n_rounds']} "
            f"rounds) does not describe the traced run.\n"
            f"Reconcile them before trusting any number: check --subdivision, "
            f"--n-devices, --partition-method and --lloyd against the run's "
            f"own manifest, and check whether that run passed a "
            f"--reorder-for different from its device count (the scorer takes "
            f"reorder_target for exactly that case).")

    steady = durations[period * args.drop_first_cycles:]
    if len(steady) < period * 2:
        raise SystemExit(
            f"only {len(steady)} calls after dropping "
            f"{args.drop_first_cycles} cycles of {period}; capture more steps "
            f"or lower --drop-first-cycles.")

    slots = []
    for pos in range(period):
        group = steady[pos::period]
        entry = profile[pos % len(profile)]
        med = st.median(group)
        slots.append({
            "position": pos,
            "fill": pos // len(profile),
            "round": entry["round"],
            "partner": entry["partner"],
            "halo_cells": entry["halo_cells"],
            "halo_edges": entry["halo_edges"],
            "n": len(group),
            "median_us": round(med, 2),
            "cv": round(st.pstdev(group) / st.mean(group), 4),
            "payload_values": payload_values(
                entry["halo_cells"], entry["halo_edges"], cell_width,
                args.nlev),
            "modeled_us": round(modeled_us(
                payload_values(entry["halo_cells"], entry["halo_edges"],
                               cell_width, args.nlev),
                args.bytes_per_value, args.comm_latency_us,
                args.comm_bandwidth_gbs), 2),
        })

    payload = [s["payload_values"] for s in slots]
    measured = [s["median_us"] for s in slots]
    r = pearson(payload, measured)
    # Sensitivity on the one assumption in the payload model (the cell/edge
    # width ratio): if the cells-only correlation agrees, the verdict does
    # not hinge on --cell-pack-width.
    r_cells = pearson([s["halo_cells"] for s in slots], measured)
    stable = [s for s in slots if s["cv"] <= 0.05]

    print()
    print(f"{'pos':>3} {'fill':>4} {'rnd':>4} {'partner':>7} {'cells':>8} "
          f"{'edges':>8} {'values':>10} {'median_us':>10} {'cv':>7} "
          f"{'model_us':>9}")
    for s in slots:
        print(f"{s['position']:3d} {s['fill']:4d} {s['round']:4d} "
              f"{s['partner']:7d} {s['halo_cells']:8d} {s['halo_edges']:8d} "
              f"{s['payload_values']:10d} {s['median_us']:10.1f} "
              f"{s['cv']:7.3f} {s['modeled_us']:9.1f}")

    print()
    print(f"[reproducibility] {len(stable)}/{len(slots)} positions have "
          f"CV <= 0.05; overall CV across all calls is {overall_cv:.3f}")
    print(f"[payload correlation] pearson r = "
          f"{'n/a' if r is None else f'{r:+.3f}'}"
          f"{'' if r is None else f'  (r^2 = {r * r:.3f})'}"
          f"   [cells-only sensitivity r = "
          f"{'n/a' if r_cells is None else f'{r_cells:+.3f}'}]")
    print(f"[totals] one cycle of medians = {sum(measured) / 1e3:.3f} ms")

    out = {
        "slots": slots,
        "period": period,
        "fills_per_cycle": fills_per_cycle,
        "period_candidates": candidates,
        "within_position_cv": round(within_cv, 4),
        "overall_cv": round(overall_cv, 4),
        "payload_correlation_r": None if r is None else round(r, 4),
        "payload_correlation_r_cells_only": (
            None if r_cells is None else round(r_cells, 4)),
        "cell_pack_width": cell_width,
        "positions_stable_cv_le_0p05": len(stable),
        "cycle_total_median_ms": round(sum(measured) / 1e3, 4),
        "n_rounds": cost["n_rounds"],
        "max_degree": cost["max_degree"],
        "device_rounds": len(profile),
        "sqlite": os.path.abspath(args.sqlite),
        # No verdict string: the interpretation belongs in the analysis after
        # the controls pass, not baked into the tool where it gets echoed
        # back as evidence.
        "metadata": scaling_metadata(
            grid="voronoi", component="atmosphere",
            resolution=f"L{args.subdivision}", n_levels=args.nlev,
            precision="n/a", decomposition="cell_partition",
            solver_variant="n/a", scaling_kind="halo-round-attribution",
            transport="nccl",
            extra={"n_devices": args.n_devices,
                   "partition_method": args.partition_method,
                   "lloyd_iterations": args.lloyd,
                   "reorder_target": cost["reorder_target"],
                   "device": args.device,
                   "cell_pack_width_assumed": cell_width,
                   "n_tracers_assumed": args.n_tracers,
                   "comm_latency_us": args.comm_latency_us,
                   "comm_bandwidth_gbs": args.comm_bandwidth_gbs}),
    }
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
