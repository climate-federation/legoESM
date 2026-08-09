"""Benchmark MPAS/Voronoi partition methods: RCB vs Hilbert-SFC vs METIS.

Scaling-audit item 8.  Two independent measurement layers:

1. **Offline partition quality** (no MPI, exact, every rank enumerated
   serially): edge cut, halo cells (max/mean, halo/owned ratio), load
   balance (cells/rank min/max, imbalance max/mean), neighbor-rank fan-out —
   for each method x rank-count on the real icosahedral mesh.  These are
   the numbers ``resolve_partition_method``'s ``auto`` policy must be
   justified by.
1b. **SPMD halo-schedule depth** (``--schedule-cost``, opt-in because it is
   the expensive layer): ``n_rounds`` — the number of SEQUENTIAL ppermute
   rounds one halo fill costs — and ``max_degree``, the communication
   graph's lower bound on it.  This is the term that binds MPAS GPU strong
   scaling above ~64 devices, and layer 1 CANNOT stand in for it: the
   neighbor fan-out above is a 1-ring proxy that reported 8 rounds for
   every method and rank count while the real depth-3-plus-closure
   schedule reported 12-14.  Scored by the production
   ``spmd_schedule_cost`` (which calls the production builders), never a
   re-derived lookalike.

   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
   The schedule is a proper EDGE colouring of the device communication
   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
   asserts properness), and ``max_degree`` is that same graph's maximum
   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
   yes/no flag.

   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
     headroom is exactly ZERO.
   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
     this neither establishes nor excludes a one-round win.
   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
     most ``gap``.

   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
   enters a device pair into ``comm_pairs`` when EITHER direction has a
   halo dependency and then emits BOTH ppermute directions, even where one
   send map is empty.  A redesigned DIRECTED schedule that exploits
   one-way exchanges is therefore not bounded by ``Delta`` at all, so
   ``gap == 0`` must never be reported as "only ownership can help" — it
   rules out a better undirected edge colouring of this graph, and
   nothing more.

   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
   flag is the MPI lane's (default 2), while the schedule is scored at the
   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
   evaluations of the integrator actually run.  When
   ``production_strategy`` is ``"allgather"`` (auto-selected below the
   cells/device threshold) there is no ppermute schedule in production and
   the round count is COUNTERFACTUAL; the row says so.
2. **Step time** (optional pointer, NOT run here): drive the existing
   MPI lane with ``bench_ocean_mpas_scaling.py --partition-method <m>``
   (ocean) or ``bench_mpas_spmd_scaling.py --partition-method <m>``
   (atmosphere SPMD) — one method per launch, same case otherwise
   (controlled comparison).

Guards: methods that are unavailable (``metis`` without ``pymetis``) are
reported as ``"unavailable"`` — never silently substituted, so a table
column can never claim METIS numbers that actually came from the RCB
fallback.  Partition CORRECTNESS is asserted per row (every cell owned by
exactly one rank; owner range valid) before any metric is recorded.

Run:
  python scripts/bench/bench_voronoi_partition_methods.py \
      --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json

  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
  python scripts/bench/bench_voronoi_partition_methods.py \
      --subdivision 9 --rank-counts 64,128 --schedule-cost \
      --out results/a1/schedule_cost_s9.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import annotate_incomplete, scaling_metadata  # noqa: E402

METHODS = ("geometric", "sfc", "metis")


def method_available(method: str) -> bool:
    if method != "metis":
        return True
    try:
        import pymetis  # noqa: F401

        return True
    except Exception:
        return False


def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
                      halo_depth: int = 2) -> dict:
    """Exact partition-quality metrics from a global owner array.

    Serial enumeration of every rank (no MPI): the same halo construction
    the runtime uses (``compute_halo_cells``), so the reported halo sizes
    are the runtime's, not an estimate.
    """
    from legoesm.parallel.voronoi_partition import compute_halo_cells

    n_cells = int(mesh.nCells)
    # Correctness: every cell owned exactly once, owners in range, no
    # empty rank (an empty rank would silently deflate the halo/owned
    # ratio through the max(counts, 1) guard).
    if n_cells == 0 or cell_owner.size == 0:
        raise AssertionError("empty mesh / owner array")
    if cell_owner.shape != (n_cells,):
        raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
    if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
        raise AssertionError("owner out of range")
    counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
    if int(counts.sum()) != n_cells:
        raise AssertionError("ownership does not cover the mesh")
    if counts.min() <= 0:
        raise AssertionError(
            f"empty rank in partition (counts.min()={counts.min():.0f}) — "
            f"a skipped rank corrupts every per-rank metric")

    # Edge cut: edges whose two cells have different owners.
    c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    valid = (c1 >= 0) & (c2 >= 0)
    edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())

    halo_sizes = []
    neighbor_counts = []
    cells_on_cell = np.asarray(mesh.cellsOnCell)
    max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
    for r in range(n_ranks):
        halo = compute_halo_cells(
            cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
        halo_sizes.append(len(halo))
        neighbor_counts.append(
            len(set(int(cell_owner[c]) for c in halo) - {r}))
    _ = max_edges
    halo_sizes = np.array(halo_sizes, dtype=float)
    return {
        "cells_per_rank_min": int(counts.min()),
        "cells_per_rank_max": int(counts.max()),
        "load_imbalance_max_over_mean": float(counts.max() / counts.mean()),
        "edge_cut": edge_cut,
        "edge_cut_fraction": float(edge_cut / max(int(valid.sum()), 1)),
        "halo_cells_max": int(halo_sizes.max()),
        "halo_cells_mean": float(halo_sizes.mean()),
        "halo_owned_ratio_max": float(
            (halo_sizes / np.maximum(counts, 1.0)).max()),
        "neighbor_ranks_max": int(max(neighbor_counts)),
    }


def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
    from legoesm.parallel.voronoi_partition import (
        partition_cells_geometric,
        partition_cells_metis,
        partition_cells_sfc,
    )

    if method == "geometric":
        return np.asarray(partition_cells_geometric(mesh, n_ranks))
    if method == "sfc":
        return np.asarray(partition_cells_sfc(mesh, n_ranks))
    if method == "metis":
        return np.asarray(partition_cells_metis(mesh, n_ranks))
    raise ValueError(f"unknown partition method {method!r}; "
                     f"expected one of {METHODS}")


def parse_expect_rounds(spec: str) -> dict:
    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.

    Raises on anything malformed rather than skipping it — a typo'd
    expectation that is silently dropped turns the gate into a no-op, which
    is exactly the failure this flag exists to prevent.  Three ways that
    could happen, all rejected here:

    * a malformed item (``geometric:2``, ``geometric=2``);
    * an unknown method;
    * a DUPLICATE key — ``geometric:2=999,geometric:2=13`` would otherwise
      let the second silently overwrite the first and pass;
    * a non-empty spec that parses to NOTHING (``",,,"``), which would make
      the caller skip the check while believing it ran.
    """
    out: dict[tuple[str, int], int] = {}
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        try:
            lhs, rounds = item.split("=")
            method, n_ranks = lhs.split(":")
            key = (method.strip(), int(n_ranks))
            value = int(rounds)
        except ValueError as exc:
            raise ValueError(
                f"--expect-rounds: cannot parse {item!r}; expected "
                f"'<method>:<n_ranks>=<n_rounds>'") from exc
        if key[0] not in METHODS:
            raise ValueError(
                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
                f"expected one of {METHODS}")
        if key in out:
            raise ValueError(
                f"--expect-rounds: duplicate expectation for "
                f"{key[0]}:{key[1]} ({out[key]} then {value}); the later one "
                f"would silently overwrite the earlier and the gate would "
                f"pass while discarding a listed expectation.")
        out[key] = value
    # Guard on `spec`, NOT `spec.strip()`: a whitespace-only value is a value
    # the caller PASSED, and silently reading it as "no gate requested" is
    # the same bypass as ",,," (codex round 3).  Only the default empty
    # string means "no gate".
    if spec and not out:
        raise ValueError(
            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
            f"would be skipped while looking like it ran.")
    return out


def check_expected_rounds(rows: list, expect: dict) -> list:
    """Compare scored rounds against *expect*; return failure strings.

    A listed pair that was never scored is a FAILURE, not a skip: otherwise
    a sweep that silently dropped a method (unavailable ``pymetis``) or a
    rank count would still report a clean gate.
    """
    scored = {
        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
        for r in rows
        if r.get("available") and "schedule" in r and "n_ranks" in r
    }
    failures = []
    for (method, n_ranks), want in sorted(expect.items()):
        got = scored.get((method, n_ranks))
        if got is None:
            failures.append(
                f"{method}:{n_ranks} expected rounds={want} but the pair was "
                f"NOT SCORED (method unavailable, or not in this sweep)")
        elif got != want:
            failures.append(
                f"{method}:{n_ranks} expected rounds={want}, got {got}")
    return failures


def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
    """SPMD halo-schedule depth for one (method, n_ranks) candidate.

    Thin wrapper over the production
    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
    lane's (2), and scoring the SPMD schedule at 2 would colour a different
    graph.

    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
    it (see the module docstring).  The headroom is reported as an INTERVAL,
    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:

    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
      if the graph is Class 1.

    It is deliberately NOT a "recolour vs ownership" verdict: at
    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.

    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
    different reorder target, which would mis-slice the owned blocks — is
    unreachable from here: this passes the raw mesh with the scorer's default
    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
    the call would therefore only swallow *unforeseen* failures into a row
    that reads like an orderly skip, which is how a missing number turns into
    a silently wrong table.  Rows already print as the sweep goes, so a raise
    keeps the completed rungs in the log.
    """
    import time

    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

    t0 = time.perf_counter()
    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
    return {
        "n_rounds": int(cost["n_rounds"]),
        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
        "max_degree": int(cost["max_degree"]),
        # The decisive column, read through Vizing (see module docstring).
        "coloring_gap": gap,
        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
        "coloring_headroom_rounds_min": max(0, gap - 1),
        "coloring_headroom_rounds_max": gap,
        "coloring_optimal_proven": gap == 0,
        # Scorer provenance, carried per row: a copied/flattened row must be
        # able to show it scored a raw mesh partitioned for THIS device
        # count, not one reordered for a different target.
        "reorder_target": cost["reorder_target"],
        "already_reordered": cost["already_reordered"],
        "coloring_method": cost["coloring_method"],
        "resolved_method": cost["resolved_method"],
        "schedule_halo_depth": int(cost["halo_depth"]),
        "cells_per_device": int(cost["cells_per_device"]),
        # "allgather" => production runs no ppermute schedule here, so the
        # round count above is COUNTERFACTUAL, not a cost production pays.
        "production_strategy": cost["production_strategy"],
        "score_seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdivision", type=int, default=5,
                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
    p.add_argument("--halo-depth", type=int, default=2,
                   help="Halo layers (runtime default 2, del4 support).")
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--schedule-cost", action="store_true",
                   help="Also score the SPMD ppermute halo-schedule depth "
                        "(n_rounds vs max_degree) per method x rank count. "
                        "Uses the production SPMD halo depth, NOT "
                        "--halo-depth. Expensive: minutes per candidate at "
                        "subdiv>=8 — run it under batch.")
    p.add_argument("--lloyd", type=int, default=50,
                   help="Lloyd relaxation iterations for the mesh. 50 = the "
                        "production SCVT key; 0 = the LABELLED synthetic "
                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
                        "masquerade as a production receipt, and it must "
                        "match the prewarmed cache key at subdiv>=9. "
                        "Must be >= 0: the builder relaxes only when this is "
                        "> 0, so a negative behaves exactly like 0 while "
                        "being recorded (and cached) under a different key — "
                        "false provenance.")
    p.add_argument("--expect-rounds", type=str, default="",
                   help="Instrument check, MECHANICAL. Comma-separated "
                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
                        "scored and match, or main() returns 1 — so a caller "
                        "that reproduces a known census can GATE on it "
                        "instead of asserting agreement in a comment. "
                        "Requires --schedule-cost.")
    p.add_argument("--out", type=str,
                   default="results/a1/voronoi_partition_quality.json")
    args = p.parse_args()

    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
    if not rank_counts or any(n < 2 for n in rank_counts):
        raise SystemExit("--rank-counts needs integers >= 2")
    if not methods:
        # Same class as an empty --expect-rounds: the loop would be skipped,
        # `rows: []` written, and 0 returned — an empty run that reads as a
        # successful one.
        raise SystemExit(
            f"--methods={args.methods!r} selects NO methods; the benchmark "
            f"would measure nothing and still exit 0. Choose from {METHODS}.")
    # Negative values in these three are all the SAME false-provenance bug:
    # the underlying code treats them exactly like 0 (no relaxation, no
    # bisection, `range(-1)` is empty), but the run is recorded under the
    # negative value, so a level-0 mesh gets filed as "L-1".
    if args.lloyd < 0:
        raise SystemExit(
            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
            f"relaxes only for > 0, so a negative is indistinguishable from "
            f"0 in the mesh but is recorded and cached under its own key.")
    if args.subdivision < 0:
        raise SystemExit(
            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
            f"builder bisects only for > 0, so a negative silently yields "
            f"the level-0 base mesh while being recorded as "
            f"L{args.subdivision}.")
    if args.halo_depth < 0:
        raise SystemExit(
            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
            f"while being recorded as {args.halo_depth}.")
    expect = parse_expect_rounds(args.expect_rounds)
    if expect and not args.schedule_cost:
        raise SystemExit(
            "--expect-rounds compares scored round counts, so it needs "
            "--schedule-cost; without it nothing is scored and the gate "
            "would pass vacuously.")


    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import resolve_partition_method

    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
                               lloyd_iterations=args.lloyd)
    if max(rank_counts) > int(mesh.nCells):
        raise SystemExit(
            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
          f"{int(mesh.nEdges)} edges; auto -> "
          f"{resolve_partition_method('auto')!r}")

    rows = []
    for method in methods:
        if not method_available(method):
            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
                  f"column omitted, never substituted")
            rows.append({"method": method, "available": False})
            continue
        for n_ranks in rank_counts:
            q = partition_quality(
                mesh, owner_for(mesh, method, n_ranks), n_ranks,
                halo_depth=args.halo_depth)
            row = {"method": method, "available": True,
                   "n_ranks": n_ranks, **q}
            rows.append(row)
            print(f"  {method:9s} np={n_ranks:3d} | "
                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
                  f"edge_cut={q['edge_cut']:6d} "
                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
                  f"halo max={q['halo_cells_max']:5d} "
                  f"mean={q['halo_cells_mean']:8.1f} | "
                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
                  f"nbrs max={q['neighbor_ranks_max']}")
            if args.schedule_cost:
                sc = schedule_cost_row(mesh, method, n_ranks)
                row["schedule"] = sc
                note = ("  [COUNTERFACTUAL: production auto-selects "
                        "allgather here, no ppermute schedule]"
                        if sc["production_strategy"] == "allgather" else "")
                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
                           if sc["coloring_optimal_proven"] else
                           f"recolour headroom "
                           f"{sc['coloring_headroom_rounds_min']}-"
                           f"{sc['coloring_headroom_rounds_max']} round(s)")
                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
                      f"rounds={sc['n_rounds']:3d} "
                      f"max_degree={sc['max_degree']:3d} "
                      f"gap={sc['coloring_gap']:+d} "
                      f"-> {verdict} "
                      f"({sc['score_seconds']:.1f}s){note}", flush=True)

    # An "empty but successful" run is the same failure class as a skipped
    # gate: --methods metis on a box without pymetis appends only an
    # unavailable row and would otherwise exit 0 having measured nothing
    # (codex round 5).
    if not any(r.get("available") and "n_ranks" in r for r in rows):
        raise SystemExit(
            f"no method was actually measured (requested {methods}); every "
            f"one was unavailable, so this run has no results and must not "
            f"report success.")

    payload = {
        "rows": rows,
        "auto_resolves_to": resolve_partition_method("auto"),
        "step_time_pointer": (
            "step-time per method: bench_ocean_mpas_scaling.py / "
            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
            "per launch, same case otherwise)"),
        "metadata": annotate_incomplete(scaling_metadata(
            grid="voronoi",
            component="partitioning",
            resolution=f"L{args.subdivision}",
            n_levels=0,
            precision="n/a",
            decomposition="cell_partition",
            solver_variant="n/a",
            scaling_kind="partition-quality",
            transport="none",
            extra={"rank_counts": rank_counts, "methods": methods,
                   "halo_depth": args.halo_depth,
                   "schedule_cost": bool(args.schedule_cost),
                   "lloyd_iterations": args.lloyd},
        )),
    }
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON: {args.out}")

    if expect:
        failures = check_expected_rounds(rows, expect)
        payload["expected_rounds_check"] = {
            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
            "failures": failures,
            "passed": not failures,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        if failures:
            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
            for line in failures:
                print(f"  {line}")
            print("The scorer did NOT reproduce the known census — treat every "
                  "unknown row in this run as UNTRUSTED.")
            return 1
        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
