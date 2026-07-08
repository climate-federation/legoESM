"""Benchmark MPAS/Voronoi partition methods: RCB vs Hilbert-SFC vs METIS.

Scaling-audit item 8.  Two independent measurement layers:

1. **Offline partition quality** (no MPI, exact, every rank enumerated
   serially): edge cut, halo cells (max/mean, halo/owned ratio), load
   balance (cells/rank min/max, imbalance max/mean), neighbor-rank fan-out —
   for each method x rank-count on the real icosahedral mesh.  These are
   the numbers ``resolve_partition_method``'s ``auto`` policy must be
   justified by.
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


    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import resolve_partition_method

    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
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
                   "halo_depth": args.halo_depth},
        )),
    }
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
