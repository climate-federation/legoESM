"""How much of a device's work is the RIM, and is hiding the halo worth it?

The interior/rim split computes the tendency twice: once on the whole
device-local mesh (the same cost the step pays today, and independent of the
halo, so it can run while the wire is busy) and once on a compact rim submesh
after the halo lands.  The second evaluation is ADDED work.  This prices it
from the plan the repository already builds, before anyone writes the split.

The proxy is exact where it matters: the rim fraction is set by how many cells
a device owns, not by how big the globe is, so subdivision 8 over 32 devices
carries the SAME 20,480 cells per device as subdivision 9 over 128 -- the
configuration whose plateau is in question -- while costing seconds to build.

Run: python scripts/bench/size_rim_split.py [--subdivision 8] [--devices 32]
"""
from __future__ import annotations

import argparse
import json


def size(subdivision: int, n_dev: int, halo_depth: int, rim_width: int,
         stencil_depth: int, method: str = "metis") -> dict:
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import (
        _build_rim_plan, _build_rim_rings, _build_voronoi_partition_infra,
    )
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    # MUST reorder with the SAME partitioner production runs before building
    # partitions. The infra takes CONTIGUOUS blocks of the reordered mesh as
    # each device's shard, so without the reorder the blocks are arbitrary
    # index ranges -- stringy, scattered pieces with NO interior, and the rim
    # would falsely swallow every owned cell. This was the instrument bug in
    # the first run of this probe.
    mesh = reorder_voronoi_for_sharding(mesh, n_dev, method=method)
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     _owner) = _build_voronoi_partition_infra(mesh, n_dev,
                                              halo_depth=halo_depth)
    cell_rim, edge_rim = _build_rim_rings(mesh, partitions, max_lc, max_le,
                                          rim_width)

    # Size the split straight from the rim RINGS, not the rim PLAN. The plan's
    # exact-cover tripwire refuses any partition that HAS an interior (it
    # cannot tell a deep interior edge from a genuinely absent one, because
    # both carry the same FAR sentinel), so it raises on exactly the compact
    # METIS partitions overlap is meant for. The rings build unconditionally
    # and carry the geometry the sizing needs.
    from legoesm.parallel.sharded_dynamics import _WIDE_RING_FAR
    per_dev = []
    for d, part in enumerate(partitions):
        n_oc = part.n_owned_cells
        n_oe = part.n_owned_edges
        cr = cell_rim[d, :n_oc]
        er = edge_rim[d, :n_oe]
        # Interior = owned entities whose radius-rim_width stencil sees no
        # ghost, i.e. FAR. Rim = 1..rim_width. (Cells at 0 are halo, not owned
        # here.) The interior is the work that can run while the wire is busy.
        rim_c = int(((cr >= 1) & (cr <= rim_width)).sum())
        int_c = int((cr == _WIDE_RING_FAR).sum())
        rim_e = int(((er >= 0) & (er <= rim_width)).sum())
        int_e = int((er == _WIDE_RING_FAR).sum())
        per_dev.append({
            "owned_cells": int(n_oc), "rim_cells": rim_c,
            "interior_cells": int_c,
            "interior_cell_fraction": int_c / n_oc if n_oc else 0.0,
            "owned_edges": int(n_oe), "rim_edges": rim_e,
            "interior_edges": int_e,
            "interior_edge_fraction": int_e / n_oe if n_oe else 0.0,
        })
    # The overlap hides the wire behind the INTERIOR pass; the rim pass is the
    # part that must wait. Cost model, per device: the interior pass runs the
    # operators on the whole local mesh (unchanged, halo-padded) and keeps the
    # interior; the rim pass reruns on a submesh whose size is ~ the rim plus
    # its stencil closure. Approximate the rim-pass added work by the rim
    # fraction inflated by the closure; use the edge fraction, the busier set.
    worst = max(per_dev, key=lambda r: 1.0 - r["interior_edge_fraction"])
    mean_interior = sum(r["interior_edge_fraction"]
                        for r in per_dev) / len(per_dev)
    return {
        "subdivision": subdivision, "n_devices": n_dev,
        "method": method, "halo_depth": halo_depth,
        "rim_width": rim_width, "stencil_depth": stencil_depth,
        "cells_per_device": mesh.nCells // n_dev,
        "mean_interior_edge_fraction": mean_interior,
        "worst_device": worst,
    }

def payoff(interior_frac: float, local_ms: float, comm_ms: float,
           closure_inflation: float = 1.6) -> dict:
    """Serial today against overlapped, from the interior fraction.

    Best case: the interior pass (fraction ``interior_frac`` of local work)
    runs concurrently with the wire, then the rim pass runs. The rim pass
    reruns the boundary shell plus its stencil closure, so its cost is the
    RIM fraction inflated by ``closure_inflation``. This is an UPPER bound on
    the achievable step and a rough one -- the real number needs the arm.
    """
    serial = local_ms + comm_ms
    rim_frac = min(1.0, (1.0 - interior_frac) * closure_inflation)
    interior_ms = interior_frac * local_ms
    rim_ms = rim_frac * local_ms
    # Interior overlaps the wire; the rim waits for the halo, then runs.
    overlapped = max(interior_ms, comm_ms) + rim_ms
    return {"serial_ms": serial, "overlapped_ms": overlapped,
            "rim_fraction": rim_frac, "change": overlapped / serial - 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivision", type=int, default=8)
    ap.add_argument("--devices", type=int, default=32)
    ap.add_argument("--halo-depth", type=int, default=4)
    ap.add_argument("--rim-width", type=int, default=2)
    ap.add_argument("--stencil-depth", type=int, default=2)
    ap.add_argument("--method", default="metis",
                    choices=["metis", "geometric", "sfc"])
    # The 128-GPU split this is meant to price, measured, not assumed.
    ap.add_argument("--local-ms", type=float, default=2.330)
    ap.add_argument("--comm-ms", type=float, default=2.660)
    args = ap.parse_args()

    out = size(args.subdivision, args.devices, args.halo_depth,
               args.rim_width, args.stencil_depth, method=args.method)
    out["payoff_mean"] = payoff(out["mean_interior_edge_fraction"],
                                args.local_ms, args.comm_ms)
    out["payoff_worst"] = payoff(out["worst_device"]["interior_edge_fraction"],
                                 args.local_ms, args.comm_ms)
    print(json.dumps(out, indent=2))
    p = out["payoff_mean"]
    print(f"\n{out['cells_per_device']} cells per device ({args.method}): "
          f"interior is {out['mean_interior_edge_fraction']:.1%} of owned "
          f"edges on average, "
          f"{out['worst_device']['interior_edge_fraction']:.1%} on the worst "
          f"device.")
    print(f"At local {args.local_ms} ms, wire {args.comm_ms} ms: {p['serial_ms']:.3f}"
          f" ms serial -> {p['overlapped_ms']:.3f} ms best-case overlapped "
          f"({p['change']:+.1%}), rim pass ~{p['rim_fraction']:.1%} of local.")
    # A large interior means the wire can hide; report it, do not gate here --
    # the real verdict is the machine arm, this only says whether to build.
    if p["change"] > -0.05:
        print("MARGINAL: interior too small or wire too cheap; overlap may "
              "not clear the arm spread. Build only if the arm confirms.")


if __name__ == "__main__":
    main()
