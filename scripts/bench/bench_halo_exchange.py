#!/usr/bin/env python
"""Micro-benchmark for MPI halo exchange across grid types.

Isolates halo exchange performance from dycore compute to diagnose
communication bottlenecks.  Measures:
  - Time per halo exchange (3D and 4D variants)
  - Message count and aggregate message size
  - Bandwidth (bytes/sec)
  - Scaling with rank count

Usage
-----
Cubed-sphere (face-only mode, np <= 6)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 2 \
        python scripts/bench_halo_exchange.py --grid cubed-sphere --n 48 --nlev 40

Cubed-sphere (tiled mode, np > 6)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 24 \
        python scripts/bench_halo_exchange.py --grid cubed-sphere --n 24 --nlev 40

Voronoi/MPAS::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 2 \
        python scripts/bench_halo_exchange.py --grid voronoi --level 4

Single-rank baseline (no MPI, measures local halo cost)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/bench_halo_exchange.py \
        --grid cubed-sphere --n 48 --nlev 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Suppress Metal warnings on macOS
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp


def _bench_cubed_sphere(n: int, nlev: int, halo: int, n_warmup: int,
                         n_iters: int, use_mpi: bool):
    """Benchmark cubed-sphere halo exchange."""
    from legoesm.grids.halo import pad_halo, pad_halo_4d, set_halo_backend

    if use_mpi:
        from mpi4py import MPI
        from legoesm.parallel.comm import build_comm_topology
        rank = MPI.COMM_WORLD.Get_rank()
        n_procs = MPI.COMM_WORLD.Get_size()
        topo = build_comm_topology(rank, n_procs)
        set_halo_backend("mpi", topo)
        barrier = MPI.COMM_WORLD.Barrier
    else:
        rank = 0
        n_procs = 1
        barrier = lambda: None

    key = jax.random.PRNGKey(rank)
    data_3d = jax.random.normal(key, (6, n, n), dtype=jnp.float64)
    data_4d = jax.random.normal(key, (6, n, n, nlev), dtype=jnp.float64)

    results = {}

    # --- 3D halo exchange ---
    for _ in range(n_warmup):
        _ = pad_halo(data_3d, halo=halo)

    barrier()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = pad_halo(data_3d, halo=halo)
    barrier()
    t1 = time.perf_counter()

    msg_bytes_3d = n * halo * 8 * 4  # 4 edges, float64=8 bytes
    results["3d"] = {
        "time_per_exchange_ms": (t1 - t0) / n_iters * 1000,
        "total_time_s": t1 - t0,
        "msg_bytes_per_edge": n * halo * 8,
        "n_edges": 4,
    }

    # --- 4D halo exchange ---
    for _ in range(n_warmup):
        _ = pad_halo_4d(data_4d, halo=halo)

    barrier()
    t2 = time.perf_counter()
    for _ in range(n_iters):
        _ = pad_halo_4d(data_4d, halo=halo)
    barrier()
    t3 = time.perf_counter()

    msg_bytes_4d = n * nlev * halo * 8 * 4
    results["4d"] = {
        "time_per_exchange_ms": (t3 - t2) / n_iters * 1000,
        "total_time_s": t3 - t2,
        "msg_bytes_per_edge": n * nlev * halo * 8,
        "n_edges": 4,
    }

    set_halo_backend("local")  # reset

    return {
        "grid": "cubed-sphere",
        "n": n,
        "nlev": nlev,
        "halo": halo,
        "n_ranks": n_procs,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        "exchanges": results,
    }


def _bench_voronoi(level: int, n_warmup: int, n_iters: int, use_mpi: bool):
    """Benchmark Voronoi halo exchange."""
    from legoesm.grids.voronoi import create_voronoi_mesh

    if use_mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        n_procs = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
    else:
        rank = 0
        n_procs = 1
        barrier = lambda: None

    mesh = create_voronoi_mesh(level)
    n_cells = mesh.nCells

    if use_mpi and n_procs > 1:
        from legoesm.parallel.voronoi_mpi import make_voronoi_partition_layout
        from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
        layout = make_voronoi_partition_layout(mesh, n_procs, rank)
        halo_ex = VoronoiHaloExchange(layout, backend="mpi")

        # Create local cell field
        key = jax.random.PRNGKey(rank)
        # n_local_cells = owned + halo (the old n_halo_cells field is gone)
        local_n = layout.partition.n_local_cells
        data = jax.random.normal(key, (local_n,), dtype=jnp.float64)

        for _ in range(n_warmup):
            _ = halo_ex.exchange_cell_field(data)

        barrier()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = halo_ex.exchange_cell_field(data)
        barrier()
        t1 = time.perf_counter()
    else:
        # Single rank: no exchange needed, measure cost of no-op
        t0 = t1 = 0.0

    return {
        "grid": "voronoi",
        "level": level,
        "n_cells": n_cells,
        "n_ranks": n_procs,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        "exchanges": {
            "cell": {
                "time_per_exchange_ms": (t1 - t0) / max(n_iters, 1) * 1000,
                "total_time_s": t1 - t0,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Halo exchange micro-benchmark")
    parser.add_argument("--grid", choices=["cubed-sphere", "voronoi"],
                        default="cubed-sphere")
    parser.add_argument("--n", type=int, default=48,
                        help="Cubed-sphere face size")
    parser.add_argument("--nlev", type=int, default=40,
                        help="Number of vertical levels")
    parser.add_argument("--halo", type=int, default=1,
                        help="Halo width (1 or 2)")
    parser.add_argument("--level", type=int, default=4,
                        help="Voronoi subdivision level")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file")
    args = parser.parse_args()

    # Detect MPI
    try:
        from mpi4py import MPI
        use_mpi = MPI.COMM_WORLD.Get_size() > 1
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        use_mpi = False
        rank = 0

    if args.grid == "cubed-sphere":
        results = _bench_cubed_sphere(
            args.n, args.nlev, args.halo,
            args.n_warmup, args.n_iters, use_mpi,
        )
    elif args.grid == "voronoi":
        results = _bench_voronoi(
            args.level, args.n_warmup, args.n_iters, use_mpi,
        )

    if rank == 0:
        print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        # Summary table
        print("\n" + "=" * 60)
        print(f"  Halo Exchange Benchmark: {results['grid']}")
        print(f"  Ranks: {results['n_ranks']}")
        print("=" * 60)
        for name, data in results["exchanges"].items():
            t = data["time_per_exchange_ms"]
            print(f"  {name:>6s}: {t:8.2f} ms/exchange")
            if "msg_bytes_per_edge" in data:
                bw = data["msg_bytes_per_edge"] * data["n_edges"] / (t / 1000) / 1e6
                print(f"         {bw:8.1f} MB/s aggregate bandwidth")
        print("=" * 60)


if __name__ == "__main__":
    main()
