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


def sched_has_neighbors(layout) -> bool:
    """Whether a Voronoi layout's batched schedule exchanges anything (an
    empty schedule makes the exchange an identity and the timing vacuous)."""
    sched = layout.batched_comm
    return bool(sched is not None and sched.neighbor_ranks)


def _bench_voronoi_state(level: int, nlev: int, n_tracers: int,
                         n_warmup: int, n_iters: int):
    """Time the PRODUCTION MPAS state exchange: one batched halo exchange of
    the edge winds (n_local_edges, nlev) plus T (n_local_cells, nlev), p_s
    (n_local_cells,) and ``n_tracers`` (n_local_cells, nlev) tracers, float64,
    compiled, each call synchronised before the next (as a dynamics stage
    consumes its exchange).  The same ``batched_halo_exchange`` call
    ``make_voronoi_mpi_step`` issues per stage, on the same layout builder.

    Also returns a sha256 per rank of the exchanged arrays: the exchange only
    copies values, so two transports (host-staged vs GPU-direct) must give
    identical digests from the same seeds.
    """
    import hashlib

    import numpy as np
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.halo_exchange_voronoi import (
        batched_halo_exchange,
        build_batched_halo_schedule,
    )
    from legoesm.parallel.voronoi_mpi import initialize_voronoi_mpi
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, n_procs = comm.Get_rank(), comm.Get_size()
    if n_procs < 2:
        raise SystemExit("--payload state needs >= 2 MPI ranks")
    mesh = create_voronoi_mesh(level)
    # The driver's own partition path (initialize_voronoi_mpi with its
    # defaults), so the halo sizes are the production run's.
    _, _, layout = initialize_voronoi_mpi(mesh)
    part = layout.partition
    if not sched_has_neighbors(layout):
        raise SystemExit(f"rank {rank}: layout has no halo neighbours; "
                         "nothing would be exchanged")
    sched = layout.batched_comm
    if sched is None:
        sched = build_batched_halo_schedule(part.cell_comm, part.edge_comm)
    keys = jax.random.split(jax.random.PRNGKey(rank), 3 + n_tracers)
    f64 = jnp.float64
    u = jax.random.normal(keys[0], (part.n_local_edges, nlev), dtype=f64)
    cells = ((jax.random.normal(keys[1], (part.n_local_cells, nlev), dtype=f64),
              jax.random.normal(keys[2], (part.n_local_cells,), dtype=f64))
             + tuple(jax.random.normal(keys[3 + i], (part.n_local_cells, nlev),
                                       dtype=f64) for i in range(n_tracers)))
    exchange = jax.jit(lambda u_, c_: batched_halo_exchange((u_,), c_, sched, rank))

    out = jax.block_until_ready(exchange(u, cells))
    # The exchange must overwrite each rank's halo with its neighbours'
    # (different) random values; an identity exchange leaves every leaf equal.
    _in = jax.tree_util.tree_leaves(((u,), cells))
    _out = jax.tree_util.tree_leaves(out)
    halo_changed = (len(_in) == len(_out) and any(
        not np.array_equal(np.asarray(o), np.asarray(i)) for o, i in zip(_out, _in)))
    digest = hashlib.sha256()
    for a in jax.tree_util.tree_leaves(out):
        digest.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    for _ in range(n_warmup):
        jax.block_until_ready(exchange(u, cells))
    comm.Barrier()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        jax.block_until_ready(exchange(u, cells))
        times.append(time.perf_counter() - t0)
    comm.Barrier()
    per_rank_ms = comm.allgather(1e3 * float(np.median(times)))
    digests = comm.allgather(digest.hexdigest())
    halo_changed_all = comm.allgather(bool(halo_changed))
    n_bytes = 8 * (part.n_local_edges * nlev
                   + part.n_local_cells * (nlev * (1 + n_tracers) + 1))
    try:
        import mpi4jax
        cuda_built = bool(mpi4jax.has_cuda_support())
    except Exception:  # noqa: BLE001 -- reported, never fatal for a timing
        cuda_built = None
    return {
        "grid": "voronoi",
        "payload": "state",
        "level": level,
        "nlev": nlev,
        "n_tracers": n_tracers,
        "n_cells": int(mesh.nCells),
        "n_ranks": n_procs,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        "backend": jax.default_backend(),
        "MPI4JAX_USE_CUDA_MPI": os.environ.get("MPI4JAX_USE_CUDA_MPI"),
        "mpi4jax_cuda_built": cuda_built,
        "mpi_library": MPI.Get_library_version().splitlines()[0],
        "local_state_bytes_rank0": n_bytes,
        "per_rank_digest": digests,
        "per_rank_halo_changed": halo_changed_all,
        "exchanges": {
            "state": {
                "time_per_exchange_ms": max(per_rank_ms),
                "per_rank_median_ms": per_rank_ms,
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
    parser.add_argument("--payload", choices=["cell", "state"], default="cell",
                        help="voronoi only: 'cell' = one 2-D cell field through "
                             "the per-entity exchange; 'state' = the production "
                             "batched MPAS state exchange (winds, T, p_s, "
                             "tracers), compiled and synchronised")
    parser.add_argument("--n-tracers", type=int, default=None,
                        help="voronoi --payload state: number of 3-D tracers "
                             "carried with T (required: name it for the deck "
                             "being modelled)")
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
    elif args.grid == "voronoi" and args.payload == "state":
        if args.n_tracers is None:
            parser.error("--payload state needs --n-tracers")
        results = _bench_voronoi_state(
            args.level, args.nlev, args.n_tracers, args.n_warmup, args.n_iters,
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
