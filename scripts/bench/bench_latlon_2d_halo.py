"""2-D-vs-1-D lat-lon halo micro-bench (the strong-scaling payoff of the
2-D pencil decomposition, quantified on the shipped primitives).

For a FIXED global grid and world size, time the wall-pole 2-D halo pad
(:func:`pad_halo_latlon_2d`) at every ``(proc_lat, proc_lon)``
factorization of the world size — including ``(N, 1)`` which is the
1-D-band-equivalent (proc_lon==1 ⇒ local lon wrap, the production 1-D
path).  As ``N`` grows the 1-D band's rows/rank ``n_lat/N`` shrinks, so
its per-rank halo (a full-lon row each side) dominates; a balanced 2-D
split keeps BOTH dims larger, trading 2 fat N/S messages for 2 thinner
N/S + 2 thin E/W.  This bench measures that trade directly.

Run (MPI):  ``srun --mpi=pmix -n N python -m ... bench_latlon_2d_halo.py``
(or via the sbatch wrapper).  Rank 0 prints one line per factorization:
``HALO2D n=.. pr=.. pc=.. rows/rank=.. cols/rank=.. t_ms=..``.

NOT a full-step bench — it isolates the halo-exchange cost (the term the
2-D decomposition attacks).  Forward-only timing (the pad is the hot-loop
primitive); AD adds a symmetric reverse pass.
"""
from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def _factorizations(n: int) -> list[tuple[int, int]]:
    """All (proc_lat, proc_lon) with proc_lat*proc_lon == n, proc_lat
    descending (so (N,1) — the 1-D band — is first)."""
    out = []
    for pr in range(n, 0, -1):
        if n % pr == 0:
            out.append((pr, n // pr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lat", type=int, default=256)
    ap.add_argument("--n-lon", type=int, default=512)
    ap.add_argument("--nlev", type=int, default=60)
    ap.add_argument("--halo", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    from mpi4py import MPI

    from legoesm.parallel.latlon_mpi import (
        make_latlon_2d_layout,
        scatter_field_latlon_2d,
        pad_halo_latlon_2d,
    )

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()

    # Global field (cell-centred scalar with vertical levels).
    g = np.asarray(
        np.random.default_rng(0).standard_normal(
            (args.n_lat, args.n_lon, args.nlev)))
    g = jnp.asarray(g)

    if rank == 0:
        print(f"# bench_latlon_2d_halo n_lat={args.n_lat} n_lon={args.n_lon} "
              f"nlev={args.nlev} halo={args.halo} reps={args.reps} "
              f"world={n}", flush=True)

    for pr, pc in _factorizations(n):
        # Skip factorizations the grid can't carry at this halo.
        if (args.n_lat // pr) < args.halo or (args.n_lon // pc) < args.halo:
            if rank == 0:
                print(f"HALO2D n={n} pr={pr} pc={pc} SKIP "
                      f"(block < halo)", flush=True)
            continue
        L = make_latlon_2d_layout(rank, pr, pc, args.n_lat, args.n_lon)
        local = scatter_field_latlon_2d(g, L)

        def _one(x):
            return pad_halo_latlon_2d(
                x, L, halo=args.halo, south_value=0.0, north_value=0.0)

        # Warmup (compile + first-touch).
        for _ in range(args.warmup):
            out = _one(local)
            out.block_until_ready()
        comm.Barrier()

        t0 = time.perf_counter()
        for _ in range(args.reps):
            out = _one(local)
            out.block_until_ready()
        # Stop the timer BEFORE the barrier: a barrier inside the timed
        # window folds barrier-wait + release skew into dt (codex MAJOR).
        dt_ms = (time.perf_counter() - t0) / args.reps * 1e3
        comm.Barrier()

        # Reduce to the SLOWEST rank (the one that gates a synchronous step).
        dt_max = comm.allreduce(dt_ms, op=MPI.MAX)
        if rank == 0:
            if pc == 1:
                tag = " (1D-band)"      # pure lat split: N/S sendrecv only
            elif pr == 1:
                tag = " (lon-only)"     # pure lon split: E/W ring only
            else:
                tag = " (2D)"           # the actual balanced 2-D set
            print(f"HALO2D n={n} pr={pr} pc={pc} "
                  f"rows/rank={args.n_lat // pr} cols/rank={args.n_lon // pc} "
                  f"t_ms={dt_max:.4f}{tag}", flush=True)


if __name__ == "__main__":
    main()
