"""Bench halo-aware operators across MPI ranks — strong + weak scaling.

Proves the halo-aware operator pipeline scales correctly under
true domain decomposition, validating the path toward a fully
domain-decomposed plane CRM dycore.

Pipeline (one "tendency step"): exchange halos + apply 8 halo-aware
operators that the slow-tendency uses (grad_x, grad_y, divergence,
laplacian, 2 interp_cell_to_face, 2 interp_face_to_cell). This is
a realistic proxy for the slow-tendency computational cost.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from legoesm.atmosphere.dynamics.les import plane_operators_halo as oh
from legoesm.grids.plane import create_plane_grid
from legoesm.parallel.plane_mpi import (
    exchange_halo_plane_yxz, make_plane_pencil_layout,
    packed_exchange_halo_plane_yxz,
)

jax.config.update("jax_enable_x64", True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--ny", type=int, default=24)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-bench", type=int, default=50)
    p.add_argument("--label", type=str, default="bench")
    return p.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if n_ranks == 12:
        nry, nrx = 3, 4
    elif n_ranks == 6:
        nry, nrx = 2, 3
    elif n_ranks == 4:
        nry, nrx = 2, 2
    elif n_ranks == 3:
        nry, nrx = 1, 3
    elif n_ranks == 2:
        nry, nrx = 1, 2
    else:
        nry, nrx = 1, n_ranks

    if args.ny % nry != 0 or args.nx % nrx != 0:
        if rank == 0:
            print(f"SKIP {args.label}: grid not divisible by {nry}x{nrx}")
        return

    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=nry, n_ranks_x=nrx,
        ny_global=args.ny, nx_global=args.nx,
    )
    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )

    # Each rank holds local slab of u, v, theta — (ny_local, nx_local, nlev).
    rng = np.random.default_rng(rank)
    shp_local = (layout.ny_local, layout.nx_local, args.nlev)
    u_local = jnp.asarray(rng.standard_normal(shp_local))
    v_local = jnp.asarray(rng.standard_normal(shp_local))
    theta_local = jnp.asarray(rng.standard_normal(shp_local))

    # Local-compute kernel (JIT-compiled). Operates on already-padded
    # arrays. Halo exchange happens outside JIT (Python-controlled MPI).
    @jax.jit
    def local_compute(u_pad, v_pad, theta_pad):
        gp_x = oh.grad_x_3d_halo(theta_pad, grid)
        gp_y = oh.grad_y_3d_halo(theta_pad, grid)
        div = oh.divergence_3d_halo(u_pad, v_pad, grid)
        lap_theta = oh.laplacian_3d_halo(theta_pad, grid)
        theta_xf = oh.interp_cell_to_xface_halo(theta_pad, grid)
        theta_yf = oh.interp_cell_to_yface_halo(theta_pad, grid)
        u_c = oh.interp_xface_to_cell_halo(u_pad, grid)
        v_c = oh.interp_yface_to_cell_halo(v_pad, grid)
        du = -gp_x + div * 0.0 + theta_xf * 0.0 - u_c * lap_theta
        return du, gp_y, v_c, theta_yf

    def pipeline(u_l, v_l, theta_l):
        # PACKED exchange: 1 MPI round for all 3 fields instead of 3.
        u_pad, v_pad, theta_pad = packed_exchange_halo_plane_yxz(
            u_l, v_l, theta_l, layout=layout,
        )
        return local_compute(u_pad, v_pad, theta_pad)

    # Warmup (JIT).
    for _ in range(args.n_warmup):
        out = pipeline(u_local, v_local, theta_local)
    jax.block_until_ready(out)
    comm.Barrier()

    # Bench.
    t0 = time.time()
    for _ in range(args.n_bench):
        out = pipeline(u_local, v_local, theta_local)
    jax.block_until_ready(out)
    comm.Barrier()
    wall = time.time() - t0

    # Local-pipeline-only (single-rank single-process control: skip
    # the MPI sendrecv branch by using a 1-rank layout to bypass it).
    layout_1 = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=layout.ny_local, nx_global=layout.nx_local,
    )

    def pipeline_local(u_l, v_l, theta_l):
        u_pad = exchange_halo_plane_yxz(u_l, layout_1)
        v_pad = exchange_halo_plane_yxz(v_l, layout_1)
        theta_pad = exchange_halo_plane_yxz(theta_l, layout_1)
        return local_compute(u_pad, v_pad, theta_pad)

    for _ in range(args.n_warmup):
        out = pipeline_local(u_local, v_local, theta_local)
    jax.block_until_ready(out)
    comm.Barrier()
    t0 = time.time()
    for _ in range(args.n_bench):
        out = pipeline_local(u_local, v_local, theta_local)
    jax.block_until_ready(out)
    comm.Barrier()
    wall_local = time.time() - t0

    if rank == 0:
        ms_total = wall / args.n_bench * 1000
        ms_local = wall_local / args.n_bench * 1000
        ms_comm = ms_total - ms_local
        print(
            f"{args.label:24s} nranks={n_ranks:2d} "
            f"grid={args.ny}x{args.nx} ny_local={layout.ny_local} "
            f"nx_local={layout.nx_local}  "
            f"total={ms_total:6.2f}ms  "
            f"local_compute={ms_local:6.2f}ms  "
            f"halo_comm={ms_comm:6.2f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
