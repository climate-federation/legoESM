"""Measure strong + weak scaling of plane CRM MPI path.

Strong scaling: fixed global problem, vary nranks.
Weak scaling: fixed per-rank problem, vary nranks.

Driver: launches the bench with mpirun -np N internally for each
data point. Reports per-step wall-clock for each N + scaling
efficiency.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    compute_total_water_mass_plane_mpi,
    remove_horizontal_mean_wind_plane_mpi,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

# Precision must be resolved BEFORE any jax array is created, so pre-parse
# ``--precision`` from argv (argparse runs later, inside main()).  Default
# float64 preserves the historical behaviour.
import sys as _sys
_PRECISION = "float64"
if "--precision" in _sys.argv:
    try:
        _PRECISION = _sys.argv[_sys.argv.index("--precision") + 1]
    except IndexError:
        pass
jax.config.update("jax_enable_x64", _PRECISION == "float64")
_DTYPE = jnp.float64 if _PRECISION == "float64" else jnp.float32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--ny", type=int, default=24)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-bench", type=int, default=30)
    p.add_argument("--label", type=str, default="bench")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64",
                   help="Array/compute precision (jax_enable_x64 set to match).")
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
            print(f"SKIP {args.label}: grid {args.ny}x{args.nx} not divisible by {nry}x{nrx}")
        return

    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks, n_ranks_y=nry, n_ranks_x=nrx,
        ny_global=args.ny, nx_global=args.nx,
    )

    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=_DTYPE,
    )
    hc = create_stretched_height_coordinate(args.nlev, H=20_000., dz_sfc=100.)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        n_acoustic_substeps=12, semi_implicit_acoustic=True,
        sponge_coeff=0.05, sponge_width=5000.,
        hyperdiff_coeff=1e6, hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
        smagorinsky_cs=0.2, use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = make_rest_state(grid, hc, dtype=_DTYPE)
    new_tr = jnp.zeros(
        (args.ny, args.nx, args.nlev, 3), dtype=_DTYPE,
    ).at[..., 0].set(0.01)
    state = state._replace(tracers=state.tracers.replace(data=new_tr))

    owned_mask = jnp.zeros((args.ny, args.nx), dtype=_DTYPE)
    owned_mask = owned_mask.at[
        layout.iy_start:layout.iy_end,
        layout.ix_start:layout.ix_end,
    ].set(1.0)

    def _bcast_state(s):
        names = ("u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers")
        new = {}
        for nm in names:
            arr = np.asarray(getattr(s, nm).data)
            if rank != 0:
                arr = np.empty(arr.shape, dtype=arr.dtype)
            comm.Bcast(arr, root=0)
            new[nm] = getattr(s, nm).replace(data=jnp.asarray(arr))
        return s._replace(**new)

    # Batch-allreduce path: combine the 2 reductions into 1 allreduce
    # to slash mpi4jax per-call overhead.
    from legoesm.parallel.reductions import batch_allreduce_mpi

    def one_step(s):
        if rank == 0:
            s = model.step(s, dt=args.dt)
        s = _bcast_state(s)
        # Compute locals for both reductions, then 1 batched allreduce.
        ny, nx, nlev = s.theta_prime.data.shape
        if n_ranks > 1:
            mask_3d = owned_mask[:, :, None]
            u_local_sum = jnp.sum(s.u.data * mask_3d, axis=(0, 1))
            v_local_sum = jnp.sum(s.v.data * mask_3d, axis=(0, 1))
            local_count = jnp.sum(owned_mask)
            rho_total = hc.rho_ref + s.rho_prime.data
            weight_2d = grid.area_T[:, :, None] * owned_mask[:, :, None]
            column_w = rho_total * weight_2d * hc.dz
            q_total = (
                s.tracers.data[..., 0]
                + s.tracers.data[..., 1]
                + s.tracers.data[..., 2]
            )
            local_water = jnp.sum(q_total * column_w)
            u_global, v_global, count, water = batch_allreduce_mpi(
                [u_local_sum, v_local_sum, local_count, local_water],
            )
            u_mean = u_global / count
            v_mean = v_global / count
            new_u = s.u.data - u_mean[None, None, :]
            new_v = s.v.data - v_mean[None, None, :]
            s = s._replace(
                u=s.u.replace(data=new_u),
                v=s.v.replace(data=new_v),
            )
        else:
            s = remove_horizontal_mean_wind_plane_mpi(s, layout, owned_mask)
            _ = compute_total_water_mass_plane_mpi(s, hc, grid, layout, owned_mask)
        return s

    # Warmup (JIT).
    for _ in range(args.n_warmup):
        state = one_step(state)
    state.w.data.block_until_ready()
    comm.Barrier()

    # Bench.
    t0 = time.time()
    for _ in range(args.n_bench):
        state = one_step(state)
    state.w.data.block_until_ready()
    comm.Barrier()
    wall = time.time() - t0

    # Also time bcast + reductions alone.
    t0 = time.time()
    for _ in range(args.n_bench):
        state = _bcast_state(state)
    state.w.data.block_until_ready()
    comm.Barrier()
    bcast_wall = time.time() - t0

    t0 = time.time()
    for _ in range(args.n_bench):
        _ = compute_total_water_mass_plane_mpi(state, hc, grid, layout, owned_mask)
        _ = remove_horizontal_mean_wind_plane_mpi(state, layout, owned_mask)
    state.w.data.block_until_ready()
    comm.Barrier()
    reduce_wall = time.time() - t0

    if rank == 0:
        ms_step = wall / args.n_bench * 1000
        ms_bcast = bcast_wall / args.n_bench * 1000
        ms_reduce = reduce_wall / args.n_bench * 1000
        ms_dycore = ms_step - ms_bcast - ms_reduce
        print(
            f"{args.label:32s} nranks={n_ranks:2d} "
            f"grid={args.ny}x{args.nx}x{args.nlev}  "
            f"total={ms_step:6.1f}ms  "
            f"dycore={ms_dycore:6.1f}ms  "
            f"bcast={ms_bcast:5.1f}ms  "
            f"reduce={ms_reduce:5.1f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
