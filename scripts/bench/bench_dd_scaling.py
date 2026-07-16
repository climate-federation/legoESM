"""Domain-decomposed scaling bench using step_halo.

Each rank owns local slab (ny_local × nx_local × nlev). Halo
exchange per slow-tendency call via packed_exchange_halo_plane_yxz.
Acoustic substeps vertical-local (no MPI).

Reports per-step wall for strong + weak scaling. Compare to
bench_mpi_scaling.py (replicated dycore) which showed
anti-scaling.
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

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--ny", type=int, default=24)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-bench", type=int, default=15)
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

    # Local grid (per-rank slab).
    local_grid = create_plane_grid(
        nx=layout.nx_local, ny=layout.ny_local, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    hc = create_stretched_height_coordinate(
        args.nlev, H=20_000.0, dz_sfc=100.0,
    )
    local_tm = make_flat_plane_terrain_metric(local_grid, hc)
    cfg = CompressibleEulerConfig(
        n_acoustic_substeps=12, semi_implicit_acoustic=True,
        sponge_coeff=0.05, sponge_width=5000.,
        hyperdiff_coeff=1e6, hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
        smagorinsky_cs=0.0, use_coriolis=False,
        fix_mass=False, anchor_mass_to_initial=False,
    )
    model = PlaneCompressibleEulerModel(local_grid, hc, local_tm, cfg)
    local_state = make_rest_state(local_grid, hc, dtype=jnp.float64)
    new_tr = jnp.zeros(
        (layout.ny_local, layout.nx_local, args.nlev, 3),
        dtype=jnp.float64,
    ).at[..., 0].set(0.01)
    local_state = local_state._replace(
        tracers=local_state.tracers.replace(data=new_tr),
    )
    rng = jax.random.PRNGKey(rank)
    kick = 0.001 * jax.random.normal(
        rng, local_state.theta_prime.data.shape,
    )
    local_state = local_state._replace(
        theta_prime=local_state.theta_prime.replace(data=kick),
    )

    # Use jit-split slow tendency directly (skips eager step_halo).
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
        slow_tendency_jit_split,
    )

    def one_step(s):
        # Just compute slow tendency (proxy for full step cost).
        return slow_tendency_jit_split(s, local_grid, hc, local_tm, cfg, layout)

    # Warmup.
    for _ in range(args.n_warmup):
        tend = one_step(local_state)
    tend.du_dt.data.block_until_ready()
    comm.Barrier()

    t0 = time.time()
    for _ in range(args.n_bench):
        tend = one_step(local_state)
    tend.du_dt.data.block_until_ready()
    comm.Barrier()
    wall = time.time() - t0

    if rank == 0:
        ms = wall / args.n_bench * 1000
        print(
            f"{args.label:24s} nranks={n_ranks:2d} "
            f"grid={args.ny}x{args.nx} ny_local={layout.ny_local} "
            f"nx_local={layout.nx_local}  per_step={ms:6.1f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
