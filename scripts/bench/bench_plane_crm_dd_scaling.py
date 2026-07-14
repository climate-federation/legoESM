"""MPI strong + weak scaling benchmark for the plane CRM step_halo path (R8).

Two modes:

* **strong** (``--mode strong``): fixed GLOBAL grid (default 48x48),
  rank count varies via the mpirun ``-np`` argument. Strong-scaling
  efficiency at N ranks = wall_at_1 / (N * wall_at_N). Ideal = 1.0.

* **weak** (``--mode weak``): fixed PER-RANK grid (default 24x24),
  global grid grows with rank count (2 ranks → 48x24 or 24x48,
  4 ranks → 48x48, etc.). Weak-scaling efficiency at N ranks =
  wall_at_1 / wall_at_N. Ideal = 1.0.

The benchmark uses the step_halo path (R4-R7 features wired in), so
the timing reflects the production multi-rank cost model: scatter →
local slabs → step_halo per outer step (with halo exchange, no
global comm beyond the per-step mass-fixer allreduce + the per-step
mean-wind allreduce) → MPI mean-wind + MPI moist-mass fixer.

No physics — bench focuses on the dynamical core scaling. Physics
is column-local on the plane so it never contributes communication;
its scaling matches single-rank ratio.

CLI
---
.. code-block:: bash

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 4 .venv/bin/python \\
        scripts/bench_plane_crm_dd_scaling.py \\
        --mode strong --nx 48 --ny 48 --nlev 20 \\
        --warmup-steps 3 --time-steps 20 \\
        --output results/dd_scaling/strong_n4.txt

Run the same with ``-np 1``, ``-np 2``, ``-np 4`` and aggregate the
output files to compute strong-scaling efficiency.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from mpi4py import MPI

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    remove_horizontal_mean_wind_plane_mpi,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_theta_ref_fn,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import (
    make_plane_pencil_grid, make_plane_pencil_layout, scatter_plane_field,
)

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong",
                   help="strong = fixed global grid; weak = fixed per-rank grid.")
    p.add_argument("--nx", type=int, default=48,
                   help="strong: global nx; weak: per-rank nx.")
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--dx", type=float, default=2000.0)
    p.add_argument("--H", type=float, default=20_000.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64",
                   help="Array/compute precision (jax_enable_x64 set to match).")
    p.add_argument("--n-acoustic-substeps", type=int, default=12)
    p.add_argument("--warmup-steps", type=int, default=3,
                   help="JIT-warmup steps NOT counted in timing.")
    p.add_argument("--time-steps", type=int, default=20,
                   help="Steps included in the timed window.")
    p.add_argument("--smag-cs", type=float, default=0.0,
                   help="Default 0 keeps Smag off so bench focuses on "
                        "halo-exchange + acoustic-substep scaling. Set "
                        "0.2 to include Smag in the cost model.")
    p.add_argument("--output", type=str, default=None,
                   help="One-line CSV: mode,n_ranks,ny_global,nx_global,"
                        "ny_local,nx_local,nlev,wall_s,steps_per_s,"
                        "wall_per_step_s. Rank 0 writes; other ranks "
                        "skip.")
    return p.parse_args()


def _decompose_ranks(n_ranks: int) -> tuple[int, int]:
    """Pick a 2D mesh that's as square as possible."""
    if n_ranks == 1:
        return 1, 1
    if n_ranks == 2:
        return 2, 1
    if n_ranks == 4:
        return 2, 2
    if n_ranks == 8:
        return 4, 2
    if n_ranks == 12:
        return 3, 4
    if n_ranks == 16:
        return 4, 4
    # Fall back to 1D pencil.
    return 1, n_ranks


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    n_ranks_y, n_ranks_x = _decompose_ranks(n_ranks)

    # Compute the GLOBAL grid for the chosen mode.
    if args.mode == "strong":
        ny_global, nx_global = args.ny, args.nx
    else:  # weak
        ny_global = args.ny * n_ranks_y
        nx_global = args.nx * n_ranks_x

    if ny_global % n_ranks_y != 0 or nx_global % n_ranks_x != 0:
        if rank == 0:
            print(
                f"ERROR: global grid {ny_global}x{nx_global} not "
                f"divisible by {n_ranks_y}x{n_ranks_x}",
            )
        return

    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=ny_global, nx_global=nx_global,
    )

    # Build global IC (deterministic on every rank).
    grid_global = create_plane_grid(
        nx=nx_global, ny=ny_global, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=_DTYPE,
    )
    # Near-EQUILIBRIUM CRM benchmark: theta reference uses T_v0=300 K to match
    # the 300 K surface setup (NOT the strict-RCEMIP fixed 295 K). The surface-
    # temp param was renamed T_sfc -> T_v0 (old kwarg TypeErrors).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=300.0, q_sfc=0.0224, z_t=15_000.0, Gamma=6.7e-3,
    )
    hc = create_height_coordinate(
        n_levels=args.nlev, H=args.H, theta_ref_fn=theta_fn,
    )
    state_global = make_rest_state(grid_global, hc, dtype=_DTYPE)
    # Small momentum kick so the slow tendency does real work
    # (zero state → most operators short-circuit to zero, hides the
    # advection + diffusion cost).
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 4)
    state_global = state_global._replace(
        u=state_global.u.replace(
            data=0.05 * jax.random.normal(keys[0], state_global.u.data.shape),
        ),
        v=state_global.v.replace(
            data=0.05 * jax.random.normal(keys[1], state_global.v.data.shape),
        ),
        theta_prime=state_global.theta_prime.replace(
            data=0.1 * jax.random.normal(
                keys[2], state_global.theta_prime.data.shape,
            ),
        ),
    )

    # Scatter to local slab.
    state_local = state_global
    if n_ranks > 1:
        new_fields = {}
        for name in (
            "u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers",
        ):
            fld = getattr(state_global, name)
            new_fields[name] = fld.replace(
                data=scatter_plane_field(fld.data, layout),
            )
        state_local = state_global._replace(**new_fields)

    grid_local = (
        make_plane_pencil_grid(
            layout, dx=args.dx, dy=args.dx, nlev=args.nlev,
            dtype=_DTYPE,
        )
        if n_ranks > 1 else grid_global
    )
    terrain_local = make_flat_plane_terrain_metric(grid_local, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=1.0e6,
        hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        use_coriolis=False,
        semi_implicit_acoustic=True,
        acoustic_off_centering=0.1,
        fix_mass=True, anchor_mass_to_initial=True,
        n_acoustic_substeps=args.n_acoustic_substeps,
    )
    model = PlaneCompressibleEulerModel(
        grid_local, hc, terrain_local, config=cfg,
    )
    owned_mask = jnp.ones(
        (layout.ny_local, layout.nx_local), dtype=_DTYPE,
    )

    # Warmup (JIT compile + first MPI sync).
    for _ in range(args.warmup_steps):
        state_local = model.step_halo(
            state_local, dt=args.dt, layout=layout, owned_mask=owned_mask,
        )
        state_local = remove_horizontal_mean_wind_plane_mpi(
            state_local, layout, owned_mask,
        )

    # Barrier so all ranks start the timed window together.
    comm.Barrier()
    t0 = time.time()
    for _ in range(args.time_steps):
        state_local = model.step_halo(
            state_local, dt=args.dt, layout=layout, owned_mask=owned_mask,
        )
        state_local = remove_horizontal_mean_wind_plane_mpi(
            state_local, layout, owned_mask,
        )
    # Force materialization of any pending traces — without this the
    # timing would just measure dispatch latency, not real compute.
    state_local.u.data.block_until_ready()
    comm.Barrier()
    wall = time.time() - t0

    steps_per_s = args.time_steps / max(wall, 1e-30)
    wall_per_step = wall / args.time_steps

    if rank == 0:
        line = (
            f"{args.mode},{n_ranks},{ny_global},{nx_global},"
            f"{layout.ny_local},{layout.nx_local},{args.nlev},"
            f"{wall:.4f},{steps_per_s:.4f},{wall_per_step:.6f}"
        )
        print(line)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                out_path.write_text(
                    "mode,n_ranks,ny_global,nx_global,ny_local,nx_local,"
                    "nlev,wall_s,steps_per_s,wall_per_step_s\n"
                )
            with out_path.open("a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
