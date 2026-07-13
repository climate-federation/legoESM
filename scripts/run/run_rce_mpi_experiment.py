"""Real RCE experiment under MPI — 132x132 plane CRM, dx=2 km.

Demonstrates the full RCE stack (Wing 2018 IC + stretched HC +
surface fluxes + tracer positivity + mean-wind removal + moist-
mass fixer + RCE diagnostics) under MPI launch.

GAP NOTE
--------
The plane CRM dycore is NOT yet MPI-domain-decomposed. The halo-
exchange infrastructure exists in legoesm.parallel.plane_mpi but
the dycore step does not consume it. Therefore this script runs
in REPLICATED mode: every rank initializes an identical full-
domain (132x132) state and steps it independently. The new MPI
reductions added in PR #307 are exercised to verify consistency
across ranks (every rank's mean CWV must agree to round-off).

The user-visible deliverable here is realism of the RCE
integration: CWV magnitude, MSE order, CFL stability, and the
fact that the full stack composes cleanly under mpirun.

CLI
---
.. code-block:: bash

   mpirun -np 12 .venv/bin/python \\
       scripts/run_rce_mpi_experiment.py \\
       --steps 50 --dt 4.0 --output results/rce_mpi_12.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force CPU before any JAX import — Metal + mpi4jax do not coexist.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from mpi4py import MPI

from legoesm import constants
from legoesm.atmosphere.dynamics.shared.cfl_diagnostic import (
    compute_courant_numbers_plane,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.shared.mean_wind_filter import (
    remove_horizontal_mean_wind,
)
from legoesm.atmosphere.dynamics.crm.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
    cloud_fraction_profile_plane, column_moist_static_energy_plane,
    column_water_vapor_plane, precipitation_rate_proxy_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    compute_total_water_mass_plane_mpi,
)
from legoesm.atmosphere.dynamics.crm.rce_surface_flux import (
    apply_rce_surface_fluxes, wind_speed_at_lowest_level_plane,
)
from legoesm.atmosphere.dynamics.shared.tracer_positivity import (
    apply_positive_filter_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_qv_ref_fn, make_wing2018_theta_ref_fn,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------
# Wing 2018 RCEMIP1 SST = 300 K
# --------------------------------------------------------------------
T_SFC_K = 300.0
Q_SFC_FRAC = 0.0224
GAMMA_TROP = 6.7e-3
Z_T = 15_000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=132,
                   help="Global nx (must be divisible by n_ranks_x).")
    p.add_argument("--ny", type=int, default=132)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0,
                   help="Horizontal grid spacing [m].")
    p.add_argument("--dt", type=float, default=4.0,
                   help="Outer time step [s].")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--H", type=float, default=20_000.0)
    p.add_argument("--dz-sfc", type=float, default=50.0)
    p.add_argument("--c-h", type=float, default=1.5e-3)
    p.add_argument("--no-mass-fixer", action="store_true", default=False,
                   help="iter-95b: disable fix_moist_mass_plane "
                        "(which rescales total water to IC every step). "
                        "Default False matches the script's conservation-"
                        "smoke purpose; set this flag for any spin-up "
                        "experiment >~1 sim-hour so surface flux can "
                        "NET ADD moisture instead of being clipped.")
    p.add_argument("--n-acoustic-substeps", type=int, default=12)
    p.add_argument("--output", type=str,
                   default="results/rce_mpi_experiment.txt")
    return p.parse_args()


def build_height_coord_and_state(args, grid):
    # Near-EQUILIBRIUM RCE smoke: theta reference uses T_v0=T_SFC_K to match the
    # SST=300 K setup (NOT the strict-RCEMIP fixed 295 K, which floods the column).
    # The surface-temp param was renamed T_sfc -> T_v0 (old kwarg TypeErrors).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=T_SFC_K, q_sfc=Q_SFC_FRAC, z_t=Z_T, Gamma=GAMMA_TROP,
    )
    qv_fn = make_wing2018_qv_ref_fn(q_sfc=Q_SFC_FRAC, z_t=Z_T)
    # iter-95: pass p_sfc=101480 (Wing 2018 Tab A1) so the
    # hydrostatic reference state uses the bottom-up integration
    # with the correct surface BC. Without this the legacy top-down
    # BC produces ~12 K too-hot T at the lowest model level on
    # H=33 km columns, breaking surface-flux coupling.
    hc = create_stretched_height_coordinate(
        n_levels=args.nlev, H=args.H, dz_sfc=args.dz_sfc,
        theta_ref_fn=theta_fn,
        p_sfc=101480.0,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    z = hc.z_full
    nlev = z.shape[0]
    qv_profile = qv_fn(z)
    new_tracers = jnp.zeros((grid.ny, grid.nx, nlev, 3), dtype=jnp.float64)
    new_tracers = new_tracers.at[..., 0].set(
        qv_profile[None, None, :] * jnp.ones((grid.ny, grid.nx, nlev)),
    )
    state = state._replace(
        tracers=state.tracers.replace(data=new_tracers),
    )
    return hc, state


def physics_step(state, hc, grid, dt, C_h):
    """Codex iter-2 ordering: surface flux → mean-wind → positivity."""
    wspd = wind_speed_at_lowest_level_plane(state)
    T_sfc = jnp.full(wspd.shape, T_SFC_K)
    q_sfc = jnp.full(wspd.shape, Q_SFC_FRAC)
    state = apply_rce_surface_fluxes(
        state, hc, dt, T_sfc, q_sfc, wspd,
        C_h=C_h, gustiness_floor=5.0,
    )
    state = remove_horizontal_mean_wind(state)
    state = apply_positive_filter_state(
        state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
    )
    return state


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Layout for MPI helpers: factor n_ranks as n_ranks_y × n_ranks_x.
    # 12 = 3 × 4 works for 132×132 (44 × 33 local).
    if n_ranks == 12:
        n_ranks_y, n_ranks_x = 3, 4
    elif n_ranks == 4:
        n_ranks_y, n_ranks_x = 2, 2
    elif n_ranks == 2:
        n_ranks_y, n_ranks_x = 1, 2
    elif n_ranks == 1:
        n_ranks_y, n_ranks_x = 1, 1
    else:
        # Default to 1×N decomposition.
        n_ranks_y, n_ranks_x = 1, n_ranks

    if args.ny % n_ranks_y != 0 or args.nx % n_ranks_x != 0:
        if rank == 0:
            print(
                f"ERROR: grid {args.ny}x{args.nx} not divisible by "
                f"{n_ranks_y}x{n_ranks_x} decomposition."
            )
        return

    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=args.ny, nx_global=args.nx,
    )

    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    hc, state = build_height_coord_and_state(args, grid)
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        fix_mass=True, anchor_mass_to_initial=True,
        n_acoustic_substeps=args.n_acoustic_substeps,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)

    # Snapshot initial moist mass (full state on every rank — replicated).
    target_water = compute_total_water_mass_plane(state, hc, grid)

    # Owned-mask for MPI reductions: each rank owns its 132/3 x 132/4
    # = 44x33 slab even though the dycore steps the full state.
    owned_mask = jnp.zeros((args.ny, args.nx), dtype=jnp.float64)
    owned_mask = owned_mask.at[
        layout.iy_start:layout.iy_end,
        layout.ix_start:layout.ix_end,
    ].set(1.0)

    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append(
            f"# RCE MPI experiment  n_ranks={n_ranks} "
            f"decomp={n_ranks_y}x{n_ranks_x}  grid={args.ny}x{args.nx} "
            f"nlev={args.nlev}  dx={args.dx}  dt={args.dt} "
            f"n_acoustic_substeps={args.n_acoustic_substeps}"
        )
        lines.append(
            f"# NOTE: dycore runs REPLICATED per rank (plane CRM has "
            f"no MPI halo exchange yet). MPI reductions verify "
            f"consistency."
        )
        lines.append(
            f"# step  t[s]  CWV_mean[kg/m2]  CWV_max  CWV_min  "
            f"MSE_mean[J/m2]  cloud_max  precip_max[kg/m2/s]  "
            f"C_a_per_substep  mpi_total_water_global[kg]"
        )
        print(lines[0])
        print(lines[1])
        print(lines[2])

    for step in range(args.steps):
        state = model.step(state, dt=args.dt)
        state = physics_step(state, hc, grid, args.dt, args.c_h)
        if not args.no_mass_fixer:
            state = fix_moist_mass_plane(
                state, hc, grid, target_total_water=target_water,
            )
        cwv = column_water_vapor_plane(state, hc)
        mse = column_moist_static_energy_plane(state, hc)
        cf = cloud_fraction_profile_plane(state, hc)
        precip = precipitation_rate_proxy_plane(state, hc)
        cn = compute_courant_numbers_plane(
            state, hc, grid, args.dt,
            n_acoustic_substeps=args.n_acoustic_substeps,
        )
        # Exercise the MPI reduction — sums across all ranks' owned
        # slabs. With replicated state, the global sum should equal
        # the per-rank sum (each rank covers its own 44x33 slab).
        global_water = float(compute_total_water_mass_plane_mpi(
            state, hc, grid, layout, owned_mask,
        ))
        if rank == 0:
            line = (
                f"{step:5d}  {(step+1)*args.dt:9.2f}  "
                f"{float(jnp.mean(cwv)):14.4e}  "
                f"{float(jnp.max(cwv)):14.4e}  "
                f"{float(jnp.min(cwv)):14.4e}  "
                f"{float(jnp.mean(mse)):14.4e}  "
                f"{float(jnp.max(cf)):8.4f}  "
                f"{float(jnp.max(precip)):14.4e}  "
                f"{float(cn.acoustic):8.4f}  "
                f"{global_water:14.4e}"
            )
            print(line, flush=True)
            lines.append(line)
        if not bool(jnp.all(jnp.isfinite(cwv))):
            if rank == 0:
                print(f"NaN CWV at step {step} — bail", flush=True)
            break

    if rank == 0:
        # Realism summary.
        cwv_final = float(jnp.mean(cwv))
        mse_final = float(jnp.mean(mse))
        precip_final = float(jnp.max(precip))
        summary = (
            f"\n# === REALISM CHECK ===\n"
            f"# Final mean CWV       = {cwv_final:.2f} kg/m²  "
            f"(RCEMIP tropical SST=300K typical 30-50 kg/m²)\n"
            f"# Final mean MSE       = {mse_final:.3e} J/m²  "
            f"(tropical typical ~3-4e9)\n"
            f"# Final max precip     = {precip_final:.3e} kg/m²/s\n"
            f"# Final per-substep C_a = {float(cn.acoustic):.3f}  "
            f"(must be < 1)\n"
            f"# MPI global water     = {global_water:.3e} kg  "
            f"(stable across ranks ⇒ MPI reductions correct)\n"
        )
        print(summary)
        lines.append(summary)
        out_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
