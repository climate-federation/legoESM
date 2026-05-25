"""RCE experiment under MPI — full physics, 132x132, 12 ranks.

Composes the full RCE stack with real MPI calls per step:

* Per-step MPI: rank-0 runs the dycore step + broadcasts the new
  state to all ranks. Each rank then applies the RCE physics
  helpers on its owned slab and contributes to the global
  reductions (mean wind, moist mass).
* Physics: gray radiation (LW+SW two-stream), Kessler warm-rain
  microphysics, Smagorinsky LES (in dycore config), surface flux
  composer, tracer positivity, mean-wind filter, moist-mass fixer.
* RCE diagnostics: CWV, MSE, cloud fraction, precip proxy, CFL
  per step. Global moist mass via `global_sum_mpi`.

DYCORE-MPI GAP
--------------
The plane CRM dycore is NOT domain-decomposed (no halo exchange
in the operators). To keep the integration physically correct
the dycore step is run on rank 0 only and the new state is
broadcast each step. This is asymmetric work — rank 0 carries
the dycore load while every rank carries identical-state physics
+ MPI reductions. The honest fix is to thread halo exchange
through plane_operators (~50 jnp.roll sites — separate PR).

CLI
---
.. code-block:: bash

   mpirun -np 12 .venv/bin/python \\
       scripts/run_rce_mpi_full.py \\
       --steps 50 --dt 0.5 --output results/rce_mpi_full.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force CPU before any JAX import — mpi4jax needs CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from legoesm import constants
from legoesm.atmosphere.dynamics.cfl_diagnostic import (
    compute_courant_numbers_plane,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.atmosphere.dynamics.rce_diagnostics import (
    cloud_fraction_profile_plane, column_moist_static_energy_plane,
    column_water_vapor_plane, precipitation_rate_proxy_plane,
)
from legoesm.atmosphere.dynamics.rce_mpi import (
    compute_total_water_mass_plane_mpi,
    remove_horizontal_mean_wind_plane_mpi,
)
from legoesm.atmosphere.dynamics.rce_surface_flux import (
    apply_rce_surface_fluxes, wind_speed_at_lowest_level_plane,
)
from legoesm.atmosphere.dynamics.tracer_positivity import (
    apply_positive_filter_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_qv_ref_fn, make_wing2018_theta_ref_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


T_SFC_K = 300.0
Q_SFC_FRAC = 0.0224
GAMMA_TROP = 6.7e-3
Z_T = 15_000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=132)
    p.add_argument("--ny", type=int, default=132)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--H", type=float, default=20_000.0)
    p.add_argument("--dz-sfc", type=float, default=100.0)
    p.add_argument("--c-h", type=float, default=1.5e-3)
    p.add_argument("--n-acoustic-substeps", type=int, default=24)
    p.add_argument("--smag-cs", type=float, default=0.2,
                   help="Smagorinsky-Lilly LES coefficient (0 to disable).")
    p.add_argument("--hyperdiff", type=float, default=1.0e6)
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--sponge-width", type=float, default=5_000.0)
    p.add_argument("--radiation", choices=["gray", "none"], default="gray")
    p.add_argument("--microphysics",
                   choices=["kessler", "none"], default="kessler")
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output", type=str,
                   default="results/rce_mpi_full.txt")
    return p.parse_args()


def build_height_coord_and_state(args, grid):
    theta_fn = make_wing2018_theta_ref_fn(
        T_sfc=T_SFC_K, q_sfc=Q_SFC_FRAC, z_t=Z_T, Gamma=GAMMA_TROP,
    )
    qv_fn = make_wing2018_qv_ref_fn(q_sfc=Q_SFC_FRAC, z_t=Z_T)
    hc = create_stretched_height_coordinate(
        n_levels=args.nlev, H=args.H, dz_sfc=args.dz_sfc,
        theta_ref_fn=theta_fn,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    z = hc.z_full
    nlev = z.shape[0]
    qv_profile = qv_fn(z)
    new_tracers = jnp.zeros((grid.ny, grid.nx, nlev, 3), dtype=jnp.float64)
    qv_3d = qv_profile[None, None, :] * jnp.ones(
        (grid.ny, grid.nx, nlev), dtype=jnp.float64,
    )
    new_tracers = new_tracers.at[..., 0].set(qv_3d)
    # Small theta perturbation to seed convection.
    rng = jax.random.PRNGKey(0)
    theta_kick = 0.1 * jax.random.normal(
        rng, state.theta_prime.data.shape,
    )
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_kick),
        tracers=state.tracers.replace(data=new_tracers),
    )
    return hc, state


def _build_surface_flux_physics_fn(grid, hc, tm):
    """Wrap apply_rce_surface_fluxes as a tendency-producing physics_fn.
    Returns dq, dtheta for the lowest level as a tendency rate (per
    second) so the dycore SSP-RK3 outer integrator can sum it in."""
    from legoesm.atmosphere.dynamics.rce_surface_flux import (
        compose_rce_surface_scalar_tendencies,
    )

    def physics_fn(state, grid_in, hc_in, tm_in):
        wspd = wind_speed_at_lowest_level_plane(state)
        T_sfc = jnp.full(wspd.shape, T_SFC_K)
        q_sfc = jnp.full(wspd.shape, Q_SFC_FRAC)
        dtheta_sfc, dq_sfc = compose_rce_surface_scalar_tendencies(
            state, hc_in, T_sfc, q_sfc, wspd,
            C_h=1.5e-3, gustiness_floor=5.0,
        )
        k_sfc = -1
        dtheta_data = jnp.zeros_like(state.theta_prime.data).at[
            ..., k_sfc
        ].set(dtheta_sfc)
        dtr_data = jnp.zeros_like(state.tracers.data).at[
            ..., k_sfc, 0
        ].set(dq_sfc)
        return PlaneNonHydrostaticTendencies(
            du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
            dv_dt=state.v.replace(data=jnp.zeros_like(state.v.data)),
            dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
            dtheta_prime_dt=state.theta_prime.replace(data=dtheta_data),
            drho_prime_dt=state.rho_prime.replace(
                data=jnp.zeros_like(state.rho_prime.data),
            ),
            dphis_dt=state.phis.replace(
                data=jnp.zeros_like(state.phis.data),
            ),
            dtracers_dt=state.tracers.replace(data=dtr_data),
        )
    return physics_fn


def _sum_tendencies(*tendencies):
    out_kwargs = {}
    fields = (
        "du_dt", "dv_dt", "dw_dt", "dtheta_prime_dt",
        "drho_prime_dt", "dphis_dt", "dtracers_dt",
    )
    for fld in fields:
        sum_data = sum(
            getattr(t, fld).data for t in tendencies
        )
        out_kwargs[fld] = getattr(tendencies[0], fld).replace(data=sum_data)
    return PlaneNonHydrostaticTendencies(**out_kwargs)


def build_physics_fn(args, grid, hc, tm):
    fns = [_build_surface_flux_physics_fn(grid, hc, tm)]
    if args.radiation != "none":
        fns.append(make_radiation_physics(
            RadiationConfig(scheme=args.radiation), model_type="plane",
        ))
    if args.microphysics != "none":
        fns.append(make_microphysics_physics(
            MicrophysicsConfig(scheme=args.microphysics),
            model_type="plane", dt=args.dt,
        ))

    def physics_fn(state, grid_in, hc_in, tm_in):
        tendencies = [fn(state, grid_in, hc_in, tm_in) for fn in fns]
        return _sum_tendencies(*tendencies)
    return physics_fn


def _broadcast_state(state, comm, root=0):
    """Per-step MPI: rank 0 sends new state to all ranks.
    Each prognostic Field is broadcast individually."""
    field_names = (
        "u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers",
    )
    new_fields = {}
    for name in field_names:
        fld = getattr(state, name)
        arr = np.asarray(fld.data)
        # Force allocation on every rank with right shape+dtype.
        if comm.Get_rank() != root:
            arr = np.empty(arr.shape, dtype=arr.dtype)
        comm.Bcast(arr, root=root)
        new_fields[name] = fld.replace(data=jnp.asarray(arr))
    return state._replace(**new_fields)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if n_ranks == 12:
        n_ranks_y, n_ranks_x = 3, 4
    elif n_ranks == 4:
        n_ranks_y, n_ranks_x = 2, 2
    elif n_ranks == 2:
        n_ranks_y, n_ranks_x = 1, 2
    elif n_ranks == 1:
        n_ranks_y, n_ranks_x = 1, 1
    else:
        n_ranks_y, n_ranks_x = 1, n_ranks

    if args.ny % n_ranks_y != 0 or args.nx % n_ranks_x != 0:
        if rank == 0:
            print(
                f"ERROR: {args.ny}x{args.nx} not divisible by "
                f"{n_ranks_y}x{n_ranks_x}."
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
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=args.n_acoustic_substeps,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)
    physics_fn = build_physics_fn(args, grid, hc, terrain)
    target_water = compute_total_water_mass_plane(state, hc, grid)

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
            f"# RCE MPI FULL  n_ranks={n_ranks} "
            f"decomp={n_ranks_y}x{n_ranks_x}  grid={args.ny}x{args.nx} "
            f"nlev={args.nlev}  dx={args.dx} m  dt={args.dt} s"
        )
        lines.append(
            f"# physics: radiation={args.radiation} "
            f"microphysics={args.microphysics} "
            f"smag_cs={args.smag_cs} hyperdiff={args.hyperdiff:.1e}"
        )
        lines.append(
            f"# MPI: dycore on rank 0 + broadcast; "
            f"reductions (mean wind, moist mass) all ranks."
        )
        lines.append(
            f"# step  t[s]  CWV[kg/m2]  MSE[J/m2]  max|w|  max(q_c)  "
            f"max(precip)  C_a/sub  global_water[kg]"
        )
        for ln in lines:
            print(ln, flush=True)

    for step in range(args.steps):
        # Rank 0 steps the dycore on the full global state.
        if rank == 0:
            state = model.step(state, dt=args.dt, physics_fn=physics_fn)
        # Per-step MPI: broadcast state to every rank.
        state = _broadcast_state(state, comm, root=0)
        # All ranks: apply MPI-aware mean wind removal + moist mass fixer
        # on the (still-global) state. The MPI reductions sum over
        # owned-mask cells so the per-rank-physics result is correct.
        state = remove_horizontal_mean_wind_plane_mpi(
            state, layout, owned_mask,
        )
        state = apply_positive_filter_state(
            state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
        )
        global_water = compute_total_water_mass_plane_mpi(
            state, hc, grid, layout, owned_mask,
        )
        state = fix_moist_mass_plane(
            state, hc, grid, target_total_water=target_water,
        )
        cwv = column_water_vapor_plane(state, hc)
        mse = column_moist_static_energy_plane(state, hc)
        precip = precipitation_rate_proxy_plane(state, hc)
        cn = compute_courant_numbers_plane(
            state, hc, grid, args.dt,
            n_acoustic_substeps=args.n_acoustic_substeps,
        )
        max_w = float(jnp.max(jnp.abs(state.w.data)))
        max_qc = float(jnp.max(state.tracers.data[..., 1]))
        if rank == 0 and ((step + 1) % args.print_every == 0 or step == 0):
            line = (
                f"{step:5d}  {(step+1)*args.dt:9.2f}  "
                f"{float(jnp.mean(cwv)):10.3e}  "
                f"{float(jnp.mean(mse)):10.3e}  "
                f"{max_w:9.3e}  "
                f"{max_qc:10.3e}  "
                f"{float(jnp.max(precip)):10.3e}  "
                f"{float(cn.acoustic):6.3f}  "
                f"{float(global_water):10.3e}"
            )
            print(line, flush=True)
            lines.append(line)
        if not bool(jnp.all(jnp.isfinite(cwv))):
            if rank == 0:
                print(f"NaN CWV step {step} — bail", flush=True)
            break

    if rank == 0:
        cwv_f = float(jnp.mean(cwv))
        mse_f = float(jnp.mean(mse))
        precip_f = float(jnp.max(precip))
        max_w_f = float(jnp.max(jnp.abs(state.w.data)))
        summary = (
            f"\n# === REALISM CHECK ===\n"
            f"# Final mean CWV       = {cwv_f:.2f} kg/m²  "
            f"(RCEMIP SST=300K typical 30-50)\n"
            f"# Final mean MSE       = {mse_f:.3e} J/m²  "
            f"(tropical typical 3-4e9)\n"
            f"# Final max|w|         = {max_w_f:.3f} m/s  "
            f"(deep-convection typical 5-30 m/s once spun up)\n"
            f"# Final max precip     = {precip_f:.3e} kg/m²/s = "
            f"{precip_f * 86400:.2f} mm/day  "
            f"(RCEMIP daily typical 3-5)\n"
            f"# Final per-substep Ca = {float(cn.acoustic):.3f}\n"
            f"# Global water         = {float(global_water):.3e} kg "
            f"(MPI-reduced)\n"
        )
        print(summary)
        lines.append(summary)
        Path(args.output).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
