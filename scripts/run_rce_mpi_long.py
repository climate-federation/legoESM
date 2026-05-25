"""Long RCE run under MPI — 12 ranks, 132x132, 100-day capable.

Production-ready harness for RCE on the plane CRM with:

* Full physics: gray radiation, Kessler microphysics, Smagorinsky
  LES, surface flux, tracer positivity, mean-wind filter (MPI),
  moist-mass fixer (MPI).
* MPI: rank 0 stepd dycore + broadcasts state; all ranks reduce
  for mean wind + moist mass.
* Outputs (all on rank 0):
  - `<output>/log.txt` — per-step CSV with diagnostics.
  - `<output>/snapshots/snap_day_NNN.npz` — every 24 hrs (sim
    time), x-y surface fields (CWV, MSE, surface T_atm, surface
    q_v, |U|_sfc, precip).
  - `<output>/profiles/prof_day_NNN.npz` — every 5 days,
    horizontally-meaned vertical profiles (T, q_v, q_c, q_r,
    cloud fraction).
  - `<output>/progress.txt` — overwritten each step with
    "current day = X.YYY" for external monitoring.

CLI
---
.. code-block:: bash

   mpirun -np 12 .venv/bin/python \\
       scripts/run_rce_mpi_long.py \\
       --days 100 --dt 2.0 --output results/rce100d

Wall-clock estimate (M5 Pro, 12 CPUs):
   100 days × 86400 s / dt=2 s = 4.32M steps
   ~1.5 s/step (rank 0 bound) → ~75 days wall clock.
   Requires cluster for full 100-day run.
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
    compose_rce_surface_scalar_tendencies,
    wind_speed_at_lowest_level_plane,
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
from legoesm.grids.vertical import (
    create_height_coordinate, create_stretched_height_coordinate,
)
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


T_SFC_K = 300.0
Q_SFC_FRAC = 0.0224          # Wing saturation q_v at SST
Q_SFC_IC_FRAC = 0.018        # IC sub-saturated (~80% RH) — avoids
                              # instant-condensation blowup at start.
GAMMA_TROP = 6.7e-3
Z_T = 15_000.0
SEC_PER_DAY = 86400.0

# Radiation call-frequency convention (literature):
# - SAM (Khairoutdinov-Randall): 600 s
# - CM1 (Bryan-Fritsch): 60-120 s
# - WRF-LES: 300-600 s
# - RCEMIP1 spec: model-default but typically 600 s
# Default here: 600 s. Tendency held constant between calls.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=132)
    p.add_argument("--ny", type=int, default=132)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0)
    p.add_argument("--dt", type=float, default=6.0,
                   help="Outer dt [s]. CFL_a=6/24·340/100=0.85, safe.")
    p.add_argument("--days", type=float, default=100.0)
    p.add_argument("--H", type=float, default=33_000.0,
                   help="Model top [m]. RCEMIP1 33 km.")
    p.add_argument("--dz-sfc", type=float, default=100.0,
                   help="Stretched-grid surface dz [m] (ignored if "
                        "--vertical-grid=uniform).")
    p.add_argument("--vertical-grid", choices=["uniform", "stretched"],
                   default="uniform",
                   help="uniform = no stretching, dz=H/nlev (~1100m for "
                        "default H=33km, nlev=30). Stable at dt=6. "
                        "stretched = ~50-100m surface layer (CFL-tight, "
                        "needs dt<=1).")
    p.add_argument("--c-h", type=float, default=1.5e-3)
    p.add_argument("--n-acoustic-substeps", type=int, default=24)
    p.add_argument("--n-physics-substeps", type=int, default=10,
                   help="Operator-split physics N x per dycore step. "
                        "Stiff sources (microphysics, surface flux) "
                        "sub-stepped at dt_outer/N.")
    p.add_argument("--rad-call-interval-s", type=float, default=600.0,
                   help="Radiation call interval [s]. Literature: SAM "
                        "600s, CM1 60-120s, WRF-LES 300-600s. "
                        "Tendency held constant between calls.")
    p.add_argument("--smag-cs", type=float, default=0.2)
    p.add_argument("--semi-implicit-acoustic", action="store_true",
                   help="Use per-column Thomas tridiagonal solve for "
                        "the acoustic substep (lifts vertical-CFL "
                        "limit; main benefit on stretched grids).")
    p.add_argument("--hyperdiff", type=float, default=1.0e6)
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--sponge-width", type=float, default=5_000.0)
    p.add_argument("--snapshot-hours", type=float, default=24.0)
    p.add_argument("--profile-days", type=float, default=5.0)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--output", type=str, default="results/rce_long")
    return p.parse_args()


def build_height_coord_and_state(args, grid):
    theta_fn = make_wing2018_theta_ref_fn(
        T_sfc=T_SFC_K, q_sfc=Q_SFC_FRAC, z_t=Z_T, Gamma=GAMMA_TROP,
    )
    qv_fn = make_wing2018_qv_ref_fn(q_sfc=Q_SFC_IC_FRAC, z_t=Z_T)
    if args.vertical_grid == "uniform":
        hc = create_height_coordinate(
            n_levels=args.nlev, H=args.H, theta_ref_fn=theta_fn,
        )
    else:
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
    rng = jax.random.PRNGKey(0)
    theta_kick = 0.1 * jax.random.normal(
        rng, state.theta_prime.data.shape,
    )
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_kick),
        tracers=state.tracers.replace(data=new_tracers),
    )
    return hc, state


def _surface_flux_physics_fn(grid, hc, tm):
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
        sum_data = sum(getattr(t, fld).data for t in tendencies)
        out_kwargs[fld] = getattr(tendencies[0], fld).replace(data=sum_data)
    return PlaneNonHydrostaticTendencies(**out_kwargs)


def build_fast_physics_fn(args, grid, hc, tm):
    """Per-step physics: surface flux + microphysics. Stiff sources
    that must respond to dycore evolution each step."""
    fns = [_surface_flux_physics_fn(grid, hc, tm)]
    fns.append(make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler"),
        model_type="plane", dt=args.dt / args.n_physics_substeps,
    ))

    def physics_fn(state, grid_in, hc_in, tm_in):
        tendencies = [fn(state, grid_in, hc_in, tm_in) for fn in fns]
        return _sum_tendencies(*tendencies)
    return physics_fn


def build_slow_physics_fn(args, grid, hc, tm):
    """Slow physics: gray radiation. Called every
    ``--rad-call-interval-s`` seconds; tendency held between calls."""
    rad_fn = make_radiation_physics(
        RadiationConfig(scheme="gray"), model_type="plane",
    )

    def physics_fn(state, grid_in, hc_in, tm_in):
        return rad_fn(state, grid_in, hc_in, tm_in)
    return physics_fn


def _broadcast_state(state, comm, root=0):
    field_names = (
        "u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers",
    )
    new_fields = {}
    for name in field_names:
        fld = getattr(state, name)
        arr = np.asarray(fld.data)
        if comm.Get_rank() != root:
            arr = np.empty(arr.shape, dtype=arr.dtype)
        comm.Bcast(arr, root=root)
        new_fields[name] = fld.replace(data=jnp.asarray(arr))
    return state._replace(**new_fields)


def save_snapshot(out_dir, day_idx, t_sim, state, hc):
    """x-y surface fields snapshot."""
    cwv = np.asarray(column_water_vapor_plane(state, hc))
    mse = np.asarray(column_moist_static_energy_plane(state, hc))
    precip = np.asarray(precipitation_rate_proxy_plane(state, hc))
    theta_sfc = np.asarray(
        hc.theta_ref[-1] + state.theta_prime.data[..., -1]
    )
    T_sfc = theta_sfc * float(hc.exner_ref[-1])
    qv_sfc = np.asarray(state.tracers.data[..., -1, 0])
    qc_sfc = np.asarray(state.tracers.data[..., -1, 1])
    qr_sfc = np.asarray(state.tracers.data[..., -1, 2])
    u_sfc = np.asarray(state.u.data[..., -1])
    v_sfc = np.asarray(state.v.data[..., -1])
    wind_sfc = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    snap_path = out_dir / "snapshots" / f"snap_day_{day_idx:04d}.npz"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snap_path,
        t_sim=t_sim, day=t_sim / SEC_PER_DAY,
        cwv=cwv, mse=mse, precip=precip,
        T_sfc=T_sfc, qv_sfc=qv_sfc, qc_sfc=qc_sfc, qr_sfc=qr_sfc,
        u_sfc=u_sfc, v_sfc=v_sfc, wind_sfc=wind_sfc,
    )


def save_profile(out_dir, day_idx, t_sim, state, hc):
    """Horizontally-averaged vertical profiles."""
    theta_tot = hc.theta_ref + state.theta_prime.data
    T = theta_tot * hc.exner_ref
    T_prof = np.asarray(jnp.mean(T, axis=(0, 1)))
    qv_prof = np.asarray(jnp.mean(state.tracers.data[..., 0], axis=(0, 1)))
    qc_prof = np.asarray(jnp.mean(state.tracers.data[..., 1], axis=(0, 1)))
    qr_prof = np.asarray(jnp.mean(state.tracers.data[..., 2], axis=(0, 1)))
    cf_prof = np.asarray(cloud_fraction_profile_plane(state, hc))
    w_var = np.asarray(jnp.var(state.w.data, axis=(0, 1)))
    z = np.asarray(hc.z_full)
    z_half = np.asarray(hc.z_half)
    prof_path = out_dir / "profiles" / f"prof_day_{day_idx:04d}.npz"
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prof_path,
        t_sim=t_sim, day=t_sim / SEC_PER_DAY,
        z=z, z_half=z_half,
        T=T_prof, qv=qv_prof, qc=qc_prof, qr=qr_prof,
        cloud_fraction=cf_prof, w_variance=w_var,
    )


def write_progress(out_dir, t_sim, total_t, step, total_steps,
                   wall_elapsed):
    day = t_sim / SEC_PER_DAY
    target_day = total_t / SEC_PER_DAY
    progress = step / max(total_steps, 1)
    eta_sec = (wall_elapsed / max(step, 1)) * (total_steps - step)
    p_path = out_dir / "progress.txt"
    p_path.write_text(
        f"current_day = {day:.3f}\n"
        f"target_day  = {target_day:.3f}\n"
        f"progress    = {progress * 100:.2f}%\n"
        f"step        = {step}/{total_steps}\n"
        f"wall_elapsed_s = {wall_elapsed:.1f}\n"
        f"eta_wall_s     = {eta_sec:.1f}\n"
        f"eta_wall_hours = {eta_sec / 3600:.2f}\n"
    )


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if n_ranks == 12:
        n_ranks_y, n_ranks_x = 3, 4
    elif n_ranks == 4:
        n_ranks_y, n_ranks_x = 2, 2
    elif n_ranks == 1:
        n_ranks_y, n_ranks_x = 1, 1
    else:
        n_ranks_y, n_ranks_x = 1, n_ranks

    if args.ny % n_ranks_y != 0 or args.nx % n_ranks_x != 0:
        if rank == 0:
            print(f"ERROR: grid not divisible by {n_ranks_y}x{n_ranks_x}.")
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
        semi_implicit_acoustic=args.semi_implicit_acoustic,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=args.n_acoustic_substeps,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)
    fast_physics_fn = build_fast_physics_fn(args, grid, hc, terrain)
    slow_physics_fn = build_slow_physics_fn(args, grid, hc, terrain)
    rad_call_every_steps = max(1, int(round(args.rad_call_interval_s / args.dt)))
    cached_rad_tend = [None]  # mutable closure for the cache

    def apply_physics_substep(state, dt_sub):
        """Forward-Euler physics tendency applied for dt_sub.
        Combines fast-physics tendency + cached radiation tendency."""
        tend = fast_physics_fn(state, grid, hc, terrain)
        if cached_rad_tend[0] is not None:
            tend = _sum_tendencies(tend, cached_rad_tend[0])
        new_u = state.u.data + dt_sub * tend.du_dt.data
        new_v = state.v.data + dt_sub * tend.dv_dt.data
        new_w = state.w.data + dt_sub * tend.dw_dt.data
        new_theta = (
            state.theta_prime.data + dt_sub * tend.dtheta_prime_dt.data
        )
        new_rho = state.rho_prime.data + dt_sub * tend.drho_prime_dt.data
        new_tr = state.tracers.data + dt_sub * tend.dtracers_dt.data
        return state._replace(
            u=state.u.replace(data=new_u),
            v=state.v.replace(data=new_v),
            w=state.w.replace(data=new_w),
            theta_prime=state.theta_prime.replace(data=new_theta),
            rho_prime=state.rho_prime.replace(data=new_rho),
            tracers=state.tracers.replace(data=new_tr),
        )

    def physics_split(state, dt_outer, n_sub):
        dt_sub = dt_outer / n_sub
        for _ in range(n_sub):
            state = apply_physics_substep(state, dt_sub)
            state = apply_positive_filter_state(
                state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
            )
        return state
    target_water = compute_total_water_mass_plane(state, hc, grid)

    owned_mask = jnp.zeros((args.ny, args.nx), dtype=jnp.float64)
    owned_mask = owned_mask.at[
        layout.iy_start:layout.iy_end,
        layout.ix_start:layout.ix_end,
    ].set(1.0)

    total_t = args.days * SEC_PER_DAY
    total_steps = int(total_t / args.dt)
    snap_dt = args.snapshot_hours * 3600.0
    prof_dt = args.profile_days * SEC_PER_DAY

    out_dir = Path(args.output)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "snapshots").mkdir(exist_ok=True)
        (out_dir / "profiles").mkdir(exist_ok=True)
        log_path = out_dir / "log.txt"
        log_f = open(log_path, "w", buffering=1)
        log_f.write(
            f"# RCE MPI LONG  n_ranks={n_ranks} grid={args.ny}x{args.nx} "
            f"nlev={args.nlev} dx={args.dx} dt={args.dt} "
            f"days={args.days} total_steps={total_steps}\n"
        )
        log_f.write(
            f"# physics: gray radiation + Kessler microphysics + "
            f"Smagorinsky LES (cs={args.smag_cs})\n"
        )
        log_f.write(
            f"# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,"
            f"max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        )

        # Save IC snapshot + profile.
        save_snapshot(out_dir, 0, 0.0, state, hc)
        save_profile(out_dir, 0, 0.0, state, hc)
        write_progress(out_dir, 0.0, total_t, 0, total_steps, 0.0)

    wall_start = time.time()
    next_snap_t = snap_dt
    next_prof_t = prof_dt

    for step in range(1, total_steps + 1):
        if rank == 0:
            # 1. Refresh radiation tendency every N steps (cached
            #    between calls — literature standard, e.g. SAM 600s).
            if (step - 1) % rad_call_every_steps == 0:
                cached_rad_tend[0] = slow_physics_fn(
                    state, grid, hc, terrain,
                )
            # 2. Dycore step (no physics — split out).
            state = model.step(state, dt=args.dt, physics_fn=None)
            # 3. Operator-split physics: N forward-Euler sub-steps
            #    at dt_outer/N (fast physics + cached radiation).
            state = physics_split(
                state, args.dt, args.n_physics_substeps,
            )
        state = _broadcast_state(state, comm, root=0)
        # Tracer positivity: clip negatives created by dycore advection
        # BEFORE the mass fixer (which raises on negative total water).
        state = apply_positive_filter_state(
            state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
        )
        state = remove_horizontal_mean_wind_plane_mpi(
            state, layout, owned_mask,
        )
        _ = compute_total_water_mass_plane_mpi(
            state, hc, grid, layout, owned_mask,
        )
        state = fix_moist_mass_plane(
            state, hc, grid, target_total_water=target_water,
        )
        t_sim = step * args.dt
        wall_elapsed = time.time() - wall_start

        if rank == 0:
            # Live progress every step.
            write_progress(
                out_dir, t_sim, total_t, step, total_steps, wall_elapsed,
            )
            # Per-step diagnostics (cheap, no extra MPI).
            if step % args.log_every_steps == 0 or step == 1:
                cwv = column_water_vapor_plane(state, hc)
                mse = column_moist_static_energy_plane(state, hc)
                precip = precipitation_rate_proxy_plane(state, hc)
                cn = compute_courant_numbers_plane(
                    state, hc, grid, args.dt,
                    n_acoustic_substeps=args.n_acoustic_substeps,
                )
                max_w = float(jnp.max(jnp.abs(state.w.data)))
                max_qc = float(jnp.max(state.tracers.data[..., 1]))
                max_qr = float(jnp.max(state.tracers.data[..., 2]))
                precip_mmday = float(jnp.max(precip)) * SEC_PER_DAY
                log_f.write(
                    f"{step},{t_sim / SEC_PER_DAY:.6f},"
                    f"{float(jnp.mean(cwv)):.4e},"
                    f"{float(jnp.max(cwv)):.4e},"
                    f"{float(jnp.mean(mse)):.4e},"
                    f"{max_w:.4e},{max_qc:.4e},{max_qr:.4e},"
                    f"{precip_mmday:.4e},"
                    f"{float(cn.acoustic):.4f}\n"
                )
                if not bool(jnp.all(jnp.isfinite(cwv))):
                    log_f.write(f"# BAIL: NaN at step {step}\n")
                    break

            # Snapshot every snapshot_hours.
            if t_sim >= next_snap_t:
                day_idx = int(round(t_sim / SEC_PER_DAY))
                save_snapshot(out_dir, day_idx, t_sim, state, hc)
                next_snap_t += snap_dt

            # Profile every profile_days.
            if t_sim >= next_prof_t:
                day_idx = int(round(t_sim / SEC_PER_DAY))
                save_profile(out_dir, day_idx, t_sim, state, hc)
                next_prof_t += prof_dt

    if rank == 0:
        log_f.close()
        write_progress(
            out_dir, t_sim, total_t, total_steps, total_steps,
            time.time() - wall_start,
        )
        print(
            f"Done. {step} steps, {t_sim / SEC_PER_DAY:.3f} days sim. "
            f"Wall: {(time.time() - wall_start) / 60:.1f} min."
        )


if __name__ == "__main__":
    main()
