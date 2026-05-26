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
    column_water_vapor_plane, moist_static_energy_3d_plane,
    precipitation_rate_proxy_plane, temperature_3d_plane,
)
from legoesm.atmosphere.dynamics.rce_mpi import (
    compute_dry_mass_plane_mpi,
    compute_total_water_mass_plane_mpi,
    fix_mass_nonhydrostatic_plane_mpi,
    fix_moist_mass_plane_mpi,
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
from legoesm.parallel.plane_mpi import (
    gather_plane_field,
    make_plane_pencil_grid,
    make_plane_pencil_layout,
    scatter_plane_field,
)

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
    p.add_argument("--dt", type=float, default=5.0,
                   help="Outer dt [s]. Bare-dycore stability bound is "
                        "BUBBLE-dependent: with the legacy 0.5 K warm "
                        "bubble at z<1 km (F1) dt was limited to ≤1 s; "
                        "with the F8/F10 clean Wing IC (no bubble + no "
                        "qv noise) dt=10 s is bit-stable bare-dycore + "
                        "dt=5 s is stable through 864 steps of full "
                        "physics (smoke iter-9, 2026-05). Production "
                        "default set to 5 s — 5× speedup of every "
                        "30-day run vs the iter-2 conservative 1 s "
                        "default.")
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
    p.add_argument("--n-acoustic-substeps", type=int, default=12,
                   help="Number of acoustic-mode substeps per outer "
                        "dycore step. Production contract (iter-14 + "
                        "iter-38): N=12 at dt=5 s on the clean Wing IC "
                        "is 132x132 1-sim-hour PASS (max|w|=6.1e-3 m/s, "
                        "MSE drift=1.7e-4 over 725 steps). The historical "
                        "default 24 was set for the iter-1 dt=1.0 s "
                        "config and produced an over-stable but 2x-cost "
                        "substep at the current dt=5.0 s default. iter-59 "
                        "refreshed to match production.")
    p.add_argument("--n-physics-substeps", type=int, default=10,
                   help="Operator-split physics N x per dycore step. "
                        "Stiff sources (microphysics, surface flux) "
                        "sub-stepped at dt_outer/N.")
    p.add_argument("--rad-call-interval-s", type=float, default=600.0,
                   help="Radiation call interval [s]. Literature: SAM "
                        "600s, CM1 60-120s, WRF-LES 300-600s. "
                        "Tendency held constant between calls.")
    p.add_argument("--no-radiation", action="store_true", default=False,
                   help="Fully disable the gray radiation tendency "
                        "(skips slow_physics_fn entirely; cached_rad_tend "
                        "stays None). Use for dycore-isolation smokes. "
                        "Codex iter-39 review: --rad-call-interval-s=1e9 "
                        "does NOT disable radiation — it still fires once "
                        "at step 1 and caches the tendency for the full "
                        "run. This flag is the only way to truly run "
                        "without radiation.")
    p.add_argument("--smag-cs", type=float, default=0.2)
    p.add_argument("--semi-implicit-acoustic", action="store_true",
                   help="Use per-column Thomas tridiagonal solve for "
                        "the acoustic substep (lifts vertical-CFL "
                        "limit; main benefit on stretched grids).")
    p.add_argument("--hyperdiff", type=float, default=5.0e6,
                   help="Biharmonic horizontal diffusion on u, v, theta', "
                        "rho', w [m^4/s]. Production default 5e6 — the "
                        "previous 1e6 default could not damp the 2-Δz "
                        "vertical mode seeded by the warm-bubble IC at "
                        "dt=1 s, max|w| growing to >20 m/s by step 250 "
                        "(see CRM_implementation.md F6).")
    p.add_argument("--bubble-theta-pert", type=float, default=0.0,
                   help="Warm-bubble convection seed amplitude [K]. "
                        "Default 0 = no bubble (standard RCEMIP1 "
                        "protocol: spin up convection from surface "
                        "fluxes + qv noise alone). Set 0.5 to "
                        "reproduce the legacy single-bubble IC. The "
                        "0.5 K bubble seeds a 2-Δz vertical mode at "
                        "dz~1.1 km that no amount of hyperdiff fully "
                        "damps — only suitable for short rising-"
                        "thermal smokes, not 30-day runs.")
    p.add_argument("--qv-noise-amp", type=float, default=0.0,
                   help="Amplitude of zero-mean random qv perturbations "
                        "[kg/kg] applied to the lowest 4 model levels "
                        "of the IC. Default 0 = clean Wing 2018 IC; "
                        "convection spins up from radiative cooling + "
                        "surface flux (~hours). Set 1e-5 - 5e-5 (0.01 - "
                        "0.05 g/kg) for a gentle stochastic seed. "
                        "Amplitudes ≥ 2.5e-4 trigger immediate Kessler "
                        "condensation hotspots that destabilise the "
                        "dycore in <5 min sim time (smoke verified — "
                        "see CRM_implementation.md F7).")
    p.add_argument("--qv-noise-seed", type=int, default=0,
                   help="RNG seed for qv noise perturbation. Same seed "
                        "→ bit-identical IC across reruns.")
    p.add_argument("--use-dd", action="store_true", default=False,
                   help="Switch from the legacy rank-0-dycore + "
                        "broadcast pattern to true per-rank domain "
                        "decomposition via step_halo + owned_mask. "
                        "Activates the R7 MPI mass fixer + MPI mean "
                        "wind. Required for any real MPI scaling claim. "
                        "Defaults False to preserve the F8-stable "
                        "config from iter-2; flip to True once you have "
                        "verified the DD path on the smoke run.")
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--sponge-width", type=float, default=10_000.0,
                   help="Sponge layer width from model top [m]. "
                        "Default 10 km matches CompressibleEulerConfig. "
                        "5 km is too thin for H=33 km when gravity-wave "
                        "wavelengths exceed sponge depth.")
    p.add_argument("--acoustic-off-centering", type=float, default=0.0,
                   help="Skamarock-Klemp off-centering parameter beta "
                        "in [0, 1). 0 = neutral forward-backward. "
                        "0.05-0.1 damps acoustic modes. Try 0.1 if "
                        "instability appears as growing rho_prime / w "
                        "oscillations.")
    p.add_argument("--vertical-theta-diffusion", type=float, default=0.0,
                   help="Explicit vertical Laplacian diffusivity on "
                        "theta_prime [m^2/s]. 0 = off. Try 1e4-5e4 to "
                        "damp the buoyancy/PG feedback that destabilises "
                        "the dycore at dt > 0.5 s on coarse vertical grids.")
    p.add_argument("--advection", choices=["upwind1", "weno5"],
                   default="upwind1",
                   help="Horizontal advection scheme for theta/u/v/w.")
    p.add_argument("--implicit-buoyancy", action="store_true", default=False,
                   help="Klemp-Wilhelmson 1978 implicit-buoyancy treatment "
                        "of the w-equation in the SI acoustic substep. Adds "
                        "three nearest-neighbour bands proportional to "
                        "dtheta_ref/dz to the tridiagonal solve. Closes the "
                        "w<->theta gravity-wave feedback that destabilises "
                        "the dycore at coarse dz (~1 km) with stratified "
                        "ICs. Only active when --semi-implicit-acoustic.")
    p.add_argument("--snapshot-hours", type=float, default=24.0)
    p.add_argument("--snapshot-3d-hours", type=float, default=0.0,
                   help="If > 0, dump full 3D MSE/qv/T volumes "
                        "(ny,nx,nlev) as compressed float32 NPZ "
                        "every N simulation hours. 0 disables.")
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

    # Convection seed.
    # --------------------------------------------------------------
    # Default (--bubble-theta-pert 0): NO bubble. Convection spins
    # up from --qv-noise-amp boundary-layer qv perturbations + the
    # surface flux. This matches the RCEMIP1 protocol and avoids
    # the 2-Δz vertical mode that the single-bubble IC seeds at
    # the coarse dz~1.1 km production grid (see CRM_implementation.md
    # F6: hyperdiff up to 5e6 only delays — does not stop — the mode
    # that grows from the bubble's single-level perturbation).
    #
    # Optional (--bubble-theta-pert > 0): legacy warm-bubble IC,
    # cosine-tapered horizontally over ~10·dx and vertically over
    # z<1 km. Density set so the initial pressure is unperturbed
    # (ρ'/ρ_ref = −θ'/θ_ref). Use ONLY for short rising-thermal
    # smokes.
    bubble_amp = float(args.bubble_theta_pert)
    if bubble_amp > 0.0:
        ny, nx = grid.ny, grid.nx
        jj = jnp.arange(ny)
        ii = jnp.arange(nx)
        yy, xx = jnp.meshgrid(jj, ii, indexing="ij")
        yc, xc = (ny - 1) / 2.0, (nx - 1) / 2.0
        r_cells = jnp.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        r0 = 10.0  # bubble radius in cell units
        horiz = jnp.where(
            r_cells < r0,
            0.5 * (1.0 + jnp.cos(jnp.pi * r_cells / r0)),
            0.0,
        )
        z_top_bubble = 1000.0
        vert = jnp.where(z < z_top_bubble,
                         0.5 * (1.0 + jnp.cos(jnp.pi * z / z_top_bubble)),
                         0.0)
        bubble_theta = bubble_amp * horiz[:, :, None] * vert[None, None, :]
        bubble_rho = -hc.rho_ref * bubble_theta / hc.theta_ref
        state = state._replace(
            theta_prime=state.theta_prime.replace(data=bubble_theta),
            rho_prime=state.rho_prime.replace(data=bubble_rho),
        )

    # RCEMIP1 qv noise seed in the lowest 4 levels.
    qv_noise_amp = float(args.qv_noise_amp)
    if qv_noise_amp > 0.0:
        key = jax.random.PRNGKey(int(args.qv_noise_seed))
        n_seed_lev = min(4, nlev)
        noise = jax.random.uniform(
            key, shape=(grid.ny, grid.nx, n_seed_lev),
            minval=-qv_noise_amp, maxval=qv_noise_amp, dtype=jnp.float64,
        )
        noise = noise - jnp.mean(noise, axis=(0, 1), keepdims=True)
        # Lowest n_seed_lev model levels = LAST indices (k goes top→bottom).
        new_qv = new_tracers[..., 0]
        new_qv = new_qv.at[..., -n_seed_lev:].add(noise)
        new_qv = jnp.maximum(new_qv, 0.0)
        new_tracers = new_tracers.at[..., 0].set(new_qv)

    state = state._replace(
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


_DD_FIELD_NAMES = (
    "u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers",
)


def _scatter_state(state, layout):
    """Build rank-local slab from a globally-replicated state.

    Each rank independently calls :func:`scatter_plane_field` (pure
    slicing — no MPI). Assumes every rank has built the same global
    IC deterministically before this call. Avoids the bcast cost of
    a "rank-0 owns; everyone else receives" pattern at IC time."""
    new_fields = {}
    for name in _DD_FIELD_NAMES:
        fld = getattr(state, name)
        local_data = scatter_plane_field(fld.data, layout)
        new_fields[name] = fld.replace(data=local_data)
    return state._replace(**new_fields)


def _gather_state(state_local, layout):
    """Gather rank-local slabs to a single global state on rank 0.

    Per-field :func:`gather_plane_field` (uses ``comm.gather`` under
    the hood — a COLLECTIVE that every rank must enter). Returns the
    gathered global state on rank 0 and ``None`` on all other ranks.

    Bug avoided: previously this returned early on the first
    ``gather_plane_field == None`` (i.e. on non-rank-0 after the
    first field), which left rank 0 blocked on the next gather
    waiting for a peer that had already exited. We now ALWAYS
    participate in every collective regardless of rank.
    """
    rank0_arrays = []
    for name in _DD_FIELD_NAMES:
        fld = getattr(state_local, name)
        global_data = gather_plane_field(fld.data, layout)
        rank0_arrays.append(global_data)
    # Non-rank-0 has None in every slot — return None to signal the
    # caller this isn't the rank that owns the gathered state.
    if rank0_arrays[0] is None:
        return None
    rank0_fields = {}
    for name, arr in zip(_DD_FIELD_NAMES, rank0_arrays):
        fld = getattr(state_local, name)
        rank0_fields[name] = fld.replace(data=arr)
    return state_local._replace(**rank0_fields)


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


def save_snapshot_3d(out_dir, hr_idx, t_sim, state, hc):
    """Full 3D MSE/q_v/T volumes [float32, compressed].

    Stored at ``snapshots_3d/snap_hr_NNNN.npz`` with arrays:
    ``mse``, ``qv``, ``T`` each shape (ny, nx, nlev); plus ``z``
    (nlev,) and scalars ``t_sim``, ``day``, ``hour``.
    """
    mse_3d = np.asarray(
        moist_static_energy_3d_plane(state, hc), dtype=np.float32,
    )
    qv_3d = np.asarray(state.tracers.data[..., 0], dtype=np.float32)
    T_3d = np.asarray(temperature_3d_plane(state, hc), dtype=np.float32)
    z = np.asarray(hc.z_full, dtype=np.float32)
    snap_path = out_dir / "snapshots_3d" / f"snap_hr_{hr_idx:04d}.npz"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snap_path,
        t_sim=t_sim, day=t_sim / SEC_PER_DAY, hour=t_sim / 3600.0,
        z=z, mse=mse_3d, qv=qv_3d, T=T_3d,
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
    if args.implicit_buoyancy and not args.semi_implicit_acoustic:
        raise SystemExit(
            "--implicit-buoyancy requires --semi-implicit-acoustic "
            "(the Klemp-Wilhelmson 1978 substitution lives inside "
            "the column tridiagonal solve)."
        )
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
        acoustic_off_centering=args.acoustic_off_centering,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=args.n_acoustic_substeps,
        vertical_theta_diffusion=args.vertical_theta_diffusion,
        horizontal_advection_scheme=args.advection,
        implicit_buoyancy=args.implicit_buoyancy,
    )
    # Save the GLOBAL state for snapshot dumping (needed on every rank
    # in legacy mode; saved on rank 0 in DD mode after gather).
    state_global_ic = state

    # ----------------------------------------------------------------- #
    # Domain-decomposition: scatter IC + rebuild local grid / model /   #
    # physics so every rank steps its own slab via step_halo.           #
    # ----------------------------------------------------------------- #
    if args.use_dd and n_ranks > 1:
        state = _scatter_state(state, layout)
        grid_local = make_plane_pencil_grid(
            layout, dx=args.dx, dy=args.dx, nlev=args.nlev,
            dtype=jnp.float64,
        )
        terrain_local = make_flat_plane_terrain_metric(grid_local, hc)
        model = PlaneCompressibleEulerModel(
            grid_local, hc, terrain_local, config=cfg,
        )
        fast_physics_fn = build_fast_physics_fn(
            args, grid_local, hc, terrain_local,
        )
        slow_physics_fn = build_slow_physics_fn(
            args, grid_local, hc, terrain_local,
        )
        grid = grid_local
        terrain = terrain_local
    else:
        model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)
        fast_physics_fn = build_fast_physics_fn(args, grid, hc, terrain)
        slow_physics_fn = build_slow_physics_fn(args, grid, hc, terrain)
    # iter-55 Codex MEDIUM#1 fix: skip the schedule helper entirely
    # when --no-radiation is set. Otherwise a bogus --rad-call-
    # interval-s (NaN/inf, rejected by physics_schedule's iter-43
    # validation) crashes the driver even though the value is
    # unused. With --no-radiation the schedule is irrelevant —
    # cached_rad_tend stays None for the full run.
    if args.no_radiation:
        rad_call_every_steps = 1  # harmless sentinel; never consulted
    else:
        from legoesm.driver.physics_schedule import (
            radiation_call_every_steps as _rad_every,
        )
        rad_call_every_steps = _rad_every(
            args.rad_call_interval_s, args.dt,
        )
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
    # owned_mask + target water mass.
    if args.use_dd and n_ranks > 1:
        # Per-rank local mask is all-ones (every cell of the local slab
        # is owned by this rank; halo is added by step_halo's exchange).
        owned_mask = jnp.ones(
            (layout.ny_local, layout.nx_local), dtype=jnp.float64,
        )
        target_water = compute_total_water_mass_plane_mpi(
            state, hc, grid, layout, owned_mask,
        )
    else:
        # Legacy rank-0-broadcast path: owned_mask is in GLOBAL coords
        # so the MPI helpers can mark only this rank's slice of the
        # replicated full state.
        owned_mask = jnp.zeros((args.ny, args.nx), dtype=jnp.float64)
        owned_mask = owned_mask.at[
            layout.iy_start:layout.iy_end,
            layout.ix_start:layout.ix_end,
        ].set(1.0)
        target_water = compute_total_water_mass_plane(state, hc, grid)

    total_t = args.days * SEC_PER_DAY
    total_steps = int(total_t / args.dt)
    snap_dt = args.snapshot_hours * 3600.0
    snap3d_dt = args.snapshot_3d_hours * 3600.0
    snap3d_enabled = args.snapshot_3d_hours > 0.0
    prof_dt = args.profile_days * SEC_PER_DAY

    out_dir = Path(args.output)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "snapshots").mkdir(exist_ok=True)
        (out_dir / "profiles").mkdir(exist_ok=True)
        if snap3d_enabled:
            (out_dir / "snapshots_3d").mkdir(exist_ok=True)
        log_path = out_dir / "log.txt"
        log_f = open(log_path, "w", buffering=1)
        log_f.write(
            f"# RCE MPI LONG  n_ranks={n_ranks} grid={args.ny}x{args.nx} "
            f"nlev={args.nlev} dx={args.dx} dt={args.dt} "
            f"days={args.days} total_steps={total_steps}\n"
        )
        log_f.write(
            f"# physics: "
            f"{'NO radiation' if args.no_radiation else 'gray radiation'}"
            f" + Kessler microphysics + "
            f"Smagorinsky LES (cs={args.smag_cs})\n"
        )
        log_f.write(
            f"# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,"
            f"max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        )

        # Save IC snapshot + profile. In DD mode the `state` variable
        # holds the rank-local slab; dump from the pre-scatter global
        # IC so snapshots stay full-domain shape.
        ic_state_for_io = state_global_ic if args.use_dd else state
        save_snapshot(out_dir, 0, 0.0, ic_state_for_io, hc)
        save_profile(out_dir, 0, 0.0, ic_state_for_io, hc)
        if snap3d_enabled:
            save_snapshot_3d(out_dir, 0, 0.0, ic_state_for_io, hc)
        write_progress(out_dir, 0.0, total_t, 0, total_steps, 0.0)

    wall_start = time.time()
    next_snap_t = snap_dt
    next_snap3d_t = snap3d_dt
    next_prof_t = prof_dt
    # iter-40: explicit counter for radiation tendency refreshes. Used
    # by test_plane_crm_production_scale_132x132_with_radiation /
    # ..._envelope (iter-38/39) to assert the radiation tick branch
    # actually fires the expected number of times. Codex iter-39
    # MEDIUM#1 / MEDIUM#3 gap: without an observable count the test
    # could pass under a broken rad_call_every_steps arithmetic.
    rad_call_count = 0
    # iter-55 Codex MEDIUM#2 fix: init step + t_sim BEFORE the loop
    # so that ``--days 0`` (used by some smoke-tests as a CLI-parse-
    # only dry-run) doesn't crash the final ``Done.`` print with a
    # NameError. With total_steps=0 the loop body is skipped and the
    # report shows ``Done. 0 steps...``.
    step = 0
    t_sim = 0.0

    def _maybe_fire_radiation(step_idx):
        """Fire gray-radiation slow tendency on schedule. Returns
        True if it fired this step (caller increments the counter)."""
        if args.no_radiation:
            return False
        if (step_idx - 1) % rad_call_every_steps != 0:
            return False
        cached_rad_tend[0] = slow_physics_fn(state, grid, hc, terrain)
        return True

    for step in range(1, total_steps + 1):
        if args.use_dd and n_ranks > 1:
            # ----------------------------------------------------- #
            # True per-rank DD path.                                #
            # ----------------------------------------------------- #
            if _maybe_fire_radiation(step):
                rad_call_count += 1
            state = model.step_halo(
                state, dt=args.dt, layout=layout,
                owned_mask=owned_mask,
            )
            state = physics_split(state, args.dt, args.n_physics_substeps)
            state = apply_positive_filter_state(
                state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
            )
            state = remove_horizontal_mean_wind_plane_mpi(
                state, layout, owned_mask,
            )
            state = fix_moist_mass_plane_mpi(
                state, hc, grid, layout, owned_mask,
                target_total_water=target_water,
            )
        else:
            # ----------------------------------------------------- #
            # Legacy rank-0-dycore + broadcast path (replicated).   #
            # ----------------------------------------------------- #
            if rank == 0:
                if _maybe_fire_radiation(step):
                    rad_call_count += 1
                state = model.step(state, dt=args.dt, physics_fn=None)
                state = physics_split(
                    state, args.dt, args.n_physics_substeps,
                )
            state = _broadcast_state(state, comm, root=0)
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

        # In DD mode, diagnostics + snapshots need a globally-assembled
        # state on rank 0. Gather lazily — only on log/snapshot/profile
        # ticks, never every step.
        need_gather = args.use_dd and n_ranks > 1 and (
            step % args.log_every_steps == 0 or step == 1
            or t_sim >= next_snap_t
            or (snap3d_enabled and t_sim >= next_snap3d_t)
            or t_sim >= next_prof_t
        )
        state_for_io = (
            _gather_state(state, layout) if need_gather else state
        )

        if rank == 0:
            # Live progress every step.
            write_progress(
                out_dir, t_sim, total_t, step, total_steps, wall_elapsed,
            )
            # Per-step diagnostics. In DD mode `state_for_io` is the
            # gathered global state on rank 0; in legacy mode it is the
            # same as `state` (rank-0 already holds the full domain).
            if step % args.log_every_steps == 0 or step == 1:
                state_io = state_for_io
                # Build a global grid for CFL diagnostics in DD mode
                # (cn.acoustic depends only on rho_ref + dx + dt).
                grid_io = grid
                if args.use_dd and n_ranks > 1:
                    grid_io = create_plane_grid(
                        nx=args.nx, ny=args.ny, nlev=args.nlev,
                        dx=args.dx, dy=args.dx, dtype=jnp.float64,
                    )
                cwv = column_water_vapor_plane(state_io, hc)
                mse = column_moist_static_energy_plane(state_io, hc)
                precip = precipitation_rate_proxy_plane(state_io, hc)
                cn = compute_courant_numbers_plane(
                    state_io, hc, grid_io, args.dt,
                    n_acoustic_substeps=args.n_acoustic_substeps,
                )
                max_w = float(jnp.max(jnp.abs(state_io.w.data)))
                max_qc = float(jnp.max(state_io.tracers.data[..., 1]))
                max_qr = float(jnp.max(state_io.tracers.data[..., 2]))
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

            # Snapshot every snapshot_hours. state_for_io is the
            # gathered global state on rank 0 in DD mode, or `state`
            # in legacy mode (rank 0 already holds the full domain).
            if t_sim >= next_snap_t:
                day_idx = int(round(t_sim / SEC_PER_DAY))
                save_snapshot(out_dir, day_idx, t_sim, state_for_io, hc)

            # 3D snapshot every snapshot_3d_hours.
            if snap3d_enabled and t_sim >= next_snap3d_t:
                hr_idx = int(round(t_sim / 3600.0))
                save_snapshot_3d(
                    out_dir, hr_idx, t_sim, state_for_io, hc,
                )

            # Profile every profile_days.
            if t_sim >= next_prof_t:
                day_idx = int(round(t_sim / SEC_PER_DAY))
                save_profile(out_dir, day_idx, t_sim, state_for_io, hc)

        # Keep gather/snapshot/profile timers identical on every rank.
        # These thresholds feed need_gather, which gates collective gathers.
        if t_sim >= next_snap_t:
            next_snap_t += snap_dt
        if snap3d_enabled and t_sim >= next_snap3d_t:
            next_snap3d_t += snap3d_dt
        if t_sim >= next_prof_t:
            next_prof_t += prof_dt

    if rank == 0:
        log_f.close()
        write_progress(
            out_dir, t_sim, total_t, total_steps, total_steps,
            time.time() - wall_start,
        )
        print(
            f"Done. {step} steps, {t_sim / SEC_PER_DAY:.3f} days sim. "
            f"Wall: {(time.time() - wall_start) / 60:.1f} min. "
            f"rad_calls={rad_call_count}."
        )


if __name__ == "__main__":
    main()
