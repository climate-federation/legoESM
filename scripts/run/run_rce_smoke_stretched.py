"""RCE smoke harness — stretched vertical grid + full RCE stack.

End-to-end demonstration of the RCE-capability stack accumulated
through PRs #296–#305 on the plane non-hydrostatic CRM dycore:

* Stretched HeightCoordinate (PR #296, ~50 m surface layer)
* Wing 2018 RCEMIP1 IC profiles (PR #298)
* Tracer positivity filter (PR #299)
* CFL diagnostic (PR #300)
* Radiative-equilibrium preconditioner (PR #301, optional)
* Mean-wind removal filter (PR #302)
* Moist-mass fixer (PR #303)
* RCE diagnostics: CWV, MSE, cloud fraction, precip proxy (PR #304)
* RCE surface flux composer (PR #305)

This is a SMOKE script — runs a small number of outer steps to
verify the full stack wires together and produces finite + physically
plausible diagnostics. Long RCE production runs (~100 days) go
through ``scripts/run/run_rcemip_long.py`` once item 13 (cloud-rad
coupling) lands.

CLI
---
.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run_rce_smoke_stretched.py \\
       --nx 16 --ny 16 --nlev 30 --dx 4000.0 --dt 6.0 \\
       --steps 20 --output results/rce_smoke_stretched.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

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
    cloud_fraction_profile_plane,
    column_moist_static_energy_plane,
    column_water_vapor_plane,
    precipitation_rate_proxy_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_surface_flux import (
    apply_rce_surface_fluxes,
    wind_speed_at_lowest_level_plane,
)
from legoesm.atmosphere.dynamics.shared.tracer_positivity import (
    apply_positive_filter_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_qv_ref_fn, make_wing2018_theta_ref_fn,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------
# Wing 2018 RCEMIP1 SST = 300 K reference parameters
# --------------------------------------------------------------------
T_SFC_K = 300.0          # SST
Q_SFC_FRAC = 0.0224      # ~saturation q at 300 K, 1015 hPa (Wing Tab A2)
GAMMA_TROP = 6.7e-3      # K/m tropospheric lapse rate
Z_T = 15_000.0           # tropopause height [m]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--dx", type=float, default=4_000.0)
    p.add_argument("--dt", type=float, default=0.5,
                   help="Outer time step [s]. Default 0.5 gives "
                        "per-substep acoustic CFL ~0.66 with 6 "
                        "acoustic substeps on the smoke-test grid "
                        "(--nx 4 --nlev 12 --dx 4000).")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--H", type=float, default=20_000.0)
    p.add_argument("--dz-sfc", type=float, default=50.0)
    p.add_argument("--c-h", type=float, default=1.5e-3)
    p.add_argument("--no-mass-fixer", action="store_true", default=False,
                   help="iter-95b: disable fix_moist_mass_plane "
                        "(rescales total water to IC every step). "
                        "Default False matches this script's conservation-"
                        "smoke purpose; set for any spin-up smoke "
                        "longer than ~10 outer steps where surface flux "
                        "should be allowed to NET ADD moisture.")
    p.add_argument("--n-acoustic-substeps", type=int, default=6,
                   help="Acoustic substeps per outer step. Must match "
                        "CompressibleEulerConfig.n_acoustic_substeps.")
    p.add_argument("--acoustic-cfl-max", type=float, default=1.0,
                   help="Per-substep acoustic CFL gate; raise on breach.")
    p.add_argument("--output", type=str, default="results/rce_smoke_stretched.txt")
    return p.parse_args()


def build_height_coord_and_state(nlev, H, dz_sfc, grid):
    """Build HeightCoordinate with Wing θ_ref so the reference state
    IS the Wing profile (Codex iter-2). theta_prime then starts at
    zero and the IC is in hydrostatic balance with the Wing profile
    — no spurious pressure/buoyancy imbalance at t=0.

    q_v is set on the tracers field (no analog of theta_ref_fn for
    moisture in HeightCoordinate; q_v lives entirely on the tracer
    pytree).
    """
    # Near-EQUILIBRIUM RCE smoke: theta reference uses T_v0=T_SFC_K to match the
    # SST=300 K setup (NOT the strict-RCEMIP fixed 295 K, which floods the column).
    # The surface-temp param was renamed T_sfc -> T_v0 (old kwarg TypeErrors).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=T_SFC_K, q_sfc=Q_SFC_FRAC, z_t=Z_T, Gamma=GAMMA_TROP,
    )
    qv_fn = make_wing2018_qv_ref_fn(q_sfc=Q_SFC_FRAC, z_t=Z_T)
    # iter-95: pass p_sfc=101480 (Wing 2018 Tab A1) for the
    # correct hydrostatic surface BC. Without this the legacy
    # top-down T_avg=250 K BC produces ~12 K too-hot T at the
    # lowest model level on H=33 km columns.
    hc = create_stretched_height_coordinate(
        n_levels=nlev, H=H, dz_sfc=dz_sfc,
        theta_ref_fn=theta_fn,
        p_sfc=101480.0,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # theta_prime starts at zero (rest state convention) since
    # theta_ref now IS the Wing profile.
    z = hc.z_full
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
    """One physics step (Codex iter-2 ordering):

    1. Surface fluxes — uses UNFILTERED wind so the bulk-aero
       coefficient sees the actual wind relative to the surface
       (filtering first would collapse fluxes to the gustiness floor
       and bias the RCE forcing).
    2. Mean-wind removal — Galilean filter applied AFTER fluxes so
       the no-mean-wind RCE convention holds going into the next
       dycore step.
    3. Tracer positivity — clip q_v, q_c, q_r negatives last so the
       advection + surface flux + filter compositions can't leave
       behind negative tracer mass.
    """
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    hc, state = build_height_coord_and_state(
        args.nlev, args.H, args.dz_sfc, grid,
    )
    terrain = make_flat_plane_terrain_metric(grid, hc)

    cfg = CompressibleEulerConfig(
        fix_mass=True, anchor_mass_to_initial=True,
        n_acoustic_substeps=args.n_acoustic_substeps,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)

    # Snapshot initial moist mass for the fixer.
    target_water_mass = compute_total_water_mass_plane(state, hc, grid)

    lines: list[str] = []
    lines.append(
        f"# RCE smoke (stretched grid)  nx={args.nx} ny={args.ny} "
        f"nlev={args.nlev} dx={args.dx} dz_sfc={args.dz_sfc} H={args.H}"
    )
    lines.append(
        f"# step  t[s]  CWV_mean[kg/m2]  MSE_mean[J/m2]  "
        f"cloud_max  precip_max[kg/m2/s]  C_h_max  C_v_max  C_a_max"
    )
    for step in range(args.steps):
        state = model.step(state, dt=args.dt)
        state = physics_step(state, hc, grid, args.dt, args.c_h)
        if not args.no_mass_fixer:
            state = fix_moist_mass_plane(
                state, hc, grid, target_total_water=target_water_mass,
            )
        # Diagnostics every step.
        cwv = column_water_vapor_plane(state, hc)
        mse = column_moist_static_energy_plane(state, hc)
        cf = cloud_fraction_profile_plane(state, hc)
        precip = precipitation_rate_proxy_plane(state, hc)
        cn = compute_courant_numbers_plane(
            state, hc, grid, args.dt,
            n_acoustic_substeps=args.n_acoustic_substeps,
        )
        per_substep_acoustic = float(cn.acoustic)
        if per_substep_acoustic > args.acoustic_cfl_max:
            raise RuntimeError(
                f"Per-substep acoustic CFL "
                f"{per_substep_acoustic:.3f} > "
                f"--acoustic-cfl-max={args.acoustic_cfl_max} at step "
                f"{step}; reduce --dt, raise --n-acoustic-substeps, "
                f"or coarsen --dx."
            )
        lines.append(
            f"{step:5d}  {(step+1)*args.dt:8.2f}  "
            f"{float(jnp.mean(cwv)):14.4e}  "
            f"{float(jnp.mean(mse)):14.4e}  "
            f"{float(jnp.max(cf)):10.4f}  "
            f"{float(jnp.max(precip)):14.4e}  "
            f"{float(cn.horizontal_advective):8.4f}  "
            f"{float(cn.vertical_advective):8.4f}  "
            f"{float(cn.acoustic):8.4f}"
        )
        # Bail on NaN / unstable.
        if not bool(jnp.all(jnp.isfinite(cwv))):
            raise RuntimeError(
                f"Non-finite CWV at step {step}; integration unstable."
            )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {out_path}")
    print("\n".join(lines[-min(5, args.steps):]))


if __name__ == "__main__":
    main()
