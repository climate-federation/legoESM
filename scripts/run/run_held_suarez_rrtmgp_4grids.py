#!/usr/bin/env python
"""Run Held-Suarez with RRTMGP radiation on all 4 grid types and compare.

Grid types: cubed-sphere, lat-lon FV, icosahedral (MPAS), spectral PE.
Each runs for a configurable number of days (default 100) and produces
time-series diagnostics (mass, mean_T, max_wind) for cross-grid comparison.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_held_suarez_rrtmgp_4grids.py [--days 100]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from legoesm import constants

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def check_finite(arrays: dict[str, jnp.ndarray]) -> bool:
    for name, arr in arrays.items():
        if not bool(jnp.all(jnp.isfinite(arr))):
            print(f"  *** NaN/Inf detected in {name}")
            return False
    return True


def _create_vertical(nlev: int, vertical_coord: str):
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    if vertical_coord == "hybrid":
        return standard_hybrid_levels(nlev)
    return create_sigma_coordinate(nlev)


def _make_rrtmgp_physics(model_type: str, dt: float, hs_fn=None):
    """Create RRTMGP physics, optionally combined with Held-Suarez forcing.

    When *hs_fn* is provided the returned function sums the HS Newtonian
    relaxation / Rayleigh drag tendencies with RRTMGP radiative tendencies.
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="rrtmgp"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    rrtmgp_fn = make_physics(phys_cfg, model_type=model_type, dt=dt)

    if hs_fn is None:
        return rrtmgp_fn

    if model_type == "spectral_pe":
        def combined_fn(state, grid, sigma_coord, phys_state=None):
            rrtmgp_result = rrtmgp_fn(state, grid, sigma_coord, phys_state=phys_state)
            rrtmgp_tend = rrtmgp_result[0] if isinstance(rrtmgp_result, tuple) else rrtmgp_result
            phys_state_out = rrtmgp_result[1] if isinstance(rrtmgp_result, tuple) else None
            hs_tend = hs_fn(state, grid, sigma_coord)
            from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
            summed = SpectralHydrostaticState(
                vor_hat=rrtmgp_tend.vor_hat.replace(
                    data=rrtmgp_tend.vor_hat.data + hs_tend.vor_hat.data),
                div_hat=rrtmgp_tend.div_hat.replace(
                    data=rrtmgp_tend.div_hat.data + hs_tend.div_hat.data),
                T_hat=rrtmgp_tend.T_hat.replace(
                    data=rrtmgp_tend.T_hat.data + hs_tend.T_hat.data),
                lnps_hat=rrtmgp_tend.lnps_hat.replace(
                    data=rrtmgp_tend.lnps_hat.data + hs_tend.lnps_hat.data),
                phis_hat=rrtmgp_tend.phis_hat.replace(
                    data=rrtmgp_tend.phis_hat.data + hs_tend.phis_hat.data),
            )
            return summed, phys_state_out
    else:
        def combined_fn(state, grid, sigma_coord, phys_state=None):
            rrtmgp_result = rrtmgp_fn(state, grid, sigma_coord, phys_state=phys_state)
            rrtmgp_tend = rrtmgp_result[0] if isinstance(rrtmgp_result, tuple) else rrtmgp_result
            phys_state_out = rrtmgp_result[1] if isinstance(rrtmgp_result, tuple) else None
            hs_tend = hs_fn(state, grid, sigma_coord)
            from legoesm.core.state import HydrostaticTendencies
            dv_dt = None
            if rrtmgp_tend.dv_dt is not None and hs_tend.dv_dt is not None:
                dv_dt = rrtmgp_tend.dv_dt.replace(
                    data=rrtmgp_tend.dv_dt.data + hs_tend.dv_dt.data)
            elif rrtmgp_tend.dv_dt is not None:
                dv_dt = rrtmgp_tend.dv_dt
            elif hs_tend.dv_dt is not None:
                dv_dt = hs_tend.dv_dt
            summed = HydrostaticTendencies(
                du_dt=rrtmgp_tend.du_dt.replace(
                    data=rrtmgp_tend.du_dt.data + hs_tend.du_dt.data),
                dT_dt=rrtmgp_tend.dT_dt.replace(
                    data=rrtmgp_tend.dT_dt.data + hs_tend.dT_dt.data),
                dp_s_dt=rrtmgp_tend.dp_s_dt.replace(
                    data=rrtmgp_tend.dp_s_dt.data + hs_tend.dp_s_dt.data),
                dphis_dt=rrtmgp_tend.dphis_dt.replace(
                    data=rrtmgp_tend.dphis_dt.data + hs_tend.dphis_dt.data),
                dv_dt=dv_dt,
                tracer_tendencies=rrtmgp_tend.tracer_tendencies,
            )
            return summed, phys_state_out

    if hasattr(rrtmgp_fn, 'set_time'):
        combined_fn.set_time = rrtmgp_fn.set_time
    if hasattr(rrtmgp_fn, 'reset_state'):
        combined_fn.reset_state = rrtmgp_fn.reset_state

    return combined_fn


# Diffusion coefficient helpers (match test matrix)
def _hyperdiff_cube(n):
    dx = constants.R_earth * np.pi / (2 * n)
    tau = 3600.0
    return dx ** 4 / tau

def _div_damp_cube(n):
    dx = constants.R_earth * np.pi / (2 * n)
    return 0.15 * dx ** 2

def _laplacian_visc_cube(n):
    dx = constants.R_earth * np.pi / (2 * n)
    return 0.01 * dx ** 2

def _laplacian_visc_latlon(n_lat):
    dx = constants.R_earth * np.pi / n_lat
    return 0.01 * dx ** 2

def _hyperdiff_ico(mesh):
    dx = np.sqrt(np.mean(np.asarray(mesh.areaCell)))
    tau = 3600.0
    return dx ** 4 / tau

def _laplacian_visc_ico(mesh):
    dx = np.sqrt(np.mean(np.asarray(mesh.areaCell)))
    return 0.01 * dx ** 2


# ---------------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------------

def run_timeloop(step_fn, state, dt, n_steps, check_fn, scalar_fn,
                 diag_every, label=""):
    """Run time loop, collecting diagnostics every diag_every steps."""
    diag = {}
    t0 = time.time()
    ok = True

    print(f"\n{'='*60}")
    print(f"  {label}: {n_steps} steps, dt={dt}s, "
          f"~{n_steps*dt/86400:.0f} days")
    print(f"{'='*60}")

    # JIT warmup
    print("  JIT compiling...", end="", flush=True)
    state = step_fn(state, dt)
    jax.block_until_ready(state)
    t_jit = time.time() - t0
    print(f" done ({t_jit:.1f}s)")

    t_run = time.time()
    for i in range(1, n_steps):
        state = step_fn(state, dt)

        if i % diag_every == 0:
            jax.block_until_ready(state)
            finite, max_val = check_fn(state)
            if not finite or max_val > 1000.0:
                print(f"  *** BLOWUP at step {i+1} "
                      f"(day {(i+1)*dt/86400:.1f}), max={max_val:.1f}")
                ok = False
                break

            scalars = scalar_fn(state)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            day = (i + 1) * dt / 86400
            print(f"  day {day:7.1f}: "
                  f"mean_T={scalars.get('mean_T', 0):.2f}K  "
                  f"max_wind={scalars.get('max_wind', 0):.1f}m/s  "
                  f"mass={scalars.get('mass', 0):.6e}")

    jax.block_until_ready(state)
    wall = time.time() - t_run
    print(f"  Wall time: {wall:.1f}s")
    return state, diag, wall, ok


# ---------------------------------------------------------------------------
# Grid-specific runners
# ---------------------------------------------------------------------------

def run_cubed_sphere(days, nlev, vertical_coord):
    """Run Held-Suarez + RRTMGP on cubed-sphere C36."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig)
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init, held_suarez_forcing)
    from legoesm.core.operators import global_integral

    n = 36
    grid = create_cubed_sphere(n)
    sigma = _create_vertical(nlev, vertical_coord)
    dt = 200.0

    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=_hyperdiff_cube(n),
        hyperdiff_ps_coeff=_hyperdiff_cube(n),
        div_damp_coeff=_div_damp_cube(n),
        A_h=_laplacian_visc_cube(n),
        use_conservation_fixer=True, fix_mass=True)
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)

    physics_fn = _make_rrtmgp_physics("hydrostatic", dt, hs_fn=held_suarez_forcing)

    def step_fn(s, dt_):
        return model.step_with_physics(s, dt_, physics_fn)

    def check_fn(s):
        return (check_finite({"T": s.T.data, "u": s.u.data}),
                float(jnp.max(jnp.abs(s.u.data))))

    def scalar_fn(s):
        return {
            "mass": float(global_integral(s.p_s, grid)),
            "max_wind": float(jnp.max(jnp.sqrt(s.u.data**2 + s.v.data**2))),
            "mean_T": float(jnp.mean(s.T.data)),
        }

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))  # every 6 hours

    return run_timeloop(step_fn, state, dt, n_steps, check_fn, scalar_fn,
                        diag_every, label="Cubed-Sphere C36 RRTMGP")


def run_latlon(days, nlev, vertical_coord, sb81_omega_conversion=False):
    """Run Held-Suarez + RRTMGP on lat-lon 72x144."""
    import math as _m
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig)
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon, held_suarez_forcing_latlon)

    n_lat, n_lon = 72, 144
    grid = create_latlon_grid(n_lat, n_lon)
    sigma = _create_vertical(nlev, vertical_coord)

    # Pole-cell CFL-safe dt and A_h for the explicit C-grid solver.
    R = float(grid.radius)
    dx_pole = R * grid.dlon * _m.cos(_m.pi / 2 - grid.dlat / 2)
    dt = min(200.0, 0.8 * dx_pole / 300.0)  # advective CFL
    A_h_max = 0.4 * dx_pole ** 2 / dt
    A_h = min(_laplacian_visc_latlon(n_lat), A_h_max)

    config = CGridLatLonPrimitiveEquationConfig(
        A_h=A_h,
        fix_mass=True,
        # #1029 omega-side SB81 conversion (hybrid lane only; default OFF).
        sb81_omega_conversion=sb81_omega_conversion)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)

    # Convert to native C-grid state once; step natively to avoid
    # lossy face↔cell re-projection every timestep.
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    hs_init = held_suarez_init_latlon(grid, sigma)
    state = hydrostatic_to_cgrid(hs_init, grid)

    physics_fn = _make_rrtmgp_physics("hydrostatic", dt, hs_fn=held_suarez_forcing_latlon)

    def step_fn(s, dt_):
        return model.step(s, dt_, physics_fn=physics_fn)

    def check_fn(s):
        return (check_finite({"T": s.T}),
                float(jnp.max(jnp.abs(s.u))))

    def scalar_fn(s):
        return {
            "mass": float(jnp.sum(s.p_s * grid.area)),
            "max_wind": float(jnp.max(jnp.sqrt(
                (0.5 * (s.u[:, :-1] + s.u[:, 1:]))**2
                + (0.5 * (s.v[:-1] + s.v[1:]))**2))),
            "mean_T": float(jnp.mean(s.T)),
        }

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))

    return run_timeloop(step_fn, state, dt, n_steps, check_fn, scalar_fn,
                        diag_every, label="Lat-Lon 72x144 RRTMGP")


def run_icosahedral(days, nlev):
    """Run Held-Suarez + RRTMGP on icosahedral ico5."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas, held_suarez_forcing_mpas)

    level = 5
    mesh = create_voronoi_mesh(level)
    sigma = create_sigma_coordinate(nlev)  # MPAS uses sigma only
    dt = 200.0

    config = MPASPrimitiveEquationConfig(
        nu_del4=_hyperdiff_ico(mesh),
        nu_del2=_laplacian_visc_ico(mesh),
        fix_mass=True)
    model = MPASPrimitiveEquationModel(mesh, sigma, config)
    state = held_suarez_init_mpas(mesh, sigma)

    physics_fn = _make_rrtmgp_physics("mpas", dt, hs_fn=held_suarez_forcing_mpas)

    def step_fn(s, dt_):
        return model.step(s, dt_, physics_fn)

    def check_fn(s):
        return (check_finite({"T": s.T.data, "u": s.u.data}),
                float(jnp.max(jnp.abs(s.u.data))))

    def scalar_fn(s):
        return {
            "mass": float(jnp.sum(s.p_s.data * mesh.areaCell)),
            "max_wind": float(jnp.max(jnp.abs(s.u.data))),
            "mean_T": float(jnp.mean(s.T.data)),
        }

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))

    return run_timeloop(step_fn, state, dt, n_steps, check_fn, scalar_fn,
                        diag_every, label="Icosahedral ico5 RRTMGP")


def run_spectral(days, nlev, vertical_coord):
    """Run Held-Suarez + RRTMGP on spectral T21."""
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_to_grid)
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_spectral

    n_max = 21
    grid = create_gaussian_grid(n_max)
    sigma = _create_vertical(nlev, vertical_coord)
    dt = 600.0

    pe_config = SpectralPEConfig(
        hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
        spectral_filter_order=8,
        spectral_filter_strength=0.01,
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

    physics_fn = _make_rrtmgp_physics("spectral_pe", dt, hs_fn=held_suarez_forcing_spectral)

    def step_fn(s, dt_):
        return model.step(s, dt_, physics_fn=physics_fn)

    def check_fn(s):
        T = sh_synthesis_3d(grid, s.T_hat.data)
        return (check_finite({"T": T}),
                float(jnp.max(jnp.abs(T))))

    def scalar_fn(s):
        fields = spectral_pe_to_grid(s, grid, sigma)
        return {
            "mass": float(jnp.mean(fields['p_s'])),
            "max_wind": float(jnp.max(jnp.sqrt(
                fields['u']**2 + fields['v']**2))),
            "mean_T": float(jnp.mean(fields['T'])),
        }

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))

    return run_timeloop(step_fn, state, dt, n_steps, check_fn, scalar_fn,
                        diag_every, label="Spectral T21 RRTMGP")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(results: dict, output_dir: Path):
    """Print comparison table and save to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  CROSS-GRID COMPARISON: Held-Suarez + RRTMGP")
    print("=" * 70)
    print(f"  {'Grid':<20} {'Status':<8} {'mean_T(K)':<12} "
          f"{'max_wind':<12} {'mass_drift':<14} {'wall(s)':<10}")
    print("-" * 70)

    # iter-119 codex iter-118-followup MEDIUM-2: apply the
    # iter-117 mass-drift tolerance to this 4-grid HS+RRTMGP
    # runner too.  Pre-iter-119 ``status = "PASS" if ok else
    # "FAIL"`` only checked finiteness + non-blown-up,
    # allowing arbitrary mass drift to PASS — same gap that
    # iter-117 closed for the matrix runner's HS branch.
    HS_RRTMGP_MASS_DRIFT_TOL = 1e-2
    import numpy as _np_iter119

    summary = {}
    for grid_name, (state, diag, wall, ok) in results.items():
        mean_T = diag.get("mean_T", [0])[-1] if diag.get("mean_T") else 0
        max_wind = diag.get("max_wind", [0])[-1] if diag.get("max_wind") else 0

        # iter-88: delegate to the shared baseline-zero-safe
        # helper (factored out of the iter-83 / iter-87 inline
        # implementations).  See
        # ``legoesm.diagnostics.conservation_drift`` for history.
        from legoesm.diagnostics.conservation_drift import compute_relative_drift
        mass_drift = compute_relative_drift(diag.get("mass", []))

        # iter-119 mass-drift gate: NaN- and threshold-aware.
        # Update both the local ``ok`` AND the entry in
        # ``results`` because ``main()`` aggregates via
        # ``all(r[3] for r in results.values())`` — relying on
        # ``r[3]`` (the original ok) to reflect the iter-119
        # mass-drift gate too.
        if ok and (
            not _np_iter119.isfinite(mass_drift)
            or mass_drift > HS_RRTMGP_MASS_DRIFT_TOL
        ):
            ok = False
            # Persist back to the results dict so main()'s
            # all_ok aggregation picks up the iter-119 fail.
            results[grid_name] = (state, diag, wall, ok)

        status = "PASS" if ok else "FAIL"

        print(f"  {grid_name:<20} {status:<8} {mean_T:<12.2f} "
              f"{max_wind:<12.1f} {mass_drift:<14.2e} {wall:<10.1f}")

        summary[grid_name] = {
            "status": status,
            "mean_T_final": mean_T,
            "max_wind_final": max_wind,
            "mass_drift": mass_drift,
            "wall_time": wall,
            "diag": {k: [float(v) for v in vs] for k, vs in diag.items()},
        }

    print("-" * 70)

    # Physical consistency checks
    temps = [s["mean_T_final"] for s in summary.values() if s["status"] == "PASS"]
    winds = [s["max_wind_final"] for s in summary.values() if s["status"] == "PASS"]

    if temps:
        T_spread = max(temps) - min(temps)
        print(f"\n  Temperature spread across grids: {T_spread:.2f} K "
              f"(range: {min(temps):.2f} - {max(temps):.2f} K)")
        if T_spread < 20:
            print("  -> Cross-grid temperature CONSISTENT")
        else:
            print("  -> WARNING: Large temperature spread, investigate grid differences")

    if winds:
        w_spread = max(winds) - min(winds)
        print(f"  Max wind spread: {w_spread:.1f} m/s "
              f"(range: {min(winds):.1f} - {max(winds):.1f} m/s)")

    # Physical soundness checks
    print("\n  Physical soundness:")
    for grid_name, s in summary.items():
        issues = []
        if s["mean_T_final"] < 200 or s["mean_T_final"] > 350:
            issues.append(f"mean_T={s['mean_T_final']:.1f}K out of [200,350]")
        if s["max_wind_final"] > 200:
            issues.append(f"max_wind={s['max_wind_final']:.1f} unrealistically high")
        if s["mass_drift"] > 1e-6:
            issues.append(f"mass_drift={s['mass_drift']:.2e} exceeds 1e-6")
        if issues:
            print(f"    {grid_name}: WARNING - {'; '.join(issues)}")
        else:
            print(f"    {grid_name}: OK")

    # Save results
    out_file = output_dir / "comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Held-Suarez + RRTMGP on all grid types")
    parser.add_argument("--days", type=float, default=100,
                        help="Simulation length in days (default: 100)")
    parser.add_argument("--nlev", type=int, default=40,
                        help="Number of vertical levels (default: 40)")
    parser.add_argument("--vertical-coord", type=str, default="hybrid",
                        choices=["sigma", "hybrid"],
                        help="Vertical coordinate type (default: hybrid)")
    parser.add_argument("--output", type=str, default="results/held_suarez_rrtmgp_4grids",
                        help="Output directory")
    parser.add_argument("--grids", type=str, nargs="+",
                        default=["cubed_sphere", "latlon", "icosahedral", "spectral"],
                        help="Which grids to run")
    parser.add_argument("--sb81-omega-conversion", action="store_true",
                        default=False,
                        help="SB81 alpha-weighted kT*omega/p energy conversion "
                             "on the hybrid lat-lon lane (#1029; latlon+hybrid "
                             "only, default OFF).")
    args = parser.parse_args()

    output_dir = Path(args.output)
    results = {}

    grid_runners = {
        "cubed_sphere": lambda: run_cubed_sphere(args.days, args.nlev, args.vertical_coord),
        "latlon": lambda: run_latlon(args.days, args.nlev, args.vertical_coord,
                             sb81_omega_conversion=args.sb81_omega_conversion),
        "icosahedral": lambda: run_icosahedral(args.days, args.nlev),
        "spectral": lambda: run_spectral(args.days, args.nlev, args.vertical_coord),
    }

    for grid_name in args.grids:
        if grid_name not in grid_runners:
            print(f"Unknown grid type: {grid_name}")
            continue
        try:
            state, diag, wall, ok = grid_runners[grid_name]()
            results[grid_name] = (state, diag, wall, ok)
        except Exception as e:
            print(f"\n  *** {grid_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[grid_name] = (None, {}, 0, False)

    if results:
        compare_results(results, output_dir)

    # Return success if all grids passed
    all_ok = all(r[3] for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
