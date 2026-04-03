#!/usr/bin/env python
"""Run Held-Suarez + RRTMGP on all 4 grid types for 100 days.

Uses the 4grids script directly (bypasses ModelDriver buffering issues).
Runs grids sequentially with C16, 36x72 latlon, T21 spectral, ico3.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

DAYS = 100
NLEV = 20

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def run_cubed_sphere(days, nlev):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig)
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.grids.vertical import standard_hybrid_levels

    n = 16
    grid = create_cubed_sphere(n)
    sigma = standard_hybrid_levels(nlev)
    dx = 6.371e6 * np.pi / (2 * n)
    dt = 300.0  # Conservative dt

    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=dx**4 / 3600.0, hyperdiff_ps_coeff=dx**4 / 3600.0,
        div_damp_coeff=0.15 * dx**2, A_h=0.01 * dx**2,
        use_conservation_fixer=True, fix_mass=True)
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme='rrtmgp'),
        convection=ConvectionConfig(scheme='none'),
        turbulence=TurbulenceConfig(scheme='none'),
        microphysics=MicrophysicsConfig(scheme='none'),
        gravity_wave_drag=GravityWaveDragConfig(scheme='none'),
    )
    physics_fn = make_physics(phys_cfg, model_type='hydrostatic', dt=dt)

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(10 * 3600 * 24 / dt))  # every 10 days

    flush_print(f"  Cubed-sphere C{n}: {n_steps} steps, dt={dt}s")
    state = model.step_with_physics(state, dt, physics_fn)
    jax.block_until_ready(state)
    flush_print("  JIT done")

    t0 = time.time()
    diag = []
    for i in range(1, n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        if (i + 1) % diag_every == 0:
            jax.block_until_ready(state)
            day = (i + 1) * dt / 86400
            T = state.T.data
            mean_T = float(jnp.mean(T))
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            if not jnp.all(jnp.isfinite(T)):
                flush_print(f"  Day {day:6.1f}: BLOWUP (NaN)")
                return None
            diag.append({"day": day, "mean_T": mean_T, "max_wind": max_wind})
            flush_print(f"  Day {day:6.1f}: mean_T={mean_T:.2f}K  max_wind={max_wind:.1f}m/s")

    wall = time.time() - t0
    flush_print(f"  DONE in {wall:.0f}s")
    return diag


def run_spectral(days, nlev):
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_to_grid)
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.grids.vertical import standard_hybrid_levels

    n_max = 21
    grid = create_gaussian_grid(n_max)
    sigma = standard_hybrid_levels(nlev)
    dt = 600.0

    pe_config = SpectralPEConfig(
        hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
        spectral_filter_order=8, spectral_filter_strength=0.01)
    model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme='rrtmgp'),
        convection=ConvectionConfig(scheme='none'),
        turbulence=TurbulenceConfig(scheme='none'),
        microphysics=MicrophysicsConfig(scheme='none'),
        gravity_wave_drag=GravityWaveDragConfig(scheme='none'),
    )
    physics_fn = make_physics(phys_cfg, model_type='spectral_pe', dt=dt)

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(10 * 3600 * 24 / dt))  # every 10 days

    flush_print(f"  Spectral T{n_max}: {n_steps} steps, dt={dt}s")
    state = model.step(state, dt, physics_fn=physics_fn)
    jax.block_until_ready(state)
    flush_print("  JIT done")

    t0 = time.time()
    diag = []
    for i in range(1, n_steps):
        state = model.step(state, dt, physics_fn=physics_fn)
        if (i + 1) % diag_every == 0:
            jax.block_until_ready(state)
            day = (i + 1) * dt / 86400
            fields = spectral_pe_to_grid(state, grid, sigma)
            mean_T = float(jnp.mean(fields['T']))
            max_wind = float(jnp.max(jnp.sqrt(fields['u']**2 + fields['v']**2)))
            T = sh_synthesis_3d(grid, state.T_hat.data)
            if not jnp.all(jnp.isfinite(T)):
                flush_print(f"  Day {day:6.1f}: BLOWUP (NaN)")
                return None
            diag.append({"day": day, "mean_T": mean_T, "max_wind": max_wind})
            flush_print(f"  Day {day:6.1f}: mean_T={mean_T:.2f}K  max_wind={max_wind:.1f}m/s")

    wall = time.time() - t0
    flush_print(f"  DONE in {wall:.0f}s")
    return diag


def run_icosahedral(days, nlev):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
    from legoesm.atmosphere.physics.held_suarez_mpas import held_suarez_init_mpas
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    level = 3
    mesh = create_voronoi_mesh(level)
    sigma = create_sigma_coordinate(nlev)
    dx = np.sqrt(np.mean(np.asarray(mesh.areaCell)))
    dt = 600.0

    config = MPASPrimitiveEquationConfig(
        nu_del4=dx**4 / 3600.0, nu_del2=0.01 * dx**2, fix_mass=True)
    model = MPASPrimitiveEquationModel(mesh, sigma, config)
    state = held_suarez_init_mpas(mesh, sigma)

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme='rrtmgp'),
        convection=ConvectionConfig(scheme='none'),
        turbulence=TurbulenceConfig(scheme='none'),
        microphysics=MicrophysicsConfig(scheme='none'),
        gravity_wave_drag=GravityWaveDragConfig(scheme='none'),
    )
    physics_fn = make_physics(phys_cfg, model_type='mpas', dt=dt)

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(10 * 3600 * 24 / dt))  # every 10 days

    flush_print(f"  Icosahedral ico{level}: {n_steps} steps, dt={dt}s, dx={dx/1000:.0f}km")
    state = model.step(state, dt, physics_fn)
    jax.block_until_ready(state)
    flush_print("  JIT done")

    t0 = time.time()
    diag = []
    for i in range(1, n_steps):
        state = model.step(state, dt, physics_fn)
        if (i + 1) % diag_every == 0:
            jax.block_until_ready(state)
            day = (i + 1) * dt / 86400
            T = state.T.data
            mean_T = float(jnp.mean(T))
            max_u = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(T)):
                flush_print(f"  Day {day:6.1f}: BLOWUP (NaN)")
                return None
            diag.append({"day": day, "mean_T": mean_T, "max_wind": max_u})
            flush_print(f"  Day {day:6.1f}: mean_T={mean_T:.2f}K  max|u|={max_u:.1f}m/s")

    wall = time.time() - t0
    flush_print(f"  DONE in {wall:.0f}s")
    return diag


def run_latlon_spectral_elements(days, nlev):
    """Lat-lon SE uses the spectral element discretization on a lat-lon grid.

    For Held-Suarez at low resolution, we use a centered difference scheme
    on a lat-lon grid which avoids the polar CFL issue.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
        LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig)
    from legoesm.atmosphere.physics.held_suarez_latlon import held_suarez_init_latlon
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.grids.vertical import standard_hybrid_levels

    n_lat, n_lon = 36, 72
    grid = create_latlon_grid(n_lat, n_lon)
    sigma = standard_hybrid_levels(nlev)
    dx = 6.371e6 * np.pi / n_lat
    # Use small dt to handle polar CFL
    dt = 120.0

    config = LatLonPrimitiveEquationConfig(
        hyperdiff_coeff=dx**4 / 3600.0, hyperdiff_ps_coeff=dx**4 / 3600.0,
        div_damp_coeff=0.15 * dx**2, A_h=0.01 * dx**2,
        use_conservation_fixer=True, fix_mass=True)
    model = LatLonPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init_latlon(grid, sigma)

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme='rrtmgp'),
        convection=ConvectionConfig(scheme='none'),
        turbulence=TurbulenceConfig(scheme='none'),
        microphysics=MicrophysicsConfig(scheme='none'),
        gravity_wave_drag=GravityWaveDragConfig(scheme='none'),
    )
    physics_fn = make_physics(phys_cfg, model_type='hydrostatic', dt=dt)

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(10 * 3600 * 24 / dt))  # every 10 days

    flush_print(f"  Lat-lon {n_lat}x{n_lon}: {n_steps} steps, dt={dt}s")
    state = model.step_with_physics(state, dt, physics_fn)
    jax.block_until_ready(state)
    flush_print("  JIT done")

    t0 = time.time()
    diag = []
    for i in range(1, n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        if (i + 1) % diag_every == 0:
            jax.block_until_ready(state)
            day = (i + 1) * dt / 86400
            T = state.T.data
            mean_T = float(jnp.mean(T))
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            if not jnp.all(jnp.isfinite(T)):
                flush_print(f"  Day {day:6.1f}: BLOWUP (NaN)")
                return None
            diag.append({"day": day, "mean_T": mean_T, "max_wind": max_wind})
            flush_print(f"  Day {day:6.1f}: mean_T={mean_T:.2f}K  max_wind={max_wind:.1f}m/s")

    wall = time.time() - t0
    flush_print(f"  DONE in {wall:.0f}s")
    return diag


def main():
    flush_print(f"JAX backend: {jax.default_backend()}, x64: {jax.config.jax_enable_x64}")
    flush_print(f"Running Held-Suarez + RRTMGP on all grids for {DAYS} days")
    flush_print()

    results = {}

    grids = [
        ("spectral", run_spectral),
        ("icosahedral", run_icosahedral),
        ("cubed_sphere", run_cubed_sphere),
        ("latlon", run_latlon_spectral_elements),
    ]

    for name, runner in grids:
        flush_print(f"{'='*60}")
        flush_print(f"  {name}")
        flush_print(f"{'='*60}")
        try:
            diag = runner(DAYS, NLEV)
            results[name] = diag
            if diag is None:
                flush_print(f"  {name}: FAILED (blowup)")
            else:
                flush_print(f"  {name}: PASSED")
        except Exception as e:
            flush_print(f"  {name}: ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = None
        flush_print()

    # Summary
    flush_print("=" * 70)
    flush_print("CROSS-GRID COMPARISON")
    flush_print("=" * 70)
    flush_print(f"{'Grid':<20} {'Status':<8} {'Final mean_T':<14} {'Final max_wind':<14}")
    flush_print("-" * 70)

    for name, diag in results.items():
        if diag is None or len(diag) == 0:
            flush_print(f"{name:<20} {'FAIL':<8}")
        else:
            last = diag[-1]
            flush_print(f"{name:<20} {'PASS':<8} {last['mean_T']:<14.2f} {last['max_wind']:<14.1f}")

    # Check cross-grid consistency
    passing = {n: d for n, d in results.items() if d is not None and len(d) > 0}
    if len(passing) >= 2:
        temps = [d[-1]["mean_T"] for d in passing.values()]
        spread = max(temps) - min(temps)
        flush_print(f"\nTemperature spread: {spread:.2f}K (range: {min(temps):.2f} - {max(temps):.2f}K)")
        if spread < 20:
            flush_print("Cross-grid CONSISTENT")
        else:
            flush_print("WARNING: Large temperature spread")

    # Save results
    out = Path("results/held_suarez_rrtmgp_allgrids")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    flush_print(f"\nResults saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
