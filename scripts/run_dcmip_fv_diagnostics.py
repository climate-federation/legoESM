#!/usr/bin/env python
"""Run DCMIP FV 3D test cases with mass conservation diagnostics.

Runs short integrations of:
1. DCMIP-2012 Test 1-1: 3D Deformational Flow (tracer transport)
2. DCMIP-2025 TC1: Mountain gravity waves (NH compressible)
3. DCMIP-2025 TC2a: Gap flow (NH, small Earth)

Reports mass conservation and stability diagnostics.
"""
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def global_mass_nh(state, height_coord, terrain_metric, grid):
    """Total atmospheric mass for non-hydrostatic model."""
    rho_0 = height_coord.rho_ref
    rho_total = rho_0 + state.rho_prime.data
    J = terrain_metric.jacobian
    dz = height_coord.dz
    # Mass per column = sum(rho * dz * J)
    mass_col = jnp.sum(rho_total * dz, axis=-1) * J
    # Global integral: sum over all cells weighted by area
    area = grid.area  # (6, n, n)
    return float(jnp.sum(mass_col * area))


def run_dcmip_transport_11():
    """DCMIP-2012 Test 1-1: 3D Deformational Flow (12-day flow reversal)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.tracer_transport import (
        TracerTransportModel,
        TracerTransportConfig,
    )
    from tests.test_cases.dcmip_transport import (
        dcmip11_wind, dcmip11_init, create_dcmip_sigma,
        compute_tracer_error_norms,
    )

    print("=" * 70)
    print("DCMIP-2012 Test 1-1: 3D Deformational Flow")
    print("=" * 70)

    resolution = 16
    n_levels = 30
    dt = 1800.0
    duration = 12.0 * 86400.0  # 12 days
    n_steps = int(duration / dt)

    grid = create_cubed_sphere(resolution)
    sigma_coord = create_dcmip_sigma(n_levels)
    state_init = dcmip11_init(grid, sigma_coord)

    config = TracerTransportConfig(hyperdiff_coeff=0.0)
    model = TracerTransportModel(grid, sigma_coord, dcmip11_wind, config)

    print(f"  C{resolution} L{n_levels}, dt={dt}s, {n_steps} steps (12 days)")

    # Initial tracer mass
    q0 = state_init.tracers.data
    n_tracers = q0.shape[-1]
    area = grid.area
    dsigma = sigma_coord.dsigma

    def tracer_mass(q_data):
        """Compute global integral of each tracer."""
        masses = []
        for i in range(q_data.shape[-1]):
            qi = q_data[..., i]  # (6, n, n, nlev)
            integral = jnp.sum(qi * dsigma, axis=-1)  # (6, n, n)
            masses.append(float(jnp.sum(integral * area)))
        return masses

    mass_init = tracer_mass(q0)
    print(f"  Initial tracer masses: {[f'{m:.6e}' for m in mass_init]}")

    # Integrate
    state = state_init
    t0 = time.time()
    state = model.step(state, dt)  # JIT warmup
    jax.block_until_ready(state.tracers.data)
    print(f"  JIT: {time.time()-t0:.1f}s")

    t0 = time.time()
    diag_every = n_steps // 4
    for i in range(1, n_steps):
        state = model.step(state, dt)
        if (i + 1) % diag_every == 0:
            q = state.tracers.data
            q1_range = f"[{float(jnp.min(q[...,0])):.4f}, {float(jnp.max(q[...,0])):.4f}]"
            print(f"    Step {i+1}/{n_steps} | q1 range: {q1_range}")

    wall_time = time.time() - t0
    print(f"  Wall time: {wall_time:.1f}s")

    # Final mass conservation
    mass_final = tracer_mass(state.tracers.data)
    print(f"\n  Final tracer masses: {[f'{m:.6e}' for m in mass_final]}")
    for i in range(n_tracers):
        if abs(mass_init[i]) > 1e-15:
            rel_err = abs(mass_final[i] - mass_init[i]) / abs(mass_init[i])
            print(f"  Tracer {i+1} mass conservation: relative error = {rel_err:.2e}")
        else:
            abs_err = abs(mass_final[i] - mass_init[i])
            print(f"  Tracer {i+1} mass conservation: absolute error = {abs_err:.2e}")

    # Error norms (flow reversal → exact = initial)
    norms = compute_tracer_error_norms(state, state_init, grid)
    print(f"\n  Flow-reversal error norms (exact = initial):")
    for i in range(n_tracers):
        print(f"    q{i+1}: l1={float(norms['l1'][i]):.4e}  "
              f"l2={float(norms['l2'][i]):.4e}  "
              f"linf={float(norms['linf'][i]):.4e}")

    return True


def run_dcmip2025_tc1():
    """DCMIP-2025 TC1: Mountain gravity waves (30 min)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel, CompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    print("\n" + "=" * 70)
    print("DCMIP-2025 TC1: Mountain Gravity Waves (30 min)")
    print("=" * 70)

    resolution = 16
    n_levels = 20
    dt = 10.0
    duration = 1800.0  # 30 min
    n_steps = int(duration / dt)

    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=n_levels)

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=10000.0,
        sponge_coeff=0.05,
    )
    model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)

    print(f"  C{resolution} L{n_levels}, dt={dt}s, {n_steps} steps")

    mass_init = global_mass_nh(state, height_coord, terrain_metric, grid)
    print(f"  Initial total mass: {mass_init:.6e}")

    # Constant-theta test: initial dtheta should be ~0
    theta_0 = height_coord.theta_ref
    theta_total = theta_0 + state.theta_prime.data
    print(f"  Initial theta range: [{float(jnp.min(theta_total)):.2f}, "
          f"{float(jnp.max(theta_total)):.2f}]")

    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT: {time.time()-t0:.1f}s")

    t0 = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)
        if (i + 1) % (n_steps // 5) == 0:
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            theta_p_max = float(jnp.max(jnp.abs(state.theta_prime.data)))
            print(f"    Step {i+1}/{n_steps} | max|w|={w_max:.4f} | "
                  f"max|u|={u_max:.2f} | max|theta'|={theta_p_max:.4f}")

    wall_time = time.time() - t0

    mass_final = global_mass_nh(state, height_coord, terrain_metric, grid)
    mass_err = abs(mass_final - mass_init) / abs(mass_init)
    print(f"\n  Final total mass: {mass_final:.6e}")
    print(f"  Mass conservation: relative error = {mass_err:.2e}")
    print(f"  Wall time: {wall_time:.1f}s")

    stable = bool(jnp.all(jnp.isfinite(state.u.data)))
    print(f"  Stable: {stable}")
    return stable


def run_dcmip2025_tc2a():
    """DCMIP-2025 TC2a: Gap flow (30 min, small Earth)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel, CompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc2_init

    print("\n" + "=" * 70)
    print("DCMIP-2025 TC2a: Gap Flow (30 min, small Earth)")
    print("=" * 70)

    resolution = 16
    n_levels = 20
    dt = 5.0  # Small-Earth needs smaller dt (dx~46km, CFL~147s)
    duration = 1800.0  # 30 min
    n_steps = int(duration / dt)

    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
        grid, n_levels=n_levels, subcase="a",
    )

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=15000.0,
        sponge_coeff=1.0 / (0.1 * 86400.0),
        small_earth_factor=20.0,
    )
    model = CompressibleEulerModel(small_grid, height_coord, terrain_metric, config)

    print(f"  C{resolution} L{n_levels}, dt={dt}s, {n_steps} steps")

    mass_init = global_mass_nh(state, height_coord, terrain_metric, small_grid)
    print(f"  Initial total mass: {mass_init:.6e}")

    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT: {time.time()-t0:.1f}s")

    t0 = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)
        if (i + 1) % (n_steps // 5) == 0:
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            print(f"    Step {i+1}/{n_steps} | max|w|={w_max:.4f} | max|u|={u_max:.2f}")

    wall_time = time.time() - t0

    mass_final = global_mass_nh(state, height_coord, terrain_metric, small_grid)
    mass_err = abs(mass_final - mass_init) / abs(mass_init)
    print(f"\n  Final total mass: {mass_final:.6e}")
    print(f"  Mass conservation: relative error = {mass_err:.2e}")
    print(f"  Wall time: {wall_time:.1f}s")

    stable = bool(jnp.all(jnp.isfinite(state.u.data)))
    print(f"  Stable: {stable}")
    return stable


if __name__ == "__main__":
    print("legoESM DCMIP FV 3D Diagnostics")
    print(f"Backend: {jax.default_backend()}")
    print(f"Float64: {jnp.zeros(1).dtype}")
    print()

    results = {}
    results["transport_11"] = run_dcmip_transport_11()
    results["tc1"] = run_dcmip2025_tc1()
    results["tc2a"] = run_dcmip2025_tc2a()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
