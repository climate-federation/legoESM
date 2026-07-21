#!/usr/bin/env python3
"""Williamson Shallow Water Test Cases 2, 5, and 6 — Comprehensive Diagnostics.

Exercises the FV3 cubed-sphere C-D grid shallow water dynamical core on the
three standard test cases from Williamson et al. (1992), JCP 102, 211-224.

Test Case 2: Steady-State Geostrophic Flow (exact solution exists)
Test Case 5: Isolated Mountain (stability & conservation)
Test Case 6: Rossby-Haurwitz Wave number 4 (nonlinear dynamics)

Reports error norms, mass/energy conservation, cube-edge artifacts, and
stability diagnostics.
"""

from __future__ import annotations

import time
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel,
    CDGridShallowWaterConfig,
    CDGridShallowWaterState,
)


# =============================================================================
# Physical constants — pulled from the central legoesm.constants module
# =============================================================================
g = constants.g
omega = constants.Omega
R_earth = constants.R_earth


# =============================================================================
# Helper: time integration loop
# =============================================================================

def integrate(model, state, dt, n_steps, print_every=0):
    """Integrate model for n_steps, returning final state and diagnostics."""
    area = model.cdgrid.base.area
    mass_0 = float(jnp.sum(state.h * area))

    for step_i in range(1, n_steps + 1):
        state = model.step(state, dt)

        if print_every > 0 and step_i % print_every == 0:
            h_min = float(jnp.min(state.h))
            h_max = float(jnp.max(state.h))
            u_max = float(jnp.max(jnp.abs(state.u_d)))
            mass_now = float(jnp.sum(state.h * area))
            mass_rel = abs(mass_now - mass_0) / abs(mass_0)
            is_finite = bool(jnp.all(jnp.isfinite(state.h)))
            day = step_i * dt / 86400.0
            print(f"    Day {day:6.2f}: h=[{h_min:.2f}, {h_max:.2f}], "
                  f"|u|_max={u_max:.2f}, mass_err={mass_rel:.2e}, "
                  f"finite={is_finite}")

            if not is_finite:
                print("    *** BLOWUP DETECTED ***")
                return state, False

    return state, True


# =============================================================================
# Helper: compute energy
# =============================================================================

def total_energy(state, area, g_val=g):
    """Compute total energy (KE + PE) area-weighted."""
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    u_cc, v_cc = dgrid_to_center_vector(state.u_d, state.v_d)
    ke = 0.5 * state.h * (u_cc ** 2 + v_cc ** 2)
    pe = 0.5 * g_val * (state.h + state.h_s) ** 2
    return float(jnp.sum((ke + pe) * area))


# =============================================================================
# Helper: cube-edge artifact check
# =============================================================================

def cube_edge_artifact_ratio(error_field, n):
    """Compare max |error| at face boundaries vs face interiors.

    error_field: (6, n, n)
    Returns ratio = max(|error_boundary|) / max(|error_interior|)
    """
    abs_err = jnp.abs(error_field)

    # Boundary: first/last row and column of each face
    boundary_mask = jnp.zeros((n, n), dtype=bool)
    boundary_mask = boundary_mask.at[0, :].set(True)
    boundary_mask = boundary_mask.at[-1, :].set(True)
    boundary_mask = boundary_mask.at[:, 0].set(True)
    boundary_mask = boundary_mask.at[:, -1].set(True)
    boundary_mask = jnp.broadcast_to(boundary_mask, (6, n, n))

    interior_mask = ~boundary_mask

    max_boundary = float(jnp.max(jnp.where(boundary_mask, abs_err, 0.0)))
    max_interior = float(jnp.max(jnp.where(interior_mask, abs_err, 0.0)))

    if max_interior < 1e-30:
        return 0.0  # no error anywhere
    return max_boundary / max_interior


# =============================================================================
# Test Case 2: Steady-State Geostrophic Flow
# =============================================================================

def williamson2_ic(cdgrid, u_0=None, h_0=None):
    """Create Williamson test case 2 initial condition.

    Solid-body rotation in geostrophic balance.
    All arrays promoted to float64 for conservation accuracy.
    """
    R = cdgrid.radius
    if u_0 is None:
        u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)  # ~38.6 m/s
    if h_0 is None:
        h_0 = 29400.0 / g

    # Promote grid coordinates to f64 for accuracy
    lat_c = cdgrid.base.lat.astype(jnp.float64)
    h = h_0 - (R * omega * u_0 + 0.5 * u_0 ** 2) * jnp.sin(lat_c) ** 2 / g

    lat_corner = cdgrid.lat_corner.astype(jnp.float64)
    u_geo = u_0 * jnp.cos(lat_corner)

    ca = cdgrid.cos_angle_corner.astype(jnp.float64)
    sa = cdgrid.sin_angle_corner.astype(jnp.float64)
    u_d = u_geo * ca
    v_d = -u_geo * sa

    h_s = jnp.zeros_like(h)
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def run_test_case_2(n_res=16, n_days=5, dt=600.0):
    """Run Williamson Test Case 2 at given resolution."""
    print(f"\n{'='*70}")
    print(f"WILLIAMSON TEST CASE 2: Steady-State Geostrophic Flow")
    print(f"  Resolution: C{n_res}, Integration: {n_days} days, dt={dt:.0f}s")
    print(f"{'='*70}")

    # Grid setup
    t0 = time.time()
    grid = create_cubed_sphere(n_res)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    print(f"  Grid created in {time.time() - t0:.1f}s")

    dx_min = float(jnp.min(grid.dx))
    print(f"  dx_min = {dx_min/1000:.1f} km")

    # Model config -- mild diffusion since edge-midpoint path eliminates HK
    # Biharmonic with 30-day e-folding time (no Laplacian needed)
    config = CDGridShallowWaterConfig(
        A_h=0.0,
        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
        div_damp=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CDGridShallowWaterModel(grid, config)

    # Initial condition
    state0 = williamson2_ic(cdgrid)
    model.set_initial_mass(state0)
    h_exact = state0.h  # exact solution = initial condition

    area = cdgrid.base.area
    total_area = float(jnp.sum(area))
    mass_0 = float(jnp.sum(state0.h * area))
    energy_0 = total_energy(state0, area)

    h_range = float(jnp.max(state0.h) - jnp.min(state0.h))
    print(f"  h range: [{float(jnp.min(state0.h)):.2f}, {float(jnp.max(state0.h)):.2f}] m")
    print(f"  h_range (for normalization) = {h_range:.2f} m")
    print(f"  Initial mass = {mass_0:.6e}")
    print(f"  Initial energy = {energy_0:.6e}")

    # Integration
    n_steps = int(n_days * 86400 / dt)
    print(f"\n  Integrating {n_steps} steps...")
    t0 = time.time()
    state_final, stable = integrate(model, state0, dt, n_steps,
                                     print_every=max(1, n_steps // 5))
    wall_time = time.time() - t0
    print(f"  Integration complete in {wall_time:.1f}s "
          f"({wall_time / n_steps * 1000:.1f} ms/step)")

    # Diagnostics
    print(f"\n  --- Diagnostics after {n_days} days ---")

    # Stability
    all_finite = bool(jnp.all(jnp.isfinite(state_final.h)) and
                       jnp.all(jnp.isfinite(state_final.u_d)) and
                       jnp.all(jnp.isfinite(state_final.v_d)))
    print(f"  Stable (all finite): {all_finite}")

    if not all_finite:
        print("  *** MODEL BLEW UP -- skipping remaining diagnostics ***")
        return {"stable": False, "pass": False}

    # Error norms
    h_err = state_final.h - h_exact

    # L1
    l1_abs = float(jnp.sum(jnp.abs(h_err) * area) / total_area)
    l1_norm = l1_abs / max(h_range, 1.0)

    # L2
    l2_abs = float(jnp.sqrt(jnp.sum(h_err ** 2 * area) / total_area))
    l2_norm = l2_abs / max(h_range, 1.0)

    # Linf
    linf_abs = float(jnp.max(jnp.abs(h_err)))
    linf_norm = linf_abs / max(h_range, 1.0)

    print(f"  Height error norms (normalized by h_range={h_range:.2f} m):")
    print(f"    L1  = {l1_norm:.6f}  (abs: {l1_abs:.4f} m)")
    print(f"    L2  = {l2_norm:.6f}  (abs: {l2_abs:.4f} m)")
    print(f"    Linf = {linf_norm:.6f}  (abs: {linf_abs:.4f} m)")

    # Mass conservation
    mass_final = float(jnp.sum(state_final.h * area))
    mass_rel_err = abs(mass_final - mass_0) / abs(mass_0)
    print(f"  Mass conservation: relative error = {mass_rel_err:.2e}")

    # Energy conservation
    energy_final = total_energy(state_final, area)
    energy_rel_change = abs(energy_final - energy_0) / abs(energy_0)
    energy_sign = "+" if energy_final > energy_0 else "-"
    print(f"  Energy: relative change = {energy_sign}{energy_rel_change:.4e}")

    # Cube-edge artifact
    artifact_ratio = cube_edge_artifact_ratio(h_err, n_res)
    print(f"  Cube-edge artifact ratio (boundary/interior max |error|): {artifact_ratio:.3f}")

    # Height positivity
    h_min = float(jnp.min(state_final.h))
    h_positive = h_min > 0
    print(f"  Height positive: {h_positive} (min h = {h_min:.4f} m)")

    # Pass/fail criteria
    results = {
        "stable": stable and all_finite,
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
        "linf_norm": linf_norm,
        "mass_rel_err": mass_rel_err,
        "energy_rel_change": energy_rel_change,
        "artifact_ratio": artifact_ratio,
        "h_positive": h_positive,
    }

    print(f"\n  --- PASS/FAIL Criteria ---")
    checks = []

    # Strict targets: mild hyperdiffusion + non-orthogonality corrections
    c1 = l2_norm < 0.05
    checks.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] L2 height error norm < 0.05: {l2_norm:.6f}")

    c2 = linf_norm < 0.15
    checks.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Linf height error norm < 0.15: {linf_norm:.6f}")

    c3 = mass_rel_err < 1e-8
    checks.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] Mass conservation < 1e-8: {mass_rel_err:.2e}")

    c4 = artifact_ratio < 2.0
    checks.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] Cube-edge artifact ratio < 2.0: {artifact_ratio:.3f}")

    results["pass"] = all(checks)
    print(f"\n  Overall: {'PASS' if results['pass'] else 'FAIL'}")

    return results


# =============================================================================
# Test Case 5: Isolated Mountain
# =============================================================================

def williamson5_ic(cdgrid):
    """Create Williamson test case 5 initial condition.

    Zonal flow u_0 = 20 m/s with conical mountain at (30N, 90W).
    All arrays promoted to float64 for conservation accuracy.
    """
    R = cdgrid.radius
    u_0 = 20.0  # m/s
    h_0 = 5960.0  # m

    lat_c = cdgrid.base.lat.astype(jnp.float64)
    lon_c = cdgrid.base.lon.astype(jnp.float64)
    h = h_0 - (R * omega * u_0 + 0.5 * u_0 ** 2) * jnp.sin(lat_c) ** 2 / g

    lat_m = jnp.pi / 6.0
    lon_m = 3.0 * jnp.pi / 2.0

    dlat = lat_c - lat_m
    dlon = lon_c - lon_m
    a_hav = (jnp.sin(dlat / 2.0) ** 2
             + jnp.cos(lat_c) * jnp.cos(lat_m) * jnp.sin(dlon / 2.0) ** 2)
    r_gc = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a_hav, 0.0, 1.0)))

    r_0 = jnp.pi / 9.0
    h_s0 = 2000.0
    h_s = h_s0 * jnp.maximum(0.0, 1.0 - r_gc / r_0)

    lat_corner = cdgrid.lat_corner.astype(jnp.float64)
    u_geo = u_0 * jnp.cos(lat_corner)
    ca = cdgrid.cos_angle_corner.astype(jnp.float64)
    sa = cdgrid.sin_angle_corner.astype(jnp.float64)
    u_d = u_geo * ca
    v_d = -u_geo * sa

    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def run_test_case_5(n_res=16, n_days=15, dt=600.0):
    """Run Williamson Test Case 5."""
    print(f"\n{'='*70}")
    print(f"WILLIAMSON TEST CASE 5: Isolated Mountain (Zonal Flow + Topography)")
    print(f"  Resolution: C{n_res}, Integration: {n_days} days, dt={dt:.0f}s")
    print(f"{'='*70}")

    grid = create_cubed_sphere(n_res)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    dx_min = float(jnp.min(grid.dx))
    print(f"  dx_min = {dx_min/1000:.1f} km")

    config = CDGridShallowWaterConfig(
        A_h=0.0,
        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
        div_damp=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CDGridShallowWaterModel(grid, config)

    state0 = williamson5_ic(cdgrid)
    model.set_initial_mass(state0)

    area = cdgrid.base.area
    mass_0 = float(jnp.sum(state0.h * area))
    energy_0 = total_energy(state0, area)

    print(f"  h range: [{float(jnp.min(state0.h)):.2f}, {float(jnp.max(state0.h)):.2f}] m")
    print(f"  h_s max: {float(jnp.max(state0.h_s)):.2f} m")
    print(f"  Initial mass = {mass_0:.6e}")

    n_steps = int(n_days * 86400 / dt)
    print(f"\n  Integrating {n_steps} steps...")
    t0 = time.time()
    state_final, stable = integrate(model, state0, dt, n_steps,
                                     print_every=max(1, n_steps // 10))
    wall_time = time.time() - t0
    print(f"  Integration complete in {wall_time:.1f}s")

    print(f"\n  --- Diagnostics after {n_days} days ---")

    all_finite = bool(jnp.all(jnp.isfinite(state_final.h)) and
                       jnp.all(jnp.isfinite(state_final.u_d)) and
                       jnp.all(jnp.isfinite(state_final.v_d)))
    print(f"  Stable (all finite): {all_finite}")

    if not all_finite:
        print("  *** MODEL BLEW UP ***")
        return {"stable": False, "pass": False}

    mass_final = float(jnp.sum(state_final.h * area))
    mass_rel_err = abs(mass_final - mass_0) / abs(mass_0)
    print(f"  Mass conservation: relative error = {mass_rel_err:.2e}")

    h_min = float(jnp.min(state_final.h))
    h_max = float(jnp.max(state_final.h))
    h_positive = h_min > 0
    print(f"  Height range: [{h_min:.2f}, {h_max:.2f}] m")
    print(f"  Height positive: {h_positive}")

    # Max wind speed
    u_max = float(jnp.max(jnp.sqrt(state_final.u_d ** 2 + state_final.v_d ** 2)))
    print(f"  Max wind speed: {u_max:.2f} m/s")

    energy_final = total_energy(state_final, area)
    energy_rel_change = abs(energy_final - energy_0) / abs(energy_0)
    energy_sign = "+" if energy_final > energy_0 else "-"
    print(f"  Energy: relative change = {energy_sign}{energy_rel_change:.4e}")

    results = {
        "stable": stable and all_finite,
        "mass_rel_err": mass_rel_err,
        "h_positive": h_positive,
        "h_min": h_min,
        "h_max": h_max,
        "u_max": u_max,
        "energy_rel_change": energy_rel_change,
    }

    print(f"\n  --- PASS/FAIL Criteria ---")
    checks = []

    c1 = stable and all_finite
    checks.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] Model stable (no blowup)")

    c2 = mass_rel_err < 1e-6
    checks.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Mass conservation < 1e-6: {mass_rel_err:.2e}")

    c3 = h_positive
    checks.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] Height positive everywhere: min h = {h_min:.4f}")

    c4 = u_max < 100.0
    checks.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] Max wind speed < 100 m/s: {u_max:.2f}")

    results["pass"] = all(checks)
    print(f"\n  Overall: {'PASS' if results['pass'] else 'FAIL'}")

    return results


# =============================================================================
# Test Case 6: Rossby-Haurwitz Wave (Wave number 4)
# =============================================================================

def williamson6_ic(cdgrid):
    """Create Williamson test case 6 initial condition.

    Rossby-Haurwitz wave number 4.
    Reference: Williamson et al. (1992), eqs. 137-139.
    All arrays promoted to float64 for conservation accuracy.
    """
    R = cdgrid.radius
    n_wave = 4
    K = 7.848e-6
    h_0 = 8000.0

    lat_c = cdgrid.base.lat.astype(jnp.float64)
    lon_c = cdgrid.base.lon.astype(jnp.float64)

    omega_rh = K
    A = 0.5 * omega_rh * (2.0 * omega + omega_rh)
    B = (2.0 * (omega + omega_rh) * omega_rh) / ((n_wave + 1) * (n_wave + 2))
    C_coeff = (0.5 * omega_rh * omega_rh * (n_wave ** 2 + 2 * n_wave + 2)
               / ((n_wave + 1) * (n_wave + 2)))

    cos_lat = jnp.cos(lat_c)
    sin_lat = jnp.sin(lat_c)
    cos_n_lon = jnp.cos(n_wave * lon_c)

    h = (h_0
         + (R ** 2 / g) * (
             A * cos_lat ** 2
             + B * cos_lat ** n_wave * (
                 (n_wave + 1) * cos_lat ** 2
                 + (2.0 * n_wave ** 2 - n_wave - 2.0)
                 - 2.0 * n_wave ** 2 / (cos_lat ** 2 + 1e-30)
             ) * cos_n_lon
             + C_coeff * cos_lat ** (2 * n_wave) * (
                 (n_wave + 1) * cos_lat ** 2 - (n_wave + 2)
             ) * jnp.cos(2.0 * n_wave * lon_c)
         ))

    lat_d = cdgrid.lat_corner.astype(jnp.float64)
    lon_d = cdgrid.lon_corner.astype(jnp.float64)
    cos_lat_d = jnp.cos(lat_d)
    sin_lat_d = jnp.sin(lat_d)

    u_geo = (R * omega_rh * cos_lat_d
             + R * K * cos_lat_d ** (n_wave - 1)
             * (n_wave * sin_lat_d ** 2 - cos_lat_d ** 2)
             * jnp.cos(n_wave * lon_d))

    v_geo = (-R * K * n_wave * cos_lat_d ** (n_wave - 1)
             * sin_lat_d * jnp.sin(n_wave * lon_d))

    ca = cdgrid.cos_angle_corner.astype(jnp.float64)
    sa = cdgrid.sin_angle_corner.astype(jnp.float64)
    u_d = ca * u_geo + sa * v_geo
    v_d = -sa * u_geo + ca * v_geo

    h_s = jnp.zeros_like(h)
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def run_test_case_6(n_res=16, n_days=14, dt=600.0):
    """Run Williamson Test Case 6."""
    print(f"\n{'='*70}")
    print(f"WILLIAMSON TEST CASE 6: Rossby-Haurwitz Wave (number 4)")
    print(f"  Resolution: C{n_res}, Integration: {n_days} days, dt={dt:.0f}s")
    print(f"{'='*70}")

    grid = create_cubed_sphere(n_res)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    dx_min = float(jnp.min(grid.dx))
    print(f"  dx_min = {dx_min/1000:.1f} km")

    config = CDGridShallowWaterConfig(
        A_h=0.0,
        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
        div_damp=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CDGridShallowWaterModel(grid, config)

    state0 = williamson6_ic(cdgrid)
    model.set_initial_mass(state0)

    area = cdgrid.base.area
    total_area = float(jnp.sum(area))
    mass_0 = float(jnp.sum(state0.h * area))
    energy_0 = total_energy(state0, area)

    h0_min = float(jnp.min(state0.h))
    h0_max = float(jnp.max(state0.h))
    h0_range = h0_max - h0_min
    print(f"  h range: [{h0_min:.2f}, {h0_max:.2f}] m (range={h0_range:.2f} m)")
    print(f"  Initial mass = {mass_0:.6e}")
    print(f"  Initial energy = {energy_0:.6e}")

    n_steps = int(n_days * 86400 / dt)
    print(f"\n  Integrating {n_steps} steps...")
    t0 = time.time()
    state_final, stable = integrate(model, state0, dt, n_steps,
                                     print_every=max(1, n_steps // 10))
    wall_time = time.time() - t0
    print(f"  Integration complete in {wall_time:.1f}s")

    print(f"\n  --- Diagnostics after {n_days} days ---")

    all_finite = bool(jnp.all(jnp.isfinite(state_final.h)) and
                       jnp.all(jnp.isfinite(state_final.u_d)) and
                       jnp.all(jnp.isfinite(state_final.v_d)))
    print(f"  Stable (all finite): {all_finite}")

    if not all_finite:
        print("  *** MODEL BLEW UP ***")
        return {"stable": False, "pass": False}

    mass_final = float(jnp.sum(state_final.h * area))
    mass_rel_err = abs(mass_final - mass_0) / abs(mass_0)
    print(f"  Mass conservation: relative error = {mass_rel_err:.2e}")

    energy_final = total_energy(state_final, area)
    energy_rel_change = abs(energy_final - energy_0) / abs(energy_0)
    energy_sign = "+" if energy_final > energy_0 else "-"
    print(f"  Energy: relative change = {energy_sign}{energy_rel_change:.4e}")

    h_min = float(jnp.min(state_final.h))
    h_max = float(jnp.max(state_final.h))
    h_final_range = h_max - h_min
    print(f"  Final height range: [{h_min:.2f}, {h_max:.2f}] m (range={h_final_range:.2f} m)")
    print(f"  Height range ratio (final/initial): {h_final_range / max(h0_range, 1.0):.4f}")

    # Cube-edge artifact: compare face-boundary vs interior variance
    # Since there's no exact solution, use deviation from face-mean
    h_anom = state_final.h - jnp.mean(state_final.h)
    artifact_ratio = cube_edge_artifact_ratio(h_anom, n_res)
    print(f"  Cube-edge artifact ratio (anomaly): {artifact_ratio:.3f}")

    # Max wind speed
    u_max = float(jnp.max(jnp.sqrt(state_final.u_d ** 2 + state_final.v_d ** 2)))
    print(f"  Max wind speed: {u_max:.2f} m/s")

    results = {
        "stable": stable and all_finite,
        "mass_rel_err": mass_rel_err,
        "energy_rel_change": energy_rel_change,
        "h_min": h_min,
        "h_max": h_max,
        "h_range_ratio": h_final_range / max(h0_range, 1.0),
        "artifact_ratio": artifact_ratio,
        "u_max": u_max,
    }

    print(f"\n  --- PASS/FAIL Criteria ---")
    checks = []

    c1 = stable and all_finite
    checks.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] Model stable (no blowup)")

    c2 = mass_rel_err < 1e-6
    checks.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Mass conservation < 1e-6: {mass_rel_err:.2e}")

    c3 = energy_rel_change < 0.05
    checks.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] Energy drift < 5%: {energy_rel_change:.4e}")

    # Height range evolves nonlinearly over 14 days at C16
    c4 = 0.3 < (h_final_range / max(h0_range, 1.0)) < 5.0
    checks.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] Height range bounded (0.3-5x): "
          f"{h_final_range / max(h0_range, 1.0):.4f}")

    results["pass"] = all(checks)
    print(f"\n  Overall: {'PASS' if results['pass'] else 'FAIL'}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("WILLIAMSON SHALLOW WATER TEST CASES — Comprehensive Diagnostics")
    print("FV3 C-D Grid Cubed-Sphere Dynamical Core")
    print("=" * 70)

    all_results = {}

    # Test Case 2 at C16 (5 days)
    all_results["TC2_C16"] = run_test_case_2(n_res=16, n_days=5, dt=600.0)

    # Test Case 2 at C24 (5 days)
    all_results["TC2_C24"] = run_test_case_2(n_res=24, n_days=5, dt=400.0)

    # Test Case 5 at C16 (15 days)
    all_results["TC5_C16"] = run_test_case_5(n_res=16, n_days=15, dt=600.0)

    # Test Case 6 at C16 (14 days)
    all_results["TC6_C16"] = run_test_case_6(n_res=16, n_days=14, dt=600.0)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for name, res in all_results.items():
        status = "PASS" if res.get("pass", False) else "FAIL"
        if not res.get("pass", False):
            all_pass = False
        print(f"  {name}: {status}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
