#!/usr/bin/env python
"""Held-Suarez realism analysis at 2.5° resolution.

Runs both spectral (T42) and cubed-sphere (C48) configurations with
sigma and hybrid-sigma-pressure coordinates, then compares zonal-mean
profiles against Held & Suarez (1994) reference climatology.

Expected climate (Held & Suarez 1994, Fig 1-3):
- Subtropical jets: ~25-30 m/s at ~250 hPa, ~30° lat
- Surface westerlies: ~5-8 m/s at ~45° lat
- Surface easterlies: ~3-5 m/s in tropics
- Equatorial T_surface: ~295 K, polar T_surface: ~230-240 K
- Tropopause: ~200 K at ~200 hPa (tropics), ~220 K at ~300 hPa (poles)
- Stratosphere: isothermal ~200 K
- Temperature lapse rate: ~6.5 K/km in troposphere
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.held_suarez import (
    held_suarez_equilibrium_temperature,
    held_suarez_forcing,
    held_suarez_forcing_spectral,
    held_suarez_init,
    K_A, K_S, K_F, SIGMA_B, DELTA_T_Y, DELTA_THETA_Z, T_MIN, P_0,
)


def run_spectral(n_max, nlev, coord_type, dt, n_days, diag_every_days=10):
    """Run spectral PE Held-Suarez and return diagnostics."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel,
        isothermal_rest_state_spectral, spectral_pe_to_grid,
    )

    grid = create_gaussian_grid(n_max)

    if coord_type == "sigma":
        sigma = create_sigma_coordinate(nlev)
    else:
        sigma = standard_hybrid_levels(nlev)

    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0, p_s_init=1e5)

    a = float(grid.radius)
    eig_max = n_max * (n_max + 1) / (a * a)
    config = SpectralPEConfig(
        hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max**2),
        hyperdiff_order=2,
        semi_implicit=True,
        time_integrator="leapfrog_si",
        spectral_filter_order=8,
        spectral_filter_strength=0.01,
        sponge_sigma=0.1,
        sponge_tau=3600.0,  # 1-hour e-folding at top
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)

    n_steps = int(n_days * 86400.0 / dt)
    diag_interval = max(1, int(diag_every_days * 86400.0 / dt))

    print(f"  T{n_max}/L{nlev} ({coord_type}): {n_steps} steps, dt={dt}s")

    # Warmup
    state = model.step_with_physics(state, dt, held_suarez_forcing_spectral)
    jax.block_until_ready(state.vor_hat.data)

    t0 = time.time()
    diag_records = []
    for step in range(1, n_steps + 1):
        if step > 1:
            state = model.step_with_physics(state, dt, held_suarez_forcing_spectral)

        if step % diag_interval == 0 or step == n_steps:
            fields = spectral_pe_to_grid(state, grid, sigma)
            u = np.asarray(fields['u'])
            v = np.asarray(fields['v'])
            T = np.asarray(fields['T'])
            p_s = np.asarray(fields['p_s'])
            day = step * dt / 86400.0
            wind = np.sqrt(u**2 + v**2)
            diag_records.append({
                'day': day,
                'u': u, 'v': v, 'T': T, 'p_s': p_s,
                'max_wind': float(np.max(wind)),
                'mean_T': float(np.mean(T)),
            })
            print(f"    day {day:.0f}: max|v|={float(np.max(wind)):.1f} m/s, "
                  f"<T>={float(np.mean(T)):.1f} K")

        if step % max(1, n_steps // 10) == 0:
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                print(f"    *** BLOWUP at step {step} ***")
                break

    wall = time.time() - t0
    print(f"    Done in {wall:.0f}s ({n_steps / wall:.0f} steps/s)")

    # Extract final zonal means
    lat_1d = np.asarray(grid.lat)  # (n_lat,)
    weights = np.asarray(grid.weights)  # (n_lat,)
    final = diag_records[-1]
    u_zonal = np.mean(final['u'], axis=1)  # (n_lat, nlev)
    T_zonal = np.mean(final['T'], axis=1)  # (n_lat, nlev)

    # Get sigma/pressure levels
    if coord_type == "sigma":
        sigma_full = np.asarray(sigma.sigma_full)
    else:
        sigma_full = np.asarray(sigma.B_full)  # approximate for hybrid

    return {
        'lat': lat_1d,
        'sigma': sigma_full,
        'u_zonal': u_zonal,
        'T_zonal': T_zonal,
        'p_s_mean': float(np.mean(final['p_s'])),
        'diag_records': diag_records,
        'wall_time': wall,
        'coord_type': coord_type,
        'grid_type': f'T{n_max}',
    }


def run_cubed_sphere(n_grid, nlev, coord_type, dt, n_days, diag_every_days=10):
    """Run cubed-sphere FV Held-Suarez and return diagnostics."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel as FVPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig as FVPrimitiveEquationConfig,
    )
    from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
    from legoesm.core.cfl import cfl_check_and_adjust

    grid = create_cubed_sphere(n_grid)

    if coord_type == "sigma":
        sigma = create_sigma_coordinate(nlev)
    else:
        sigma = standard_hybrid_levels(nlev)

    # CFL check
    dt = cfl_check_and_adjust(dt, n_grid, model_type="primitive_eq",
                               max_wind=60.0, gravity_wave_speed=0.0)

    nu2, nu4 = default_div_damp_coeffs(grid, dt=dt)
    ref_coeff = 5e16
    hyperdiff = ref_coeff * (48 / n_grid) ** 4

    config = FVPrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        div_damp_2=nu2,
        div_damp_4=nu4,
        use_conservation_fixer=True,
        fix_mass=True,
        time_integrator="ssp45",
        use_limiter=True,
    )
    model = FVPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)

    n_steps = int(n_days * 86400.0 / dt)
    diag_interval = max(1, int(diag_every_days * 86400.0 / dt))

    print(f"  C{n_grid}/L{nlev} ({coord_type}): {n_steps} steps, dt={dt}s")

    # Warmup
    state = model.step_with_physics(state, dt, held_suarez_forcing)
    jax.block_until_ready(state.u.data)

    t0 = time.time()
    diag_records = []
    for step in range(1, n_steps + 1):
        if step > 1:
            state = model.step_with_physics(state, dt, held_suarez_forcing)

        if step % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 500:
                day = (step + 1) * dt / 86400.0
                print(f"    *** BLOWUP at day {day:.1f}, u_max={u_max:.1f} ***")
                break

        if step % diag_interval == 0 or step == n_steps:
            day = step * dt / 86400.0
            u = np.asarray(state.u.data)
            v = np.asarray(state.v.data)
            T = np.asarray(state.T.data)
            p_s = np.asarray(state.p_s.data)
            wind = np.sqrt(u**2 + v**2)
            diag_records.append({
                'day': day,
                'u': u, 'v': v, 'T': T, 'p_s': p_s,
                'max_wind': float(np.max(wind)),
                'mean_T': float(np.mean(T)),
            })
            print(f"    day {day:.0f}: max|v|={float(np.max(wind)):.1f} m/s, "
                  f"<T>={float(np.mean(T)):.1f} K")

    wall = time.time() - t0
    print(f"    Done in {wall:.0f}s ({n_steps / wall:.0f} steps/s)")

    # Compute zonal means on cubed-sphere
    lat_np = np.asarray(grid.lat).flatten()
    area_np = np.asarray(grid.area).flatten()
    final = diag_records[-1]

    n_bins = 64
    lat_bins = np.linspace(-np.pi / 2, np.pi / 2, n_bins + 1)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    u_zonal = np.full((n_bins, nlev), np.nan)
    T_zonal = np.full((n_bins, nlev), np.nan)

    u_flat = final['u'].reshape(-1, nlev)
    T_flat = final['T'].reshape(-1, nlev)

    for i in range(n_bins):
        mask = (lat_np >= lat_bins[i]) & (lat_np < lat_bins[i + 1])
        if np.sum(mask) > 0:
            w = area_np[mask]
            w = w / w.sum()
            u_zonal[i] = np.sum(w[:, None] * u_flat[mask], axis=0)
            T_zonal[i] = np.sum(w[:, None] * T_flat[mask], axis=0)

    # Interpolate any empty bins
    for k in range(nlev):
        valid = ~np.isnan(T_zonal[:, k])
        if valid.any() and not valid.all():
            T_zonal[:, k] = np.interp(lat_centers, lat_centers[valid], T_zonal[valid, k])
            u_zonal[:, k] = np.interp(lat_centers, lat_centers[valid], u_zonal[valid, k])

    if coord_type == "sigma":
        sigma_full = np.asarray(sigma.sigma_full)
    else:
        sigma_full = np.asarray(sigma.B_full)

    return {
        'lat': lat_centers,
        'sigma': sigma_full,
        'u_zonal': u_zonal,
        'T_zonal': T_zonal,
        'p_s_mean': float(np.mean(final['p_s'])),
        'diag_records': diag_records,
        'wall_time': wall,
        'coord_type': coord_type,
        'grid_type': f'C{n_grid}',
    }


def run_icosahedral(subdiv_level, nlev, coord_type, dt, n_days, diag_every_days=10):
    """Run MPAS icosahedral PE Held-Suarez and return diagnostics."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez_mpas import (
        held_suarez_forcing_mpas, held_suarez_init_mpas,
    )

    mesh = create_voronoi_mesh(subdiv_level)
    nCells = mesh.nCells
    dx_km = np.sqrt(4 * np.pi * mesh.radius**2 / nCells) / 1e3

    if coord_type == "sigma":
        sigma = create_sigma_coordinate(nlev)
    else:
        sigma = standard_hybrid_levels(nlev)

    dx_mean = np.sqrt(4 * np.pi * mesh.radius**2 / nCells)
    nu_del4 = dx_mean**4 / (48.0 * 3600.0)

    config = MPASPrimitiveEquationConfig(nu_del4=nu_del4, fix_mass=True)
    model = MPASPrimitiveEquationModel(mesh, sigma, config)
    state = held_suarez_init_mpas(mesh, sigma)

    n_steps = int(n_days * 86400.0 / dt)
    diag_interval = max(1, int(diag_every_days * 86400.0 / dt))

    print(f"  MPAS level {subdiv_level} ({nCells} cells, ~{dx_km:.0f} km)/L{nlev} ({coord_type}): "
          f"{n_steps} steps, dt={dt}s")

    state = model.step(state, dt, held_suarez_forcing_mpas)
    jax.block_until_ready(state.T.data)

    t0 = time.time()
    diag_records = []
    for step in range(1, n_steps + 1):
        if step > 1:
            state = model.step(state, dt, held_suarez_forcing_mpas)

        if step % 100 == 0:
            if not jnp.all(jnp.isfinite(state.T.data)):
                day = (step + 1) * dt / 86400.0
                print(f"    *** BLOWUP at day {day:.1f} ***")
                break

        if step % diag_interval == 0 or step == n_steps:
            day = step * dt / 86400.0
            T = np.asarray(state.T.data)
            u = np.asarray(state.u.data)
            p_s = np.asarray(state.p_s.data)
            diag_records.append({
                'day': day, 'T': T, 'u': u, 'p_s': p_s,
                'max_wind': float(np.max(np.abs(u))),
                'mean_T': float(np.mean(T)),
            })
            print(f"    day {day:.0f}: max|u|={float(np.max(np.abs(u))):.1f} m/s, "
                  f"<T>={float(np.mean(T)):.1f} K")

    wall = time.time() - t0
    print(f"    Done in {wall:.0f}s ({n_steps / wall:.0f} steps/s)")

    # Zonal means on unstructured grid
    lat_cells = np.asarray(mesh.grid_lat)
    area_cells = np.asarray(mesh.grid_area)
    lat_deg_cells = np.degrees(lat_cells)
    final = diag_records[-1]

    n_bins = max(36, nCells // 100)
    lat_bins = np.linspace(-90, 90, n_bins + 1)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    T_zonal = np.full((n_bins, nlev), np.nan)

    for i in range(n_bins):
        mask = (lat_deg_cells >= lat_bins[i]) & (lat_deg_cells < lat_bins[i + 1])
        if np.sum(mask) > 0:
            w = area_cells[mask]
            w = w / w.sum()
            T_zonal[i] = np.sum(w[:, None] * final['T'][mask], axis=0)

    if coord_type == "sigma":
        sigma_full = np.asarray(sigma.sigma_full)
    else:
        sigma_full = np.asarray(sigma.B_full)

    # No u_zonal for MPAS (edge-normal velocity, not u/v decomposition)
    # Use T_zonal for temperature analysis
    return {
        'lat': np.radians(lat_centers),
        'sigma': sigma_full,
        'u_zonal': np.zeros_like(T_zonal),  # placeholder — MPAS has edge-normal u
        'T_zonal': T_zonal,
        'p_s_mean': float(np.mean(final['p_s'])),
        'diag_records': diag_records,
        'wall_time': wall,
        'coord_type': coord_type,
        'grid_type': f'MPAS-L{subdiv_level}({nCells})',
    }


def check_realism(result, label):
    """Check profiles against Held-Suarez reference climate."""
    lat_deg = np.degrees(result['lat'])
    u_z = result['u_zonal']
    T_z = result['T_zonal']
    sigma = result['sigma']
    p_hPa = sigma * 1000  # approximate

    issues = []

    # --- Zonal wind checks ---
    # Find jet maximum
    jet_idx = np.unravel_index(np.nanargmax(u_z), u_z.shape)
    jet_lat = lat_deg[jet_idx[0]]
    jet_p = p_hPa[jet_idx[1]]
    jet_speed = u_z[jet_idx]

    print(f"\n  [{label}] Jet: {jet_speed:.1f} m/s at {jet_lat:.0f}°, {jet_p:.0f} hPa")

    if jet_speed < 15:
        issues.append(f"Jet too weak: {jet_speed:.1f} m/s (expected ~25-30 m/s)")
    elif jet_speed > 45:
        issues.append(f"Jet too strong: {jet_speed:.1f} m/s (expected ~25-30 m/s)")

    if abs(jet_lat) < 15 or abs(jet_lat) > 50:
        issues.append(f"Jet at wrong latitude: {jet_lat:.0f}° (expected ~30°)")

    if jet_p > 500 or jet_p < 50:
        issues.append(f"Jet at wrong level: {jet_p:.0f} hPa (expected ~200-300 hPa)")

    # --- Surface wind structure ---
    sfc_idx = -1  # lowest level
    u_sfc = u_z[:, sfc_idx]

    # Check for surface westerlies at midlatitudes
    midlat_mask = (np.abs(lat_deg) > 30) & (np.abs(lat_deg) < 60)
    if midlat_mask.any():
        midlat_u = np.nanmean(np.abs(u_sfc[midlat_mask]))
        if midlat_u < 1:
            issues.append(f"Surface midlat winds too weak: {midlat_u:.1f} m/s (expected ~5-8 m/s)")

    # --- Temperature checks ---
    # Equatorial surface temperature
    eq_mask = np.abs(lat_deg) < 10
    if eq_mask.any():
        T_eq_sfc = np.nanmean(T_z[eq_mask, sfc_idx])
        print(f"  [{label}] T_eq_surface: {T_eq_sfc:.1f} K (expected ~290-300 K)")
        if T_eq_sfc < 275 or T_eq_sfc > 310:
            issues.append(f"Equatorial surface T: {T_eq_sfc:.1f} K (expected ~290-300 K)")

    # Polar surface temperature
    pole_mask = np.abs(lat_deg) > 70
    if pole_mask.any():
        T_pole_sfc = np.nanmean(T_z[pole_mask, sfc_idx])
        print(f"  [{label}] T_pole_surface: {T_pole_sfc:.1f} K (expected ~230-250 K)")
        if T_pole_sfc < 200 or T_pole_sfc > 265:
            issues.append(f"Polar surface T: {T_pole_sfc:.1f} K (expected ~230-250 K)")

    # Stratosphere temperature (should be ~200 K)
    top_idx = 0
    T_strat = np.nanmean(T_z[:, top_idx])
    print(f"  [{label}] T_stratosphere: {T_strat:.1f} K (expected ~200 K)")
    if T_strat < 180 or T_strat > 220:
        issues.append(f"Stratosphere T: {T_strat:.1f} K (expected ~200 K)")

    # Meridional temperature gradient at surface
    if eq_mask.any() and pole_mask.any():
        dT_merid = T_eq_sfc - T_pole_sfc
        print(f"  [{label}] Meridional dT: {dT_merid:.1f} K (expected ~40-60 K)")
        if dT_merid < 20:
            issues.append(f"Meridional dT too small: {dT_merid:.1f} K (expected ~40-60 K)")
        elif dT_merid > 80:
            issues.append(f"Meridional dT too large: {dT_merid:.1f} K (expected ~40-60 K)")

    # Global mean temperature
    mean_T = np.nanmean(T_z)
    print(f"  [{label}] Global mean T: {mean_T:.1f} K (expected ~250-270 K)")

    # Hemispheric symmetry (NH ≈ SH)
    nh_mask = lat_deg > 10
    sh_mask = lat_deg < -10
    if nh_mask.any() and sh_mask.any():
        T_nh = np.nanmean(T_z[nh_mask])
        T_sh = np.nanmean(T_z[sh_mask])
        asym = abs(T_nh - T_sh)
        if asym > 5:
            issues.append(f"Hemispheric T asymmetry: {asym:.1f} K (should be <5 K)")

    # Mean surface pressure
    print(f"  [{label}] Mean p_s: {result['p_s_mean']:.0f} Pa (expected ~100000 Pa)")

    if issues:
        print(f"  [{label}] ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  [{label}] All checks PASSED")

    return issues


def run_latlon(n_lat, nlev, coord_type, dt, n_days, diag_every_days=10):
    """Run lat-lon PE Held-Suarez and return diagnostics."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
        LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez_latlon import (
        held_suarez_forcing_latlon, held_suarez_init_latlon,
    )

    n_lon = 2 * n_lat
    grid = create_latlon_grid(n_lat, n_lon)

    if coord_type == "sigma":
        sigma = create_sigma_coordinate(nlev)
    else:
        sigma = standard_hybrid_levels(nlev)

    # Scale hyperdiffusion
    ref_coeff = 2e16
    hyperdiff = max(ref_coeff * (64 / n_lat) ** 4, 1e16)

    config = LatLonPrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = LatLonPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init_latlon(grid, sigma)

    n_steps = int(n_days * 86400.0 / dt)
    diag_interval = max(1, int(diag_every_days * 86400.0 / dt))

    print(f"  {n_lat}x{n_lon}/L{nlev} ({coord_type}): {n_steps} steps, dt={dt}s")

    # Warmup
    state = model.step_with_physics(state, dt, held_suarez_forcing_latlon)
    jax.block_until_ready(state.u.data)

    t0 = time.time()
    diag_records = []
    for step in range(1, n_steps + 1):
        if step > 1:
            state = model.step_with_physics(state, dt, held_suarez_forcing_latlon)

        if step % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 500:
                day = (step + 1) * dt / 86400.0
                print(f"    *** BLOWUP at day {day:.1f}, u_max={u_max:.1f} ***")
                break

        if step % diag_interval == 0 or step == n_steps:
            day = step * dt / 86400.0
            u = np.asarray(state.u.data)
            v = np.asarray(state.v.data)
            T = np.asarray(state.T.data)
            p_s = np.asarray(state.p_s.data)
            wind = np.sqrt(u**2 + v**2)
            diag_records.append({
                'day': day,
                'u': u, 'v': v, 'T': T, 'p_s': p_s,
                'max_wind': float(np.max(wind)),
                'mean_T': float(np.mean(T)),
            })
            print(f"    day {day:.0f}: max|v|={float(np.max(wind)):.1f} m/s, "
                  f"<T>={float(np.mean(T)):.1f} K")

    wall = time.time() - t0
    print(f"    Done in {wall:.0f}s ({n_steps / wall:.0f} steps/s)")

    # Zonal means on lat-lon grid (simple longitude average)
    lat_1d = np.asarray(grid.lat)  # (n_lat,)
    final = diag_records[-1]
    u_zonal = np.mean(final['u'], axis=1)  # (n_lat, nlev)
    T_zonal = np.mean(final['T'], axis=1)  # (n_lat, nlev)

    if coord_type == "sigma":
        sigma_full = np.asarray(sigma.sigma_full)
    else:
        sigma_full = np.asarray(sigma.B_full)

    return {
        'lat': lat_1d,
        'sigma': sigma_full,
        'u_zonal': u_zonal,
        'T_zonal': T_zonal,
        'p_s_mean': float(np.mean(final['p_s'])),
        'diag_records': diag_records,
        'wall_time': wall,
        'coord_type': coord_type,
        'grid_type': f'{n_lat}x{n_lon}',
    }


def check_equilibrium_temperature():
    """Verify T_eq formula matches Held & Suarez (1994) Table 1."""
    print("\n--- Equilibrium Temperature Verification ---")

    lat = jnp.array([0.0, jnp.pi / 6, jnp.pi / 4, jnp.pi / 3, jnp.pi / 2])
    lat_deg = np.degrees(np.asarray(lat))
    p = jnp.array([100e2, 200e2, 500e2, 850e2, 1000e2])  # Pa

    for i, la in enumerate(lat):
        for j, pr in enumerate(p):
            T_eq = float(held_suarez_equilibrium_temperature(la, pr))
            sigma = float(pr / 1e5)
            print(f"  lat={lat_deg[i]:5.1f}°, p={float(pr)/100:7.0f} hPa (σ={sigma:.3f}): "
                  f"T_eq = {T_eq:.1f} K")

    # Check specific reference values from HS94
    # At equator, surface: T_eq ≈ 315 * (1000/1000)^0.286 = 315 K -> capped behavior
    T_eq_eq_sfc = float(held_suarez_equilibrium_temperature(jnp.array(0.0), jnp.array(1e5)))
    T_eq_pole_sfc = float(held_suarez_equilibrium_temperature(jnp.array(jnp.pi / 2), jnp.array(1e5)))
    T_eq_eq_top = float(held_suarez_equilibrium_temperature(jnp.array(0.0), jnp.array(100e2)))

    print(f"\n  Reference checks:")
    print(f"    T_eq(eq, sfc)  = {T_eq_eq_sfc:.1f} K (should be 315.0 K)")
    print(f"    T_eq(pole, sfc) = {T_eq_pole_sfc:.1f} K (should be 255.0 K)")
    print(f"    T_eq(eq, 100hPa) = {T_eq_eq_top:.1f} K (should be 200.0 K, capped)")

    issues = []
    if abs(T_eq_eq_sfc - 315.0) > 0.1:
        issues.append(f"T_eq(eq,sfc) = {T_eq_eq_sfc:.1f}, expected 315.0")
    if abs(T_eq_pole_sfc - 255.0) > 0.1:
        issues.append(f"T_eq(pole,sfc) = {T_eq_pole_sfc:.1f}, expected 255.0")
    if T_eq_eq_top > 200.1:
        issues.append(f"T_eq(eq,100hPa) = {T_eq_eq_top:.1f}, should be capped at 200.0")

    return issues


def check_forcing_coefficients():
    """Verify k_T and k_v match HS94 Table 1."""
    print("\n--- Forcing Coefficient Verification ---")

    print(f"  k_a = {K_A:.6e} s⁻¹ (1/40 day = {1/(40*86400):.6e})")
    print(f"  k_s = {K_S:.6e} s⁻¹ (1/4 day = {1/(4*86400):.6e})")
    print(f"  k_f = {K_F:.6e} s⁻¹ (1/1 day = {1/(1*86400):.6e})")
    print(f"  σ_b = {SIGMA_B}")
    print(f"  ΔT_y = {DELTA_T_Y} K")
    print(f"  Δθ_z = {DELTA_THETA_Z} K")
    print(f"  T_min = {T_MIN} K")
    print(f"  P_0 = {P_0:.0f} Pa")

    issues = []
    if abs(K_A - 1 / (40 * 86400)) > 1e-12:
        issues.append(f"k_a wrong: {K_A}")
    if abs(K_S - 1 / (4 * 86400)) > 1e-12:
        issues.append(f"k_s wrong: {K_S}")
    if abs(K_F - 1 / (1 * 86400)) > 1e-12:
        issues.append(f"k_f wrong: {K_F}")
    if SIGMA_B != 0.7:
        issues.append(f"σ_b wrong: {SIGMA_B}")

    # Check k_T at specific points
    # At σ=1.0, equator: k_T = k_a + (k_s - k_a) * (1-0.7)/(1-0.7) * cos⁴(0) = k_s
    # At σ=0.5 (above BL): k_T = k_a (no BL contribution)
    # At σ=1.0, pole: k_T = k_a + (k_s - k_a) * 1 * cos⁴(90°) = k_a
    sigma_1 = 1.0
    sigma_05 = 0.5
    sf_1 = max(0, (sigma_1 - SIGMA_B) / (1 - SIGMA_B))
    sf_05 = max(0, (sigma_05 - SIGMA_B) / (1 - SIGMA_B))

    k_T_eq_sfc = K_A + (K_S - K_A) * sf_1 * np.cos(0) ** 4
    k_T_eq_mid = K_A + (K_S - K_A) * sf_05 * np.cos(0) ** 4
    k_T_pole_sfc = K_A + (K_S - K_A) * sf_1 * np.cos(np.pi / 2) ** 4

    print(f"\n  k_T(σ=1, eq) = {k_T_eq_sfc:.6e} (should = k_s = {K_S:.6e})")
    print(f"  k_T(σ=0.5, eq) = {k_T_eq_mid:.6e} (should = k_a = {K_A:.6e})")
    print(f"  k_T(σ=1, pole) = {k_T_pole_sfc:.6e} (should = k_a = {K_A:.6e})")

    if abs(k_T_eq_sfc - K_S) > 1e-12:
        issues.append("k_T(σ=1,eq) should equal k_s")
    if abs(k_T_eq_mid - K_A) > 1e-12:
        issues.append("k_T(σ=0.5,eq) should equal k_a")
    if abs(k_T_pole_sfc - K_A) > 1e-12:
        issues.append("k_T(σ=1,pole) should equal k_a")

    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified Held-Suarez driver and profile analyzer.",
        epilog="""
Examples:
  # Run all grid types (spectral, cubed-sphere, lat-lon) with defaults:
  python analyze_held_suarez.py

  # Run only spectral T42 with sigma:
  python analyze_held_suarez.py --grid spectral --resolution 42 --coord sigma

  # Run cubed-sphere C48 with hybrid sigma-pressure:
  python analyze_held_suarez.py --grid cubed_sphere --resolution 48 --coord hybrid

  # Run lat-lon 72x144:
  python analyze_held_suarez.py --grid latlon --resolution 72 --coord sigma

  # Quick test (30 days):
  python analyze_held_suarez.py --days 30 --grid spectral --resolution 21
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--grid", type=str, default="all",
                        choices=["all", "spectral", "cubed_sphere", "latlon", "icosahedral"],
                        help="Grid type (default: all)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Resolution (T-number for spectral, C-number for cubed-sphere, n_lat for latlon)")
    parser.add_argument("--levels", type=int, default=20,
                        help="Number of vertical levels (default: 20)")
    parser.add_argument("--coord", type=str, default="sigma",
                        choices=["sigma", "hybrid"],
                        help="Vertical coordinate type (default: sigma)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step in seconds (default: auto)")
    parser.add_argument("--days", type=int, default=200,
                        help="Integration duration in days (default: 200)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip forcing formula verification")
    args = parser.parse_args()

    print("=" * 70)
    print("Held-Suarez Realism Analysis")
    print("=" * 70)

    all_issues = {}

    # Phase 1: Verify forcing formulas
    if not args.skip_verify:
        teq_issues = check_equilibrium_temperature()
        coeff_issues = check_forcing_coefficients()
        if teq_issues:
            all_issues['T_eq formula'] = teq_issues
        if coeff_issues:
            all_issues['Coefficients'] = coeff_issues

    # Phase 2: Build configuration list
    n_days = args.days
    nlev = args.levels

    if args.grid == "all":
        configs = [
            ("spectral", 42, "sigma", 300.0),
            ("spectral", 42, "hybrid", 300.0),
            ("cubed_sphere", 48, "sigma", 600.0),
            ("latlon", 72, "sigma", 600.0),
            ("icosahedral", 4, "sigma", 300.0),  # level 4 = 2562 cells (~4°)
        ]
    else:
        default_res = {"spectral": 42, "cubed_sphere": 48, "latlon": 72, "icosahedral": 4}
        default_dt = {"spectral": 300.0, "cubed_sphere": 600.0, "latlon": 600.0, "icosahedral": 300.0}
        res = args.resolution or default_res[args.grid]
        dt = args.dt or default_dt[args.grid]
        configs = [(args.grid, res, args.coord, dt)]

    results = {}
    for grid_type, res, coord, dt in configs:
        label = f"{grid_type}_{coord}"
        print(f"\n{'='*70}")
        print(f"Running: {label} (res={res}, L{nlev}, dt={dt}s, {n_days} days)")
        print(f"{'='*70}")

        try:
            if grid_type == "spectral":
                result = run_spectral(res, nlev, coord, dt, n_days)
            elif grid_type == "latlon":
                result = run_latlon(res, nlev, coord, dt, n_days)
            elif grid_type == "icosahedral":
                result = run_icosahedral(res, nlev, coord, dt, n_days)
            else:
                result = run_cubed_sphere(res, nlev, coord, dt, n_days)
            results[label] = result
            issues = check_realism(result, label)
            if issues:
                all_issues[label] = issues
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_issues[label] = [f"Runtime error: {e}"]

    # Phase 3: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not all_issues:
        print("All checks passed! Profiles are realistic.")
    else:
        print(f"Found issues in {len(all_issues)} categories:")
        for cat, issues in all_issues.items():
            print(f"\n  {cat}:")
            for issue in issues:
                print(f"    - {issue}")

    # Phase 4: Cross-compare spectral vs cubed-sphere
    if 'spectral_sigma' in results and 'cubed_sphere_sigma' in results:
        print("\n--- Cross-comparison: Spectral vs Cubed-Sphere (sigma) ---")
        sp = results['spectral_sigma']
        cs = results['cubed_sphere_sigma']

        # Compare jet speeds
        sp_jet = np.nanmax(sp['u_zonal'])
        cs_jet = np.nanmax(cs['u_zonal'])
        print(f"  Jet speed: spectral={sp_jet:.1f} m/s, cubed-sphere={cs_jet:.1f} m/s")
        print(f"  Difference: {abs(sp_jet - cs_jet):.1f} m/s")

        # Compare global mean T
        sp_meanT = np.nanmean(sp['T_zonal'])
        cs_meanT = np.nanmean(cs['T_zonal'])
        print(f"  Mean T: spectral={sp_meanT:.1f} K, cubed-sphere={cs_meanT:.1f} K")

    # Phase 5: Compare sigma vs hybrid
    if 'spectral_sigma' in results and 'spectral_hybrid' in results:
        print("\n--- Cross-comparison: Sigma vs Hybrid (spectral T42) ---")
        sig = results['spectral_sigma']
        hyb = results['spectral_hybrid']

        sig_jet = np.nanmax(sig['u_zonal'])
        hyb_jet = np.nanmax(hyb['u_zonal'])
        print(f"  Jet speed: sigma={sig_jet:.1f} m/s, hybrid={hyb_jet:.1f} m/s")

        sig_meanT = np.nanmean(sig['T_zonal'])
        hyb_meanT = np.nanmean(hyb['T_zonal'])
        print(f"  Mean T: sigma={sig_meanT:.1f} K, hybrid={hyb_meanT:.1f} K")

    return all_issues


if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues else 0)
