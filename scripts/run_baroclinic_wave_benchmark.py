#!/usr/bin/env python
"""Dry baroclinic wave benchmark replicating CliMA Figure 3 (Yatunin et al. 2026).

Runs a 10-day Jablonowski-Williamson (2006) baroclinic instability test on the
C-D grid cubed-sphere dynamical core and generates:

1. **6-panel figure** (baroclinic_wave_benchmark.png):
   - Top row: Surface pressure perturbation at days 8 and 10
   - Middle row: 850 hPa temperature at days 8 and 10
   - Bottom row: 850 hPa relative vorticity at days 8 and 10
   Northern Hemisphere only (0-90N), PlateCarree projection.

2. **Conservation timeseries** (baroclinic_wave_conservation.png):
   - Relative dry air mass deviation
   - Relative total energy deviation
   - Min surface pressure vs time
   - Max wind speed vs time

3. **NPZ diagnostics** (baroclinic_wave_diagnostics.npz)

Usage:
    JAX_ENABLE_X64=1 python scripts/run_baroclinic_wave_benchmark.py
    JAX_ENABLE_X64=1 python scripts/run_baroclinic_wave_benchmark.py --resolution 48 --nlev 26 --dt 450 --days 10

References:
    Jablonowski & Williamson (2006), QJRMS 132, 2943-2975.
    Ullrich et al. (2016), DCMIP2016 Test Case Document.
    Yatunin et al. (2026), JAMES, Figure 3.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# legoESM imports
# ---------------------------------------------------------------------------
from legoesm.grids.cubed_sphere import (
    CubedSphereGrid,
    create_cubed_sphere,
    rotate_winds_grid_to_geo,
)
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    SigmaCoordinate,
    create_sigma_coordinate,
    pressure_from_sigma,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, FV3HydrostaticState
from legoesm.core.operators import global_integral
from legoesm.core.operators_cdgrid import dgrid_vorticity, dgrid_to_center_vector
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    CDGridPrimitiveEquationConfig,
    hydrostatic_to_fv3,
    fv3_to_hydrostatic,
)
from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init, P0
from legoesm.core.cfl import (
    adaptive_hyperdiff_coeff,
    estimate_min_dx_cubed_sphere,
    cfl_check_and_adjust,
)
from legoesm import constants


# ---------------------------------------------------------------------------
# Cubed-sphere to lat-lon regridding
# ---------------------------------------------------------------------------

def regrid_cubed_sphere_to_latlon(
    field_cs: np.ndarray,
    lon_cs: np.ndarray,
    lat_cs: np.ndarray,
    lon_ll: np.ndarray,
    lat_ll: np.ndarray,
) -> np.ndarray:
    """Regrid a cubed-sphere field to a regular lat-lon grid.

    Uses inverse-distance-weighted interpolation from the nearest
    cubed-sphere points for each lat-lon target point. This is fast
    and sufficient for visualization.

    Parameters
    ----------
    field_cs : np.ndarray, shape (6, n, n) or (6*n*n,)
        Scalar field on cubed-sphere.
    lon_cs, lat_cs : np.ndarray, shape (6, n, n)
        Cell-center coordinates in radians.
    lon_ll : np.ndarray, shape (n_lon,)
        Target longitudes in radians.
    lat_ll : np.ndarray, shape (n_lat,)
        Target latitudes in radians.

    Returns
    -------
    np.ndarray, shape (n_lat, n_lon)
    """
    from scipy.spatial import cKDTree

    # Flatten cubed-sphere coordinates to Cartesian (for k-d tree on sphere)
    lon_flat = lon_cs.ravel()
    lat_flat = lat_cs.ravel()
    field_flat = field_cs.ravel()

    # Convert to 3D Cartesian
    cos_lat = np.cos(lat_flat)
    x_cs = cos_lat * np.cos(lon_flat)
    y_cs = cos_lat * np.sin(lon_flat)
    z_cs = np.sin(lat_flat)

    tree = cKDTree(np.column_stack([x_cs, y_cs, z_cs]))

    # Target grid
    lon2d, lat2d = np.meshgrid(lon_ll, lat_ll)
    cos_lat_ll = np.cos(lat2d.ravel())
    x_ll = cos_lat_ll * np.cos(lon2d.ravel())
    y_ll = cos_lat_ll * np.sin(lon2d.ravel())
    z_ll = np.sin(lat2d.ravel())

    target_xyz = np.column_stack([x_ll, y_ll, z_ll])

    # Query k nearest neighbors and do inverse-distance weighting
    k = 4
    dist, idx = tree.query(target_xyz, k=k)

    # Handle exact matches (dist=0)
    dist = np.maximum(dist, 1e-15)
    weights = 1.0 / dist
    weights /= weights.sum(axis=1, keepdims=True)

    result = np.sum(weights * field_flat[idx], axis=1)
    return result.reshape(len(lat_ll), len(lon_ll))


# ---------------------------------------------------------------------------
# Energy diagnostics
# ---------------------------------------------------------------------------

def compute_total_energy(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> float:
    """Compute global total energy (kinetic + internal).

    E = integral over (KE + c_v * T) * dp/g * dA

    For the dry baroclinic wave (no moisture, no topography):
        KE = 0.5 * (u^2 + v^2)
        Internal = c_vd * T
    """
    g = constants.g
    c_vd = constants.c_vd
    area = grid.area  # (6, n, n)

    u = state.u.data
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data
    dsigma = sigma_coord.dsigma  # (nlev,)

    # dp = dsigma * p_s for each column
    dp = p_s[..., None] * dsigma  # (6, n, n, nlev)

    KE = 0.5 * (u ** 2 + v ** 2)
    energy_density = (KE + c_vd * T) * dp / g  # (6, n, n, nlev)

    # Sum over levels, then area-weighted global sum
    column_energy = jnp.sum(energy_density, axis=-1)  # (6, n, n)
    total = jnp.sum(column_energy * area)
    return float(total)


def compute_dry_mass(
    state: HydrostaticState,
    grid: CubedSphereGrid,
) -> float:
    """Compute global dry air mass = integral(p_s / g * dA)."""
    return float(jnp.sum(state.p_s.data * grid.area) / constants.g)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dry baroclinic wave benchmark (CliMA Figure 3 reproduction)"
    )
    parser.add_argument(
        "--resolution", type=str, default="C48",
        help="Cubed-sphere resolution, e.g. C48, C24 (default: C48)"
    )
    parser.add_argument(
        "--nlev", type=int, default=26,
        help="Number of sigma levels (default: 26)"
    )
    parser.add_argument(
        "--dt", type=float, default=450.0,
        help="Time step in seconds (default: 450)"
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="Total simulation days (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory (default: output/)"
    )
    args = parser.parse_args()

    # Parse resolution string (accept "C48" or "48")
    res_str = args.resolution.upper().lstrip("C")
    N_GRID = int(res_str)
    N_LEV = args.nlev
    DT = args.dt
    N_DAYS = args.days
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CFL check
    DT = cfl_check_and_adjust(
        DT, N_GRID, model_type="primitive_eq",
        max_wind=60.0, gravity_wave_speed=300.0,
    )

    # Hyperdiffusion: physically tuned via adaptive_hyperdiff_coeff
    dx_min = estimate_min_dx_cubed_sphere(N_GRID)
    nu4 = adaptive_hyperdiff_coeff(dx_min, DT, order=4, safety=0.5)

    print("=" * 72)
    print("  Dry Baroclinic Wave Benchmark (CliMA Figure 3)")
    print("=" * 72)
    print(f"  Resolution:      C{N_GRID} ({N_GRID}x{N_GRID} per face, 6 faces)")
    print(f"  Vertical levels: {N_LEV}")
    print(f"  Time step:       {DT:.0f} s")
    print(f"  Duration:        {N_DAYS} days")
    print(f"  dx_min:          {dx_min / 1000:.1f} km")
    print(f"  Hyperdiffusion:  {nu4:.3e} m^4/s")
    print(f"  Output:          {OUTPUT_DIR}/")
    print()

    # -----------------------------------------------------------------------
    # Grid and initial conditions
    # -----------------------------------------------------------------------
    print("Creating grid...")
    grid = create_cubed_sphere(N_GRID)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(N_LEV)

    print("Initializing Jablonowski-Williamson baroclinic wave...")
    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)

    # Store initial surface pressure for perturbation computation
    ps_init = np.array(state_cc.p_s.data)

    print(f"  Initial max |u|: {float(jnp.max(jnp.abs(state_cc.u.data))):.1f} m/s")
    print(f"  Initial mean T:  {float(jnp.mean(state_cc.T.data)):.1f} K")
    print(f"  Initial p_s:     {float(jnp.mean(state_cc.p_s.data)) / 100:.1f} hPa")

    # Convert to FV3 D-grid state for the C-D grid dycore
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    # -----------------------------------------------------------------------
    # Model configuration
    # -----------------------------------------------------------------------
    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=nu4,
        hyperdiff_ps_coeff=nu4,
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
        time_integrator="ssp_rk3",
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)

    # -----------------------------------------------------------------------
    # Time integration with diagnostic collection
    # -----------------------------------------------------------------------
    n_steps_total = int(N_DAYS * 86400 / DT)
    diag_interval_steps = max(1, int(3600 / DT))  # every hour

    # Snapshot days for the 6-panel figure
    snapshot_days = [4, 6, 8, 10]
    snapshot_steps = {int(d * 86400 / DT): d for d in snapshot_days if d <= N_DAYS}

    # Diagnostic storage
    diag_times = []
    diag_dry_mass = []
    diag_total_energy = []
    diag_ps_min = []
    diag_max_wind = []

    # Initial diagnostics
    state_cc_init = fv3_to_hydrostatic(state, cdgrid)
    mass_init = compute_dry_mass(state_cc_init, grid)
    energy_init = compute_total_energy(state_cc_init, grid, sigma)

    diag_times.append(0.0)
    diag_dry_mass.append(mass_init)
    diag_total_energy.append(energy_init)
    diag_ps_min.append(float(jnp.min(state.p_s.data)))
    u_cc, v_cc = dgrid_to_center_vector(state.u_d.data, state.v_d.data)
    diag_max_wind.append(float(jnp.max(jnp.sqrt(u_cc ** 2 + v_cc ** 2))))

    # Snapshot storage: day -> HydrostaticState (cell-centre)
    snapshots = {}

    print(f"\nIntegrating for {n_steps_total} steps...")

    # JIT warmup
    print("  JIT compiling (first step)...", end=" ", flush=True)
    t_jit = time.time()
    state = model.step(state, DT)
    jax.block_until_ready(state.p_s.data)
    print(f"done ({time.time() - t_jit:.1f}s)")

    t_start = time.time()
    last_print = t_start

    for step in range(1, n_steps_total):
        state = model.step(state, DT)

        current_step = step + 1  # 1-indexed (we already did step 0 warmup)
        day = current_step * DT / 86400.0

        # Blowup check every 100 steps
        if current_step % 100 == 0:
            ps_max_check = float(jnp.max(state.p_s.data))
            if not jnp.all(jnp.isfinite(state.p_s.data)) or ps_max_check > 2e5:
                print(f"\n  *** BLOWUP at day {day:.2f}, step {current_step} ***")
                sys.exit(1)

        # Hourly diagnostics
        if current_step % diag_interval_steps == 0:
            state_cc_now = fv3_to_hydrostatic(state, cdgrid)
            mass_now = compute_dry_mass(state_cc_now, grid)
            energy_now = compute_total_energy(state_cc_now, grid, sigma)
            ps_min_now = float(jnp.min(state.p_s.data))
            u_cc_now, v_cc_now = dgrid_to_center_vector(
                state.u_d.data, state.v_d.data
            )
            wind_max_now = float(
                jnp.max(jnp.sqrt(u_cc_now ** 2 + v_cc_now ** 2))
            )

            diag_times.append(day)
            diag_dry_mass.append(mass_now)
            diag_total_energy.append(energy_now)
            diag_ps_min.append(ps_min_now)
            diag_max_wind.append(wind_max_now)

        # Save snapshots
        if current_step in snapshot_steps:
            snap_day = snapshot_steps[current_step]
            state_cc_snap = fv3_to_hydrostatic(state, cdgrid)
            snapshots[snap_day] = jax.tree.map(lambda x: np.array(x), state_cc_snap)
            print(f"  Snapshot saved at day {snap_day}")

        # Progress reporting every 30 seconds
        now = time.time()
        if now - last_print > 30:
            elapsed = now - t_start
            steps_done = step
            steps_per_sec = steps_done / elapsed
            eta = (n_steps_total - current_step) / steps_per_sec
            ps_min_cur = float(jnp.min(state.p_s.data))
            print(
                f"  Day {day:6.2f}/{N_DAYS} | "
                f"ps_min={ps_min_cur / 100:.1f} hPa | "
                f"{steps_per_sec:.1f} steps/s | "
                f"ETA {eta / 60:.0f} min"
            )
            last_print = now

    elapsed_total = time.time() - t_start
    print(f"\nIntegration complete: {elapsed_total:.0f}s "
          f"({n_steps_total / elapsed_total:.1f} steps/s)")

    # Convert arrays
    diag_times = np.array(diag_times)
    diag_dry_mass = np.array(diag_dry_mass)
    diag_total_energy = np.array(diag_total_energy)
    diag_ps_min = np.array(diag_ps_min)
    diag_max_wind = np.array(diag_max_wind)

    # -----------------------------------------------------------------------
    # Save NPZ diagnostics
    # -----------------------------------------------------------------------
    npz_path = OUTPUT_DIR / "baroclinic_wave_diagnostics.npz"
    np.savez(
        npz_path,
        times_days=diag_times,
        dry_mass=diag_dry_mass,
        total_energy=diag_total_energy,
        ps_min=diag_ps_min,
        max_wind=diag_max_wind,
        ps_init=ps_init,
        resolution=N_GRID,
        nlev=N_LEV,
        dt=DT,
    )
    print(f"  Diagnostics saved to {npz_path}")

    # -----------------------------------------------------------------------
    # Regridding setup for plotting
    # -----------------------------------------------------------------------
    print("\nRegridding snapshots to lat-lon for plotting...")

    lon_cs = np.array(grid.lon)  # (6, n, n), radians
    lat_cs = np.array(grid.lat)  # (6, n, n), radians
    angle_cs = np.array(grid.angle)

    # Target lat-lon grid for plotting
    n_lon_plot = 360
    n_lat_plot = 180
    lon_ll = np.linspace(0, 2 * np.pi, n_lon_plot, endpoint=False)
    lat_ll = np.linspace(-np.pi / 2, np.pi / 2, n_lat_plot)

    # Find sigma level closest to 850 hPa (sigma = 0.85)
    sigma_full = np.array(sigma.sigma_full)
    k_850 = int(np.argmin(np.abs(sigma_full - 0.85)))
    p_850_actual = sigma_full[k_850] * 1000  # hPa
    print(f"  850 hPa level: k={k_850}, actual p={p_850_actual:.0f} hPa")

    # -----------------------------------------------------------------------
    # Figure 1: 6-panel CliMA Figure 3 reproduction
    # -----------------------------------------------------------------------
    print("Generating 6-panel benchmark figure...")

    plot_days = [8, 10]
    # Verify we have the needed snapshots
    missing = [d for d in plot_days if d not in snapshots]
    if missing:
        print(f"  WARNING: Missing snapshots for days {missing}. "
              f"Available: {sorted(snapshots.keys())}")
        # Fall back to whatever we have
        plot_days = sorted([d for d in plot_days if d in snapshots])

    if len(plot_days) >= 2:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            has_cartopy = True
        except ImportError:
            print("  WARNING: cartopy not installed, using basic projection")
            has_cartopy = False

        fig, axes = plt.subplots(
            3, 2,
            figsize=(14, 15),
            subplot_kw={"projection": ccrs.PlateCarree()} if has_cartopy else {},
        )

        lon_deg = np.degrees(lon_ll)
        lat_deg = np.degrees(lat_ll)
        # Northern hemisphere mask
        nh_mask = lat_deg >= 0
        lat_nh = lat_deg[nh_mask]

        for col, day in enumerate(plot_days[:2]):
            snap = snapshots[day]

            # --- Surface pressure perturbation ---
            ps_pert = np.array(snap.p_s.data) - ps_init
            ps_pert_ll = regrid_cubed_sphere_to_latlon(
                ps_pert, lon_cs, lat_cs, lon_ll, lat_ll
            )
            ps_pert_nh = ps_pert_ll[nh_mask, :] / 100  # convert to hPa

            ax = axes[0, col]
            if has_cartopy:
                vmax_ps = max(np.abs(ps_pert_nh).max(), 1.0)
                cf = ax.contourf(
                    lon_deg, lat_nh, ps_pert_nh,
                    levels=np.linspace(-vmax_ps, vmax_ps, 21),
                    cmap="RdBu_r",
                    transform=ccrs.PlateCarree(),
                    extend="both",
                )
                ax.coastlines(linewidth=0.5, color="gray")
                ax.set_extent([0, 360, 0, 90], crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.05, shrink=0.8)
                cb.set_label("hPa")
            else:
                vmax_ps = max(np.abs(ps_pert_nh).max(), 1.0)
                cf = ax.contourf(
                    lon_deg, lat_nh, ps_pert_nh,
                    levels=np.linspace(-vmax_ps, vmax_ps, 21),
                    cmap="RdBu_r", extend="both",
                )
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.08, shrink=0.8)
                cb.set_label("hPa")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 90)
            ax.set_title(f"Surface pressure perturbation, day {day}", fontsize=11)

            # --- 850 hPa temperature ---
            T_850 = np.array(snap.T.data)[..., k_850]
            T_850_ll = regrid_cubed_sphere_to_latlon(
                T_850, lon_cs, lat_cs, lon_ll, lat_ll
            )
            T_850_nh = T_850_ll[nh_mask, :]

            ax = axes[1, col]
            if has_cartopy:
                cf = ax.contourf(
                    lon_deg, lat_nh, T_850_nh,
                    levels=20, cmap="RdYlBu_r",
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines(linewidth=0.5, color="gray")
                ax.set_extent([0, 360, 0, 90], crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.05, shrink=0.8)
                cb.set_label("K")
            else:
                cf = ax.contourf(lon_deg, lat_nh, T_850_nh,
                                 levels=20, cmap="RdYlBu_r")
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.08, shrink=0.8)
                cb.set_label("K")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 90)
            ax.set_title(f"850 hPa temperature, day {day}", fontsize=11)

            # --- 850 hPa relative vorticity ---
            # Compute vorticity from cell-centre winds using finite differences
            u_grid = np.array(snap.u.data)
            v_grid = np.array(snap.v.data)

            # Rotate grid-aligned winds to geographic (east, north)
            u_east_3d = np.zeros_like(u_grid)
            v_north_3d = np.zeros_like(v_grid)
            for k in range(N_LEV):
                u_e, v_n = rotate_winds_grid_to_geo(
                    jnp.array(u_grid[..., k]),
                    jnp.array(v_grid[..., k]),
                    grid.angle,
                )
                u_east_3d[..., k] = np.array(u_e)
                v_north_3d[..., k] = np.array(v_n)

            # Compute relative vorticity at 850 hPa:
            # zeta = (1/(a*cos(lat))) * dv/dlon - (1/a) * d(u*cos(lat))/dlat
            # Approximate on the regridded lat-lon grid
            u_east_850 = u_east_3d[..., k_850]
            v_north_850 = v_north_3d[..., k_850]

            u_east_850_ll = regrid_cubed_sphere_to_latlon(
                u_east_850, lon_cs, lat_cs, lon_ll, lat_ll
            )
            v_north_850_ll = regrid_cubed_sphere_to_latlon(
                v_north_850, lon_cs, lat_cs, lon_ll, lat_ll
            )

            a = float(constants.R_earth)
            dlon = lon_ll[1] - lon_ll[0]
            dlat = lat_ll[1] - lat_ll[0]
            cos_lat_2d = np.cos(lat_ll)[:, None]

            # dv/dlon
            dvdlon = np.gradient(v_north_850_ll, dlon, axis=1)
            # d(u*cos(lat))/dlat
            u_cos = u_east_850_ll * cos_lat_2d
            du_cos_dlat = np.gradient(u_cos, dlat, axis=0)

            # Avoid division by zero at poles
            cos_lat_safe = np.maximum(cos_lat_2d, 1e-6)
            vort_ll = (1.0 / (a * cos_lat_safe)) * dvdlon - (1.0 / a) * du_cos_dlat / cos_lat_safe
            vort_nh = vort_ll[nh_mask, :]

            ax = axes[2, col]
            vmax_vort = 2e-4
            if has_cartopy:
                cf = ax.contourf(
                    lon_deg, lat_nh, vort_nh,
                    levels=np.linspace(-vmax_vort, vmax_vort, 21),
                    cmap="RdBu_r",
                    transform=ccrs.PlateCarree(),
                    extend="both",
                )
                ax.coastlines(linewidth=0.5, color="gray")
                ax.set_extent([0, 360, 0, 90], crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.05, shrink=0.8)
                cb.set_label("s$^{-1}$")
            else:
                cf = ax.contourf(
                    lon_deg, lat_nh, vort_nh,
                    levels=np.linspace(-vmax_vort, vmax_vort, 21),
                    cmap="RdBu_r", extend="both",
                )
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.08, shrink=0.8)
                cb.set_label("s$^{-1}$")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 90)
            ax.set_title(f"850 hPa relative vorticity, day {day}", fontsize=11)

        plt.suptitle(
            f"Dry Baroclinic Wave Benchmark  |  C{N_GRID} L{N_LEV}  |  "
            f"dt={DT:.0f}s  |  legoESM C-D grid PE",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "baroclinic_wave_benchmark.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fig_path}")
    else:
        print("  Skipping 6-panel figure (not enough snapshots)")

    # -----------------------------------------------------------------------
    # Figure 2: Conservation timeseries (CliMA Figure 9 style)
    # -----------------------------------------------------------------------
    print("Generating conservation timeseries figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Relative dry mass deviation
    mass_rel = (diag_dry_mass - mass_init) / mass_init
    axes[0, 0].plot(diag_times, mass_rel, "b-", linewidth=1.2)
    axes[0, 0].set_ylabel("Relative deviation")
    axes[0, 0].set_title("(a) Dry air mass conservation")
    axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel("Time [days]")

    # (b) Relative total energy deviation
    energy_rel = (diag_total_energy - energy_init) / abs(energy_init)
    axes[0, 1].plot(diag_times, energy_rel, "r-", linewidth=1.2)
    axes[0, 1].set_ylabel("Relative deviation")
    axes[0, 1].set_title("(b) Total energy conservation")
    axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel("Time [days]")

    # (c) Minimum surface pressure
    axes[1, 0].plot(diag_times, diag_ps_min / 100, "k-", linewidth=1.2)
    axes[1, 0].set_ylabel("Min p$_s$ [hPa]")
    axes[1, 0].set_title("(c) Minimum surface pressure")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Time [days]")

    # (d) Maximum wind speed
    axes[1, 1].plot(diag_times, diag_max_wind, "g-", linewidth=1.2)
    axes[1, 1].set_ylabel("Max |v| [m/s]")
    axes[1, 1].set_title("(d) Maximum wind speed")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Time [days]")

    plt.suptitle(
        f"Baroclinic Wave Conservation  |  C{N_GRID} L{N_LEV}  |  "
        f"dt={DT:.0f}s",
        fontsize=13,
    )
    plt.tight_layout()
    cons_path = OUTPUT_DIR / "baroclinic_wave_conservation.png"
    plt.savefig(cons_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {cons_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  Resolution:      C{N_GRID} L{N_LEV}")
    print(f"  Duration:        {N_DAYS} days ({n_steps_total} steps)")
    print(f"  Wall time:       {elapsed_total:.0f}s "
          f"({n_steps_total / elapsed_total:.1f} steps/s)")
    print(f"  Mass drift:      {mass_rel[-1]:+.3e} (relative)")
    print(f"  Energy drift:    {energy_rel[-1]:+.3e} (relative)")
    print(f"  Min p_s (final): {diag_ps_min[-1] / 100:.1f} hPa")
    print(f"  Max wind (final):{diag_max_wind[-1]:.1f} m/s")
    print(f"\n  Outputs:")
    print(f"    {OUTPUT_DIR / 'baroclinic_wave_benchmark.png'}")
    print(f"    {OUTPUT_DIR / 'baroclinic_wave_conservation.png'}")
    print(f"    {OUTPUT_DIR / 'baroclinic_wave_diagnostics.npz'}")
    print()


if __name__ == "__main__":
    main()
