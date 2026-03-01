#!/usr/bin/env python
"""Realistic ocean model experiments with visualization.

Three physically meaningful test cases at moderate resolution:

  1. Stommel Gyre — wind-driven double gyre with western intensification
  2. Baroclinic Adjustment — density-driven geostrophic flow from a thermal front
  3. Equatorial Kelvin Wave — equatorial SSH perturbation propagating eastward

Default: C16, 20 levels, 30-day integrations (adjustable with --days).

Usage:
    cd /Users/pierregentine/legoESM
    source .venv/bin/activate

    # All three tests (C16, 20 levels, 30 days, ~15 min)
    python scripts/run_ocean_realistic.py

    # Higher resolution, longer run
    python scripts/run_ocean_realistic.py -n 24 -l 30 -d 100

    # Single test
    python scripts/run_ocean_realistic.py --test gyre --days 100
"""

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from legoesm.grids.cubed_sphere import create_cubed_sphere, CubedSphereGrid
from legoesm.ocean import (
    OceanModel,
    OceanConfig,
    OceanState,
    create_ocean_z_star,
)
from legoesm.ocean.eos import wright_eos, rho_0
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.init import idealized_bathymetry, rest_state_ocean
from legoesm.ocean.conservation import _ocean_area_sum, _ocean_volume_sum
from legoesm.core.field import Field

IDEALIZED_LAND_LAT_THRESHOLD = 90.0  # Fully oceanic mask for idealized cases


# =====================================================================
# Helpers
# =====================================================================

def safe_max(arr, default=0.001):
    """NaN-safe max of absolute values."""
    finite = arr[np.isfinite(arr)]
    return float(np.max(np.abs(finite))) if len(finite) > 0 else default


def scatter_field(ax, lon_deg, lat_deg, data, cmap, vmin, vmax,
                  point_size=2.0, alpha=0.9):
    """Scatter plot a 2D field on all 6 cubed-sphere faces."""
    for face in range(6):
        sc = ax.scatter(
            lon_deg[face].ravel(), lat_deg[face].ravel(),
            c=np.asarray(data[face]).ravel(),
            s=point_size, cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), edgecolors="none", alpha=alpha,
        )
    return sc


def _idealized_projection(central_longitude: float = 0.0):
    """Map projection for idealized all-ocean experiments."""
    return ccrs.Mollweide(central_longitude=central_longitude)


def _style_idealized_axes(ax):
    """Style map axes for idealized all-ocean plots (no real coastlines)."""
    ax.gridlines(linewidth=0.3, alpha=0.4)
    ax.set_global()


def _rest_state_all_ocean(grid, z_coord, **kwargs):
    """Rest state with no land mask for idealized ocean cases."""
    return rest_state_ocean(
        grid,
        z_coord,
        land_lat_threshold=IDEALIZED_LAND_LAT_THRESHOLD,
        **kwargs,
    )


def _assert_all_ocean(state, label: str):
    """Guard that idealized cases use a fully oceanic mask."""
    ocean_fraction = float(jnp.mean(state.land_mask.data))
    if ocean_fraction < 0.999999:
        raise ValueError(
            f"{label}: expected fully oceanic mask, got ocean_fraction={ocean_fraction:.6f}",
        )


def compute_diagnostics(state, grid, z_coord):
    """Compute scalar diagnostics for the ocean state."""
    mask = state.land_mask.data
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)

    vol = float(_ocean_area_sum(state.eta.data, mask, grid))
    heat = float(_ocean_volume_sum(state.T.data, h_k, mask, grid))
    salt = float(_ocean_volume_sum(state.S.data, h_k, mask, grid))
    ke = float(_ocean_volume_sum(
        0.5 * (state.u.data**2 + state.v.data**2), h_k, mask, grid,
    ))
    ocean_cells = float(jnp.sum(mask))

    return {
        "volume": vol,
        "heat": heat,
        "salt": salt,
        "kinetic_energy": ke,
        "SST_mean": float(jnp.sum(state.T.data[..., 0] * mask)
                          / jnp.maximum(ocean_cells, 1.0)),
        "SST_max": float(jnp.max(jnp.where(mask > 0.5, state.T.data[..., 0], -999.0))),
        "SST_min": float(jnp.min(jnp.where(mask > 0.5, state.T.data[..., 0], 999.0))),
        "SSH_min": float(jnp.min(jnp.where(mask > 0.5, state.eta.data, 999.0))),
        "SSH_max": float(jnp.max(jnp.where(mask > 0.5, state.eta.data, -999.0))),
        "u_max": float(jnp.max(jnp.abs(state.u.data))),
        "v_max": float(jnp.max(jnp.abs(state.v.data))),
        "speed_max": float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))),
    }


def zonal_mean(field_2d, grid, n_bins=60):
    """Compute zonal mean of a 2D field."""
    lat_flat = np.asarray(grid.lat).flatten()
    area_flat = np.asarray(grid.area).flatten()
    data_flat = np.asarray(field_2d).flatten()

    lat_bins = np.linspace(-np.pi / 2, np.pi / 2, n_bins + 1)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    result = np.full(n_bins, np.nan)

    for i in range(n_bins):
        idx = (lat_flat >= lat_bins[i]) & (lat_flat < lat_bins[i + 1])
        if np.sum(idx) > 0:
            w = area_flat[idx]
            result[i] = np.sum(w * data_flat[idx]) / np.sum(w)

    # Interpolate NaN bins
    valid = ~np.isnan(result)
    if valid.any() and not valid.all():
        result = np.interp(lat_centers, lat_centers[valid], result[valid])

    return np.degrees(lat_centers), result


def run_integration(model, state, grid, z_coord, dt, n_steps,
                    diag_interval, snapshot_steps, label,
                    external_forcing_fn=None):
    """Generic integration loop with diagnostics and snapshots."""
    diagnostics = [compute_diagnostics(state, grid, z_coord)]
    snapshots = {0: state}

    print(f"  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = model.step(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"\n{'Step':>7s}  {'Day':>7s}  {'SSH_min':>9s}  {'SSH_max':>9s}  "
          f"{'|v|_max':>9s}  {'SST_mean':>9s}  {'KE':>12s}  {'wall':>6s}")
    print("-" * 80)

    t_start = time.time()
    t_lap = t_start
    for step in range(1, n_steps + 1):
        # External forcing (e.g., wind stress)
        if external_forcing_fn is not None:
            state = external_forcing_fn(state, dt)

        state = model.step(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            elapsed = time.time() - t_lap
            t_lap = time.time()
            print(f"{step:7d}  {day:7.1f}  {diag['SSH_min']:9.3f}  "
                  f"{diag['SSH_max']:9.3f}  {diag['speed_max']:9.4f}  "
                  f"{diag['SST_mean']:9.4f}  {diag['kinetic_energy']:12.4e}  "
                  f"{elapsed:6.1f}")

        if step in snapshot_steps:
            snapshots[step] = state

        # Blowup detection
        if step % max(1, n_steps // 20) == 0:
            if not jnp.all(jnp.isfinite(state.eta.data)):
                print(f"\n  *** BLOWUP at step {step} (day {step*dt/86400:.1f}) ***")
                break

    total_time = time.time() - t_start
    print(f"\n  Completed in {total_time:.1f}s ({n_steps/total_time:.0f} steps/s)")

    return state, diagnostics, snapshots, total_time


def plot_ssh_snapshots(snapshots, grid, dt, output_path, title, point_size,
                       central_longitude=0):
    """4-panel SSH snapshot figure."""
    if not HAS_CARTOPY:
        return
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    sorted_snaps = sorted(snapshots.items())
    n_panels = min(len(sorted_snaps), 4)
    # Pick evenly spaced snapshots
    indices = np.linspace(0, len(sorted_snaps) - 1, n_panels, dtype=int)
    panels = [sorted_snaps[i] for i in indices]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                              subplot_kw={"projection": _idealized_projection(
                                  central_longitude=central_longitude)})
    axes = axes.ravel()

    all_ssh = [np.asarray(snap.eta.data) for _, snap in panels]
    ssh_abs = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
    ssh_abs = max(ssh_abs, 0.001)

    for idx, (step_num, snap) in enumerate(panels):
        day = step_num * dt / 86400.0
        ax = axes[idx]
        ssh = np.asarray(snap.eta.data)
        sc = scatter_field(ax, lon_deg, lat_deg, ssh, "RdBu_r",
                           -ssh_abs, ssh_abs, point_size=point_size)
        _style_idealized_axes(ax)
        ax.set_title(f"Day {day:.0f}", fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, pad=0.02, orientation="vertical",
                 label="SSH [m]")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_field_map(data, grid, output_path, title, cmap, vmin, vmax,
                   colorbar_label, point_size, central_longitude=0):
    """Single global map figure."""
    if not HAS_CARTOPY:
        return
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    fig, ax = plt.subplots(1, 1, figsize=(14, 7),
                            subplot_kw={"projection": _idealized_projection(
                                central_longitude=central_longitude)})
    sc = scatter_field(ax, lon_deg, lat_deg, np.asarray(data), cmap,
                       vmin, vmax, point_size=point_size)
    _style_idealized_axes(ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical",
                 label=colorbar_label, pad=0.02)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_diagnostics_timeseries(diagnostics, dt, diag_interval, n_steps,
                                output_path, title):
    """4-panel time series of scalar diagnostics."""
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ssh_max = [d["SSH_max"] for d in diagnostics]
    ssh_min = [d["SSH_min"] for d in diagnostics]
    axes[0, 0].plot(times_days, ssh_max, "b-", label="SSH max", linewidth=1.5)
    axes[0, 0].plot(times_days, ssh_min, "r-", label="SSH min", linewidth=1.5)
    axes[0, 0].set_ylabel("SSH [m]")
    axes[0, 0].set_title("Sea Surface Height Extremes")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    ke = np.array([d["kinetic_energy"] for d in diagnostics])
    axes[0, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[0, 1].set_ylabel("KE")
    axes[0, 1].set_title("Kinetic Energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    axes[1, 0].plot(times_days, [d["speed_max"] for d in diagnostics],
                    "r-", linewidth=1.5)
    axes[1, 0].set_ylabel("|v| max [m/s]")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_title("Maximum Current Speed")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times_days, [d["SST_mean"] for d in diagnostics],
                    "g-", linewidth=1.5)
    axes[1, 1].set_ylabel("Mean SST [degC]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Mean Sea Surface Temperature")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_zonal_sections(state, state_init, grid, z_coord, output_path, title):
    """Zonal-mean T, u sections (depth vs latitude)."""
    lat_flat = np.asarray(grid.lat).flatten()
    area_flat = np.asarray(grid.area).flatten()
    mask_flat = np.asarray(state.land_mask.data).flatten()

    n_lat_bins = 40
    lat_bins = np.linspace(-np.pi / 2, np.pi / 2, n_lat_bins + 1)
    lat_centers = np.degrees(0.5 * (lat_bins[:-1] + lat_bins[1:]))
    z_ref = np.asarray(z_coord.z_full_ref)
    nlev = z_coord.n_levels

    # Compute zonal means (area-weighted, ocean-only)
    T_zonal = np.full((n_lat_bins, nlev), np.nan)
    u_zonal = np.full((n_lat_bins, nlev), np.nan)
    T_init_zonal = np.full((n_lat_bins, nlev), np.nan)

    T_flat = np.asarray(state.T.data).reshape(-1, nlev)
    u_flat = np.asarray(state.u.data).reshape(-1, nlev)
    T_init_flat = np.asarray(state_init.T.data).reshape(-1, nlev)

    for i in range(n_lat_bins):
        idx = (lat_flat >= lat_bins[i]) & (lat_flat < lat_bins[i + 1]) & (mask_flat > 0.5)
        if np.sum(idx) > 0:
            w = area_flat[idx]
            w = w / w.sum()
            for k in range(nlev):
                T_zonal[i, k] = np.sum(w * T_flat[idx, k])
                u_zonal[i, k] = np.sum(w * u_flat[idx, k])
                T_init_zonal[i, k] = np.sum(w * T_init_flat[idx, k])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Temperature
    T_zonal_clean = np.where(np.isfinite(T_zonal), T_zonal, 0.0)
    cs0 = axes[0].contourf(lat_centers, z_ref, T_zonal_clean.T,
                            levels=20, cmap="RdYlBu_r")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_xlabel("Latitude [deg]")
    axes[0].set_title("Zonal-Mean Temperature [degC]")
    plt.colorbar(cs0, ax=axes[0])

    # Temperature anomaly
    T_anom = T_zonal - T_init_zonal
    T_anom = np.where(np.isfinite(T_anom), T_anom, 0.0)
    vabs = max(float(np.nanmax(np.abs(T_anom[np.isfinite(T_anom)]))) if np.any(np.isfinite(T_anom)) else 0.01, 0.01)
    cs1 = axes[1].contourf(lat_centers, z_ref, T_anom.T,
                            levels=np.linspace(-vabs, vabs, 21), cmap="RdBu_r")
    axes[1].set_xlabel("Latitude [deg]")
    axes[1].set_title("Temperature Anomaly [degC]")
    plt.colorbar(cs1, ax=axes[1])

    # Zonal velocity
    u_zonal_clean = np.where(np.isfinite(u_zonal), u_zonal, 0.0)
    vabs_u = max(float(np.max(np.abs(u_zonal_clean))) if u_zonal_clean.size > 0 else 0.001, 0.001)
    cs2 = axes[2].contourf(lat_centers, z_ref, u_zonal_clean.T,
                            levels=np.linspace(-vabs_u, vabs_u, 21), cmap="RdBu_r")
    axes[2].set_xlabel("Latitude [deg]")
    axes[2].set_title("Zonal-Mean Zonal Current [m/s]")
    plt.colorbar(cs2, ax=axes[2])

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_conservation(diagnostics, dt, diag_interval, n_steps,
                      output_path, title):
    """Conservation diagnostics (volume, heat, salt)."""
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    vol = np.array([d["volume"] for d in diagnostics])
    heat = np.array([d["heat"] for d in diagnostics])
    salt = np.array([d["salt"] for d in diagnostics])

    axes[0].plot(times_days, vol, "b-", linewidth=1.5)
    axes[0].set_ylabel("Volume integral [m^3]")
    axes[0].set_title("Volume (eta integral)")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    if abs(heat[0]) > 1e-10:
        heat_rel = (heat - heat[0]) / abs(heat[0])
        axes[1].plot(times_days, heat_rel, "r-", linewidth=1.5)
        axes[1].set_ylabel("Relative change")
    else:
        axes[1].plot(times_days, heat, "r-", linewidth=1.5)
        axes[1].set_ylabel("Heat integral")
    axes[1].set_title("Heat Conservation")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    if abs(salt[0]) > 1e-10:
        salt_rel = (salt - salt[0]) / abs(salt[0])
        axes[2].plot(times_days, salt_rel, "g-", linewidth=1.5)
        axes[2].set_ylabel("Relative change")
    else:
        axes[2].plot(times_days, salt, "g-", linewidth=1.5)
        axes[2].set_ylabel("Salt integral")
    axes[2].set_title("Salt Conservation")
    axes[2].set_xlabel("Time [days]")
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# Test Case 1: Stommel Wind-Driven Gyre
# =====================================================================

def stommel_gyre(args, grid, z_coord, config, output_dir, point_size):
    """Wind-driven double gyre: Stommel/Munk western boundary current.

    Idealized zonal wind stress drives a subtropical and subpolar gyre
    with expected western boundary intensification via beta effect.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Stommel Wind-Driven Double Gyre")
    print("=" * 70)

    # Uniform temperature avoids pressure gradient errors from stratification
    # on the cubed sphere. Wind-driven gyre is a barotropic test.
    state = _rest_state_all_ocean(grid, z_coord, T_surface=15.0, T_deep=15.0)
    _assert_all_ocean(state, "stommel_gyre")
    state_init = state

    # Wind stress parameters
    tau_max = 0.1  # N/m^2
    lat_south = 15.0  # degrees N
    lat_north = 75.0  # degrees N
    lat_center = 0.5 * (lat_south + lat_north)  # 45N
    lat_width = lat_north - lat_south  # 60 degrees

    print(f"  Wind: tau_x = -{tau_max} * cos(pi*(lat-{lat_south})/{lat_width:.0f})")
    print(f"  Band: {lat_south}N to {lat_north}N")
    print(f"  SST: 15 degC (uniform, barotropic test)")

    lat_deg_grid = grid.lat * (180.0 / jnp.pi)
    mask = state.land_mask.data

    # Precompute wind stress tendency per timestep
    # tau_x = -tau_max * cos(pi * (lat - lat_south) / lat_width)
    lat_frac = (lat_deg_grid - lat_south) / lat_width
    tau_x = -tau_max * jnp.cos(jnp.pi * lat_frac)
    wind_mask = jnp.where(
        (lat_deg_grid > lat_south) & (lat_deg_grid < lat_north) & (mask > 0.5),
        1.0, 0.0,
    )
    tau_x = tau_x * wind_mask

    # Applied as acceleration in top layer: du/dt = tau / (rho_0 * h_surface)
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
    h_surface = jnp.maximum(h_k[..., 0], 1.0)
    du_surf_per_s = tau_x / (rho_0 * h_surface)

    def wind_forcing(st, dt_val):
        du_3d = jnp.zeros_like(st.u.data)
        du_3d = du_3d.at[..., 0].set(du_surf_per_s * dt_val)
        return st._replace(u=st.u.replace(data=st.u.data + du_3d))

    model = OceanModel(grid, z_coord, config)
    dt = args.dt
    n_steps = int(args.days * 86400 / dt)
    diag_interval = max(1, n_steps // 50)
    snapshot_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    snapshot_steps = {max(1, int(f * n_steps)): f for f in snapshot_fracs if f > 0}
    snapshot_steps[0] = 0.0

    state_final, diagnostics, snapshots, wall_time = run_integration(
        model, state, grid, z_coord, dt, n_steps,
        diag_interval, snapshot_steps, "Stommel Gyre",
        external_forcing_fn=wind_forcing,
    )

    # --- Visualization ---
    outdir = Path(output_dir) / "stommel_gyre"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_ssh_snapshots(snapshots, grid, dt,
                       outdir / "ssh_snapshots.png",
                       "Stommel Gyre — SSH [m]", point_size)

    # Surface speed
    speed_surf = np.sqrt(np.asarray(state_final.u.data[..., 0])**2 +
                         np.asarray(state_final.v.data[..., 0])**2)
    plot_field_map(speed_surf, grid, outdir / "speed_surface.png",
                   f"Surface Current Speed [m/s] — Day {args.days}",
                   "magma", 0, safe_max(speed_surf),
                   "Speed [m/s]", point_size)

    # SST
    sst = np.asarray(state_final.T.data[..., 0])
    plot_field_map(sst, grid, outdir / "sst_final.png",
                   f"SST [degC] — Day {args.days}",
                   "RdYlBu_r", 0, 27, "Temperature [degC]", point_size)

    # Wind stress pattern
    plot_field_map(np.asarray(tau_x), grid, outdir / "wind_stress.png",
                   "Zonal Wind Stress [N/m^2]",
                   "RdBu_r", -tau_max, tau_max,
                   "tau_x [N/m^2]", point_size)

    plot_diagnostics_timeseries(diagnostics, dt, diag_interval, n_steps,
                                outdir / "diagnostics.png",
                                "Stommel Gyre — Diagnostics")

    plot_conservation(diagnostics, dt, diag_interval, n_steps,
                      outdir / "conservation.png",
                      "Stommel Gyre — Conservation")

    plot_zonal_sections(state_final, state_init, grid, z_coord,
                        outdir / "zonal_sections.png",
                        f"Stommel Gyre — Zonal-Mean Sections (Day {args.days})")

    # Zonal-mean SSH (latitude profile)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    lat_c, ssh_zm = zonal_mean(state_final.eta.data, grid)
    _, ssh_zm_init = zonal_mean(state_init.eta.data, grid)
    ax.plot(lat_c, ssh_zm, "b-", linewidth=2, label=f"Day {args.days}")
    ax.plot(lat_c, ssh_zm_init, "k--", linewidth=1, label="Initial")
    ax.set_xlabel("Latitude [deg]", fontsize=12)
    ax.set_ylabel("SSH [m]", fontsize=12)
    ax.set_title("Zonal-Mean SSH", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ssh_zonal_mean.png", dpi=150, bbox_inches="tight")
    plt.close()

    return state_final, diagnostics


# =====================================================================
# Test Case 2: Baroclinic Adjustment
# =====================================================================

def baroclinic_adjustment(args, grid, z_coord, config, output_dir, point_size):
    """Baroclinic adjustment: density-driven flow from a thermal front.

    A sharp meridional temperature gradient (warm south, cold north) across
    a front at 30N drives geostrophic adjustment. The resulting baroclinic
    flow intensifies and may develop instabilities at sufficient resolution.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Baroclinic Adjustment (Thermal Front)")
    print("=" * 70)

    # Uniform background temperature with a HORIZONTAL front.
    # No vertical stratification to avoid pressure gradient errors on cubed sphere.
    # The only density variation is the meridional temperature front.
    state_base = _rest_state_all_ocean(grid, z_coord, T_surface=15.0, T_deep=15.0)
    _assert_all_ocean(state_base, "baroclinic_adjustment")
    mask = state_base.land_mask.data

    # Temperature front at 30N: warm south, cold north (DEPTH-UNIFORM)
    lat_deg = grid.lat * (180.0 / jnp.pi)
    front_lat = 30.0    # degrees N
    front_width = 15.0   # degrees, transition width (wider for stability)

    # Smoothed step function: 0 in south, 1 in north
    step_fn = 0.5 * (1.0 + jnp.tanh((lat_deg - front_lat) / (0.5 * front_width)))

    T_south = 20.0  # degC
    T_north = 10.0  # degC

    # Depth-uniform temperature with meridional front
    n = grid.n
    nlev = z_coord.n_levels
    T_2d = T_south + (T_north - T_south) * step_fn  # (6, n, n)
    T_3d = jnp.broadcast_to(T_2d[..., jnp.newaxis], (6, n, n, nlev)).astype(jnp.float32)

    state = state_base._replace(
        T=state_base.T.replace(data=T_3d),
    )
    state_init = state

    print(f"  Front latitude: {front_lat}N (width {front_width} deg)")
    print(f"  South T: {T_south} degC, North T: {T_north} degC (depth-uniform)")
    print(f"  Density difference: ~{float(wright_eos(jnp.array(T_north), jnp.array(35.0), jnp.array(0.0)) - wright_eos(jnp.array(T_south), jnp.array(35.0), jnp.array(0.0))):.2f} kg/m^3")

    model = OceanModel(grid, z_coord, config)
    dt = args.dt
    n_steps = int(args.days * 86400 / dt)
    diag_interval = max(1, n_steps // 50)
    snapshot_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    snapshot_steps = {max(1, int(f * n_steps)): f for f in snapshot_fracs if f > 0}
    snapshot_steps[0] = 0.0

    state_final, diagnostics, snapshots, wall_time = run_integration(
        model, state, grid, z_coord, dt, n_steps,
        diag_interval, snapshot_steps, "Baroclinic Adjustment",
    )

    # --- Visualization ---
    outdir = Path(output_dir) / "baroclinic_adjustment"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_ssh_snapshots(snapshots, grid, dt,
                       outdir / "ssh_snapshots.png",
                       "Baroclinic Adjustment — SSH [m]", point_size)

    # SST initial and final
    if HAS_CARTOPY:
        lon_deg_np = np.asarray(grid.lon) * 180 / np.pi
        lat_deg_np = np.asarray(grid.lat) * 180 / np.pi

        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  subplot_kw={"projection": _idealized_projection()})
        for ax, (label, snap) in zip(axes, [("Initial", state_init), (f"Day {args.days}", state_final)]):
            sst = np.asarray(snap.T.data[..., 0])
            sc = scatter_field(ax, lon_deg_np, lat_deg_np, sst, "RdYlBu_r",
                               vmin=0, vmax=27, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SST — {label}", fontsize=13, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                     label="Temperature [degC]", pad=0.02)
        fig.suptitle("Baroclinic Adjustment — SST", fontsize=15, fontweight="bold")
        plt.savefig(outdir / "sst_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Surface velocity
    speed_surf = np.sqrt(np.asarray(state_final.u.data[..., 0])**2 +
                         np.asarray(state_final.v.data[..., 0])**2)
    plot_field_map(speed_surf, grid, outdir / "speed_surface.png",
                   f"Surface Current Speed [m/s] — Day {args.days}",
                   "magma", 0, safe_max(speed_surf),
                   "Speed [m/s]", point_size)

    # Zonal current
    u_surf = np.asarray(state_final.u.data[..., 0])
    vabs = safe_max(u_surf)
    plot_field_map(u_surf, grid, outdir / "u_surface.png",
                   f"Surface Zonal Current [m/s] — Day {args.days}",
                   "RdBu_r", -vabs, vabs, "u [m/s]", point_size)

    plot_diagnostics_timeseries(diagnostics, dt, diag_interval, n_steps,
                                outdir / "diagnostics.png",
                                "Baroclinic Adjustment — Diagnostics")

    plot_conservation(diagnostics, dt, diag_interval, n_steps,
                      outdir / "conservation.png",
                      "Baroclinic Adjustment — Conservation")

    plot_zonal_sections(state_final, state_init, grid, z_coord,
                        outdir / "zonal_sections.png",
                        f"Baroclinic Adjustment — Zonal Sections (Day {args.days})")

    # Density section at equator and at front
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    z_ref = np.asarray(z_coord.z_full_ref)
    mid = grid.n // 2

    # T profile at equator vs at front
    T_equator_init = np.asarray(state_init.T.data[0, mid, mid, :])
    T_equator_final = np.asarray(state_final.T.data[0, mid, mid, :])
    axes[0].plot(T_equator_init, z_ref, "b-o", markersize=3, label="Initial", linewidth=1.5)
    axes[0].plot(T_equator_final, z_ref, "r--s", markersize=3, label="Final", linewidth=1.5)
    axes[0].set_xlabel("Temperature [degC]")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_title("T Profile (near equator)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Density at equator
    rho_init = np.asarray(wright_eos(
        state_init.T.data[0, mid, mid, :],
        state_init.S.data[0, mid, mid, :],
        jnp.zeros(nlev),
    ))
    rho_final = np.asarray(wright_eos(
        state_final.T.data[0, mid, mid, :],
        state_final.S.data[0, mid, mid, :],
        jnp.zeros(nlev),
    ))
    axes[1].plot(rho_init - 1000, z_ref, "b-o", markersize=3, label="Initial", linewidth=1.5)
    axes[1].plot(rho_final - 1000, z_ref, "r--s", markersize=3, label="Final", linewidth=1.5)
    axes[1].set_xlabel("sigma_0 [kg/m^3]")
    axes[1].set_ylabel("Depth [m]")
    axes[1].set_title("Density Profile (near equator)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Baroclinic Adjustment — Vertical Profiles", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "profiles.png", dpi=150, bbox_inches="tight")
    plt.close()

    return state_final, diagnostics


# =====================================================================
# Test Case 3: Equatorial Kelvin Wave
# =====================================================================

def equatorial_kelvin_wave(args, grid, z_coord, config, output_dir, point_size):
    """Equatorial Kelvin wave: SSH perturbation propagating eastward.

    A Gaussian SSH perturbation at the equator in the western hemisphere
    excites barotropic Kelvin waves that propagate eastward along the
    equator and poleward along coastlines. The Kelvin wave speed is
    c = sqrt(gH) ~ 230 m/s for barotropic mode.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Equatorial Kelvin Wave")
    print("=" * 70)

    state = _rest_state_all_ocean(grid, z_coord, T_surface=15.0, T_deep=15.0)
    _assert_all_ocean(state, "equatorial_kelvin_wave")

    # SSH perturbation: Gaussian in longitude at equator
    lon_rad = grid.lon
    lat_rad = grid.lat
    mask = state.land_mask.data

    # Perturbation centered at (lon=240E = -120W, lat=0)
    lon0 = 4.0 * jnp.pi / 3.0   # 240 degrees = western Pacific
    lat0 = 0.0

    # Gaussian: narrow in latitude (equatorial trapping), wider in longitude
    sigma_lon = 15.0 * jnp.pi / 180.0  # 15 degrees
    sigma_lat = 5.0 * jnp.pi / 180.0   # 5 degrees (equatorial trapping)

    dlon = lon_rad - lon0
    # Wrap to [-pi, pi]
    dlon = jnp.arctan2(jnp.sin(dlon), jnp.cos(dlon))

    eta_pert = 0.5 * jnp.exp(-0.5 * (dlon / sigma_lon)**2
                               - 0.5 * (lat_rad / sigma_lat)**2)
    eta_pert = eta_pert * mask

    state = state._replace(
        eta=state.eta.replace(data=eta_pert.astype(jnp.float32)),
    )
    state_init = state

    c_baro = float(jnp.sqrt(9.81 * 5500.0))
    R_eq = 6.371e6
    crossing_time = jnp.pi * R_eq / c_baro / 86400.0  # half circumference

    print(f"  Perturbation: 0.5 m SSH at (240E, 0N)")
    print(f"  Gaussian width: sigma_lon={15}deg, sigma_lat={5}deg")
    print(f"  Barotropic wave speed: {c_baro:.0f} m/s")
    print(f"  Equatorial crossing time: {float(crossing_time):.1f} days")
    print(f"  SST: 15 degC (uniform, barotropic test)")

    model = OceanModel(grid, z_coord, config)
    dt = args.dt
    n_steps = int(args.days * 86400 / dt)
    diag_interval = max(1, n_steps // 50)

    # More snapshots to see wave propagation
    n_snaps = min(8, n_steps)
    snap_steps_list = sorted(set([0] + [int(i * n_steps / (n_snaps - 1))
                                         for i in range(n_snaps)]))
    snapshot_steps = {s: s for s in snap_steps_list}

    state_final, diagnostics, snapshots, wall_time = run_integration(
        model, state, grid, z_coord, dt, n_steps,
        diag_interval, snapshot_steps, "Kelvin Wave",
    )

    # --- Visualization ---
    outdir = Path(output_dir) / "kelvin_wave"
    outdir.mkdir(parents=True, exist_ok=True)

    # SSH snapshots centered on Pacific
    plot_ssh_snapshots(snapshots, grid, dt,
                       outdir / "ssh_snapshots.png",
                       "Equatorial Kelvin Wave — SSH [m]", point_size,
                       central_longitude=180)

    # SSH at equator as Hovmoller diagram (longitude vs time)
    if HAS_CARTOPY:
        lon_deg_np = np.asarray(grid.lon) * 180 / np.pi
        lat_deg_np = np.asarray(grid.lat) * 180 / np.pi

        # All 8 snapshots
        sorted_snaps = sorted(snapshots.items())
        n_panel = min(len(sorted_snaps), 8)
        ncols = 4
        nrows = (n_panel + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                  subplot_kw={"projection": _idealized_projection(
                                      central_longitude=180)})
        axes = np.array(axes).ravel()

        all_ssh = [np.asarray(s.eta.data) for _, s in sorted_snaps[:n_panel]]
        ssh_abs = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
        ssh_abs = max(ssh_abs, 0.01)

        for idx, (step_num, snap) in enumerate(sorted_snaps[:n_panel]):
            day = step_num * dt / 86400.0
            ax = axes[idx]
            ssh = np.asarray(snap.eta.data)
            sc = scatter_field(ax, lon_deg_np, lat_deg_np, ssh, "RdBu_r",
                               -ssh_abs, ssh_abs, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"Day {day:.1f}", fontsize=11, fontweight="bold")

        for idx in range(n_panel, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Equatorial Kelvin Wave — SSH Evolution",
                     fontsize=15, fontweight="bold")
        fig.colorbar(sc, ax=axes[:n_panel].tolist(), shrink=0.78, pad=0.02,
                     orientation="vertical", label="SSH [m]")
        plt.savefig(outdir / "ssh_all_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Surface velocity at final time
    speed_surf = np.sqrt(np.asarray(state_final.u.data[..., 0])**2 +
                         np.asarray(state_final.v.data[..., 0])**2)
    plot_field_map(speed_surf, grid, outdir / "speed_surface.png",
                   f"Surface Current Speed [m/s] — Day {args.days}",
                   "magma", 0, safe_max(speed_surf),
                   "Speed [m/s]", point_size, central_longitude=180)

    plot_diagnostics_timeseries(diagnostics, dt, diag_interval, n_steps,
                                outdir / "diagnostics.png",
                                "Equatorial Kelvin Wave — Diagnostics")

    plot_conservation(diagnostics, dt, diag_interval, n_steps,
                      outdir / "conservation.png",
                      "Equatorial Kelvin Wave — Conservation")

    return state_final, diagnostics


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Realistic ocean model experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resolution", "-n", type=int, default=16,
                        help="Cubed-sphere resolution (default: 16)")
    parser.add_argument("--levels", "-l", type=int, default=20,
                        help="Number of vertical levels (default: 20)")
    parser.add_argument("--dt", type=float, default=1800.0,
                        help="Time step in seconds (default: 1800)")
    parser.add_argument("--days", "-d", type=float, default=30.0,
                        help="Integration time in days (default: 30)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--test", "-t", type=str, default="all",
                        choices=["all", "gyre", "front", "kelvin"],
                        help="Which test to run (default: all)")
    args = parser.parse_args()

    output_dir = args.output or f"results/ocean_realistic_C{args.resolution}_L{args.levels}_{args.days:.0f}d"
    os.makedirs(output_dir, exist_ok=True)

    n_steps = int(args.days * 86400 / args.dt)
    point_size = max(1.0, 120 / args.resolution)

    # Banner
    print("=" * 70)
    print("legoESM — Realistic Ocean Experiments")
    print("=" * 70)
    print(f"  Backend:      {jax.default_backend()}")
    print(f"  Devices:      {jax.device_count()}")
    print(f"  Float dtype:  {jnp.zeros(1).dtype}")

    # Grid
    print(f"\nCreating C{args.resolution} cubed-sphere grid...")
    t0 = time.time()
    grid = create_cubed_sphere(args.resolution)
    print(f"  Grid created in {time.time()-t0:.1f}s")
    print(f"  Resolution: ~{grid.resolution_km:.0f} km")
    print(f"  Total cells: {grid.n_cells:,}")

    # Vertical
    print(f"\nCreating ocean z-star coordinate ({args.levels} levels)...")
    z_coord = create_ocean_z_star(n_levels=args.levels)
    print(f"  Surface dz: {float(z_coord.dz_ref[0]):.1f} m")
    print(f"  Bottom dz:  {float(z_coord.dz_ref[-1]):.1f} m")
    print(f"  Total depth: {float(z_coord.H_max):.0f} m")

    # Physics parameters scaled for resolution
    mean_dx = float(jnp.mean(grid.dx))
    hyperdiff_coeff = 1e-6 * mean_dx**4 / args.dt

    # Viscosity/diffusivity scaled with resolution
    # At coarse resolution, large A_h needed to control pressure gradient errors
    # from discrete representation of stratified flow on cubed sphere.
    # Reference: A_h ~ 1e5 at 1-degree (~100km). Scale as dx^1.5 (sub-gridscale).
    dx_ref = 100e3   # 100 km reference
    A_h_ref = 1e5    # m^2/s at 100km
    A_h = A_h_ref * (mean_dx / dx_ref)**1.5
    K_h = A_h / 10.0
    A_h = max(A_h, 1e4)
    K_h = max(K_h, 1e3)

    # Barotropic substeps: CFL for c = sqrt(gH) ~ 230 m/s
    # Use generous safety factor (4x) for stability of split-explicit coupling
    c_baro = float(jnp.sqrt(9.81 * 5500.0))
    n_baro = max(30, int(4.0 * c_baro * args.dt / mean_dx) + 1)

    # Vertical diffusivities: enhanced at coarse resolution to damp
    # discrete pressure gradient errors from stratification
    A_v = 1e-2  # m^2/s (enhanced vertical viscosity)
    K_v = 1e-3  # m^2/s (enhanced vertical tracer diffusion)

    print(f"\n  Physics parameters:")
    print(f"    Mean dx: {mean_dx/1000:.0f} km")
    print(f"    Horizontal viscosity A_h: {A_h:.2e} m^2/s")
    print(f"    Horizontal diffusivity K_h: {K_h:.2e} m^2/s")
    print(f"    Vertical viscosity A_v: {A_v:.2e} m^2/s")
    print(f"    Vertical diffusivity K_v: {K_v:.2e} m^2/s")
    print(f"    Hyperdiffusion: {hyperdiff_coeff:.2e}")
    print(f"    Barotropic substeps: {n_baro}")
    print(f"    Barotropic wave speed: {c_baro:.0f} m/s")

    config = OceanConfig(
        A_h=A_h,
        K_h=K_h,
        A_v=A_v,
        K_v=K_v,
        hyperdiff_coeff=hyperdiff_coeff,
        n_barotropic_substeps=n_baro,
        use_conservation_fixer=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
    )

    print(f"\n  Integration: {args.days} days ({n_steps} steps, dt={args.dt:.0f}s)")

    # =====================================================================
    # Run tests
    # =====================================================================
    results = {}

    if args.test in ("all", "gyre"):
        state, diag = stommel_gyre(args, grid, z_coord, config, output_dir, point_size)
        results["stommel_gyre"] = {"state": state, "diagnostics": diag}

    if args.test in ("all", "front"):
        state, diag = baroclinic_adjustment(args, grid, z_coord, config, output_dir, point_size)
        results["baroclinic_adjustment"] = {"state": state, "diagnostics": diag}

    if args.test in ("all", "kelvin"):
        state, diag = equatorial_kelvin_wave(args, grid, z_coord, config, output_dir, point_size)
        results["kelvin_wave"] = {"state": state, "diagnostics": diag}

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Resolution:  C{args.resolution} x {args.levels}L ({grid.n_cells:,} columns)")
    print(f"  Duration:    {args.days} days ({n_steps} steps, dt={args.dt:.0f}s)")
    print(f"  Physics:     A_h={A_h:.0e}, K_h={K_h:.0e}, A_v={A_v:.0e}, K_v={K_v:.0e}")

    for name, res in results.items():
        diag = res["diagnostics"]
        state = res["state"]
        d = diag[-1]
        print(f"\n  --- {name} ---")
        print(f"    SSH range:        [{d['SSH_min']:.4f}, {d['SSH_max']:.4f}] m")
        print(f"    Max current:      {d['speed_max']:.4f} m/s")
        print(f"    SST range:        [{d['SST_min']:.2f}, {d['SST_max']:.2f}] degC")
        print(f"    Mean SST:         {d['SST_mean']:.4f} degC")
        print(f"    Kinetic energy:   {d['kinetic_energy']:.4e}")

        # Conservation
        if abs(diag[0]["heat"]) > 1e-10:
            heat_drift = (diag[-1]["heat"] - diag[0]["heat"]) / abs(diag[0]["heat"])
            print(f"    Heat drift:       {heat_drift:.2e}")
        if abs(diag[0]["salt"]) > 1e-10:
            salt_drift = (diag[-1]["salt"] - diag[0]["salt"]) / abs(diag[0]["salt"])
            print(f"    Salt drift:       {salt_drift:.2e}")

        all_finite = bool(
            jnp.all(jnp.isfinite(state.u.data)) &
            jnp.all(jnp.isfinite(state.v.data)) &
            jnp.all(jnp.isfinite(state.T.data)) &
            jnp.all(jnp.isfinite(state.eta.data))
        )
        print(f"    All finite:       {all_finite}")

    # Summary text file
    with open(Path(output_dir) / "summary.txt", "w") as f:
        f.write("legoESM Ocean Realistic Experiments\n")
        f.write("=" * 50 + "\n")
        f.write(f"Resolution: C{args.resolution}, L{args.levels}\n")
        f.write(f"Time step: {args.dt:.0f} s\n")
        f.write(f"Duration: {args.days} days ({n_steps} steps)\n")
        f.write(f"A_h={A_h:.2e}, K_h={K_h:.2e}, A_v={A_v:.2e}, K_v={K_v:.2e}\n")
        f.write(f"Hyperdiffusion: {hyperdiff_coeff:.2e}\n")
        f.write(f"Barotropic substeps: {n_baro}\n\n")
        for name, res in results.items():
            d = res["diagnostics"][-1]
            f.write(f"{name}:\n")
            f.write(f"  SSH: [{d['SSH_min']:.4f}, {d['SSH_max']:.4f}] m\n")
            f.write(f"  Max speed: {d['speed_max']:.4f} m/s\n")
            f.write(f"  SST: [{d['SST_min']:.2f}, {d['SST_max']:.2f}] degC\n")
            f.write(f"  KE: {d['kinetic_energy']:.4e}\n\n")

    print(f"\n  Output: {output_dir}/")
    for name in results:
        subdir = {"stommel_gyre": "stommel_gyre",
                  "baroclinic_adjustment": "baroclinic_adjustment",
                  "kelvin_wave": "kelvin_wave"}[name]
        print(f"    {subdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
