#!/usr/bin/env python
"""Run standardized ocean model tests with visualization.

Three test cases:
  1. Rest-state adjustment — verify tendency magnitudes and conservation
  2. Barotropic gravity wave — Gaussian SSH perturbation, wave propagation
  3. Wind-driven gyre — idealized zonal wind stress, Sverdrup balance

Each test produces PNG visualizations and a summary of diagnostics.

Usage:
    cd /Users/pierregentine/legoESM
    source .venv/bin/activate

    # Quick test (C8, 10 levels, 5 days)
    python scripts/run_ocean_tests.py

    # Higher resolution (C16, 20 levels, 30 days)
    python scripts/run_ocean_tests.py --resolution 16 --levels 20 --days 30
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


IDEALIZED_LAND_LAT_THRESHOLD = 90.0  # Fully oceanic mask for idealized cases

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean import (
    OceanModel,
    OceanConfig,
    OceanState,
    create_ocean_z_star,
    rest_state_ocean,
    idealized_bathymetry,
)
from legoesm.ocean.eos import wright_eos, rho_0
from legoesm.ocean.vertical import compute_layer_thickness, compute_ocean_jacobian
from legoesm.ocean.conservation import _ocean_area_sum, _ocean_volume_sum
from legoesm.core.field import Field


# =====================================================================
# Helpers
# =====================================================================

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


def compute_ocean_diagnostics(state, grid, z_coord):
    """Compute scalar diagnostics for the ocean state."""
    mask = state.land_mask.data
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)

    # Volume: integral of eta over ocean
    vol = float(_ocean_area_sum(state.eta.data, mask, grid))

    # Heat: integral of T * h_k over ocean
    heat = float(_ocean_volume_sum(state.T.data, h_k, mask, grid))

    # Salt: integral of S * h_k over ocean
    salt = float(_ocean_volume_sum(state.S.data, h_k, mask, grid))

    # Kinetic energy: 0.5 * integral of (u^2 + v^2) * h_k over ocean
    ke = float(_ocean_volume_sum(
        0.5 * (state.u.data**2 + state.v.data**2), h_k, mask, grid,
    ))

    return {
        "volume": vol,
        "heat": heat,
        "salt": salt,
        "kinetic_energy": ke,
        "SST_mean": float(jnp.sum(state.T.data[..., 0] * mask) /
                          jnp.maximum(jnp.sum(mask), 1.0)),
        "SST_max": float(jnp.max(state.T.data[..., 0] * mask)),
        "SSH_min": float(jnp.min(jnp.where(mask > 0.5, state.eta.data, 0.0))),
        "SSH_max": float(jnp.max(state.eta.data * mask)),
        "u_max": float(jnp.max(jnp.abs(state.u.data))),
        "v_max": float(jnp.max(jnp.abs(state.v.data))),
    }


def add_wind_stress_tendency(state, grid, z_coord, config, tau_max=0.1):
    """Compute wind stress tendency for wind-driven gyre.

    Idealized zonal wind stress: tau_x = -tau_max * cos(2*pi*lat/60)
    applied in the band 15N-75N, zero elsewhere.
    Applied as du/dt = tau_x / (rho_0 * h_surface) in the top layer.
    """
    lat_deg = grid.lat * (180.0 / jnp.pi)

    # Wind stress: sinusoidal profile over Northern Hemisphere mid-latitudes
    tau_x = -tau_max * jnp.cos(2.0 * jnp.pi * (lat_deg - 45.0) / 60.0)
    # Restrict to 15N-75N ocean band
    wind_mask = jnp.where(
        (lat_deg > 15.0) & (lat_deg < 75.0) & (state.land_mask.data > 0.5),
        1.0, 0.0,
    )
    tau_x = tau_x * wind_mask

    # Apply to surface layer only
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
    h_surface = jnp.maximum(h_k[..., 0], 1.0)
    du_surface = tau_x / (rho_0 * h_surface)

    # Build full 3D tendency (only surface layer)
    du_3d = jnp.zeros_like(state.u.data)
    du_3d = du_3d.at[..., 0].set(du_surface)

    return du_3d


# =====================================================================
# Test Case 1: Rest-state adjustment
# =====================================================================

def run_rest_state_test(grid, z_coord, config, dt, n_steps, output_dir, point_size):
    """Rest-state adjustment: start from rest, verify small tendencies."""
    print("\n" + "=" * 70)
    print("TEST 1: Rest-State Adjustment")
    print("=" * 70)

    model = OceanModel(grid, z_coord, config)
    state = _rest_state_all_ocean(grid, z_coord)
    _assert_all_ocean(state, "rest_state_test")
    state_init = state

    print(f"  SST range: [{float(jnp.min(state.T.data[..., 0])):.1f}, "
          f"{float(jnp.max(state.T.data[..., 0])):.1f}] degC")
    ocean_S = state.S.data * state.land_mask.data[..., jnp.newaxis]
    ocean_count = jnp.sum(state.land_mask.data) * z_coord.n_levels
    print(f"  Salinity: {float(jnp.sum(ocean_S) / jnp.maximum(ocean_count, 1.0)):.1f} PSU")

    # JIT warmup
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = model.step(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Diagnostics storage
    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    diag_interval = max(1, n_steps // 50)

    # Integration
    print(f"\n{'Step':>6s}  {'Day':>6s}  {'SSH_max':>10s}  {'|u|_max':>10s}  "
          f"{'SST_mean':>10s}  {'KE':>12s}")
    print("-" * 70)

    t_start = time.time()
    for step in range(1, n_steps + 1):
        state = model.step(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:6.1f}  {diag['SSH_max']:10.2e}  "
                  f"{diag['u_max']:10.2e}  {diag['SST_mean']:10.4f}  "
                  f"{diag['kinetic_energy']:12.2e}")

    total_time = time.time() - t_start
    print(f"\nCompleted in {total_time:.1f}s ({n_steps/total_time:.0f} steps/s)")

    # --- Visualization ---
    os.makedirs(f"{output_dir}/rest_state", exist_ok=True)
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    if HAS_CARTOPY:
        # SST initial and final
        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  subplot_kw={"projection": _idealized_projection()})
        for ax, (label, snap) in zip(axes, [("Initial", state_init), ("Final", state)]):
            sst = np.asarray(snap.T.data[..., 0])
            sc = scatter_field(ax, lon_deg, lat_deg, sst, "RdYlBu_r",
                               vmin=0, vmax=22, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SST — {label}", fontsize=13, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                     label="Temperature [degC]", pad=0.02)
        fig.suptitle("Rest-State Test — Sea Surface Temperature", fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/rest_state/sst.png", dpi=150, bbox_inches="tight")
        plt.close()

        # SSH final
        fig, ax = plt.subplots(1, 1, figsize=(14, 7),
                                subplot_kw={"projection": _idealized_projection()})
        ssh = np.asarray(state.eta.data)
        vabs = max(abs(float(np.nanmin(ssh))), abs(float(np.nanmax(ssh))), 1e-10)
        sc = scatter_field(ax, lon_deg, lat_deg, ssh, "RdBu_r",
                           -vabs, vabs, point_size=point_size)
        _style_idealized_axes(ax)
        ax.set_title("Sea Surface Height [m]", fontsize=14, fontweight="bold")
        fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical",
                     label="SSH [m]", pad=0.02)
        plt.savefig(f"{output_dir}/rest_state/ssh.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Conservation time series
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Volume (eta integral)
    vol = np.array([d["volume"] for d in diagnostics])
    axes[0, 0].plot(times_days, vol, "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("Volume integral [m^3]")
    axes[0, 0].set_title("Volume Conservation (eta integral)")
    axes[0, 0].grid(True, alpha=0.3)

    # Heat
    heat = np.array([d["heat"] for d in diagnostics])
    if abs(heat[0]) > 0:
        heat_rel = (heat - heat[0]) / abs(heat[0])
        axes[0, 1].plot(times_days, heat_rel, "r-", linewidth=1.5)
        axes[0, 1].set_ylabel("Relative change")
    else:
        axes[0, 1].plot(times_days, heat, "r-", linewidth=1.5)
        axes[0, 1].set_ylabel("Heat integral")
    axes[0, 1].set_title("Heat Conservation")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    # Salt
    salt = np.array([d["salt"] for d in diagnostics])
    if abs(salt[0]) > 0:
        salt_rel = (salt - salt[0]) / abs(salt[0])
        axes[1, 0].plot(times_days, salt_rel, "g-", linewidth=1.5)
        axes[1, 0].set_ylabel("Relative change")
    else:
        axes[1, 0].plot(times_days, salt, "g-", linewidth=1.5)
        axes[1, 0].set_ylabel("Salt integral")
    axes[1, 0].set_title("Salt Conservation")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    # Kinetic energy
    ke = np.array([d["kinetic_energy"] for d in diagnostics])
    axes[1, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[1, 1].set_ylabel("Kinetic energy [J/m]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Kinetic Energy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    fig.suptitle("Rest-State Test — Conservation Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rest_state/conservation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Vertical temperature profile
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    z_ref = np.asarray(z_coord.z_full_ref)
    # Pick a mid-latitude ocean point (face 0, center)
    mid = grid.n // 2
    T_init = np.asarray(state_init.T.data[0, mid, mid, :])
    T_final = np.asarray(state.T.data[0, mid, mid, :])
    ax.plot(T_init, z_ref, "b-o", markersize=3, label="Initial", linewidth=1.5)
    ax.plot(T_final, z_ref, "r--s", markersize=3, label="Final", linewidth=1.5)
    ax.set_xlabel("Temperature [degC]", fontsize=12)
    ax.set_ylabel("Depth [m]", fontsize=12)
    ax.set_title("Vertical Temperature Profile (mid-latitude point)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rest_state/T_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    return state, diagnostics


# =====================================================================
# Test Case 2: Barotropic gravity wave
# =====================================================================

def run_gravity_wave_test(grid, z_coord, config, dt, n_steps, output_dir, point_size):
    """Barotropic gravity wave: Gaussian SSH perturbation."""
    print("\n" + "=" * 70)
    print("TEST 2: Barotropic Gravity Wave")
    print("=" * 70)

    # Start from rest state, add Gaussian SSH perturbation
    state = _rest_state_all_ocean(grid, z_coord)
    _assert_all_ocean(state, "gravity_wave_test")

    # Gaussian SSH perturbation centered at (lon=180, lat=0)
    lon_rad = grid.lon
    lat_rad = grid.lat
    lon0 = jnp.pi  # 180 degrees
    lat0 = 0.0     # equator
    R_earth = 6.371e6
    sigma = 10.0 * jnp.pi / 180.0  # 10 degrees

    # Great-circle distance
    dlon = lon_rad - lon0
    dist_angle = jnp.arccos(
        jnp.clip(jnp.sin(lat_rad) * jnp.sin(lat0) +
                 jnp.cos(lat_rad) * jnp.cos(lat0) * jnp.cos(dlon), -1.0, 1.0)
    )
    eta_pert = 1.0 * jnp.exp(-0.5 * (dist_angle / sigma) ** 2)  # 1m amplitude
    eta_pert = eta_pert * state.land_mask.data

    state = state._replace(
        eta=state.eta.replace(data=eta_pert.astype(jnp.float32)),
    )
    state_init = state

    print(f"  SSH perturbation: {float(jnp.max(eta_pert)):.3f} m (Gaussian, sigma=10deg)")
    c_wave = float(jnp.sqrt(9.81 * 5500.0))
    print(f"  Expected wave speed: {c_wave:.0f} m/s")

    model = OceanModel(grid, z_coord, config)

    # JIT warmup
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = model.step(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Diagnostics + snapshots
    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
    snapshot_steps = {int(f * n_steps): f for f in snapshot_fracs}
    snapshots = {0: state}
    diag_interval = max(1, n_steps // 50)

    print(f"\n{'Step':>6s}  {'Day':>6s}  {'SSH_min':>10s}  {'SSH_max':>10s}  "
          f"{'|u|_max':>10s}  {'KE':>12s}")
    print("-" * 70)

    t_start = time.time()
    for step in range(1, n_steps + 1):
        state = model.step(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:6.2f}  {diag['SSH_min']:10.4f}  "
                  f"{diag['SSH_max']:10.4f}  {diag['u_max']:10.4f}  "
                  f"{diag['kinetic_energy']:12.4e}")

        if step in snapshot_steps:
            snapshots[step] = state

    total_time = time.time() - t_start
    print(f"\nCompleted in {total_time:.1f}s ({n_steps/total_time:.0f} steps/s)")

    # --- Visualization ---
    os.makedirs(f"{output_dir}/gravity_wave", exist_ok=True)
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    if HAS_CARTOPY:
        # SSH snapshots (4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                                  subplot_kw={"projection": _idealized_projection(central_longitude=180)})
        axes = axes.ravel()

        # Find global SSH range across all snapshots
        all_ssh = [np.asarray(s.eta.data) for s in snapshots.values()]
        ssh_max = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
        ssh_max = max(ssh_max, 0.01)

        for idx, (step_num, snap) in enumerate(sorted(snapshots.items())):
            if idx >= 4:
                break
            day = step_num * dt / 86400.0
            ax = axes[idx]
            ssh_data = np.asarray(snap.eta.data)
            sc = scatter_field(ax, lon_deg, lat_deg, ssh_data, "RdBu_r",
                               -ssh_max, ssh_max, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"Day {day:.2f}", fontsize=13, fontweight="bold")

        fig.suptitle("Barotropic Gravity Wave — Sea Surface Height [m]",
                     fontsize=15, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, pad=0.02, orientation="vertical",
                     label="SSH [m]")
        plt.savefig(f"{output_dir}/gravity_wave/ssh_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Velocity field at final step
        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  subplot_kw={"projection": _idealized_projection(central_longitude=180)})
        for ax, (data, title, cmap) in zip(axes, [
            (np.asarray(state.u.data[..., 0]), "Surface u [m/s]", "RdBu_r"),
            (np.asarray(state.v.data[..., 0]), "Surface v [m/s]", "RdBu_r"),
        ]):
            vabs = max(float(np.nanmax(np.abs(data))), 1e-10)
            sc = scatter_field(ax, lon_deg, lat_deg, data, cmap, -vabs, vabs,
                               point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(title, fontsize=13, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical", pad=0.02)
        fig.suptitle("Barotropic Gravity Wave — Surface Velocity (Final)",
                     fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/gravity_wave/velocity_final.png", dpi=150, bbox_inches="tight")
        plt.close()

    # SSH time series
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    axes[0].plot(times_days, [d["SSH_max"] for d in diagnostics], "b-", label="SSH max")
    axes[0].plot(times_days, [d["SSH_min"] for d in diagnostics], "r-", label="SSH min")
    axes[0].set_ylabel("SSH [m]")
    axes[0].set_title("SSH Extremes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_days, [d["kinetic_energy"] for d in diagnostics], "m-", linewidth=1.5)
    axes[1].set_ylabel("KE [J/m]")
    axes[1].set_xlabel("Time [days]")
    axes[1].set_title("Kinetic Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    fig.suptitle("Barotropic Gravity Wave — Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gravity_wave/diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    return state, diagnostics


# =====================================================================
# Test Case 3: Wind-driven gyre
# =====================================================================

def run_wind_driven_gyre_test(grid, z_coord, config, dt, n_steps, output_dir, point_size):
    """Wind-driven double gyre with idealized zonal wind stress."""
    print("\n" + "=" * 70)
    print("TEST 3: Wind-Driven Gyre")
    print("=" * 70)

    state = _rest_state_all_ocean(grid, z_coord)
    _assert_all_ocean(state, "wind_driven_gyre_test")
    state_init = state

    tau_max = 0.1  # N/m^2
    print(f"  Wind stress: tau_max = {tau_max} N/m^2")
    print(f"  Wind pattern: -tau_max * cos(2*pi*(lat-45)/60), 15N-75N")

    model = OceanModel(grid, z_coord, config)

    # JIT warmup
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = model.step(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Diagnostics + snapshots
    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
    snapshot_steps = {int(f * n_steps): f for f in snapshot_fracs}
    snapshots = {0: state}
    diag_interval = max(1, n_steps // 50)

    print(f"\n{'Step':>6s}  {'Day':>6s}  {'SSH_max':>10s}  {'|u|_max':>10s}  "
          f"{'SST_mean':>10s}  {'KE':>12s}")
    print("-" * 70)

    t_start = time.time()
    for step in range(1, n_steps + 1):
        # Apply wind forcing to surface layer velocity
        du_wind = add_wind_stress_tendency(state, grid, z_coord, config, tau_max)
        state = state._replace(
            u=state.u.replace(data=state.u.data + dt * du_wind),
        )

        # Model step (dynamics + barotropic + conservation)
        state = model.step(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:6.1f}  {diag['SSH_max']:10.4f}  "
                  f"{diag['u_max']:10.4f}  {diag['SST_mean']:10.4f}  "
                  f"{diag['kinetic_energy']:12.4e}")

        if step in snapshot_steps:
            snapshots[step] = state

    total_time = time.time() - t_start
    print(f"\nCompleted in {total_time:.1f}s ({n_steps/total_time:.0f} steps/s)")

    # --- Visualization ---
    os.makedirs(f"{output_dir}/wind_gyre", exist_ok=True)
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    if HAS_CARTOPY:
        # SSH snapshots (4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                                  subplot_kw={"projection": _idealized_projection()})
        axes = axes.ravel()

        all_ssh = [np.asarray(s.eta.data) for s in snapshots.values()]
        ssh_abs = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
        ssh_abs = max(ssh_abs, 0.001)

        for idx, (step_num, snap) in enumerate(sorted(snapshots.items())):
            if idx >= 4:
                break
            day = step_num * dt / 86400.0
            ax = axes[idx]
            ssh_data = np.asarray(snap.eta.data)
            sc = scatter_field(ax, lon_deg, lat_deg, ssh_data, "RdBu_r",
                               -ssh_abs, ssh_abs, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"Day {day:.0f}", fontsize=13, fontweight="bold")

        fig.suptitle("Wind-Driven Gyre — Sea Surface Height [m]",
                     fontsize=15, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, pad=0.02, orientation="vertical",
                     label="SSH [m]")
        plt.savefig(f"{output_dir}/wind_gyre/ssh_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Surface velocity + speed at final time
        fig, axes = plt.subplots(1, 3, figsize=(22, 5),
                                  subplot_kw={"projection": _idealized_projection()})

        u_surf = np.asarray(state.u.data[..., 0])
        v_surf = np.asarray(state.v.data[..., 0])
        speed = np.sqrt(u_surf**2 + v_surf**2)

        for ax, (data, title, cmap, sym) in zip(axes, [
            (u_surf, "Surface u [m/s]", "RdBu_r", True),
            (v_surf, "Surface v [m/s]", "RdBu_r", True),
            (speed, "Surface speed [m/s]", "magma", False),
        ]):
            if sym:
                vabs = max(float(np.nanmax(np.abs(data))), 1e-10)
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = 0, max(float(np.nanmax(data)), 1e-10)
            sc = scatter_field(ax, lon_deg, lat_deg, data, cmap, vmin, vmax,
                               point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(title, fontsize=12, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical", pad=0.02)

        fig.suptitle("Wind-Driven Gyre — Surface Velocity (Final)",
                     fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/wind_gyre/velocity_final.png", dpi=150, bbox_inches="tight")
        plt.close()

        # SST evolution
        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  subplot_kw={"projection": _idealized_projection()})
        for ax, (label, snap) in zip(axes, [("Initial", state_init), ("Final", state)]):
            sst = np.asarray(snap.T.data[..., 0])
            sc = scatter_field(ax, lon_deg, lat_deg, sst, "RdYlBu_r",
                               vmin=0, vmax=22, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SST — {label}", fontsize=13, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                     label="Temperature [degC]", pad=0.02)
        fig.suptitle("Wind-Driven Gyre — Sea Surface Temperature",
                     fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/wind_gyre/sst.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Diagnostics time series
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(times_days, [d["SSH_max"] for d in diagnostics], "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("SSH max [m]")
    axes[0, 0].set_title("Maximum SSH")
    axes[0, 0].grid(True, alpha=0.3)

    ke = np.array([d["kinetic_energy"] for d in diagnostics])
    axes[0, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[0, 1].set_ylabel("KE [J/m]")
    axes[0, 1].set_title("Kinetic Energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    axes[1, 0].plot(times_days, [d["u_max"] for d in diagnostics], "r-", linewidth=1.5)
    axes[1, 0].set_ylabel("|u| max [m/s]")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_title("Maximum Velocity")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times_days, [d["SST_mean"] for d in diagnostics], "g-", linewidth=1.5)
    axes[1, 1].set_ylabel("Mean SST [degC]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Mean Sea Surface Temperature")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Wind-Driven Gyre — Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wind_gyre/diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Vertical temperature profile at final time
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    z_ref = np.asarray(z_coord.z_full_ref)
    mid = grid.n // 2
    T_init = np.asarray(state_init.T.data[0, mid, mid, :])
    T_final = np.asarray(state.T.data[0, mid, mid, :])
    ax.plot(T_init, z_ref, "b-o", markersize=3, label="Initial", linewidth=1.5)
    ax.plot(T_final, z_ref, "r--s", markersize=3, label="Final", linewidth=1.5)
    ax.set_xlabel("Temperature [degC]", fontsize=12)
    ax.set_ylabel("Depth [m]", fontsize=12)
    ax.set_title("Vertical T Profile (mid-latitude point)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wind_gyre/T_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    return state, diagnostics


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Ocean model standardized tests")
    parser.add_argument("--resolution", "-n", type=int, default=8,
                        help="Cubed-sphere resolution (default: 8)")
    parser.add_argument("--levels", "-l", type=int, default=10,
                        help="Number of vertical levels (default: 10)")
    parser.add_argument("--dt", type=float, default=3600.0,
                        help="Time step in seconds (default: 3600)")
    parser.add_argument("--days", "-d", type=float, default=5.0,
                        help="Integration time in days (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: results/ocean_tests_C{n}_L{l})")
    parser.add_argument("--test", "-t", type=str, default="all",
                        choices=["all", "rest", "wave", "gyre"],
                        help="Which test to run (default: all)")
    args = parser.parse_args()

    output_dir = args.output or f"results/ocean_tests_C{args.resolution}_L{args.levels}"
    os.makedirs(output_dir, exist_ok=True)

    n_steps = int(args.days * 86400 / args.dt)
    point_size = max(1.0, 120 / args.resolution)

    # Banner
    print("=" * 70)
    print("legoESM — Ocean Model Standardized Tests")
    print("=" * 70)
    print(f"  Backend:      {jax.default_backend()}")
    print(f"  Devices:      {jax.device_count()}")
    print(f"  Float dtype:  {jnp.zeros(1).dtype}")

    # Grid setup
    print(f"\nCreating C{args.resolution} cubed-sphere grid...")
    t0 = time.time()
    grid = create_cubed_sphere(args.resolution)
    print(f"  Grid created in {time.time()-t0:.1f}s")
    print(f"  Resolution: ~{grid.resolution_km:.0f} km")
    print(f"  Total cells: {grid.n_cells:,}")

    # Vertical coordinate
    print(f"\nCreating ocean z-star coordinate ({args.levels} levels)...")
    z_coord = create_ocean_z_star(n_levels=args.levels)
    print(f"  Surface dz: {float(z_coord.dz_ref[0]):.1f} m")
    print(f"  Bottom dz:  {float(z_coord.dz_ref[-1]):.1f} m")
    print(f"  Total depth: {float(z_coord.H_max):.0f} m")

    # Hyperdiffusion: scale with grid spacing
    # Ocean uses weaker hyperdiffusion than atmosphere (slower velocities)
    mean_dx = float(jnp.mean(grid.dx))
    hyperdiff_coeff = 1e-6 * mean_dx**4 / args.dt
    print(f"  Mean dx: {mean_dx/1000:.0f} km")
    print(f"  Hyperdiffusion coeff: {hyperdiff_coeff:.2e}")

    # Barotropic substeps: CFL for barotropic gravity waves
    # c = sqrt(gH) ~ 230 m/s, need dt_sub < dx / c
    c_baro = float(jnp.sqrt(9.81 * 5500.0))
    n_baro = max(10, int(2.0 * c_baro * args.dt / mean_dx) + 1)
    print(f"  Barotropic wave speed: {c_baro:.0f} m/s")
    print(f"  Barotropic substeps: {n_baro}")

    config = OceanConfig(
        hyperdiff_coeff=hyperdiff_coeff,
        n_barotropic_substeps=n_baro,
        use_conservation_fixer=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
    )

    print(f"\nIntegration: {args.days} days ({n_steps} steps, dt={args.dt:.0f}s)")

    # =====================================================================
    # Run tests
    # =====================================================================
    results = {}

    if args.test in ("all", "rest"):
        state, diag = run_rest_state_test(
            grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
        )
        results["rest_state"] = {"state": state, "diagnostics": diag}

    if args.test in ("all", "wave"):
        state, diag = run_gravity_wave_test(
            grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
        )
        results["gravity_wave"] = {"state": state, "diagnostics": diag}

    if args.test in ("all", "gyre"):
        state, diag = run_wind_driven_gyre_test(
            grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
        )
        results["wind_gyre"] = {"state": state, "diagnostics": diag}

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Resolution:  C{args.resolution} x {args.levels}L ({grid.n_cells:,} columns)")
    print(f"  Duration:    {args.days} days ({n_steps} steps, dt={args.dt:.0f}s)")

    for name, res in results.items():
        diag = res["diagnostics"]
        state = res["state"]
        print(f"\n  --- {name} ---")
        print(f"    Final SSH range:  [{diag[-1]['SSH_min']:.4e}, {diag[-1]['SSH_max']:.4e}] m")
        print(f"    Final |u| max:    {diag[-1]['u_max']:.4e} m/s")
        print(f"    Final SST mean:   {diag[-1]['SST_mean']:.4f} degC")
        print(f"    Final KE:         {diag[-1]['kinetic_energy']:.4e}")

        # Check finiteness
        all_finite = bool(
            jnp.all(jnp.isfinite(state.u.data)) &
            jnp.all(jnp.isfinite(state.v.data)) &
            jnp.all(jnp.isfinite(state.T.data)) &
            jnp.all(jnp.isfinite(state.eta.data))
        )
        print(f"    All fields finite: {all_finite}")

        # Land still zero
        mask = state.land_mask.data
        land = mask < 0.5
        land_3d = jnp.broadcast_to(land[..., jnp.newaxis], state.u.data.shape)
        land_zero = bool(
            jnp.all(jnp.where(land_3d, state.u.data, 0.0) == 0) &
            jnp.all(jnp.where(land, state.eta.data, 0.0) == 0)
        ) if jnp.any(land) else True
        print(f"    Land cells zero:   {land_zero}")

    print(f"\n  Output directory: {output_dir}/")
    for name in results:
        subdir = {"rest_state": "rest_state", "gravity_wave": "gravity_wave",
                  "wind_gyre": "wind_gyre"}[name]
        print(f"    {subdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
