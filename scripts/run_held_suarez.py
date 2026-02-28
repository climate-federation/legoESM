#!/usr/bin/env python
"""Run the Held-Suarez benchmark with the hydrostatic PE dynamical core.

This script integrates the hydrostatic primitive equations with Held-Suarez
forcing at specified resolution on the cubed-sphere.

Expected results (C48, 1200 days):
- Subtropical jets ~30 m/s at ~200 hPa and ~30° latitude
- Hadley cell circulation
- Mass conservation to float32 precision

Usage:
    cd /Users/pierregentine/legoESM
    source .venv/bin/activate

    # Quick test (C16, 30 days, ~10 min)
    python scripts/run_held_suarez.py --resolution 16 --days 30

    # Full benchmark (C48, 1200 days)
    python scripts/run_held_suarez.py --resolution 48 --days 1200
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

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate, pressure_from_sigma
from legoesm.core.operators import global_integral
from legoesm.atmosphere.dynamics.primitive_eq import (
    PrimitiveEquationModel,
    PrimitiveEquationConfig,
)
from legoesm.atmosphere.physics.held_suarez import (
    held_suarez_forcing,
    held_suarez_init,
)
from legoesm import constants


def compute_hyperdiff_coeff(n_grid: int, reference_n: int = 48,
                            reference_coeff: float = 5e16) -> float:
    """Scale hyperdiffusion coefficient with resolution.

    For ∇⁴ diffusion, ν scales as (dx)⁴ / τ. Since dx ∝ 1/n_grid,
    ν ∝ (1/n_grid)⁴. We use C48 as the reference.

    Parameters
    ----------
    n_grid : int
        Grid resolution.
    reference_n : int
        Reference resolution (C48).
    reference_coeff : float
        Hyperdiffusion coefficient at reference resolution.
    """
    return reference_coeff * (reference_n / n_grid) ** 4


def main():
    parser = argparse.ArgumentParser(description="Held-Suarez benchmark")
    parser.add_argument("--resolution", "-n", type=int, default=48,
                        help="Cubed-sphere resolution (default: 48)")
    parser.add_argument("--levels", "-l", type=int, default=20,
                        help="Number of vertical levels (default: 20)")
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Time step in seconds (default: 600)")
    parser.add_argument("--days", "-d", type=int, default=1200,
                        help="Integration time in days (default: 1200)")
    parser.add_argument("--hyperdiff", type=float, default=None,
                        help="Hyperdiffusion coefficient (default: auto-scaled)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    N_GRID = args.resolution
    N_LEVELS = args.levels
    DT = args.dt
    N_DAYS = args.days
    DIAG_INTERVAL = 6   # Save diagnostics every N hours
    SNAP_INTERVAL = max(10, N_DAYS // 10)  # Save snapshots at ~10 points

    # Auto-scale hyperdiffusion if not specified
    HYPERDIFF_COEFF = args.hyperdiff or compute_hyperdiff_coeff(N_GRID)

    # Output directory
    OUTPUT_DIR = Path(args.output or f"results/held_suarez_C{N_GRID}_L{N_LEVELS}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Held-Suarez Benchmark: Hydrostatic PE on Cubed-Sphere")
    print("=" * 70)
    print(f"Resolution: C{N_GRID} ({N_GRID}x{N_GRID} per face)")
    print(f"Levels: {N_LEVELS}")
    print(f"Time step: {DT:.0f} s")
    print(f"Duration: {N_DAYS} days")
    print(f"Hyperdiffusion: {HYPERDIFF_COEFF:.2e}")
    print()

    # --- Setup ---
    print("Creating grid and initial conditions...")
    grid = create_cubed_sphere(N_GRID)
    sigma = create_sigma_coordinate(N_LEVELS)
    print(f"  Grid resolution: ~{float(grid.resolution_km):.0f} km")
    print(f"  Sigma levels: {N_LEVELS} (uniform)")
    print(f"  Sigma range: [{float(sigma.sigma_half[0]):.3f}, {float(sigma.sigma_half[-1]):.3f}]")

    config = PrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF_COEFF,
        hyperdiff_ps_coeff=HYPERDIFF_COEFF,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = PrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)
    print(f"  Initial T: {float(jnp.mean(state.T.data)):.1f} K")
    print(f"  Initial p_s: {float(jnp.mean(state.p_s.data)):.0f} Pa")

    # --- Integration ---
    n_steps_total = int(N_DAYS * 86400 / DT)
    diag_every = int(DIAG_INTERVAL * 3600 / DT)
    snap_every = int(SNAP_INTERVAL * 86400 / DT)

    print(f"\nIntegrating for {n_steps_total} steps...")
    print(f"  Diagnostics every {diag_every} steps ({DIAG_INTERVAL}h)")
    print(f"  Snapshots every {snap_every} steps ({SNAP_INTERVAL} days)")

    # Storage for diagnostics
    diag_times = []
    diag_mass = []
    diag_max_wind = []
    diag_mean_T = []

    # Storage for snapshots
    snap_days_target = sorted(set([0] + list(range(SNAP_INTERVAL, N_DAYS + 1, SNAP_INTERVAL)) + [N_DAYS]))
    snap_steps = {int(d * 86400 / DT): d for d in snap_days_target}
    snapshots = {}

    # Save initial snapshot
    snapshots[0] = jax.tree.map(lambda x: np.array(x), state)

    mass_initial = float(global_integral(state.p_s, grid))
    t_start = time.time()
    last_print = t_start

    # JIT warmup
    print("  JIT compiling (first step)...", end=" ", flush=True)
    t_jit = time.time()
    state = model.step_with_physics(state, DT, held_suarez_forcing)
    jax.block_until_ready(state.u.data)
    print(f"done ({time.time() - t_jit:.1f}s)")

    for step in range(1, n_steps_total):
        state = model.step_with_physics(state, DT, held_suarez_forcing)

        # Check for blowup
        if step % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 1000:
                day = (step + 1) * DT / 86400.0
                print(f"\n  *** BLOWUP at day {day:.1f}, step {step+1}, u_max={u_max:.1f} ***")
                break

        # Diagnostics
        if (step + 1) % diag_every == 0:
            day = (step + 1) * DT / 86400.0
            mass = float(global_integral(state.p_s, grid))
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            mean_T = float(jnp.mean(state.T.data))

            diag_times.append(day)
            diag_mass.append(mass)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)

            now = time.time()
            if now - last_print > 30:  # Print progress every 30 seconds
                elapsed = now - t_start
                steps_per_sec = (step + 1) / elapsed
                eta = (n_steps_total - step - 1) / steps_per_sec
                mass_drift = (mass - mass_initial) / mass_initial
                print(
                    f"  Day {day:7.1f}/{N_DAYS} | "
                    f"max |v|={max_wind:6.1f} m/s | "
                    f"<T>={mean_T:6.1f} K | "
                    f"mass drift={mass_drift:+.2e} | "
                    f"{steps_per_sec:.0f} steps/s | "
                    f"ETA {eta/60:.0f} min"
                )
                last_print = now

        # Save snapshots
        if (step + 1) in snap_steps:
            day = snap_steps[step + 1]
            snapshots[day] = jax.tree.map(lambda x: np.array(x), state)
            print(f"  *** Snapshot saved at day {day} ***")

    elapsed = time.time() - t_start
    print(f"\nIntegration complete: {elapsed:.0f}s ({n_steps_total/elapsed:.0f} steps/s)")

    # Final diagnostics
    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = (mass_final - mass_initial) / mass_initial
    print(f"  Mass drift: {mass_drift:+.2e} (relative)")
    print(f"  Max wind: {float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))):.1f} m/s")
    print(f"  Mean T: {float(jnp.mean(state.T.data)):.1f} K")

    # ===========================================================================
    # Generate plots
    # ===========================================================================
    print("\nGenerating plots...")

    diag_times = np.array(diag_times)
    diag_mass = np.array(diag_mass)
    diag_max_wind = np.array(diag_max_wind)
    diag_mean_T = np.array(diag_mean_T)

    # --- 1. Conservation time series ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    mass_drift_ts = (diag_mass - mass_initial) / mass_initial
    axes[0].plot(diag_times, mass_drift_ts)
    axes[0].set_ylabel("Mass drift (relative)")
    axes[0].set_title("Global mass conservation")
    axes[0].grid(True)

    axes[1].plot(diag_times, diag_max_wind)
    axes[1].set_ylabel("Max |v| [m/s]")
    axes[1].set_title("Maximum wind speed")
    axes[1].grid(True)

    axes[2].plot(diag_times, diag_mean_T)
    axes[2].set_xlabel("Time [days]")
    axes[2].set_ylabel("Mean T [K]")
    axes[2].set_title("Global mean temperature")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conservation_timeseries.png", dpi=150)
    plt.close()
    print("  Saved conservation_timeseries.png")

    # --- 2. Zonal mean diagnostics ---
    final_state = jax.tree.map(lambda x: np.array(x), state)

    lat_np = np.array(grid.lat)  # (6, n, n)
    lat_flat = lat_np.flatten()
    area_flat = np.array(grid.area).flatten()

    # Adaptive binning: fewer bins at coarse resolution to ensure coverage
    n_lat_bins = min(91, max(30, 2 * N_GRID))
    lat_bins = np.linspace(-np.pi/2, np.pi/2, n_lat_bins + 1)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    u_final = np.array(final_state.u.data)  # (6,n,n,nlev)
    T_final = np.array(final_state.T.data)
    nlev = sigma.n_levels

    # Initialize with NaN so empty bins are detectable (not 0 K!)
    u_zonal = np.full((len(lat_centers), nlev), np.nan)
    T_zonal = np.full((len(lat_centers), nlev), np.nan)

    for i in range(len(lat_centers)):
        mask = (lat_flat >= lat_bins[i]) & (lat_flat < lat_bins[i+1])
        if np.sum(mask) > 0:
            w = area_flat[mask]
            w = w / w.sum()  # area-weighted average
            for k in range(nlev):
                u_zonal[i, k] = np.sum(w * u_final[..., k].flatten()[mask])
                T_zonal[i, k] = np.sum(w * T_final[..., k].flatten()[mask])

    # Interpolate through any remaining empty bins
    for k in range(nlev):
        valid = ~np.isnan(T_zonal[:, k])
        if valid.any() and not valid.all():
            T_zonal[:, k] = np.interp(lat_centers, lat_centers[valid], T_zonal[valid, k])
            u_zonal[:, k] = np.interp(lat_centers, lat_centers[valid], u_zonal[valid, k])

    sigma_full_np = np.array(sigma.sigma_full)
    p_levels = sigma_full_np * 1000  # Approximate pressure in hPa

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cs = axes[0].contourf(
        np.degrees(lat_centers), p_levels, u_zonal.T,
        levels=np.arange(-20, 45, 5), cmap="RdBu_r", extend="both"
    )
    axes[0].set_ylim(1000, 0)
    axes[0].set_xlabel("Latitude [°]")
    axes[0].set_ylabel("Pressure [hPa]")
    axes[0].set_title(f"Zonal-mean zonal wind [m/s] (day {N_DAYS})")
    plt.colorbar(cs, ax=axes[0])

    cs = axes[1].contourf(
        np.degrees(lat_centers), p_levels, T_zonal.T,
        levels=20, cmap="RdYlBu_r"
    )
    axes[1].set_ylim(1000, 0)
    axes[1].set_xlabel("Latitude [°]")
    axes[1].set_ylabel("Pressure [hPa]")
    axes[1].set_title(f"Zonal-mean temperature [K] (day {N_DAYS})")
    plt.colorbar(cs, ax=axes[1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zonal_mean.png", dpi=150)
    plt.close()
    print("  Saved zonal_mean.png")

    # --- 3. Surface pressure snapshots ---
    n_snap_plot = min(len(snapshots), 4)
    fig, axes = plt.subplots(1, n_snap_plot, figsize=(5 * n_snap_plot, 5))
    if n_snap_plot == 1:
        axes = [axes]
    for idx, (day, snap) in enumerate(sorted(snapshots.items())[:n_snap_plot]):
        ax = axes[idx]
        ps = np.array(snap.p_s.data)
        im = ax.pcolormesh(ps[0] / 100, cmap="viridis")
        ax.set_title(f"p_s [hPa] face 0, day {day}")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "surface_pressure.png", dpi=150)
    plt.close()
    print("  Saved surface_pressure.png")

    # --- 4. Temperature profile at equator ---
    eq_mask = np.abs(lat_flat) < np.radians(5.0)
    eq_area = area_flat[eq_mask]
    eq_w = eq_area / eq_area.sum()
    T_eq_profile = np.zeros(nlev)
    for k in range(nlev):
        T_eq_profile[k] = np.sum(eq_w * T_final[..., k].flatten()[eq_mask])

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(T_eq_profile, p_levels, "b-o", markersize=4)
    ax.set_ylim(1000, 0)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [hPa]")
    ax.set_title(f"Equatorial temperature profile (day {N_DAYS})")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equatorial_T_profile.png", dpi=150)
    plt.close()
    print("  Saved equatorial_T_profile.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
