#!/usr/bin/env python
"""Run the Held-Suarez benchmark on a latitude-longitude grid.

This script integrates the hydrostatic primitive equations with Held-Suarez
forcing on a regular lat-lon grid with polar filtering.

Expected results (~64x128, 200 days):
- Subtropical jets ~25-40 m/s at ~200 hPa and ~30° latitude
- T_eq convergence: T_atm mean ~255 K
- Midlatitude baroclinic eddies
- No blowup (polar filter keeps poles stable)

Usage:
    cd /Users/pierregentine/legoESM
    source .venv/bin/activate

    # Quick test (32x64, 30 days)
    python scripts/run_held_suarez_latlon.py --n_lat 32 --days 30

    # Standard (64x128, 200 days)
    python scripts/run_held_suarez_latlon.py --n_lat 64 --days 200
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate, pressure_from_sigma
from legoesm.core.operators_latlon import global_integral
from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
    LatLonPrimitiveEquationModel,
    LatLonPrimitiveEquationConfig,
)
from legoesm.atmosphere.physics.held_suarez_latlon import (
    held_suarez_forcing_latlon,
    held_suarez_init_latlon,
)


def compute_hyperdiff_coeff(
    n_lat: int,
    reference_n: int = 64,
    reference_coeff: float = 2e16,
    min_coeff: float = 1e16,
) -> float:
    """Scale hyperdiffusion coefficient with resolution.

    For nabla^4 diffusion, nu scales as (dx)^4.
    A minimum floor keeps high-resolution lat-lon runs stable in SSP45 mode.
    """
    scaled = reference_coeff * (reference_n / n_lat) ** 4
    return max(scaled, min_coeff)


def main():
    parser = argparse.ArgumentParser(description="Held-Suarez on lat-lon grid")
    parser.add_argument("--n_lat", type=int, default=64,
                        help="Number of latitude points (default: 64)")
    parser.add_argument("--n_lon", type=int, default=None,
                        help="Number of longitude points (default: 2*n_lat)")
    parser.add_argument("--levels", "-l", type=int, default=20,
                        help="Number of vertical levels (default: 20)")
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Time step in seconds (default: 600)")
    parser.add_argument("--days", "-d", type=int, default=200,
                        help="Integration time in days (default: 200)")
    parser.add_argument("--hyperdiff", type=float, default=None,
                        help="Hyperdiffusion coefficient (default: auto-scaled)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    N_LAT = args.n_lat
    N_LON = args.n_lon or 2 * N_LAT
    N_LEVELS = args.levels
    DT = args.dt
    N_DAYS = args.days
    DIAG_INTERVAL = 6  # hours

    if args.hyperdiff is None:
        HYPERDIFF_COEFF = compute_hyperdiff_coeff(N_LAT)
    else:
        HYPERDIFF_COEFF = float(args.hyperdiff)

    OUTPUT_DIR = Path(args.output or f"results/atmosphere/hydrostatic/held_suarez_latlon_{N_LAT}x{N_LON}_L{N_LEVELS}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Held-Suarez Benchmark: Hydrostatic PE on Lat-Lon Grid")
    print("=" * 70)
    print(f"Resolution: {N_LAT} x {N_LON}")
    print(f"Levels: {N_LEVELS}")
    print(f"Time step: {DT:.0f} s")
    print(f"Duration: {N_DAYS} days")
    print(f"Hyperdiffusion: {HYPERDIFF_COEFF:.2e}")
    print()

    # --- Setup ---
    print("Creating grid and initial conditions...")
    grid = create_latlon_grid(N_LAT, N_LON)
    sigma = create_sigma_coordinate(N_LEVELS)

    dlat_deg = float(jnp.degrees(grid.dlat))
    print(f"  Grid spacing: ~{dlat_deg:.1f}° ({float(grid.radius * grid.dlat / 1e3):.0f} km)")
    print(f"  Sigma levels: {N_LEVELS}")

    config = LatLonPrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF_COEFF,
        hyperdiff_ps_coeff=HYPERDIFF_COEFF,
        use_conservation_fixer=True,
        fix_mass=True,
        use_polar_filter=True,
        polar_filter_cutoff_deg=60.0,
    )
    model = LatLonPrimitiveEquationModel(grid, sigma, config, dt=DT)
    state = held_suarez_init_latlon(grid, sigma)
    print(f"  Initial T: {float(jnp.mean(state.T.data)):.1f} K")
    print(f"  Initial p_s: {float(jnp.mean(state.p_s.data)):.0f} Pa")

    # --- Integration ---
    n_steps_total = int(N_DAYS * 86400 / DT)
    diag_every = int(DIAG_INTERVAL * 3600 / DT)

    print(f"\nIntegrating for {n_steps_total} steps...")

    diag_times = []
    diag_mass = []
    diag_max_wind = []
    diag_mean_T = []

    mass_initial = float(global_integral(state.p_s, grid))
    t_start = time.time()
    last_print = t_start

    # JIT warmup
    print("  JIT compiling (first step)...", end=" ", flush=True)
    t_jit = time.time()
    state = model.step_with_physics(state, DT, held_suarez_forcing_latlon)
    jax.block_until_ready(state.u.data)
    print(f"done ({time.time() - t_jit:.1f}s)")

    for step in range(1, n_steps_total):
        state = model.step_with_physics(state, DT, held_suarez_forcing_latlon)

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
            if now - last_print > 30:
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

    elapsed = time.time() - t_start
    print(f"\nIntegration complete: {elapsed:.0f}s ({n_steps_total/elapsed:.0f} steps/s)")

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
    lat_np = np.array(grid.lat)  # (n_lat,)
    u_final = np.array(state.u.data)  # (n_lat, n_lon, nlev)
    T_final = np.array(state.T.data)
    nlev = sigma.n_levels

    # Simple zonal mean (average over longitude)
    u_zonal = np.mean(u_final, axis=1)  # (n_lat, nlev)
    T_zonal = np.mean(T_final, axis=1)

    sigma_full_np = np.array(sigma.sigma_full)
    p_levels = sigma_full_np * 1000  # Approximate pressure in hPa

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cs = axes[0].contourf(
        np.degrees(lat_np), p_levels, u_zonal.T,
        levels=np.arange(-20, 45, 5), cmap="RdBu_r", extend="both"
    )
    axes[0].set_ylim(1000, 0)
    axes[0].set_xlabel("Latitude [°]")
    axes[0].set_ylabel("Pressure [hPa]")
    axes[0].set_title(f"Zonal-mean zonal wind [m/s] (day {N_DAYS})")
    plt.colorbar(cs, ax=axes[0])

    cs = axes[1].contourf(
        np.degrees(lat_np), p_levels, T_zonal.T,
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

    # --- 3. Surface pressure snapshot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ps_np = np.array(state.p_s.data)
    lon_np = np.degrees(np.array(grid.lon))
    im = ax.pcolormesh(lon_np, np.degrees(lat_np), ps_np / 100, cmap="viridis")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"Surface pressure [hPa] (day {N_DAYS})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "surface_pressure.png", dpi=150)
    plt.close()
    print("  Saved surface_pressure.png")

    # --- 4. Equatorial temperature profile ---
    eq_mask = np.abs(lat_np) < np.radians(5.0)
    T_eq_profile = np.mean(T_zonal[eq_mask, :], axis=0)

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
