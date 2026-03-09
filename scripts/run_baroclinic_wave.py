#!/usr/bin/env python
"""Run the Jablonowski-Williamson baroclinic wave benchmark.

This script runs both components of the DCMIP baroclinic wave test:

1. **Steady-state preservation** (no perturbation):
   Initializes with the analytically balanced state and integrates
   for a short period. The model should maintain the balance — surface
   pressure perturbations indicate discretization error.

2. **Baroclinic wave growth** (with perturbation):
   Adds a localized perturbation to trigger baroclinic instability.
   Wave growth begins ~day 4, explosive cyclogenesis ~day 8-9.

Usage:
    cd /Users/pierregentine/legoESM
    source .venv/bin/activate

    # Quick test (C16, 10 days)
    python scripts/run_baroclinic_wave.py --resolution 16 --days 10

    # Standard benchmark (C48, 10 days)
    python scripts/run_baroclinic_wave.py --resolution 48 --days 10

    # Steady-state only (no perturbation)
    python scripts/run_baroclinic_wave.py --resolution 48 --days 30 --no-perturbation
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
    hydrostatic_tendencies,
)
from legoesm.atmosphere.physics.baroclinic_wave import (
    baroclinic_wave_init,
    P0,
)
from legoesm.core.cfl import cfl_check_and_adjust
from legoesm import constants


def compute_hyperdiff_coeff(n_grid: int, reference_n: int = 48,
                            reference_coeff: float = 5e16) -> float:
    """Scale hyperdiffusion coefficient with resolution.

    Uses the same reference coefficient as Held-Suarez (5e16 at C48)
    to ensure stability. The ∇⁴ coefficient scales as (dx)⁴ ∝ (1/n_grid)⁴.
    The grid-scale damping time at this coefficient is ~0.4 days at C48,
    which damps discretization noise without significantly affecting
    the large-scale baroclinic wave (wavelength ~4000 km >> Δx ~200 km).
    """
    return reference_coeff * (reference_n / n_grid) ** 4


def compute_850hPa_vorticity(state, grid, sigma_coord):
    """Compute relative vorticity at the level closest to 850 hPa."""
    from legoesm.atmosphere.dynamics.primitive_eq import _vorticity_3d

    sigma_full = np.array(sigma_coord.sigma_full)
    # 850 hPa corresponds to sigma = 0.85 (since p_s ≈ 1000 hPa)
    k_850 = int(np.argmin(np.abs(sigma_full - 0.85)))
    actual_p = sigma_full[k_850] * 1000  # hPa

    u = np.array(state.u.data)
    v = np.array(state.v.data)
    zeta = np.array(_vorticity_3d(state.u.data, state.v.data, grid))

    return zeta[..., k_850], actual_p, k_850


def main():
    parser = argparse.ArgumentParser(description="Baroclinic wave benchmark")
    parser.add_argument("--resolution", "-n", type=int, default=48,
                        help="Cubed-sphere resolution (default: 48)")
    parser.add_argument("--levels", "-l", type=int, default=26,
                        help="Number of vertical levels (default: 26)")
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Time step in seconds (default: 600)")
    parser.add_argument("--days", "-d", type=int, default=10,
                        help="Integration time in days (default: 10)")
    parser.add_argument("--no-perturbation", action="store_true",
                        help="Run steady-state test (no perturbation)")
    parser.add_argument("--hyperdiff", type=float, default=None,
                        help="Hyperdiffusion coefficient (default: auto-scaled)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    N_GRID = args.resolution
    N_LEVELS = args.levels
    DT = args.dt
    N_DAYS = args.days
    PERTURBED = not args.no_perturbation
    DIAG_INTERVAL = 6   # Save diagnostics every N hours
    SNAP_INTERVAL = max(1, N_DAYS // 10)  # Save snapshots

    HYPERDIFF_COEFF = args.hyperdiff or compute_hyperdiff_coeff(N_GRID)

    # CFL check: PE uses semi-implicit stepping (gravity waves implicit).
    # Only advective CFL matters. Baroclinic wave jet can reach ~50 m/s.
    DT = cfl_check_and_adjust(
        DT, N_GRID, model_type="primitive_eq",
        max_wind=60.0, gravity_wave_speed=0.0,
    )

    test_type = "perturbed" if PERTURBED else "steady_state"
    OUTPUT_DIR = Path(args.output or f"results/baroclinic_wave_{test_type}_C{N_GRID}_L{N_LEVELS}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Baroclinic Wave Test: {'Perturbed' if PERTURBED else 'Steady-State'}")
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
    print(f"  Sigma levels: {N_LEVELS}")
    print(f"  Sigma range: [{float(sigma.sigma_half[0]):.3f}, {float(sigma.sigma_half[-1]):.3f}]")

    state = baroclinic_wave_init(grid, sigma, perturbed=PERTURBED)
    print(f"  Initial max |u|: {float(jnp.max(jnp.abs(state.u.data))):.1f} m/s")
    print(f"  Initial mean T: {float(jnp.mean(state.T.data)):.1f} K")
    print(f"  Initial p_s: {float(jnp.mean(state.p_s.data)):.0f} Pa (uniform)")

    # Verify initial balance by computing tendencies
    config = PrimitiveEquationConfig(
        hyperdiff_coeff=0.0,  # No diffusion for balance check
        use_conservation_fixer=True,
        fix_mass=True,
    )
    tend_init = hydrostatic_tendencies(state, grid, sigma, config)
    print(f"\n  Initial tendency magnitudes (no diffusion):")
    print(f"    max |du/dt|:   {float(jnp.max(jnp.abs(tend_init.du_dt.data))):.4e} m/s^2")
    print(f"    max |dv/dt|:   {float(jnp.max(jnp.abs(tend_init.dv_dt.data))):.4e} m/s^2")
    print(f"    max |dT/dt|:   {float(jnp.max(jnp.abs(tend_init.dT_dt.data))):.4e} K/s")
    print(f"    max |dp_s/dt|: {float(jnp.max(jnp.abs(tend_init.dp_s_dt.data))):.4e} Pa/s")

    # --- Model with diffusion ---
    config_run = PrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF_COEFF,
        hyperdiff_ps_coeff=HYPERDIFF_COEFF,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = PrimitiveEquationModel(grid, sigma, config_run)

    # --- Integration ---
    n_steps_total = int(N_DAYS * 86400 / DT)
    diag_every = int(DIAG_INTERVAL * 3600 / DT)
    snap_every = max(1, int(SNAP_INTERVAL * 86400 / DT))

    print(f"\nIntegrating for {n_steps_total} steps...")
    print(f"  Diagnostics every {diag_every} steps ({DIAG_INTERVAL}h)")

    # Storage for diagnostics
    diag_times = []
    diag_mass = []
    diag_max_wind = []
    diag_mean_T = []
    diag_ps_min = []
    diag_ps_max = []
    diag_ps_perturbation = []  # max |p_s - p_s_init|

    # Storage for snapshots
    snap_days_target = sorted(set(
        [0] + list(range(SNAP_INTERVAL, N_DAYS + 1, SNAP_INTERVAL)) + [N_DAYS]
    ))
    snap_steps = {int(d * 86400 / DT): d for d in snap_days_target}
    snapshots = {}

    # Save initial snapshot
    snapshots[0] = jax.tree.map(lambda x: np.array(x), state)

    mass_initial = float(global_integral(state.p_s, grid))
    ps_initial = float(jnp.mean(state.p_s.data))
    t_start = time.time()
    last_print = t_start

    # JIT warmup
    print("  JIT compiling (first step)...", end=" ", flush=True)
    t_jit = time.time()
    state = model.step(state, DT)
    jax.block_until_ready(state.u.data)
    print(f"done ({time.time() - t_jit:.1f}s)")

    for step in range(1, n_steps_total):
        state = model.step(state, DT)

        # Check for blowup
        if step % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 500:
                day = (step + 1) * DT / 86400.0
                print(f"\n  *** BLOWUP at day {day:.1f}, step {step+1}, u_max={u_max:.1f} ***")
                break

        # Diagnostics
        if (step + 1) % diag_every == 0:
            day = (step + 1) * DT / 86400.0
            mass = float(global_integral(state.p_s, grid))
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            mean_T = float(jnp.mean(state.T.data))
            ps_data = np.array(state.p_s.data)
            ps_min_val = float(np.min(ps_data))
            ps_max_val = float(np.max(ps_data))
            ps_pert = float(np.max(np.abs(ps_data - ps_initial)))

            diag_times.append(day)
            diag_mass.append(mass)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)
            diag_ps_min.append(ps_min_val)
            diag_ps_max.append(ps_max_val)
            diag_ps_perturbation.append(ps_pert)

            now = time.time()
            if now - last_print > 30:
                elapsed = now - t_start
                steps_per_sec = (step + 1) / elapsed
                eta = (n_steps_total - step - 1) / steps_per_sec
                mass_drift = (mass - mass_initial) / mass_initial
                print(
                    f"  Day {day:7.1f}/{N_DAYS} | "
                    f"max |v|={max_wind:6.1f} m/s | "
                    f"p_s: [{ps_min_val/100:.0f}, {ps_max_val/100:.0f}] hPa | "
                    f"pert={ps_pert/100:.2f} hPa | "
                    f"mass={mass_drift:+.2e} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"ETA {eta/60:.0f} min"
                )
                last_print = now

        # Save snapshots
        if (step + 1) in snap_steps:
            day = snap_steps[step + 1]
            snapshots[day] = jax.tree.map(lambda x: np.array(x), state)
            print(f"  *** Snapshot saved at day {day} ***")

    elapsed = time.time() - t_start
    print(f"\nIntegration complete: {elapsed:.0f}s ({n_steps_total/elapsed:.1f} steps/s)")

    # Final diagnostics
    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = (mass_final - mass_initial) / mass_initial
    ps_final = np.array(state.p_s.data)
    print(f"  Mass drift: {mass_drift:+.2e} (relative)")
    print(f"  Max wind: {float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))):.1f} m/s")
    print(f"  Mean T: {float(jnp.mean(state.T.data)):.1f} K")
    print(f"  p_s range: [{np.min(ps_final)/100:.1f}, {np.max(ps_final)/100:.1f}] hPa")
    print(f"  Max p_s perturbation: {np.max(np.abs(ps_final - ps_initial))/100:.2f} hPa")

    # ===========================================================================
    # Generate plots
    # ===========================================================================
    print("\nGenerating plots...")

    diag_times = np.array(diag_times)
    diag_mass = np.array(diag_mass)
    diag_max_wind = np.array(diag_max_wind)
    diag_mean_T = np.array(diag_mean_T)
    diag_ps_min = np.array(diag_ps_min)
    diag_ps_max = np.array(diag_ps_max)
    diag_ps_perturbation = np.array(diag_ps_perturbation)

    # --- 1. Time series diagnostics ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    mass_drift_ts = (diag_mass - mass_initial) / mass_initial
    axes[0].plot(diag_times, mass_drift_ts)
    axes[0].set_ylabel("Mass drift (relative)")
    axes[0].set_title("Global mass conservation")
    axes[0].grid(True)

    axes[1].plot(diag_times, diag_max_wind)
    axes[1].set_ylabel("Max |v| [m/s]")
    axes[1].set_title("Maximum wind speed")
    axes[1].grid(True)

    axes[2].plot(diag_times, diag_ps_min / 100, label="min p_s")
    axes[2].plot(diag_times, diag_ps_max / 100, label="max p_s")
    axes[2].set_ylabel("Surface pressure [hPa]")
    axes[2].set_title("Surface pressure range")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(diag_times, diag_ps_perturbation / 100)
    axes[3].set_xlabel("Time [days]")
    axes[3].set_ylabel("Max |Δp_s| [hPa]")
    axes[3].set_title("Surface pressure perturbation (from initial)")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "timeseries.png", dpi=150)
    plt.close()
    print("  Saved timeseries.png")

    # --- 2. Zonal-mean diagnostics (final state) ---
    lat_np = np.array(grid.lat)
    lat_flat = lat_np.flatten()
    area_flat = np.array(grid.area).flatten()

    # Adaptive binning: fewer bins at coarse resolution to ensure coverage
    n_lat_bins = min(91, max(30, 2 * N_GRID))
    lat_bins = np.linspace(-np.pi/2, np.pi/2, n_lat_bins + 1)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    u_final = np.array(state.u.data)
    T_final = np.array(state.T.data)
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
        levels=np.arange(-10, 50, 5), cmap="RdBu_r", extend="both"
    )
    axes[0].set_ylim(1000, 0)
    axes[0].set_xlabel("Latitude [deg]")
    axes[0].set_ylabel("Pressure [hPa]")
    axes[0].set_title(f"Zonal-mean zonal wind [m/s] (day {N_DAYS})")
    plt.colorbar(cs, ax=axes[0])

    cs = axes[1].contourf(
        np.degrees(lat_centers), p_levels, T_zonal.T,
        levels=20, cmap="RdYlBu_r"
    )
    axes[1].set_ylim(1000, 0)
    axes[1].set_xlabel("Latitude [deg]")
    axes[1].set_ylabel("Pressure [hPa]")
    axes[1].set_title(f"Zonal-mean temperature [K] (day {N_DAYS})")
    plt.colorbar(cs, ax=axes[1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zonal_mean.png", dpi=150)
    plt.close()
    print("  Saved zonal_mean.png")

    # --- 3. Surface pressure maps (snapshots) ---
    n_snap = min(len(snapshots), 6)
    snap_sorted = sorted(snapshots.items())
    if n_snap > 0:
        ncols = min(n_snap, 3)
        nrows = (n_snap + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes_flat = np.array(axes).flatten() if n_snap > 1 else [axes]

        for idx, (day, snap) in enumerate(snap_sorted[:n_snap]):
            ax = axes_flat[idx]
            ps = np.array(snap.p_s.data)
            # Show face 0 (equatorial face)
            im = ax.pcolormesh(ps[0] / 100, cmap="RdBu_r",
                               vmin=980, vmax=1020)
            ax.set_title(f"p_s [hPa] face 0, day {day}")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax)

        # Hide unused axes
        for idx in range(n_snap, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "surface_pressure_snapshots.png", dpi=150)
        plt.close()
        print("  Saved surface_pressure_snapshots.png")

    # --- 4. 850 hPa vorticity (for perturbed test) ---
    if PERTURBED and N_DAYS >= 4:
        try:
            zeta_850, p_actual, k_850 = compute_850hPa_vorticity(state, grid, sigma)
            zeta_np = np.array(zeta_850)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for face in range(6):
                ax = axes[face // 3, face % 3]
                vmax = max(1e-4, np.max(np.abs(zeta_np[face])))
                im = ax.pcolormesh(
                    zeta_np[face], cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax
                )
                ax.set_title(f"Face {face}")
                ax.set_aspect("equal")
                plt.colorbar(im, ax=ax, format="%.1e")

            plt.suptitle(
                f"Relative vorticity at {p_actual:.0f} hPa (day {N_DAYS})",
                fontsize=14
            )
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "vorticity_850hPa.png", dpi=150)
            plt.close()
            print("  Saved vorticity_850hPa.png")
        except Exception as e:
            print(f"  Warning: Could not compute 850hPa vorticity: {e}")

    # --- 5. Save summary to text file ---
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write("Baroclinic Wave Test Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test type: {'Perturbed' if PERTURBED else 'Steady-state'}\n")
        f.write(f"Resolution: C{N_GRID}, L{N_LEVELS}\n")
        f.write(f"Time step: {DT:.0f} s\n")
        f.write(f"Duration: {N_DAYS} days\n")
        f.write(f"Hyperdiffusion: {HYPERDIFF_COEFF:.2e}\n")
        f.write(f"Wall time: {elapsed:.0f} s\n")
        f.write(f"Throughput: {n_steps_total/elapsed:.1f} steps/s\n\n")
        f.write("Final state:\n")
        f.write(f"  Mass drift: {mass_drift:+.2e}\n")
        f.write(f"  Max wind: {float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))):.1f} m/s\n")
        f.write(f"  Mean T: {float(jnp.mean(state.T.data)):.1f} K\n")
        f.write(f"  p_s range: [{np.min(ps_final)/100:.1f}, {np.max(ps_final)/100:.1f}] hPa\n")
        f.write(f"  Max p_s perturbation: {np.max(np.abs(ps_final - ps_initial))/100:.2f} hPa\n")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
