"""Run the spectral shallow water model for 100 days — Williamson Test Case 5.

Uses spherical harmonic transform method at T42 resolution (~2.8 degrees).
This is the spectral counterpart of run_100day.py (cubed-sphere finite diff).

Produces:
  - Height field snapshots at days 0, 25, 50, 75, 100
  - Height perturbation at day 100
  - Wind speed and vorticity maps
  - Conservation diagnostics time series
  - Mountain topography
"""

import sys
import time
import os

import jax
import jax.numpy as jnp
import numpy as np

# Ensure unbuffered output when piped
sys.stdout.reconfigure(line_buffering=True)

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.spectral_sw import (
    SpectralSWConfig,
    spectral_sw_tendencies,
    williamson_test5_spectral,
    spectral_to_grid,
    compute_spectral_diagnostics,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants


def main():
    # =====================================================================
    # Configuration
    # =====================================================================
    N_MAX = 42              # T42 spectral truncation (~2.8 deg)
    DT = 60.0               # 60s timestep [s] (within CFL for explicit RK3)
    DURATION_DAYS = 100
    SNAPSHOT_DAYS = [0, 25, 50, 75, 100]

    OUTPUT_DIR = "results/atmosphere/shallow_water/100day_spectral"

    # =====================================================================
    # Setup
    # =====================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("legoESM — 100-Day Spectral Shallow Water (Williamson Test 5)")
    print("=" * 70)

    print(f"\nCreating T{N_MAX} Gaussian grid...")
    t0 = time.time()
    grid = create_gaussian_grid(N_MAX)
    print(f"  Grid created in {time.time()-t0:.1f}s")
    print(f"  Latitudes: {grid.n_lat}, Longitudes: {grid.n_lon}")
    print(f"  Spectral coefficients: {grid.n_sh}")
    print(f"  Resolution: ~{360.0/grid.n_lon:.1f} degrees")

    # Hyperdiffusion: 1-hour e-folding at truncation wavenumber
    # Explicit RK3 with T42 mountain flow needs stronger diffusion than
    # semi-implicit models (which use 4-hour e-folding).
    a = grid.radius
    eig_max = N_MAX * (N_MAX + 1) / (a * a)
    hyperdiff_coeff = 1.0 / (1.0 * 3600.0 * eig_max**2)
    print(f"  Hyperdiffusion coeff: {hyperdiff_coeff:.3e}")

    config = SpectralSWConfig(
        mean_depth=5960.0,
        hyperdiff_coeff=hyperdiff_coeff,
        hyperdiff_order=2,
    )

    print("\nInitializing Williamson Test Case 5 (mountain)...")
    state = williamson_test5_spectral(grid)
    state_init = state

    # Initial grid-point fields for verification
    fields_init = spectral_to_grid(state, grid)
    print(f"  Height range: [{float(jnp.min(fields_init['h'])):.0f}, "
          f"{float(jnp.max(fields_init['h'])):.0f}] m")
    print(f"  Wind speed max: {float(jnp.max(jnp.sqrt(fields_init['u']**2 + fields_init['v']**2))):.1f} m/s")
    print(f"  Mountain peak: {float(jnp.max(fields_init['h_s'])):.0f} m")

    # =====================================================================
    # Integration
    # =====================================================================
    n_steps = int(DURATION_DAYS * 86400 / DT)
    STEPS_PER_DAY = int(86400 / DT)
    snapshot_days_set = set(SNAPSHOT_DAYS)

    print(f"\nIntegrating for {DURATION_DAYS} days ({n_steps:,} steps, dt={DT:.0f}s)...")

    # Storage
    diagnostics = []
    snapshots = {}

    # Save initial state
    diag = compute_spectral_diagnostics(state, grid)
    diagnostics.append(diag)
    snapshots[0] = state

    # JIT-compiled single step
    def tendency_fn(s):
        return spectral_sw_tendencies(s, grid, config)

    step_jit = jax.jit(lambda s, dt: ssp_rk3_step(s, tendency_fn, dt))

    # Warm up JIT
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = step_jit(state, DT)
    jax.block_until_ready(_.vor_hat.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Main loop
    print(f"\n{'Day':>6s}  {'h_min':>8s}  {'h_max':>8s}  {'|u|_max':>8s}  "
          f"{'mass_Δ':>10s}  {'energy_Δ':>10s}  {'wall_s':>7s}")
    print("-" * 75)

    t_start = time.time()
    t_lap = t_start
    step_count = 0

    for day_num in range(1, DURATION_DAYS + 1):
        for _ in range(STEPS_PER_DAY):
            state = step_jit(state, DT)
        step_count += STEPS_PER_DAY

        jax.block_until_ready(state.vor_hat.data)

        # Diagnostics every day
        diag = compute_spectral_diagnostics(state, grid)
        diagnostics.append(diag)

        mass_rel = (diag['mass'] - diagnostics[0]['mass']) / diagnostics[0]['mass']
        energy_rel = (diag['energy'] - diagnostics[0]['energy']) / diagnostics[0]['energy']

        fields = spectral_to_grid(state, grid)
        h_min = float(jnp.min(fields['h']))
        h_max = float(jnp.max(fields['h']))
        umax = float(jnp.max(jnp.sqrt(fields['u']**2 + fields['v']**2)))

        elapsed = time.time() - t_lap
        t_lap = time.time()

        if day_num % 5 == 0 or day_num <= 5 or day_num == DURATION_DAYS:
            print(f"{day_num:6d}  {h_min:8.0f}  {h_max:8.0f}  "
                  f"{umax:8.1f}  {mass_rel:10.2e}  {energy_rel:10.2e}  {elapsed:7.1f}")

        # Save snapshot
        if day_num in snapshot_days_set:
            snapshots[step_count] = state

    total_time = time.time() - t_start
    steps_per_sec = n_steps / total_time
    print("-" * 75)
    print(f"Completed in {total_time:.1f}s ({steps_per_sec:.0f} steps/s)")

    # =====================================================================
    # Visualization
    # =====================================================================
    print("\n\nGenerating visualizations...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11,
        'figure.facecolor': 'white',
    })

    # Helper to get lat/lon in degrees
    lon2d_deg = np.asarray(grid.lon2d) * 180 / np.pi
    lat2d_deg = np.asarray(grid.lat2d) * 180 / np.pi

    def pcolor_field(ax, data, cmap, vmin, vmax):
        """Plot a lat-lon field using pcolormesh."""
        pc = ax.pcolormesh(
            lon2d_deg, lat2d_deg,
            np.asarray(data),
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto",
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        return pc

    # ------------------------------------------------------------------
    # 1. Height field snapshots
    # ------------------------------------------------------------------
    print("  [1/5] Height field snapshots...")
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 4))
    if n_snaps == 1:
        axes = [axes]

    for idx, (step_num, snap) in enumerate(snapshots.items()):
        day = step_num * DT / 86400.0
        ax = axes[idx]
        fields = spectral_to_grid(snap, grid)
        pc = pcolor_field(ax, fields['h'], "RdYlBu_r", 5000, 6100)
        ax.set_title(f"Fluid depth h [m] — Day {day:.0f}", fontsize=12, fontweight="bold")

    fig.suptitle(f"Spectral T{N_MAX} — Fluid Depth h [m]  (100 days)",
                 fontsize=14, fontweight="bold", y=1.02)
    cbar = fig.colorbar(pc, ax=axes, shrink=0.8, orientation="horizontal",
                         label="Fluid depth [m]", pad=0.15)
    plt.savefig(f"{OUTPUT_DIR}/height_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 2. Height perturbation at day 100
    # ------------------------------------------------------------------
    print("  [2/5] Height perturbation at day 100...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    last_step = max(snapshots.keys())
    fields_final = spectral_to_grid(snapshots[last_step], grid)
    h_pert = np.asarray(fields_final['h'] - fields_init['h'])
    vabs = min(float(np.max(np.abs(h_pert))), 500)

    pc = pcolor_field(ax, h_pert, "RdBu_r", -vabs, vabs)
    ax.set_title(f"Height Perturbation h(day 100) - h(day 0)  [m]",
                 fontsize=13, fontweight="bold")
    plt.colorbar(pc, ax=ax, shrink=0.8, orientation="horizontal",
                 label="Delta h [m]", pad=0.15)
    plt.savefig(f"{OUTPUT_DIR}/height_perturbation_day100.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 3. Wind speed + vorticity at day 50
    # ------------------------------------------------------------------
    print("  [3/5] Wind speed and vorticity at day 50...")
    day50_step = int(50 * 86400 / DT)
    snap50 = snapshots.get(day50_step, snapshots[last_step])
    fields50 = spectral_to_grid(snap50, grid)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Wind speed
    speed = np.asarray(np.sqrt(fields50['u']**2 + fields50['v']**2))
    pc1 = pcolor_field(axes[0], speed, "magma", 0, float(np.max(speed)))
    axes[0].set_title("Wind Speed [m/s] — Day 50", fontsize=12, fontweight="bold")
    plt.colorbar(pc1, ax=axes[0], shrink=0.8, orientation="horizontal",
                 label="Speed [m/s]", pad=0.15)

    # Vorticity
    vort = np.asarray(fields50['vor']) * 1e5
    vort_abs = min(float(np.max(np.abs(vort))), 5.0)
    pc2 = pcolor_field(axes[1], vort, "RdBu_r", -vort_abs, vort_abs)
    axes[1].set_title("Relative Vorticity [x1e-5 s-1] — Day 50",
                      fontsize=12, fontweight="bold")
    plt.colorbar(pc2, ax=axes[1], shrink=0.8, orientation="horizontal",
                 label="Vorticity [x1e-5 s-1]", pad=0.15)

    fig.suptitle(f"Spectral T{N_MAX} — Williamson Test 5",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(f"{OUTPUT_DIR}/wind_vorticity_day50.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 4. Conservation time series
    # ------------------------------------------------------------------
    print("  [4/5] Conservation time series...")
    n_diag = len(diagnostics)
    times_days = np.arange(n_diag)  # diagnostics saved once per day

    mass = np.array([d['mass'] for d in diagnostics])
    energy = np.array([d['energy'] for d in diagnostics])

    mass_rel = (mass - mass[0]) / mass[0]
    energy_rel = (energy - energy[0]) / energy[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(times_days, mass_rel, 'b-', linewidth=1.5)
    axes[0].set_ylabel("Relative mass change", fontsize=12)
    axes[0].set_title("Mass Conservation", fontsize=13, fontweight="bold")
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_days, energy_rel, 'r-', linewidth=1.5)
    axes[1].set_ylabel("Relative energy change", fontsize=12)
    axes[1].set_xlabel("Time [days]", fontsize=12)
    axes[1].set_title("Energy Conservation", fontsize=13, fontweight="bold")
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Conservation — Spectral T{N_MAX} Williamson Test 5 ({DURATION_DAYS} days)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/conservation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 5. Mountain topography + initial height
    # ------------------------------------------------------------------
    print("  [5/5] Mountain topography...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    pc1 = pcolor_field(axes[0], fields_init['h_s'], "terrain",
                       0, float(np.max(np.asarray(fields_init['h_s']))))
    axes[0].set_title("Surface Topography h_s [m]", fontsize=12, fontweight="bold")
    plt.colorbar(pc1, ax=axes[0], shrink=0.8, orientation="horizontal",
                 label="h_s [m]", pad=0.15)

    pc2 = pcolor_field(axes[1], fields_init['h'], "RdYlBu_r", 5000, 6100)
    axes[1].set_title("Initial Fluid Depth h [m]", fontsize=12, fontweight="bold")
    plt.colorbar(pc2, ax=axes[1], shrink=0.8, orientation="horizontal",
                 label="h [m]", pad=0.15)

    fig.suptitle(f"Spectral T{N_MAX} — Setup", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(f"{OUTPUT_DIR}/setup.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Resolution:        T{N_MAX} ({grid.n_lat}x{grid.n_lon} grid, {grid.n_sh} coefficients)")
    print(f"  Duration:          {DURATION_DAYS} days ({n_steps:,} steps)")
    print(f"  Wall time:         {total_time:.1f}s ({steps_per_sec:.0f} steps/s)")
    print(f"  Final mass Δ:      {mass_rel[-1]:.2e}")
    print(f"  Final energy Δ:    {energy_rel[-1]:.2e}")

    fields_final = spectral_to_grid(state, grid)
    print(f"  Final h range:     [{float(jnp.min(fields_final['h'])):.0f}, "
          f"{float(jnp.max(fields_final['h'])):.0f}] m")
    print(f"  Final |u| max:     "
          f"{float(jnp.max(jnp.sqrt(fields_final['u']**2 + fields_final['v']**2))):.1f} m/s")

    print(f"\n  Figures saved to: {OUTPUT_DIR}/")
    print(f"    - height_snapshots.png")
    print(f"    - height_perturbation_day100.png")
    print(f"    - wind_vorticity_day50.png")
    print(f"    - conservation.png")
    print(f"    - setup.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
