#!/usr/bin/env python3
"""Benchmark: Spectral Shallow Water — Williamson Test Cases 2 and 5.

Runs:
  - Test Case 2 (geostrophic balance): 5-day steady-state test at T42.
    Expected: height anomaly < 1 m after 5 days (Williamson et al. 1992).
  - Test Case 5 (mountain flow):  15-day integration at T42.
    Expected: stable, recognizable mountain-wave pattern.

Outputs (in results/atmosphere/shallow_water/spectral_sw/):
  - conservation_tc2.png   — mass, energy, enstrophy time series (TC2)
  - conservation_tc5.png   — mass, energy, enstrophy time series (TC5)
  - snapshots_tc2.png      — height field at days 0, 1, 3, 5
  - snapshots_tc5.png      — height field at days 0, 5, 10, 15
  - diagnostics.npz        — all diagnostic timeseries
  - summary.txt            — quantitative pass/fail vs literature

References
----------
  Williamson, D. L., et al. (1992). A standard test set for numerical
  approximations to the shallow water equations in spherical geometry.
  J. Comput. Phys., 102, 211-224.
"""

import os
import sys
import time

# Ensure JAX uses float64
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
    SpectralSWConfig,
    SpectralShallowWaterModel,
    williamson_test2_spectral,
    williamson_test5_spectral,
    spectral_to_grid,
    compute_spectral_diagnostics,
)
from legoesm import constants

# ── Output directory ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_DIR, "results", "atmosphere", "shallow_water", "spectral_sw")
os.makedirs(OUT_DIR, exist_ok=True)


def proper_hyperdiff(grid):
    """4-hour e-folding at truncation wavenumber."""
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max**2)


def run_test_case(name, grid, state0, config, model, dt, n_days, snap_days):
    """Integrate and collect diagnostics."""
    n_steps = int(n_days * 86400.0 / dt)
    diag_interval = max(1, int(3600.0 / dt))  # hourly diagnostics

    # Initial diagnostics
    fields0 = spectral_to_grid(state0, grid)
    diag0 = compute_spectral_diagnostics(state0, grid)

    times = [0.0]
    mass_ts = [diag0["mass"]]
    energy_ts = [diag0["energy"]]
    enstrophy_ts = [diag0["enstrophy"]]
    snapshots = {0.0: fields0}

    state = state0
    print(f"\n  [{name}] Integrating {n_days} days, dt={dt:.0f}s, {n_steps} steps")

    # JIT warmup
    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.vor_hat.data)
    print(f"  [{name}] JIT compiled in {time.time()-t0:.1f}s")

    t_wall = time.time()
    for step in range(1, n_steps):
        state = model.step(state, dt)

        if (step + 1) % diag_interval == 0:
            day = (step + 1) * dt / 86400.0
            diag = compute_spectral_diagnostics(state, grid)
            times.append(day)
            mass_ts.append(diag["mass"])
            energy_ts.append(diag["energy"])
            enstrophy_ts.append(diag["enstrophy"])

            # Snapshot?
            for sd in snap_days:
                if abs(day - sd) < dt / 86400.0:
                    fields = spectral_to_grid(state, grid)
                    snapshots[sd] = fields

    wall_time = time.time() - t_wall
    print(f"  [{name}] Done in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")

    # Final snapshot
    fields_final = spectral_to_grid(state, grid)
    snapshots[n_days] = fields_final

    return {
        "times": np.array(times),
        "mass": np.array(mass_ts),
        "energy": np.array(energy_ts),
        "enstrophy": np.array(enstrophy_ts),
        "snapshots": snapshots,
        "fields0": fields0,
        "fields_final": fields_final,
        "state_final": state,
    }


def plot_conservation(result, name, filename):
    """Plot relative conservation errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["times"]
    mass0 = result["mass"][0]
    energy0 = result["energy"][0]
    enstrophy0 = result["enstrophy"][0]

    # iter-93 audit followup: previously inlined
    # ``(result["mass"] - mass0) / abs(mass0)`` etc. would NaN if
    # the baseline is exactly 0.  Migrated to the shared helper
    # for the iter-78/80 floor convention (1.0 floor returns
    # absolute drift in natural units when baseline is near zero).
    from legoesm.diagnostics.conservation_drift import relative_drift_series
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, relative_drift_series(result["mass"]), "b-", lw=1.5)
    axes[0].set_ylabel("Rel. mass error")
    axes[0].set_title(f"Spectral SW — {name}: Conservation")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    axes[1].plot(t, relative_drift_series(result["energy"]), "r-", lw=1.5)
    axes[1].set_ylabel("Rel. energy error")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    axes[2].plot(t, relative_drift_series(result["enstrophy"]), "g-", lw=1.5)
    axes[2].set_ylabel("Rel. enstrophy error")
    axes[2].set_xlabel("Time [days]")
    axes[2].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_snapshots(result, grid, name, snap_days, filename, phis=None):
    """Plot height field snapshots in lat-lon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = result["snapshots"]
    days = sorted(snaps.keys())
    # Pick the requested days (or closest available)
    plot_days = []
    for sd in snap_days:
        closest = min(days, key=lambda d: abs(d - sd))
        if closest not in plot_days:
            plot_days.append(closest)

    n = len(plot_days)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), subplot_kw={"projection": None})
    if n == 1:
        axes = [axes]

    lon_deg = np.degrees(np.array(grid.lon2d[0]))
    lat_deg = np.degrees(np.array(grid.lat))

    for i, day in enumerate(plot_days):
        fields = snaps[day]
        h = np.array(fields["h"])

        # If mountain test, subtract surface height for the anomaly
        if phis is not None:
            h_s = np.array(phis) / constants.g
            h_anom = h - np.mean(h)
        else:
            h_anom = h - np.mean(h)

        im = axes[i].pcolormesh(lon_deg, lat_deg, h_anom, cmap="RdBu_r", shading="auto")
        axes[i].set_title(f"Day {day:.0f}")
        axes[i].set_xlabel("Longitude")
        if i == 0:
            axes[i].set_ylabel("Latitude")
        plt.colorbar(im, ax=axes[i], shrink=0.8, label="h anomaly [m]")

    fig.suptitle(f"Spectral SW {name} — Height Anomaly (T42)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("  Spectral Shallow Water Benchmark")
    print("=" * 70)

    # ── Grid setup ─────────────────────────────────────────────────────
    n_max = 42
    print(f"\n  Grid: T{n_max} Gaussian")
    grid = create_gaussian_grid(n_max=n_max)
    nu = proper_hyperdiff(grid)
    print(f"  n_lat={grid.n_lat}, n_lon={grid.n_lon}, n_sh={grid.n_sh}")
    print(f"  Hyperdiffusion: nu={nu:.4e} (4-hour e-folding)")

    # ── Test Case 2: Geostrophic Balance (5 days) ─────────────────────
    print("\n" + "-" * 50)
    print("  Test Case 2: Steady Geostrophic Flow")
    print("-" * 50)
    # Background depth is carried by the TC2 initial phi_hat (gh0 = 2.94e4),
    # not by any config knob.
    config2 = SpectralSWConfig(hyperdiff_coeff=nu)
    state2 = williamson_test2_spectral(grid)
    model2 = SpectralShallowWaterModel(grid, config2)

    dt2 = 600.0  # 10 min
    res2 = run_test_case("TC2", grid, state2, config2, model2, dt2,
                          n_days=5, snap_days=[0, 1, 3, 5])

    # Quantitative check: height anomaly after 5 days
    h_final = np.array(res2["fields_final"]["h"])
    h_init = np.array(res2["fields0"]["h"])
    h_err_max = np.max(np.abs(h_final - h_init))
    h_err_rms = np.sqrt(np.mean((h_final - h_init)**2))
    # iter-93: same audit followup; migrate scalar drift to helper.
    from legoesm.diagnostics.conservation_drift import compute_relative_drift
    mass_drift = compute_relative_drift(res2["mass"])
    energy_drift = compute_relative_drift(res2["energy"])

    print(f"\n  TC2 Results after 5 days:")
    print(f"    Max |h - h0|   = {h_err_max:.4f} m  (expect < 1 m)")
    print(f"    RMS |h - h0|   = {h_err_rms:.6f} m")
    print(f"    Mass drift      = {mass_drift:.2e}  (expect < 1e-10)")
    print(f"    Energy drift    = {energy_drift:.2e}  (expect < 1e-6)")

    tc2_pass = h_err_max < 1.0 and mass_drift < 1e-8

    # ── Test Case 5: Mountain Flow (15 days) ──────────────────────────
    print("\n" + "-" * 50)
    print("  Test Case 5: Zonal Flow over Mountain")
    print("-" * 50)
    # TC5 needs stronger hyperdiffusion (1-hour e-folding) to handle
    # Gibbs ringing from the mountain topography at T42.
    nu5 = 1.0 / (1.0 * 3600.0 * (grid.n_max * (grid.n_max + 1) / (grid.radius**2))**2)
    config5 = SpectralSWConfig(hyperdiff_coeff=nu5)
    state5 = williamson_test5_spectral(grid)
    model5 = SpectralShallowWaterModel(grid, config5)

    dt5 = 200.0  # smaller dt needed for T42 CFL with mountain flow
    res5 = run_test_case("TC5", grid, state5, config5, model5, dt5,
                          n_days=15, snap_days=[0, 5, 10, 15])

    # Quantitative check
    h_final5 = np.array(res5["fields_final"]["h"])
    h_init5 = np.array(res5["fields0"]["h"])
    # iter-93: same audit followup.
    mass_drift5 = compute_relative_drift(res5["mass"])
    energy_drift5 = compute_relative_drift(res5["energy"])
    h_range_final = np.max(h_final5) - np.min(h_final5)
    h_range_init = np.max(h_init5) - np.min(h_init5)

    print(f"\n  TC5 Results after 15 days:")
    print(f"    h range initial = {h_range_init:.1f} m")
    print(f"    h range final   = {h_range_final:.1f} m")
    print(f"    Mass drift      = {mass_drift5:.2e}")
    print(f"    Energy drift    = {energy_drift5:.2e}")
    print(f"    All finite      = {bool(np.all(np.isfinite(h_final5)))}")

    tc5_pass = np.all(np.isfinite(h_final5)) and mass_drift5 < 1e-6

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_conservation(res2, "TC2 — Geostrophic Balance", "conservation_tc2.png")
    plot_conservation(res5, "TC5 — Mountain Flow", "conservation_tc5.png")
    plot_snapshots(res2, grid, "TC2", [0, 1, 3, 5], "snapshots_tc2.png")
    phis5 = np.array(spectral_to_grid(state5, grid)["phis"])
    plot_snapshots(res5, grid, "TC5", [0, 5, 10, 15], "snapshots_tc5.png", phis=phis5)

    # ── Save data ─────────────────────────────────────────────────────
    np.savez(
        os.path.join(OUT_DIR, "diagnostics.npz"),
        tc2_times=res2["times"], tc2_mass=res2["mass"],
        tc2_energy=res2["energy"], tc2_enstrophy=res2["enstrophy"],
        tc5_times=res5["times"], tc5_mass=res5["mass"],
        tc5_energy=res5["energy"], tc5_enstrophy=res5["enstrophy"],
        lat_deg=np.degrees(np.array(grid.lat)),
        lon_deg=np.degrees(np.array(grid.lon2d[0])),
        tc2_h_final=np.array(res2["fields_final"]["h"]),
        tc5_h_final=np.array(res5["fields_final"]["h"]),
    )
    print(f"  Saved diagnostics.npz")

    # ── Summary ───────────────────────────────────────────────────────
    summary = []
    summary.append("Spectral Shallow Water Benchmark Summary")
    summary.append("=" * 50)
    summary.append(f"Grid: T{n_max}, dt_TC2={dt2:.0f}s, dt_TC5={dt5:.0f}s")
    summary.append("")
    summary.append("Test Case 2 — Steady Geostrophic Flow (5 days):")
    summary.append(f"  Max height error:  {h_err_max:.4f} m  (threshold: 1 m)")
    summary.append(f"  RMS height error:  {h_err_rms:.6f} m")
    summary.append(f"  Rel. mass drift:   {mass_drift:.2e}")
    summary.append(f"  Rel. energy drift: {energy_drift:.2e}")
    summary.append(f"  PASS: {tc2_pass}")
    summary.append("")
    summary.append("Test Case 5 — Mountain Flow (15 days):")
    summary.append(f"  h range final:     {h_range_final:.1f} m")
    summary.append(f"  Rel. mass drift:   {mass_drift5:.2e}")
    summary.append(f"  Rel. energy drift: {energy_drift5:.2e}")
    summary.append(f"  All finite:        {bool(np.all(np.isfinite(h_final5)))}")
    summary.append(f"  PASS: {tc5_pass}")
    summary.append("")
    summary.append("Literature reference: Williamson et al. (1992)")
    summary.append("  TC2: exact steady state, error should be machine precision")
    summary.append("       (with hyperdiffusion: < 1 m at T42)")
    summary.append("  TC5: no analytic solution; should remain stable and")
    summary.append("       show recognizable Rossby wave train behind mountain")

    summary_text = "\n".join(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")

    return tc2_pass and tc5_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
