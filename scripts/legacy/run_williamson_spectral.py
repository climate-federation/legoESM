#!/usr/bin/env python
"""Run Williamson spectral shallow water test cases 2 and 5.

Produces for each case:
  - Height field snapshots at key days
  - Wind speed and vorticity snapshots
  - Conservation diagnostics time series (mass, energy, enstrophy)
  - Integrated values saved as .npz for post-processing

Usage:
    python scripts/run_williamson_spectral.py
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


def _preparse_precision(argv):
    """Parse precision flags before importing spectral modules."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--x64", dest="x64", action="store_true")
    parser.add_argument("--no-x64", dest="x64", action="store_false")
    parser.set_defaults(x64=None)
    args, _ = parser.parse_known_args(argv)
    if args.x64 is not None:
        return bool(args.x64)
    return not args.float32


USE_X64 = _preparse_precision(sys.argv[1:])

sys.stdout.reconfigure(line_buffering=True)
jax.config.update("jax_enable_x64", USE_X64)

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.spectral_sw import (
    SpectralSWConfig,
    spectral_sw_tendencies,
    williamson_test2_spectral,
    williamson_test5_spectral,
    spectral_to_grid,
    compute_spectral_diagnostics,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# Configuration
# =========================================================================
DEFAULT_T_TRUNC = 42       # T42 spectral truncation
DEFAULT_DT = 60.0          # Timestep [s]

CASES = {
    "williamson2": {
        "name": "Williamson Test Case 2 — Steady Geostrophic Flow",
        "init_fn": williamson_test2_spectral,
        "duration_days": 5,
        "snapshot_days": [0, 1, 2, 3, 4, 5],
        "hyperdiff": False,
    },
    "williamson5": {
        "name": "Williamson Test Case 5 — Zonal Flow over Mountain",
        "init_fn": williamson_test5_spectral,
        "duration_days": 15,
        "snapshot_days": [0, 3, 5, 7, 10, 15],
        "hyperdiff": True,
    },
}

DEFAULT_OUTPUT_BASE = "results/atmosphere/shallow_water/williamson_spectral"


# =========================================================================
# Plotting helpers
# =========================================================================

def pcolor_field(ax, lon2d_deg, lat2d_deg, data, cmap, vmin, vmax, title):
    pc = ax.pcolormesh(lon2d_deg, lat2d_deg, np.asarray(data),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return pc


def save_snapshots_figure(output_dir, case_tag, snap_fields, lon2d, lat2d,
                          field_key, cmap, label, unit, title_prefix):
    """Save multi-panel snapshot figure for one field."""
    days = sorted(snap_fields.keys())
    n = len(days)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    # Global color limits
    all_vals = np.concatenate([np.asarray(snap_fields[d][field_key]).ravel()
                               for d in days])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = max(abs(vmax - vmin) * 0.02, 1e-6)
    vmin -= pad
    vmax += pad

    for i, day in enumerate(days):
        pc = pcolor_field(axes[i], lon2d, lat2d, snap_fields[day][field_key],
                          cmap, vmin, vmax, f"Day {day}")

    fig.suptitle(f"{title_prefix} — {label}", fontsize=13, fontweight="bold", y=1.02)
    cbar = fig.colorbar(pc, ax=list(axes), shrink=0.8, orientation="horizontal",
                        label=f"{label} [{unit}]", pad=0.15)
    plt.savefig(f"{output_dir}/{case_tag}_{field_key}_snapshots.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def save_timeseries_figure(output_dir, case_tag, diagnostics, title_prefix):
    """Save conservation time series figure and .npz data."""
    days = np.arange(len(diagnostics), dtype=np.float64)

    mass = np.array([d['mass'] for d in diagnostics])
    energy = np.array([d['energy'] for d in diagnostics])
    enstrophy = np.array([d['enstrophy'] for d in diagnostics])

    mass_rel = (mass - mass[0]) / (abs(mass[0]) + 1e-30)
    energy_rel = (energy - energy[0]) / (abs(energy[0]) + 1e-30)
    enstrophy_rel = (enstrophy - enstrophy[0]) / (abs(enstrophy[0]) + 1e-30)

    # Save raw data
    np.savez(
        f"{output_dir}/{case_tag}_timeseries.npz",
        days=days,
        mass=mass, energy=energy, enstrophy=enstrophy,
        mass_rel=mass_rel, energy_rel=energy_rel, enstrophy_rel=enstrophy_rel,
    )

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, data, name, color in [
        (axes[0], mass_rel, "Mass", "b"),
        (axes[1], energy_rel, "Energy", "r"),
        (axes[2], enstrophy_rel, "Enstrophy", "g"),
    ]:
        ax.plot(days, data, f"{color}-", linewidth=1.5)
        ax.set_ylabel(f"Relative {name} change", fontsize=11)
        ax.set_title(f"{name} Conservation", fontsize=12, fontweight="bold")
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Time [days]", fontsize=11)
    fig.suptitle(f"{title_prefix} — Conservation Diagnostics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{case_tag}_conservation.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    return mass_rel, energy_rel, enstrophy_rel


def _weighted_stats_2d(field_2d: np.ndarray, lat_weights: np.ndarray) -> tuple[float, float]:
    """Return area-weighted mean and std for (lat, lon) field."""
    lon_mean = np.mean(field_2d, axis=1)
    lon_mean_sq = np.mean(field_2d * field_2d, axis=1)
    wsum = float(np.sum(lat_weights))
    mean = float(np.sum(lat_weights * lon_mean) / wsum)
    mean_sq = float(np.sum(lat_weights * lon_mean_sq) / wsum)
    std = float(np.sqrt(max(mean_sq - mean * mean, 0.0)))
    return mean, std


def _compute_slab_values(day: int, fields, lat_weights: np.ndarray) -> dict:
    """Compute slab/global-mean diagnostics from gridded fields."""
    h = np.asarray(fields["h"], dtype=np.float64)
    u = np.asarray(fields["u"], dtype=np.float64)
    v = np.asarray(fields["v"], dtype=np.float64)
    vor = np.asarray(fields["vor"], dtype=np.float64)
    speed = np.sqrt(u * u + v * v)

    h_mean, h_std = _weighted_stats_2d(h, lat_weights)
    speed_mean, speed_std = _weighted_stats_2d(speed, lat_weights)
    vor_mean, vor_std = _weighted_stats_2d(vor, lat_weights)
    vor_rms = float(np.sqrt(vor_mean * vor_mean + vor_std * vor_std))

    return {
        "day": float(day),
        "h_mean": h_mean,
        "h_std": h_std,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "vor_mean": vor_mean,
        "vor_rms": vor_rms,
    }


def save_slab_timeseries(output_dir, case_tag, slab_values, title_prefix):
    """Save slab/global-mean time series to CSV/NPZ and figure."""
    days = np.asarray([d["day"] for d in slab_values], dtype=np.float64)
    h_mean = np.asarray([d["h_mean"] for d in slab_values], dtype=np.float64)
    h_std = np.asarray([d["h_std"] for d in slab_values], dtype=np.float64)
    speed_mean = np.asarray([d["speed_mean"] for d in slab_values], dtype=np.float64)
    speed_std = np.asarray([d["speed_std"] for d in slab_values], dtype=np.float64)
    vor_mean = np.asarray([d["vor_mean"] for d in slab_values], dtype=np.float64)
    vor_rms = np.asarray([d["vor_rms"] for d in slab_values], dtype=np.float64)

    np.savez(
        f"{output_dir}/{case_tag}_slab_timeseries.npz",
        days=days,
        h_mean=h_mean,
        h_std=h_std,
        speed_mean=speed_mean,
        speed_std=speed_std,
        vor_mean=vor_mean,
        vor_rms=vor_rms,
    )

    csv_path = f"{output_dir}/{case_tag}_slab_timeseries.csv"
    with open(csv_path, "w") as f:
        f.write("day,h_mean_m,h_std_m,wind_mean_ms,wind_std_ms,vort_mean_s-1,vort_rms_s-1\n")
        for i in range(len(days)):
            f.write(
                f"{days[i]:.6f},{h_mean[i]:.8e},{h_std[i]:.8e},"
                f"{speed_mean[i]:.8e},{speed_std[i]:.8e},"
                f"{vor_mean[i]:.8e},{vor_rms[i]:.8e}\n"
            )

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(days, h_mean, "k-", lw=1.8, label="mean h")
    axes[0].plot(days, h_std, "k--", lw=1.3, label="std(h)")
    axes[0].set_ylabel("Fluid depth [m]")
    axes[0].set_title("Slab/Global Mean Depth", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(days, speed_mean, color="tab:orange", lw=1.8, label="mean |u|")
    axes[1].plot(days, speed_std, color="tab:orange", ls="--", lw=1.3, label="std(|u|)")
    axes[1].set_ylabel("Wind speed [m/s]")
    axes[1].set_title("Slab/Global Mean Wind", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(days, vor_mean, color="tab:blue", lw=1.8, label="mean vort")
    axes[2].plot(days, vor_rms, color="tab:blue", ls="--", lw=1.3, label="rms vort")
    axes[2].set_ylabel("Vorticity [1/s]")
    axes[2].set_title("Slab/Global Mean Vorticity", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Time [days]")

    fig.suptitle(f"{title_prefix} — Slab/Global Mean Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{case_tag}_slab_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Run one case
# =========================================================================

def run_case(case_tag, case_cfg, grid, output_dir, trunc, dt):
    """Run a single Williamson test case."""
    case_dir = f"{output_dir}/{case_tag}"
    os.makedirs(case_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  {case_cfg['name']}")
    print(f"{'=' * 70}")

    # --- Config ---
    if case_cfg["hyperdiff"]:
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        hyperdiff_coeff = 1.0 / (1.0 * 3600.0 * eig_max**2)
    else:
        hyperdiff_coeff = 0.0

    config = SpectralSWConfig(
        mean_depth=5960.0,
        hyperdiff_coeff=hyperdiff_coeff,
        hyperdiff_order=2,
    )

    # --- Init ---
    print("  Initializing...")
    state_init = case_cfg["init_fn"](grid)
    state = state_init

    fields_init = spectral_to_grid(state, grid)
    h_init = fields_init['h']
    print(f"  Height range: [{float(jnp.min(h_init)):.0f}, {float(jnp.max(h_init)):.0f}] m")
    speed_init = jnp.sqrt(fields_init['u']**2 + fields_init['v']**2)
    print(f"  Wind speed max: {float(jnp.max(speed_init)):.1f} m/s")

    # --- Time stepping setup ---
    duration_days = case_cfg["duration_days"]
    snapshot_days_set = set(case_cfg["snapshot_days"])
    steps_per_day = int(round(86400.0 / dt))
    n_steps = duration_days * steps_per_day

    def tendency_fn(s):
        return spectral_sw_tendencies(s, grid, config)
    step_jit = jax.jit(lambda s, dt: ssp_rk3_step(s, tendency_fn, dt))

    # Warm up
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = step_jit(state, dt)
    jax.block_until_ready(_.vor_hat.data)
    print(f"done ({time.time() - t0:.1f}s)")

    # --- Integration ---
    print(f"  Integrating for {duration_days} days ({n_steps:,} steps, dt={dt:.0f}s)...")

    diagnostics = []
    snap_fields = {}
    slab_values = []
    lat_weights = np.asarray(grid.weights, dtype=np.float64)

    # Save day 0
    diag = compute_spectral_diagnostics(state, grid)
    diagnostics.append(diag)
    slab_values.append(_compute_slab_values(day=0, fields=fields_init, lat_weights=lat_weights))
    if 0 in snapshot_days_set:
        snap_fields[0] = {
            "height": np.asarray(fields_init["h"]),
            "wind_speed": np.sqrt(np.asarray(fields_init["u"])**2 + np.asarray(fields_init["v"])**2),
            "vorticity": np.asarray(fields_init["vor"]),
        }

    print(f"\n  {'Day':>6s}  {'h_min':>8s}  {'h_max':>8s}  {'|u|_max':>8s}  "
          f"{'mass_d':>10s}  {'energy_d':>10s}  {'wall_s':>7s}")
    print(f"  {'-' * 68}")

    t_start = time.time()
    t_lap = t_start

    for day_num in range(1, duration_days + 1):
        for _ in range(steps_per_day):
            state = step_jit(state, dt)

        jax.block_until_ready(state.vor_hat.data)

        diag = compute_spectral_diagnostics(state, grid)
        diagnostics.append(diag)

        mass_rel = (diag['mass'] - diagnostics[0]['mass']) / abs(diagnostics[0]['mass'])
        energy_rel = (diag['energy'] - diagnostics[0]['energy']) / abs(diagnostics[0]['energy'])

        fields = spectral_to_grid(state, grid)
        h_min = float(jnp.min(fields['h']))
        h_max = float(jnp.max(fields['h']))
        u_max = float(jnp.max(jnp.sqrt(fields['u']**2 + fields['v']**2)))
        slab_values.append(_compute_slab_values(day=day_num, fields=fields, lat_weights=lat_weights))

        elapsed = time.time() - t_lap
        t_lap = time.time()

        print(f"  {day_num:6d}  {h_min:8.0f}  {h_max:8.0f}  "
              f"{u_max:8.1f}  {mass_rel:10.2e}  {energy_rel:10.2e}  {elapsed:7.1f}")

        if day_num in snapshot_days_set:
            snap_fields[day_num] = {
                "height": np.asarray(fields['h']),
                "wind_speed": np.sqrt(np.asarray(fields['u'])**2
                                       + np.asarray(fields['v'])**2),
                "vorticity": np.asarray(fields['vor']),
            }

    total_time = time.time() - t_start
    steps_per_sec = n_steps / total_time
    print(f"  {'-' * 68}")
    print(f"  Completed in {total_time:.1f}s ({steps_per_sec:.0f} steps/s)")

    # --- Save snapshot data as .npz ---
    for day, flds in snap_fields.items():
        np.savez(f"{case_dir}/snapshot_day{day:03d}.npz", **flds)

    # --- Visualization ---
    lon2d_deg = np.asarray(grid.lon2d) * 180 / np.pi
    lat2d_deg = np.asarray(grid.lat2d) * 180 / np.pi

    print("  Generating figures...")

    save_snapshots_figure(
        case_dir, case_tag, snap_fields, lon2d_deg, lat2d_deg,
        "height", "RdYlBu_r", "Fluid Depth h", "m",
        f"Spectral T{trunc} — {case_cfg['name']}",
    )
    save_snapshots_figure(
        case_dir, case_tag, snap_fields, lon2d_deg, lat2d_deg,
        "wind_speed", "magma", "Wind Speed", "m/s",
        f"Spectral T{trunc} — {case_cfg['name']}",
    )
    save_snapshots_figure(
        case_dir, case_tag, snap_fields, lon2d_deg, lat2d_deg,
        "vorticity", "RdBu_r", "Relative Vorticity", "1/s",
        f"Spectral T{trunc} — {case_cfg['name']}",
    )

    mass_rel, energy_rel, enstrophy_rel = save_timeseries_figure(
        case_dir, case_tag, diagnostics,
        f"Spectral T{trunc} — {case_cfg['name']}",
    )
    save_slab_timeseries(
        case_dir, case_tag, slab_values,
        f"Spectral T{trunc} — {case_cfg['name']}",
    )

    # --- Height perturbation map ---
    fields_final = spectral_to_grid(state, grid)
    h_pert = np.asarray(fields_final['h'] - fields_init['h'])
    vabs = max(float(np.max(np.abs(h_pert))), 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    pc = pcolor_field(ax, lon2d_deg, lat2d_deg, h_pert, "RdBu_r",
                      -vabs, vabs, f"h(day {duration_days}) - h(day 0)  [m]")
    plt.colorbar(pc, ax=ax, shrink=0.8, orientation="horizontal",
                 label="Delta h [m]", pad=0.15)
    fig.suptitle(f"Spectral T{trunc} — {case_cfg['name']} — Height Perturbation",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(f"{case_dir}/{case_tag}_height_perturbation.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- Summary results.txt ---
    with open(f"{case_dir}/results.txt", "w") as f:
        f.write(f"case: {case_cfg['name']}\n")
        f.write(f"resolution: T{trunc}\n")
        f.write(f"duration_days: {duration_days}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"wall_time_s: {total_time:.1f}\n")
        f.write(f"steps_per_sec: {steps_per_sec:.0f}\n")
        f.write(f"final_h_min: {float(jnp.min(fields_final['h'])):.2f}\n")
        f.write(f"final_h_max: {float(jnp.max(fields_final['h'])):.2f}\n")
        f.write(f"final_umax: {float(jnp.max(jnp.sqrt(fields_final['u']**2 + fields_final['v']**2))):.2f}\n")
        f.write(f"mass_drift_rel: {mass_rel[-1]:.6e}\n")
        f.write(f"energy_drift_rel: {energy_rel[-1]:.6e}\n")
        f.write(f"enstrophy_drift_rel: {enstrophy_rel[-1]:.6e}\n")
        f.write(f"final_h_mean: {slab_values[-1]['h_mean']:.6e}\n")
        f.write(f"final_speed_mean: {slab_values[-1]['speed_mean']:.6e}\n")
        f.write(f"final_vort_rms: {slab_values[-1]['vor_rms']:.6e}\n")

    print(f"\n  Results saved to {case_dir}/")
    print(f"    Snapshots: {len(snap_fields)} .npz files")
    print(f"    Time series: {case_tag}_timeseries.npz")
    print(f"    Slab means: {case_tag}_slab_timeseries.csv/.npz/.png")
    print(f"    Figures: *_snapshots.png, *_conservation.png, *_height_perturbation.png")
    print(f"    Summary: results.txt")
    print(f"    Final mass drift:     {mass_rel[-1]:.2e}")
    print(f"    Final energy drift:   {energy_rel[-1]:.2e}")
    print(f"    Final enstrophy drift: {enstrophy_rel[-1]:.2e}")

    return {
        "case": case_tag,
        "wall_time": total_time,
        "mass_drift": mass_rel[-1],
        "energy_drift": energy_rel[-1],
        "enstrophy_drift": enstrophy_rel[-1],
        "h_mean_final": slab_values[-1]["h_mean"],
        "wind_mean_final": slab_values[-1]["speed_mean"],
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Williamson spectral shallow-water tests with diagnostics."
    )
    parser.add_argument(
        "--trunc",
        type=int,
        default=DEFAULT_T_TRUNC,
        help=f"Spectral truncation (default: {DEFAULT_T_TRUNC})",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"Timestep in seconds (default: {DEFAULT_DT})",
    )
    parser.add_argument(
        "--case",
        choices=("all", "williamson2", "williamson5"),
        default="all",
        help="Which Williamson case to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Disable x64 and run in float32 (unsupported accuracy mode for spectral transforms).",
    )
    parser.add_argument(
        "--x64",
        dest="x64",
        action="store_true",
        help="Enable float64/complex128 (recommended for spectral solver).",
    )
    parser.add_argument(
        "--no-x64",
        dest="x64",
        action="store_false",
        help="Disable float64/complex128 (same effect as --float32).",
    )
    parser.set_defaults(x64=None)
    args = parser.parse_args()

    use_x64_runtime = USE_X64
    use_x64_requested = args.x64 if args.x64 is not None else (not args.float32)
    if bool(use_x64_requested) != bool(use_x64_runtime):
        raise RuntimeError("Precision flag parse mismatch. Re-run with consistent --x64/--float32 flags.")

    if abs((86400.0 / args.dt) - round(86400.0 / args.dt)) > 1e-10:
        raise ValueError("--dt must divide 86400 exactly so diagnostics remain daily.")

    output_base = args.output
    os.makedirs(output_base, exist_ok=True)

    print("=" * 70)
    print("legoESM — Williamson Spectral Shallow Water Test Cases")
    print("=" * 70)
    print(f"Precision mode: {'x64 (float64/complex128)' if use_x64_runtime else 'x32 (float32/complex64)'}")
    if not use_x64_runtime:
        print("WARNING: Spectral transforms are validated for x64; x32 is exploratory only.")

    print(f"\nCreating T{args.trunc} Gaussian grid...")
    t0 = time.time()
    grid = create_gaussian_grid(
        args.trunc,
        allow_unsupported_backend=not use_x64_runtime,
    )
    print(f"  Grid: {grid.n_lat}x{grid.n_lon}, {grid.n_sh} spectral coefficients")
    print(f"  Resolution: ~{360.0 / grid.n_lon:.1f} degrees")
    print(f"  Created in {time.time() - t0:.1f}s")

    if args.case == "all":
        case_items = list(CASES.items())
    else:
        case_items = [(args.case, CASES[args.case])]

    results = []
    for case_tag, case_cfg in case_items:
        res = run_case(case_tag, case_cfg, grid, output_base, args.trunc, args.dt)
        results.append(res)

    # --- Overall summary ---
    print(f"\n\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Case':<20s}  {'Wall[s]':>8s}  {'Mass drift':>12s}  "
          f"{'Energy drift':>12s}  {'Enstrophy drift':>15s}")
    print(f"  {'-' * 75}")
    for r in results:
        print(f"  {r['case']:<20s}  {r['wall_time']:8.1f}  "
              f"{r['mass_drift']:12.2e}  {r['energy_drift']:12.2e}  "
              f"{r['enstrophy_drift']:15.2e}")
    print(f"\n  All results in: {output_base}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
