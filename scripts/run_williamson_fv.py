#!/usr/bin/env python
"""Run Williamson finite-volume shallow water test cases 2 and 5.

Produces for each case:
  - Height, wind speed, vorticity snapshots at key days
  - Conservation diagnostics time series (mass, energy)
  - Error norms (L1, L2, Linf) time series for Test 2
  - Integrated values saved as .npz for post-processing

Usage:
    python scripts/run_williamson_fv.py
"""

import sys
import time
import os
from pathlib import Path

# Ensure project root is on sys.path for `tests.test_cases` imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water import (
    ShallowWaterModel,
    ShallowWaterConfig,
)
from tests.test_cases.williamson import (
    williamson_test2,
    williamson_test2_exact,
    williamson_test5,
    compute_error_norms,
)
from legoesm.core.conservation import compute_conservation_diagnostics
from legoesm.core.operators import curl_z

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# Configuration
# =========================================================================
RESOLUTION = 48         # C48 ~ 200 km
DT = 600.0              # 10 min timestep

CASES = {
    "williamson2": {
        "name": "Williamson Test Case 2 — Steady Geostrophic Flow",
        "init_fn": williamson_test2,
        "exact_fn": williamson_test2_exact,
        "duration_days": 5,
        "snapshot_days": [0, 1, 2, 3, 4, 5],
        "has_exact": True,
        "hyperdiff": True,
    },
    "williamson5": {
        "name": "Williamson Test Case 5 — Zonal Flow over Mountain",
        "init_fn": williamson_test5,
        "exact_fn": None,
        "duration_days": 15,
        "snapshot_days": [0, 3, 5, 7, 10, 15],
        "has_exact": False,
        "hyperdiff": True,
    },
}

OUTPUT_BASE = "results/williamson_fv"


# =========================================================================
# Plotting helpers
# =========================================================================

def _regrid_to_latlon(field_faces, grid, n_lon=360, n_lat=180):
    """Regrid cubed-sphere (6, n, n) to lat-lon using nearest-neighbor IDW."""
    from scipy.spatial import cKDTree

    lon = np.asarray(grid.lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(grid.lat, dtype=np.float64).reshape(-1)
    val = np.asarray(field_faces, dtype=np.float64).reshape(-1)

    cos_lat = np.cos(lat)
    src_xyz = np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])

    lon_1d = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat_1d = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    cos_lat_t = np.cos(lat2d.ravel())
    tgt_xyz = np.column_stack([
        cos_lat_t * np.cos(lon2d.ravel()),
        cos_lat_t * np.sin(lon2d.ravel()),
        np.sin(lat2d.ravel()),
    ])

    k = min(8, src_xyz.shape[0])
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(tgt_xyz, k=k)
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist
    w /= np.sum(w, axis=1, keepdims=True)
    field_ll = np.sum(val[idx] * w, axis=1).reshape(n_lat, n_lon)

    return lon2d * 180 / np.pi, lat2d * 180 / np.pi, field_ll


def pcolor_ll(ax, lon2d, lat2d, data, cmap, vmin, vmax, title):
    pc = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return pc


def save_snapshot_panels(output_dir, case_tag, snap_data, grid, field_key,
                         cmap, label, unit, title_prefix):
    """Save multi-panel lat-lon snapshot figure."""
    days = sorted(snap_data.keys())
    n = len(days)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    # Regrid all snapshots
    panels = {}
    for d in days:
        lon2d, lat2d, field_ll = _regrid_to_latlon(snap_data[d][field_key], grid)
        panels[d] = field_ll

    all_vals = np.concatenate([p.ravel() for p in panels.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = max(abs(vmax - vmin) * 0.02, 1e-6)
    vmin -= pad
    vmax += pad

    for i, day in enumerate(days):
        pc = pcolor_ll(axes[i], lon2d, lat2d, panels[day], cmap, vmin, vmax, f"Day {day}")

    fig.suptitle(f"{title_prefix} — {label}", fontsize=13, fontweight="bold", y=1.02)
    fig.colorbar(pc, ax=list(axes), shrink=0.8, orientation="horizontal",
                 label=f"{label} [{unit}]", pad=0.15)
    plt.savefig(f"{output_dir}/{case_tag}_{field_key}_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_timeseries(output_dir, case_tag, diagnostics, error_norms, title_prefix):
    """Save conservation + error time series figure and .npz."""
    n = len(diagnostics)
    days = np.arange(n, dtype=np.float64)

    mass = np.array([float(d['total_mass']) for d in diagnostics])
    energy = np.array([float(d['total_energy']) for d in diagnostics])
    mass_rel = (mass - mass[0]) / (abs(mass[0]) + 1e-30)
    energy_rel = (energy - energy[0]) / (abs(energy[0]) + 1e-30)

    save_dict = dict(days=days, mass=mass, energy=energy,
                     mass_rel=mass_rel, energy_rel=energy_rel)

    n_panels = 2
    if error_norms:
        n_panels = 3
        l1 = np.array([e['l1'] for e in error_norms])
        l2 = np.array([e['l2'] for e in error_norms])
        linf = np.array([e['linf'] for e in error_norms])
        save_dict.update(l1=l1, l2=l2, linf=linf)

    np.savez(f"{output_dir}/{case_tag}_timeseries.npz", **save_dict)

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)

    axes[0].plot(days, mass_rel, "b-", linewidth=1.5)
    axes[0].set_ylabel("Relative mass change")
    axes[0].set_title("Mass Conservation", fontweight="bold")
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(days, energy_rel, "r-", linewidth=1.5)
    axes[1].set_ylabel("Relative energy change")
    axes[1].set_title("Energy Conservation", fontweight="bold")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
    axes[1].grid(True, alpha=0.3)

    if error_norms:
        axes[2].plot(days, l1, "b-", linewidth=1.5, label="L1")
        axes[2].plot(days, l2, "r-", linewidth=1.5, label="L2")
        axes[2].plot(days, linf, "g-", linewidth=1.5, label="Linf")
        axes[2].set_ylabel("Normalized error")
        axes[2].set_title("Height Error Norms vs Exact Solution", fontweight="bold")
        axes[2].legend()
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [days]")
    fig.suptitle(f"{title_prefix} — Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{case_tag}_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    return mass_rel, energy_rel


# =========================================================================
# Run one case
# =========================================================================

def run_case(case_tag, case_cfg, grid, output_dir, fv_limiter="mc"):
    case_dir = f"{output_dir}/{case_tag}_{fv_limiter}"
    os.makedirs(case_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  {case_cfg['name']}  [limiter={fv_limiter}]")
    print(f"{'=' * 70}")

    # --- FV transport with velocity hyperdiffusion ---
    # The FV mass equation provides inherent upwind diffusion for h.
    # The vector-invariant momentum (centered differences) still needs
    # scale-selective damping on velocity, calibrated like the spectral model:
    #   ν₄ = 1 / (τ_damp * λ_max²)  where λ_max = n_eff*(n_eff+1)/a²
    mean_dx = float(jnp.mean(grid.dx))
    if case_cfg.get("hyperdiff", False):
        a = 6.371e6
        n_eff = 2 * RESOLUTION // 3
        eig_max = n_eff * (n_eff + 1) / (a * a)
        tau_damp = 1.0 * 3600.0  # 1-hour damping timescale
        hyperdiff_coeff = 1.0 / (tau_damp * eig_max**2)
    else:
        hyperdiff_coeff = 0.0
    print(f"  Mean dx: {mean_dx / 1000:.0f} km, limiter={fv_limiter}, ν₄={hyperdiff_coeff:.2e}")

    config = ShallowWaterConfig(
        hyperdiff_coeff=hyperdiff_coeff,
        use_conservation_fixer=True,
        fix_mass=True,
        fix_energy=False,
        fv_limiter=fv_limiter,
    )
    model = ShallowWaterModel(grid, config)

    # --- Init ---
    print("  Initializing...")
    state_init = case_cfg["init_fn"](grid)
    state = state_init

    print(f"  Height range: [{float(jnp.min(state.h.data)):.0f}, "
          f"{float(jnp.max(state.h.data)):.0f}] m")
    speed = jnp.sqrt(state.u.data**2 + state.v.data**2)
    print(f"  Wind speed max: {float(jnp.max(speed)):.1f} m/s")

    # --- Time stepping ---
    duration_days = case_cfg["duration_days"]
    snapshot_days_set = set(case_cfg["snapshot_days"])
    steps_per_day = int(86400 / DT)
    n_steps = duration_days * steps_per_day

    # Warm up
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = model.step(state, DT)
    jax.block_until_ready(_.h.data)
    print(f"done ({time.time() - t0:.1f}s)")

    # --- Integration ---
    print(f"  Integrating for {duration_days} days ({n_steps:,} steps, dt={DT:.0f}s)...")

    diagnostics = []
    error_norms = []
    snap_data = {}

    # Day 0
    diag = compute_conservation_diagnostics(state, grid)
    diagnostics.append(diag)
    if case_cfg["has_exact"]:
        enorms = compute_error_norms(state, case_cfg["exact_fn"](grid, 0.0), grid)
        error_norms.append(enorms)

    if 0 in snapshot_days_set:
        zeta = curl_z(state.u, state.v, grid).data
        snap_data[0] = {
            "height": np.asarray(state.h.data),
            "wind_speed": np.asarray(jnp.sqrt(state.u.data**2 + state.v.data**2)),
            "vorticity": np.asarray(zeta),
        }

    print(f"\n  {'Day':>6s}  {'h_min':>8s}  {'h_max':>8s}  {'|u|_max':>8s}  "
          f"{'mass_d':>10s}  {'energy_d':>10s}  {'wall_s':>7s}")
    print(f"  {'-' * 68}")

    t_start = time.time()
    t_lap = t_start

    for day_num in range(1, duration_days + 1):
        for _ in range(steps_per_day):
            state = model.step(state, DT)

        jax.block_until_ready(state.h.data)

        diag = compute_conservation_diagnostics(state, grid)
        diagnostics.append(diag)

        mass_rel = (float(diag['total_mass']) - float(diagnostics[0]['total_mass'])) / abs(float(diagnostics[0]['total_mass']))
        energy_rel = (float(diag['total_energy']) - float(diagnostics[0]['total_energy'])) / abs(float(diagnostics[0]['total_energy']))

        h_min = float(jnp.min(state.h.data))
        h_max = float(jnp.max(state.h.data))
        u_max = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))

        elapsed = time.time() - t_lap
        t_lap = time.time()

        if case_cfg["has_exact"]:
            t_sec = day_num * 86400.0
            enorms = compute_error_norms(state, case_cfg["exact_fn"](grid, t_sec), grid)
            error_norms.append(enorms)
            extra = f"  L2={enorms['l2']:.2e}"
        else:
            extra = ""

        print(f"  {day_num:6d}  {h_min:8.0f}  {h_max:8.0f}  "
              f"{u_max:8.1f}  {mass_rel:10.2e}  {energy_rel:10.2e}  {elapsed:7.1f}{extra}")

        if day_num in snapshot_days_set:
            zeta = curl_z(state.u, state.v, grid).data
            snap_data[day_num] = {
                "height": np.asarray(state.h.data),
                "wind_speed": np.asarray(jnp.sqrt(state.u.data**2 + state.v.data**2)),
                "vorticity": np.asarray(zeta),
            }

    total_time = time.time() - t_start
    steps_per_sec = n_steps / total_time
    print(f"  {'-' * 68}")
    print(f"  Completed in {total_time:.1f}s ({steps_per_sec:.0f} steps/s)")

    # --- Save snapshots ---
    for day, flds in snap_data.items():
        np.savez(f"{case_dir}/snapshot_day{day:03d}.npz", **flds)

    # --- Visualization ---
    print("  Generating figures...")

    title_prefix = f"FV-{fv_limiter.upper()} C{RESOLUTION} — {case_cfg['name']}"

    save_snapshot_panels(case_dir, case_tag, snap_data, grid,
                         "height", "RdYlBu_r", "Fluid Depth h", "m", title_prefix)
    save_snapshot_panels(case_dir, case_tag, snap_data, grid,
                         "wind_speed", "magma", "Wind Speed", "m/s", title_prefix)
    save_snapshot_panels(case_dir, case_tag, snap_data, grid,
                         "vorticity", "RdBu_r", "Relative Vorticity", "1/s", title_prefix)

    mass_rel_arr, energy_rel_arr = save_timeseries(
        case_dir, case_tag, diagnostics,
        error_norms if case_cfg["has_exact"] else None,
        title_prefix,
    )

    # --- Height perturbation ---
    h_pert_faces = np.asarray(state.h.data - state_init.h.data)
    lon2d, lat2d, h_pert_ll = _regrid_to_latlon(h_pert_faces, grid)
    vabs = max(float(np.max(np.abs(h_pert_ll))), 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    pc = pcolor_ll(ax, lon2d, lat2d, h_pert_ll, "RdBu_r", -vabs, vabs,
                   f"h(day {duration_days}) - h(day 0)  [m]")
    plt.colorbar(pc, ax=ax, shrink=0.8, orientation="horizontal",
                 label="Delta h [m]", pad=0.15)
    fig.suptitle(f"{title_prefix} — Height Perturbation", fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(f"{case_dir}/{case_tag}_height_perturbation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Summary ---
    with open(f"{case_dir}/results.txt", "w") as f:
        f.write(f"case: {case_cfg['name']}\n")
        f.write(f"fv_limiter: {fv_limiter}\n")
        f.write(f"resolution: C{RESOLUTION}\n")
        f.write(f"duration_days: {duration_days}\n")
        f.write(f"dt: {DT}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"hyperdiff_coeff: {hyperdiff_coeff:.2e}\n")
        f.write(f"wall_time_s: {total_time:.1f}\n")
        f.write(f"steps_per_sec: {steps_per_sec:.0f}\n")
        f.write(f"final_h_min: {float(jnp.min(state.h.data)):.2f}\n")
        f.write(f"final_h_max: {float(jnp.max(state.h.data)):.2f}\n")
        f.write(f"final_umax: {float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))):.2f}\n")
        f.write(f"mass_drift_rel: {mass_rel_arr[-1]:.6e}\n")
        f.write(f"energy_drift_rel: {energy_rel_arr[-1]:.6e}\n")
        if case_cfg["has_exact"] and error_norms:
            f.write(f"final_L1: {error_norms[-1]['l1']:.6e}\n")
            f.write(f"final_L2: {error_norms[-1]['l2']:.6e}\n")
            f.write(f"final_Linf: {error_norms[-1]['linf']:.6e}\n")

    print(f"\n  Results saved to {case_dir}/")
    print(f"    Final mass drift:   {mass_rel_arr[-1]:.2e}")
    print(f"    Final energy drift: {energy_rel_arr[-1]:.2e}")
    if case_cfg["has_exact"] and error_norms:
        print(f"    Final L2 error:     {error_norms[-1]['l2']:.2e}")

    return {
        "case": case_tag,
        "limiter": fv_limiter,
        "wall_time": total_time,
        "mass_drift": mass_rel_arr[-1],
        "energy_drift": energy_rel_arr[-1],
        "l2_error": error_norms[-1]['l2'] if error_norms else None,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 70)
    print("legoESM — Williamson FV Shallow Water Test Cases")
    print("=" * 70)

    print(f"\nCreating C{RESOLUTION} cubed-sphere grid...")
    t0 = time.time()
    grid = create_cubed_sphere(RESOLUTION)
    print(f"  Grid: 6x{grid.n}x{grid.n} = {grid.n_cells:,} cells")
    print(f"  Resolution: ~{grid.resolution_km:.0f} km")
    print(f"  Created in {time.time() - t0:.1f}s")

    LIMITERS = ["mc", "ppm", "weno5"]

    results = []
    for fv_limiter in LIMITERS:
        for case_tag, case_cfg in CASES.items():
            res = run_case(case_tag, case_cfg, grid, OUTPUT_BASE,
                           fv_limiter=fv_limiter)
            results.append(res)

    # --- Overall summary ---
    print(f"\n\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Case':<20s}  {'Limiter':<8s}  {'Wall[s]':>8s}  {'Mass drift':>12s}  "
          f"{'Energy drift':>12s}  {'L2 error':>12s}")
    print(f"  {'-' * 80}")
    for r in results:
        l2_str = f"{r['l2_error']:.2e}" if r['l2_error'] is not None else "N/A"
        print(f"  {r['case']:<20s}  {r['limiter']:<8s}  {r['wall_time']:8.1f}  "
              f"{r['mass_drift']:12.2e}  {r['energy_drift']:12.2e}  "
              f"{l2_str:>12s}")
    print(f"\n  All results in: {OUTPUT_BASE}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
