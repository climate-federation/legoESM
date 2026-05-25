"""Jet diagnostics for flat-bottom F experiments, focused on South Pacific.

Computes for each restart:
1. Zonal-mean KE fraction (global and South Pacific)
2. 2Δy neighbor variance ratio (global and South Pacific)

South Pacific region: 40°S-60°S, 150°E-280°E (150°E to 80°W)

Usage:
    python scripts/global_overturning/_jet_diagnostics_F.py
"""
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.latlon import create_latlon_grid


# South Pacific box
SP_LAT_MIN, SP_LAT_MAX = -60.0, -40.0
SP_LON_MIN, SP_LON_MAX = 150.0, 280.0


def jet_metrics_latlon(restart_path, grid):
    """Compute jet metrics for lat-lon restart (global + South Pacific)."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    mask = d["land_mask"] > 0.5
    u = d["u"]  # (n_lat, n_lon+1, nlev)
    v = d["v"]  # (n_lat+1, n_lon, nlev)

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))
    LON, LAT = np.meshgrid(lon_deg, lat_deg)

    # Cell-centered surface velocity
    u_sfc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_sfc = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    area = np.asarray(grid.area)

    # South Pacific mask
    sp_mask = mask & (LAT >= SP_LAT_MIN) & (LAT <= SP_LAT_MAX) & \
              (LON >= SP_LON_MIN) & (LON <= SP_LON_MAX)

    results = {"day": day}

    for region_name, rmask in [("global", mask), ("south_pacific", sp_mask)]:
        if rmask.sum() == 0:
            results[f"{region_name}_frac_zm"] = 0.0
            results[f"{region_name}_ratio_2dy"] = 0.0
            continue

        # --- Metric 1: Zonal-mean KE fraction ---
        n_per_lat = rmask.sum(axis=1)
        n_per_lat_safe = np.maximum(n_per_lat, 1)
        u_zm = (u_sfc * rmask).sum(axis=1) / n_per_lat_safe
        v_zm = (v_sfc * rmask).sum(axis=1) / n_per_lat_safe
        area_lat = (area * rmask).sum(axis=1)

        KE_zm = 0.5 * np.sum((u_zm**2 + v_zm**2) * area_lat)
        KE_total = 0.5 * np.sum((u_sfc**2 + v_sfc**2) * area * rmask)
        results[f"{region_name}_frac_zm"] = KE_zm / max(KE_total, 1e-30)

        # --- Metric 2: 2Δy neighbor variance ratio ---
        du_dy = u_sfc[1:, :] - u_sfc[:-1, :]
        both_ocean = rmask[1:, :] & rmask[:-1, :]
        if both_ocean.sum() > 0:
            var_du = np.mean(du_dy[both_ocean]**2)
            var_u = np.mean(u_sfc[rmask]**2)
            results[f"{region_name}_ratio_2dy"] = var_du / max(var_u, 1e-30)
        else:
            results[f"{region_name}_ratio_2dy"] = 0.0

    return results


def main():
    grid = create_latlon_grid(180, 360)
    base = Path("results/ocean/comparison_mpas_v_latlon")

    experiments = {}
    for tag in ["F1_flat_lap", "F5_flat_production", "e4_fp64"]:
        d = base / f"latlon_{tag}" / "restarts"
        if d.exists() and list(d.glob("restart_day*.npz")):
            experiments[tag] = d

    print(f"Found {len(experiments)} experiments")
    for name, d in experiments.items():
        n = len(list(d.glob("restart_day*.npz")))
        print(f"  {name}: {n} restarts")

    all_data = {}
    for name, d in experiments.items():
        print(f"\nProcessing {name}...")
        restarts = sorted(d.glob("restart_day*.npz"))
        rows = []
        for i, f in enumerate(restarts):
            r = jet_metrics_latlon(f, grid)
            rows.append(r)
            if i % 20 == 0 or i == len(restarts) - 1:
                print(f"  day {r['day']:7.0f}: "
                      f"SP frac_zm={r['south_pacific_frac_zm']:.3f}, "
                      f"SP 2dy={r['south_pacific_ratio_2dy']:.3f}, "
                      f"global frac_zm={r['global_frac_zm']:.3f}, "
                      f"global 2dy={r['global_ratio_2dy']:.3f}")
        all_data[name] = rows

    # Plot: 2 rows (global, south pacific) x 2 cols (frac_zm, ratio_2dy)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    colors = {"F1_flat_lap": "C0", "F5_flat_production": "C1", "e4_fp64": "C2"}
    labels = {
        "F1_flat_lap": "F1: flat, A_h=1e4, Csmag=0.33",
        "F5_flat_production": "F5: flat, A_h=2e5+latscale, B_h=5e9",
        "e4_fp64": "baseline: ETOPO, A_h=1e4, Csmag=0.33",
    }

    for name, rows in all_data.items():
        days = np.array([r["day"] for r in rows]) / 365.25
        c = colors.get(name, "C3")
        lb = labels.get(name, name)

        axes[0, 0].plot(days, [r["global_frac_zm"] for r in rows],
                        color=c, label=lb, lw=1.2)
        axes[0, 1].plot(days, [r["global_ratio_2dy"] for r in rows],
                        color=c, label=lb, lw=1.2)
        axes[1, 0].plot(days, [r["south_pacific_frac_zm"] for r in rows],
                        color=c, label=lb, lw=1.2)
        axes[1, 1].plot(days, [r["south_pacific_ratio_2dy"] for r in rows],
                        color=c, label=lb, lw=1.2)

    axes[0, 0].set_ylabel("KE_zm / KE_total")
    axes[0, 0].set_title("Global: Zonal-mean KE fraction")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_ylabel("var(Δu_2Δy) / var(u)")
    axes[0, 1].set_title("Global: 2Δy neighbor variance")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_ylabel("KE_zm / KE_total")
    axes[1, 0].set_title(f"South Pacific ({SP_LAT_MIN}°-{SP_LAT_MAX}°, "
                          f"{SP_LON_MIN}°-{SP_LON_MAX}°E)")
    axes[1, 0].set_xlabel("Sim year")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_ylabel("var(Δu_2Δy) / var(u)")
    axes[1, 1].set_title(f"South Pacific: 2Δy neighbor variance")
    axes[1, 1].set_xlabel("Sim year")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out = base / "jet_diagnostics_F.png"
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
