"""Jet formation diagnostics for MPAS-vs-LatLon comparison.

Computes two metrics from restart files:
1. Zonal-mean KE fraction: KE_zm / KE_total (high = zonally coherent jets)
2. 2Δy neighbor variance ratio: var(u[j+1]-u[j]) / var(u) (high = grid-scale banding)

Both metrics are computed at the surface level.

Usage:
    python scripts/global_overturning/_jet_diagnostics.py
"""
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_mpas import reconstruct_cell_velocity


def jet_metrics_latlon(restart_path, grid):
    """Compute jet metrics for lat-lon restart."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    mask = d["land_mask"] > 0.5
    u = d["u"]  # (n_lat, n_lon+1, nlev)
    v = d["v"]  # (n_lat+1, n_lon, nlev)

    # Cell-centered surface velocity
    u_sfc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])  # (n_lat, n_lon)
    v_sfc = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])  # (n_lat, n_lon)

    area = np.asarray(grid.area)  # (n_lat, n_lon)

    # --- Metric 1: Zonal-mean KE fraction ---
    # Zonal-mean u at each latitude (ocean cells only)
    n_ocean_per_lat = mask.sum(axis=1)  # (n_lat,)
    n_ocean_per_lat = np.maximum(n_ocean_per_lat, 1)
    u_zm = (u_sfc * mask).sum(axis=1) / n_ocean_per_lat  # (n_lat,)
    v_zm = (v_sfc * mask).sum(axis=1) / n_ocean_per_lat

    # KE from zonal mean
    area_lat = (area * mask).sum(axis=1)  # (n_lat,)
    KE_zm = 0.5 * np.sum((u_zm**2 + v_zm**2) * area_lat)

    # Total KE
    KE_total = 0.5 * np.sum((u_sfc**2 + v_sfc**2) * area * mask)

    frac_zm = KE_zm / max(KE_total, 1e-30)

    # --- Metric 2: 2Δy neighbor variance ratio ---
    # Meridional difference of u (surface, ocean only)
    du_dy = u_sfc[1:, :] - u_sfc[:-1, :]  # (n_lat-1, n_lon)
    # Both neighbors must be ocean
    both_ocean = mask[1:, :] & mask[:-1, :]
    if both_ocean.sum() > 0:
        var_du = np.mean(du_dy[both_ocean]**2)
        var_u = np.mean(u_sfc[mask]**2)
        ratio_2dy = var_du / max(var_u, 1e-30)
    else:
        ratio_2dy = 0.0

    return day, frac_zm, ratio_2dy


def jet_metrics_mpas(restart_path, mesh):
    """Compute jet metrics for MPAS restart."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    mask = d["land_mask"] > 0.5

    u_edge = d["u"]  # (nEdges, nlev)
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    u_east = np.asarray(u_east)[:, 0]   # surface
    v_north = np.asarray(v_north)[:, 0]

    lat_deg = np.degrees(np.asarray(mesh.latCell))
    area = np.asarray(mesh.areaCell)

    # --- Metric 1: Zonal-mean KE fraction ---
    lat_bins = np.arange(-80, 81, 2)
    u_zm = np.zeros(len(lat_bins) - 1)
    v_zm = np.zeros(len(lat_bins) - 1)
    area_zm = np.zeros(len(lat_bins) - 1)

    for i in range(len(lat_bins) - 1):
        in_bin = mask & (lat_deg >= lat_bins[i]) & (lat_deg < lat_bins[i+1])
        if in_bin.sum() > 0:
            u_zm[i] = u_east[in_bin].mean()
            v_zm[i] = v_north[in_bin].mean()
            area_zm[i] = area[in_bin].sum()

    KE_zm = 0.5 * np.sum((u_zm**2 + v_zm**2) * area_zm)
    KE_total = 0.5 * np.sum((u_east**2 + v_north**2) * area * mask)
    frac_zm = KE_zm / max(KE_total, 1e-30)

    # --- Metric 2: Neighbor variance ratio ---
    # On MPAS, use cellsOnCell connectivity
    cellsOnCell = np.asarray(mesh.cellsOnCell)  # (maxNeighbors, nCells)
    n_nbrs = cellsOnCell.shape[0]

    du_sq_sum = 0.0
    n_pairs = 0
    for k in range(n_nbrs):
        nbr = cellsOnCell[k]
        both = mask & (mask[nbr]) & (nbr >= 0)
        du = u_east[both] - u_east[nbr[both]]
        du_sq_sum += np.sum(du**2)
        n_pairs += both.sum()

    var_du = du_sq_sum / max(n_pairs, 1)
    var_u = np.mean(u_east[mask]**2)
    ratio_2dy = var_du / max(var_u, 1e-30)

    return day, frac_zm, ratio_2dy


def main():
    # Collect all experiments
    experiments = {}

    base = Path("results/ocean/comparison_mpas_v_latlon")

    # Lat-lon experiments
    ll_grid = None
    for tag in ["e4_fp64", "e8_bih", "e9_bih_fresh",
                 "F1_flat_lap", "F2_flat_bih", "F3_flat_bih_floor",
                 "F4_flat_bih_Ah1e4", "F5_flat_production"]:
        d = base / f"latlon_{tag}" / "restarts"
        if not d.exists():
            d = base / "latlon" / "restarts" if tag == "e4_fp64" else d
        if d.exists() and list(d.glob("restart_day*.npz")):
            experiments[f"ll_{tag}"] = ("latlon", d)

    # Also check the original latlon dir
    d = base / "latlon_e4_fp64" / "restarts"
    if d.exists() and list(d.glob("restart_day*.npz")):
        experiments["ll_e4_fp64"] = ("latlon", d)

    # MPAS experiments
    for tag in ["e4_fp64", "e7a_apvm", "e7b_apvm_bh",
                 "e5_kzb1e12", "e6_kzb1e15"]:
        d = base / f"mpas_{tag}" / "restarts"
        if d.exists() and list(d.glob("restart_day*.npz")):
            experiments[f"mpas_{tag}"] = ("mpas", d)

    # Also check original mpas dir
    d = base / "mpas_e4_fp64" / "restarts"
    if not d.exists():
        d = base / "mpas" / "restarts"
    if d.exists() and list(d.glob("restart_day*.npz")):
        experiments["mpas_e4_fp64"] = ("mpas", d)

    print(f"Found {len(experiments)} experiments:")
    for name, (grid_type, d) in experiments.items():
        n = len(list(d.glob("restart_day*.npz")))
        print(f"  {name}: {grid_type}, {n} restarts")

    # Compute metrics
    ll_grid = create_latlon_grid(180, 360)
    mpas_mesh = create_voronoi_mesh(subdivision_level=5)

    all_results = {}
    for name, (grid_type, d) in experiments.items():
        print(f"\nProcessing {name}...")
        restarts = sorted(d.glob("restart_day*.npz"))
        results = []
        for f in restarts:
            if grid_type == "latlon":
                day, frac, ratio = jet_metrics_latlon(f, ll_grid)
            else:
                day, frac, ratio = jet_metrics_mpas(f, mpas_mesh)
            results.append((day, frac, ratio))
        all_results[name] = np.array(results)
        if len(results) > 0:
            print(f"  day {results[-1][0]:.0f}: KE_zm_frac={results[-1][1]:.3f}, "
                  f"2dy_ratio={results[-1][2]:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for name, data in all_results.items():
        if len(data) == 0:
            continue
        style = "-" if "mpas" in name else "--"
        axes[0].plot(data[:, 0] / 365.25, data[:, 1], style,
                     label=name, ms=2, lw=1.2)
        axes[1].plot(data[:, 0] / 365.25, data[:, 2], style,
                     label=name, ms=2, lw=1.2)

    axes[0].set_ylabel("KE_zm / KE_total")
    axes[0].set_title("Zonal-mean KE fraction (high = jets)")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0.3, color="r", ls=":", alpha=0.5, label="jet threshold")

    axes[1].set_ylabel("var(Δu_2Δy) / var(u)")
    axes[1].set_title("2Δy neighbor variance ratio (high = grid-scale noise)")
    axes[1].set_xlabel("Sim year")
    axes[1].legend(fontsize=7, ncol=2, loc="best")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = base / "jet_diagnostics.png"
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")

    # Save data
    np.savez(out.with_suffix(".npz"),
             **{f"{k}_day": v[:, 0] for k, v in all_results.items()},
             **{f"{k}_frac_zm": v[:, 1] for k, v in all_results.items()},
             **{f"{k}_ratio_2dy": v[:, 2] for k, v in all_results.items()})
    print(f"Saved data: {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
