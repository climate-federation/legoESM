"""Plot the surface forcing fields: wind stress, T/S restoring targets.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_plot_forcing.py
"""
from __future__ import annotations

import os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/ocean/global_overturning_mpas_etopo_production/snapshots_nice/forcing.png")
    p.add_argument("--subdivision", type=int, default=5)
    args = p.parse_args()

    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))
    lon_shifted = np.where(lon > 180, lon - 360, lon)

    # Wind stress
    cfg = PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1)
    tau_x, tau_y = compute_wind_stress(mesh.grid_lat, cfg)
    tau_x = np.asarray(tau_x)
    tau_y = np.asarray(tau_y)
    tau_mag = np.sqrt(tau_x**2 + tau_y**2)

    # Restoring targets: T_star = T_eq + (T_pole - T_eq) * (1 - cos(lat)) / 2
    # with cosine profile
    lat_rad = np.asarray(mesh.latCell)
    T_star_eq = 25.0; T_star_pole = 0.0
    T_star = T_star_eq + (T_star_pole - T_star_eq) * (1 - np.cos(lat_rad)) / 2
    S_star = np.full_like(T_star, 35.0)

    # Wind stress curl (approximate via finite differences on cells)
    # Use the zonal component's meridional gradient as proxy
    lat_deg = np.degrees(lat_rad)

    tri = mtri.Triangulation(lon_shifted, lat)

    proj = ccrs.Robinson(central_longitude=200)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12),
                             subplot_kw={"projection": proj})

    def plot_tri(ax, field, title, cmap, vmin=None, vmax=None, symmetric=False):
        if vmin is None:
            vmin = np.nanpercentile(field, 1)
        if vmax is None:
            vmax = np.nanpercentile(field, 99)
        if symmetric:
            vm = max(abs(vmin), abs(vmax))
            vmin, vmax = -vm, vm
        tc = ax.tripcolor(tri, field, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
        ax.coastlines(linewidth=0.3, color="0.4")
        ax.set_global()
        ax.set_title(title, fontsize=11)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

    # Row 1: Wind
    plot_tri(axes[0, 0], tau_x, "Wind stress τ_x (zonal) [Pa]", "RdBu_r", symmetric=True)
    plot_tri(axes[0, 1], tau_y, "Wind stress τ_y (meridional) [Pa]", "RdBu_r", symmetric=True)
    plot_tri(axes[0, 2], tau_mag, "Wind stress magnitude |τ| [Pa]", "YlOrRd", vmin=0)

    # Row 2: Restoring targets
    plot_tri(axes[1, 0], T_star, "SST restoring target T* [°C]", "RdYlBu_r")
    plot_tri(axes[1, 1], S_star, "SSS restoring target S* [PSU]", "YlGnBu",
             vmin=34.9, vmax=35.1)

    # Zonal wind profile (line plot in the last panel)
    ax_line = fig.add_subplot(2, 3, 6)
    # Sort by latitude and bin
    lat_sorted = np.sort(lat_deg)
    lat_bins = np.linspace(-90, 90, 181)
    tau_x_binned = np.zeros(len(lat_bins) - 1)
    for i in range(len(lat_bins) - 1):
        in_bin = (lat_deg >= lat_bins[i]) & (lat_deg < lat_bins[i+1])
        if in_bin.sum() > 0:
            tau_x_binned[i] = tau_x[in_bin].mean()
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    ax_line.plot(tau_x_binned, lat_centers, 'b-', linewidth=2)
    ax_line.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax_line.set_xlabel("τ_x [Pa]")
    ax_line.set_ylabel("Latitude [°]")
    ax_line.set_title("Zonal wind stress profile")
    ax_line.set_ylim(-90, 90)
    ax_line.grid(True, alpha=0.3)

    # Also plot T* profile
    ax2 = ax_line.twiny()
    T_binned = np.zeros(len(lat_bins) - 1)
    for i in range(len(lat_bins) - 1):
        in_bin = (lat_deg >= lat_bins[i]) & (lat_deg < lat_bins[i+1])
        if in_bin.sum() > 0:
            T_binned[i] = T_star[in_bin].mean()
    ax2.plot(T_binned, lat_centers, 'r-', linewidth=2, alpha=0.7)
    ax2.set_xlabel("T* [°C]", color='red')
    ax2.tick_params(axis='x', labelcolor='red')

    fig.suptitle("Surface forcing: wind stress + T/S restoring targets", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
