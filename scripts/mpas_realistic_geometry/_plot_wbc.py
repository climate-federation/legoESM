"""WBC-focused plots: exclude equatorial band, zoom into basins.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_plot_wbc.py results/ocean/global_overturning_mpas_etopo_Av1e2/restarts/restart_day002922.npz
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
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import reconstruct_cell_velocity


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("restart", help="Path to restart .npz file")
    p.add_argument("--output", default=None)
    p.add_argument("--subdivision", type=int, default=5)
    args = p.parse_args()

    data = np.load(args.restart)
    day = float(data["time_days"])
    year = day / 365.25

    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))
    lon_shifted = np.where(lon > 180, lon - 360, lon)

    ocean_mask = data.get("land_mask", np.ones(len(lon)))
    mask = ocean_mask > 0.5

    T = data["T"]
    eta = data["eta"]
    u_edge = data["u"]

    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    speed = np.asarray(jnp.sqrt(u_east**2 + v_north**2))
    spd_sfc = speed[:, 0]

    # Mask equatorial band for the colorbar computation
    extratropical = np.abs(lat) > 10
    spd_extratrop = np.where(mask & extratropical, spd_sfc, np.nan)
    vmax_spd = max(0.05, np.nanpercentile(spd_extratrop, 99))

    # SSH — remove global mean for better contrast
    eta_ocean = np.where(mask, eta, np.nan)
    eta_centered = eta - np.nanmean(eta_ocean)

    tri = mtri.Triangulation(lon_shifted, lat)
    mask_tri = np.all(mask[tri.triangles], axis=1)
    tri.set_mask(~mask_tri)

    def ocean_field(f):
        return np.where(mask, f, np.nan)

    # Basin zoom regions
    basins = {
        "Gulf Stream": {"lon": (-85, -30), "lat": (20, 55)},
        "Kuroshio": {"lon": (110, 180), "lat": (15, 50)},
        "Agulhas + S. Indian": {"lon": (10, 80), "lat": (-50, -20)},
        "ACC + Drake": {"lon": (-80, 0), "lat": (-70, -30)},
    }

    fig = plt.figure(figsize=(24, 20))

    # Top row: global maps (speed, SSH) with equator-excluded colorbar
    ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.Robinson(central_longitude=200))
    tc = ax1.tripcolor(tri, ocean_field(spd_sfc), cmap="magma", vmin=0, vmax=vmax_spd,
                       transform=ccrs.PlateCarree(), rasterized=True)
    ax1.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
    ax1.coastlines(linewidth=0.3, color="0.4")
    ax1.set_global()
    ax1.set_title(f"Surface speed [m/s] (colorbar: |lat|>10°, max={vmax_spd:.2f})", fontsize=11)
    plt.colorbar(tc, ax=ax1, shrink=0.7, pad=0.02)

    ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.Robinson(central_longitude=200))
    eta_extratrop = np.where(mask & extratropical, eta_centered, np.nan)
    vm_eta = max(0.05, np.nanpercentile(np.abs(eta_extratrop), 99))
    tc = ax2.tripcolor(tri, ocean_field(eta_centered), cmap="RdBu_r",
                       vmin=-vm_eta, vmax=vm_eta,
                       transform=ccrs.PlateCarree(), rasterized=True)
    ax2.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
    ax2.coastlines(linewidth=0.3, color="0.4")
    ax2.set_global()
    ax2.set_title(f"SSH anomaly [m] (colorbar: |lat|>10°)", fontsize=11)
    plt.colorbar(tc, ax=ax2, shrink=0.7, pad=0.02)

    # Basin zoom panels
    for i, (name, bbox) in enumerate(basins.items()):
        ax = fig.add_subplot(3, 2, i + 3,
                             projection=ccrs.PlateCarree())
        ax.set_extent([bbox["lon"][0], bbox["lon"][1],
                       bbox["lat"][0], bbox["lat"][1]],
                      crs=ccrs.PlateCarree())

        # Speed in this region for local colorbar
        in_box = ((lon_shifted >= bbox["lon"][0]) & (lon_shifted <= bbox["lon"][1])
                  & (lat >= bbox["lat"][0]) & (lat <= bbox["lat"][1]) & mask)
        local_spd = spd_sfc[in_box]
        local_vmax = max(0.02, np.percentile(local_spd[local_spd > 0], 99)) if len(local_spd[local_spd > 0]) > 0 else 0.1

        tc = ax.tripcolor(tri, ocean_field(spd_sfc), cmap="magma",
                          vmin=0, vmax=local_vmax,
                          transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.5)
        ax.coastlines(linewidth=0.5, color="0.3")
        ax.set_title(f"{name} — surface speed [m/s]", fontsize=11)
        plt.colorbar(tc, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle(f"MPAS ETOPO — WBC diagnostics, year {year:.1f}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.restart).parent.parent / "snapshots_nice" / f"wbc_year{year:.1f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
