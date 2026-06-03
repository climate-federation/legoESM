"""Plot using actual Voronoi cell polygons instead of tripcolor.

Slower to render but shows the true cell geometry — no triangulation artifacts.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_plot_voronoi_cells.py <restart.npz>
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import reconstruct_cell_velocity


def build_voronoi_polygons(mesh):
    """Build polygon vertices for each Voronoi cell.

    Returns list of (N_vertices, 2) arrays in (lon_deg, lat_deg).
    """
    lon_v = np.degrees(np.asarray(mesh.lonVertex))
    lat_v = np.degrees(np.asarray(mesh.latVertex))
    lon_v = np.where(lon_v > 180, lon_v - 360, lon_v)

    # verticesOnCell: (maxEdges, nCells) — vertex indices for each cell
    voc = np.asarray(mesh.verticesOnCell)  # (maxEdges, nCells)
    nec = np.asarray(mesh.nEdgesOnCell)    # (nCells,)

    polygons = []
    for c in range(mesh.nCells):
        n = int(nec[c])
        vidx = voc[:n, c]
        lons = lon_v[vidx]
        lats = lat_v[vidx]

        # Handle dateline wrapping: if polygon spans > 180° in lon,
        # shift negative lons by 360
        lon_range = lons.max() - lons.min()
        if lon_range > 180:
            lons = np.where(lons < 0, lons + 360, lons)

        polygons.append(np.column_stack([lons, lats]))

    return polygons


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("restart")
    p.add_argument("--output", default=None)
    p.add_argument("--subdivision", type=int, default=5)
    args = p.parse_args()

    data = np.load(args.restart)
    day = float(data["time_days"])
    year = day / 365.25

    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))

    ocean_mask = data.get("land_mask", np.ones(len(lon)))
    mask = ocean_mask > 0.5

    u_edge = data["u"]
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    speed = np.asarray(jnp.sqrt(u_east**2 + v_north**2))
    eta = np.asarray(data["eta"])
    sst = np.asarray(data["T"][:, 0])

    print("Building Voronoi polygons...")
    polygons = build_voronoi_polygons(mesh)
    print(f"  {len(polygons)} cells")

    # Prepare fields
    spd_sfc = np.where(mask, speed[:, 0], np.nan)
    eta_p = np.where(mask, eta, np.nan)
    sst_p = np.where(mask, sst, np.nan)

    # Filter to ocean cells only
    ocean_idx = np.where(mask)[0]
    ocean_polys = [polygons[i] for i in ocean_idx]

    proj = ccrs.Robinson(central_longitude=200)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8),
                             subplot_kw={"projection": proj})

    def plot_voronoi(ax, values, title, cmap, vmin=None, vmax=None, symmetric=False):
        vals = values[ocean_idx]
        if vmin is None:
            vmin = np.nanpercentile(vals, 1)
        if vmax is None:
            vmax = np.nanpercentile(vals, 99)
        if symmetric:
            vm = max(abs(vmin), abs(vmax))
            vmin, vmax = -vm, vm

        pc = PolyCollection(ocean_polys, transform=ccrs.PlateCarree(),
                           edgecolors='none', linewidths=0)
        pc.set_array(vals)
        pc.set_cmap(cmap)
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
        ax.coastlines(linewidth=0.3, color="0.4")
        ax.set_global()
        ax.set_title(title, fontsize=11)
        plt.colorbar(pc, ax=ax, shrink=0.7, pad=0.02)

    # Extratropical colorbar for speed
    extratrop = np.abs(lat) > 10
    spd_et = np.where(mask & extratrop, speed[:, 0], np.nan)
    vmax_spd = max(0.01, np.nanpercentile(spd_et, 99))

    plot_voronoi(axes[0], spd_sfc,
                 f"Surface speed [m/s] (cbar: |lat|>10°, max={vmax_spd:.2f})",
                 "magma", vmin=0, vmax=vmax_spd)
    plot_voronoi(axes[1], eta_p, "SSH [m]", "RdBu_r", symmetric=True)
    plot_voronoi(axes[2], sst_p, "SST [°C]", "RdYlBu_r")

    fig.suptitle(f"MPAS ETOPO (Voronoi cells) — year {year:.1f}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.restart).parent.parent / "snapshots_nice" / f"voronoi_year{year:.1f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
