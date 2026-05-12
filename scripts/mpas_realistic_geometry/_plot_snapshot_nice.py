"""Re-plot MPAS snapshots with tripcolor + cartopy for proper map visualization.

Usage:
    python scripts/mpas_realistic_geometry/_plot_snapshot_nice.py results/ocean/global_overturning_mpas_etopo_Av1e2/restarts/restart_day002922.npz
    python scripts/mpas_realistic_geometry/_plot_snapshot_nice.py results/ocean/global_overturning_mpas_etopo_Av1e2/restarts/restart_day001826.npz
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
from legoesm.ocean.vertical import create_ocean_z_star


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("restart", help="Path to restart .npz file")
    p.add_argument("--output", default=None, help="Output PNG path (default: next to restart)")
    p.add_argument("--subdivision", type=int, default=5)
    args = p.parse_args()

    # Load restart
    data = np.load(args.restart)
    day = float(data["time_days"])
    year = day / 365.25

    # Mesh
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))

    # Shift lon to [-180, 180] for cartopy PlateCarree
    lon_shifted = np.where(lon > 180, lon - 360, lon)

    # Build Delaunay triangulation from cell centers
    tri = mtri.Triangulation(lon_shifted, lat)

    # Fields
    ocean_mask = data.get("land_mask", np.ones(len(lon)))
    mask = ocean_mask > 0.5
    T = data["T"]       # (nCells, nlev)
    S = data["S"]
    eta = data["eta"]    # (nCells,)
    u_edge = data["u"]   # (nEdges, nlev)

    # Reconstruct cell velocity
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    speed = np.asarray(jnp.sqrt(u_east**2 + v_north**2))

    # Mask triangles where any vertex is land
    mask_tri = np.all(mask[tri.triangles], axis=1)
    tri.set_mask(~mask_tri)

    # Helper: make masked array for ocean-only
    def ocean_field(f):
        return np.where(mask, f, np.nan)

    # --- Plot ---
    proj = ccrs.Robinson(central_longitude=200)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12),
                             subplot_kw={"projection": proj})

    def plot_panel(ax, field, title, cmap, vmin=None, vmax=None, symmetric=False):
        f = ocean_field(field)
        if vmin is None:
            vmin = np.nanpercentile(f, 1)
        if vmax is None:
            vmax = np.nanpercentile(f, 99)
        if symmetric:
            vm = max(abs(vmin), abs(vmax))
            vmin, vmax = -vm, vm
        tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
        ax.coastlines(linewidth=0.3, color="0.4")
        ax.set_global()
        ax.set_title(title, fontsize=12)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

    # Surface speed
    spd_sfc = speed[:, 0]
    plot_panel(axes[0, 0], spd_sfc, "Surface speed [m/s]", "magma",
               vmin=0, vmax=max(0.1, np.nanpercentile(ocean_field(spd_sfc), 99)))

    # SSH
    plot_panel(axes[0, 1], eta, "SSH [m]", "RdBu_r", symmetric=True)

    # SST
    sst = T[:, 0]
    plot_panel(axes[0, 2], sst, "SST [°C]", "RdYlBu_r")

    # Max-depth speed
    spd_max = np.max(speed, axis=1)
    plot_panel(axes[1, 0], spd_max, "Max-depth speed [m/s]", "magma",
               vmin=0, vmax=max(0.1, np.nanpercentile(ocean_field(spd_max), 99)))

    # SSS
    sss = S[:, 0]
    plot_panel(axes[1, 1], sss, "SSS [PSU]", "YlGnBu")

    # Deep T
    nlev = T.shape[1]
    deep_lev = min(15, nlev - 1)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=5500.0,
                                  dz_surface=20.0, dz_deep=500.0)
    depth = float(np.asarray(z_coord.z_full_ref)[deep_lev])
    plot_panel(axes[1, 2], T[:, deep_lev], f"T at {depth:.0f}m [°C]", "RdYlBu_r")

    fig.suptitle(f"MPAS ETOPO global overturning — year {year:.1f} (day {day:.0f})",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.restart).parent.parent / "snapshots_nice" / f"nice_year{year:.1f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
