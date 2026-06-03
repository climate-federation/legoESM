"""Plot reconstructed (u_east, v_north) velocity components at the surface.

Shows the zonal and meridional flow patterns — useful for identifying
equatorial currents, WBCs, ACC, and gyre circulation.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_plot_uv.py <restart.npz> [--output out.png]
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
    lon_shifted = np.where(lon > 180, lon - 360, lon)

    ocean_mask = data.get("land_mask", np.ones(len(lon)))
    mask = ocean_mask > 0.5

    u_edge = data["u"]
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    u_east = np.asarray(u_east)
    v_north = np.asarray(v_north)

    tri = mtri.Triangulation(lon_shifted, lat)
    mask_tri = np.all(mask[tri.triangles], axis=1)
    tri.set_mask(~mask_tri)

    def ocean_field(f):
        return np.where(mask, f, np.nan)

    proj = ccrs.Robinson(central_longitude=200)
    fig, axes = plt.subplots(3, 2, figsize=(24, 18),
                             subplot_kw={"projection": proj})

    depths = [0, 5, 12]  # surface, ~450m, ~2200m
    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0,
                                  dz_surface=20.0, dz_deep=500.0)
    z_full = np.asarray(z_coord.z_full_ref)

    for row, k in enumerate(depths):
        depth_m = z_full[k]
        u_k = ocean_field(u_east[:, k])
        v_k = ocean_field(v_north[:, k])

        # Symmetric colorbar
        u_vm = max(0.01, np.nanpercentile(np.abs(u_k), 99))
        v_vm = max(0.01, np.nanpercentile(np.abs(v_k), 99))

        # u_east (zonal)
        ax = axes[row, 0]
        tc = ax.tripcolor(tri, u_k, cmap="RdBu_r", vmin=-u_vm, vmax=u_vm,
                          transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
        ax.coastlines(linewidth=0.3, color="0.4")
        ax.set_global()
        ax.set_title(f"u (eastward) at {depth_m:.0f}m  [m/s]", fontsize=11)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

        # v_north (meridional)
        ax = axes[row, 1]
        tc = ax.tripcolor(tri, v_k, cmap="RdBu_r", vmin=-v_vm, vmax=v_vm,
                          transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
        ax.coastlines(linewidth=0.3, color="0.4")
        ax.set_global()
        ax.set_title(f"v (northward) at {depth_m:.0f}m  [m/s]", fontsize=11)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

    fig.suptitle(f"MPAS ETOPO — velocity components, year {year:.1f}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.restart).parent.parent / "snapshots_nice" / f"uv_year{year:.1f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
