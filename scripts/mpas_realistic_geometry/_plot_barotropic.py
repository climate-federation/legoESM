"""Plot depth-averaged (barotropic) velocity and transport streamfunction.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_plot_barotropic.py <restart.npz>
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
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.vertical import compute_layer_thickness, compute_ocean_jacobian
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import min_cell_to_edge


def snap_partial_cells(H_bathy, z_coord, min_frac=0.30):
    abs_z_half = jnp.abs(z_coord.z_half_ref); nlev = z_coord.n_levels
    n_above = jnp.sum(abs_z_half[None, :] < H_bathy[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]; dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_bathy - abs_z_at_bottom; frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]; z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_bathy - z_upper < z_lower - H_bathy, z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


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
    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0,
                                  dz_surface=20.0, dz_deep=500.0)

    # Bathymetry for layer thickness
    bathy_cfg = BathymetryConfig(source="file",
        path="/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc",
        H_max=5500.0, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True)
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord, min_frac=0.30)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))
    lon_shifted = np.where(lon > 180, lon - 360, lon)
    mask = np.asarray(ocean_mask_new) > 0.5

    u_edge = jnp.asarray(data["u"])
    eta = jnp.asarray(data["eta"])

    # Compute layer thickness for depth-averaging
    h_k = compute_layer_thickness(eta, H_snapped, pc_coord, min_water_column_m=0.1)

    # Reconstruct cell velocity at each level, then depth-average
    u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
    # Depth-average: u_bar = sum(u * h) / sum(h)
    H_cell = jnp.maximum(jnp.sum(h_k, axis=1), 1e-10)  # (nCells,)
    u_bar = jnp.sum(u_east * h_k, axis=1) / H_cell  # (nCells,)
    v_bar = jnp.sum(v_north * h_k, axis=1) / H_cell

    # Transport: U = u_bar * H, V = v_bar * H  [m²/s]
    U_transport = np.asarray(u_bar * H_cell)
    V_transport = np.asarray(v_bar * H_cell)

    u_bar_np = np.asarray(u_bar)
    v_bar_np = np.asarray(v_bar)
    speed_bar = np.sqrt(u_bar_np**2 + v_bar_np**2)

    tri = mtri.Triangulation(lon_shifted, lat)
    mask_tri = np.all(mask[tri.triangles], axis=1)
    tri.set_mask(~mask_tri)

    def ocean_field(f):
        return np.where(mask, f, np.nan)

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
        ax.set_title(title, fontsize=11)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

    # Row 1: depth-averaged velocity
    plot_panel(axes[0, 0], u_bar_np, "Depth-avg u (eastward) [m/s]", "RdBu_r", symmetric=True)
    plot_panel(axes[0, 1], v_bar_np, "Depth-avg v (northward) [m/s]", "RdBu_r", symmetric=True)

    # Speed with extratropical colorbar
    extratrop = np.abs(lat) > 10
    spd_et = np.where(mask & extratrop, speed_bar, np.nan)
    vmax_spd = max(0.01, np.nanpercentile(spd_et, 99))
    plot_panel(axes[0, 2], speed_bar, f"Depth-avg speed [m/s] (cbar: |lat|>10°)",
               "magma", vmin=0, vmax=vmax_spd)

    # Row 2: depth-integrated transport
    plot_panel(axes[1, 0], U_transport, "Zonal transport U·H [m²/s]", "RdBu_r", symmetric=True)
    plot_panel(axes[1, 1], V_transport, "Meridional transport V·H [m²/s]", "RdBu_r", symmetric=True)

    # SSH for context
    eta_np = np.asarray(eta)
    plot_panel(axes[1, 2], eta_np, "SSH [m]", "RdBu_r", symmetric=True)

    fig.suptitle(f"MPAS ETOPO — barotropic (depth-averaged) flow, year {year:.1f}",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.restart).parent.parent / "snapshots_nice" / f"barotropic_year{year:.1f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
