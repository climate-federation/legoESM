#!/usr/bin/env python
"""Render visual snapshots of the MPAS OMIP (JRA55-do forced) run.

Adapted from ``global_overturning/run_comparison_mpas.py:save_snapshot``
with cartopy Robinson projection + tripcolor rendering.  Replaces the
max-depth speed panel with a zonal-mean T(lat, z) section.

Layout (2x3):
  Surface speed | SSH          | SST
  Zonal-mean T  | SSS          | Deep T

Usage:
    python scripts/plot_mpas_omip_snapshot.py restart_day000450.npz
    python scripts/plot_mpas_omip_snapshot.py results/mpas_jra55_etopo_100yr/mpas/ico5/*.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _reconstruct_cell_speed(u_edges: np.ndarray, mesh) -> np.ndarray:
    """Cell-centered |U| from edge-normal velocities."""
    import jax.numpy as jnp
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edges), mesh)
    speed = np.sqrt(np.asarray(u_east) ** 2 + np.asarray(v_north) ** 2)
    return speed


def _zonal_mean_T(T_3d, lat_cell_rad, land_mask, n_bins=72):
    """Bin T by latitude and average over ocean cells in each bin."""
    lat_deg = np.asarray(lat_cell_rad) * 180.0 / np.pi
    edges = np.linspace(-90.0, 90.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nlev = T_3d.shape[1]
    T_zm = np.full((n_bins, nlev), np.nan)
    is_ocean = np.asarray(land_mask) > 0.5
    for i in range(n_bins):
        in_bin = is_ocean & (lat_deg >= edges[i]) & (lat_deg < edges[i + 1])
        if in_bin.any():
            T_zm[i, :] = np.mean(T_3d[in_bin, :], axis=0)
    return centers, T_zm


def plot_snapshot(restart_path, mesh, z_coord, output_dir=None):
    """Generate a 6-panel diagnostic PNG from a restart file."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False

    npz = np.load(restart_path, allow_pickle=False)
    day = float(npz["time_days"])
    print(f"Loading {restart_path.name} (sim day {day:.0f}, year {day/365:.2f})")

    eta = np.asarray(npz["eta"])
    T = np.asarray(npz["T"])
    u = np.asarray(npz["u"])
    S = np.asarray(npz["S"])
    land_mask = np.asarray(npz["land_mask"])
    nlev = T.shape[1]

    mask_np = land_mask > 0.5
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))
    lon_shifted = np.where(lon > 180, lon - 360, lon)

    speed = _reconstruct_cell_speed(u, mesh)
    sst = T[:, 0]
    sss = S[:, 0]
    deep_lev = min(15, nlev - 1)
    T_deep = T[:, deep_lev]
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)

    print(f"  nCells={T.shape[0]}, nlev={nlev}")
    print(f"  eta: [{eta[mask_np].min():.3f}, {eta[mask_np].max():.3f}] m")
    print(f"  SST: [{sst[mask_np].min():.1f}, {sst[mask_np].max():.1f}] C")
    print(f"  SSS: [{sss[mask_np].min():.2f}, {sss[mask_np].max():.2f}] PSU")
    print(f"  |U|_sfc: [{speed[mask_np, 0].min():.3f}, "
          f"{speed[mask_np, 0].max():.3f}] m/s")

    # Build Delaunay triangulation; mask land triangles
    tri = mtri.Triangulation(lon_shifted, lat)
    mask_tri = np.all(mask_np[tri.triangles], axis=1)
    # Remove triangles whose edges span >90° in longitude — these are
    # spurious Delaunay connections across the projection seam (at 20°E
    # for central_longitude=200) that tripcolor renders as stretched
    # artifacts.
    tri_lons = lon_shifted[tri.triangles]  # (n_tri, 3)
    seam_tri = (np.max(tri_lons, axis=1) - np.min(tri_lons, axis=1)) > 90.0
    tri.set_mask(~mask_tri | seam_tri)

    def ocean_field(f):
        return np.where(mask_np, f, np.nan)

    # --- Figure ---
    if _has_cartopy:
        proj = ccrs.Robinson(central_longitude=200)
        fig, axes = plt.subplots(2, 3, figsize=(24, 12),
                                 subplot_kw={"projection": proj})
    else:
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    def plot_panel(ax, field, title, cmap, vmin=None, vmax=None,
                   symmetric=False):
        f = ocean_field(field)
        if vmin is None:
            vmin = float(np.nanpercentile(f, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(f, 99))
        if symmetric:
            vm = max(abs(vmin), abs(vmax))
            vmin, vmax = -vm, vm
        if _has_cartopy:
            tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree(), rasterized=True)
            ax.add_feature(cfeature.LAND, facecolor="0.85",
                           edgecolor="0.5", linewidth=0.3)
            ax.coastlines(linewidth=0.3, color="0.4")
            ax.set_global()
        else:
            tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                              rasterized=True)
        ax.set_title(title, fontsize=12)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

    # Row 0: Surface speed | SSH | SST
    spd_sfc = speed[:, 0]
    plot_panel(axes[0, 0], spd_sfc, "Surface speed [m/s]", "magma",
               vmin=0, vmax=max(0.1, float(np.nanpercentile(
                   ocean_field(spd_sfc), 99))))
    plot_panel(axes[0, 1], eta, "SSH [m]", "RdBu_r", symmetric=True)
    plot_panel(axes[0, 2], sst, "SST [C]", "RdYlBu_r")

    # Row 1: Zonal-mean T section | SSS | Deep T
    # -- Zonal-mean T (replaces max-depth speed) --
    ax_zm = axes[1, 0]
    if _has_cartopy:
        # Remove the cartopy projection for this panel — it's a section
        ax_zm.remove()
        ax_zm = fig.add_subplot(2, 3, 4)
    lat_centers, T_zm = _zonal_mean_T(T, mesh.latCell, land_mask, n_bins=72)
    depth = -z_full
    LAT, Z = np.meshgrid(lat_centers, depth, indexing="ij")
    pcm = ax_zm.pcolormesh(LAT, Z, T_zm, cmap="RdYlBu_r", shading="auto")
    ax_zm.invert_yaxis()
    ax_zm.set_xlabel("Latitude")
    ax_zm.set_ylabel("Depth (m)")
    ax_zm.set_title("Zonal-mean T (C)", fontsize=12)
    plt.colorbar(pcm, ax=ax_zm, shrink=0.7, pad=0.02)

    # SSS
    plot_panel(axes[1, 1], sss, "SSS [PSU]", "YlGnBu")
    # Deep T
    plot_panel(axes[1, 2], T_deep,
               f"T at level {deep_lev} [~{int(-z_full[deep_lev])} m] (C)",
               "RdYlBu_r")

    fig.suptitle(
        f"MPAS OMIP (JRA55 RYF) — day {day:.0f} "
        f"(year {day/365:.1f}), ico{int(np.log2(mesh.nCells/10+1)):.0f} "
        f"({T.shape[0]} cells, ETOPO)",
        fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir is not None:
        out_dir = Path(output_dir)
    elif restart_path.parent.name == "restarts":
        # Convention: restarts/ and snapshots/ are siblings
        out_dir = restart_path.parent.parent / "snapshots"
    else:
        out_dir = restart_path.parent / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"snapshot_day{int(round(day)):06d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("restarts", type=Path, nargs="+",
                   help="Path(s) to restart_dayXXXXXX.npz")
    p.add_argument("--sub", type=int, default=5,
                   help="Icosahedral subdivision level (default 5)")
    p.add_argument("--H-max", type=float, default=5500.0,
                   help="Maximum ocean depth [m] (default 5500)")
    p.add_argument("--dz-surface", type=float, default=20.0,
                   help="Surface layer thickness [m] (default 20)")
    p.add_argument("--dz-deep", type=float, default=500.0,
                   help="Deep layer thickness [m] (default 500)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: same as restart)")
    args = p.parse_args()

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(args.sub)
    # Infer nlev from first restart
    npz0 = np.load(args.restarts[0], allow_pickle=False)
    nlev = int(npz0["T"].shape[1])
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=args.H_max,
        dz_surface=args.dz_surface, dz_deep=args.dz_deep)

    for restart_path in sorted(args.restarts):
        if not restart_path.exists():
            print(f"  Skipping: {restart_path} (not found)")
            continue
        plot_snapshot(restart_path, mesh, z_coord, args.output_dir)


if __name__ == "__main__":
    main()
