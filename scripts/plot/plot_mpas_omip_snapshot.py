#!/usr/bin/env python
"""Render visual snapshots of the MPAS OMIP (JRA55-do forced) run, and of
FESOM2-JAX (fesom_jax) monthly/daily output stores on the same panel layout.

Adapted from ``global_overturning/run_comparison_mpas.py:save_snapshot``
with cartopy Robinson projection + tripcolor rendering.  Replaces the
max-depth speed panel with a zonal-mean T(lat, z) section.

Layout (2x3):
  Surface speed | SSH          | SST
  Zonal-mean T  | SSS          | Deep T

Usage:
    python scripts/plot_mpas_omip_snapshot.py restart_day000450.npz
    python scripts/plot_mpas_omip_snapshot.py results/mpas_jra55_etopo_100yr/mpas/ico5/*.npz
    python scripts/plot_mpas_omip_snapshot.py --fesom-store RUN/monthly/1958_01 \
        --mesh-dir /path/to/mesh_forca20 --output-dir RUN/snapshots
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


def _zonal_mean_T(T_3d, lat_cell_rad, land_mask, n_bins=72, reduce=np.mean):
    """Bin T by latitude and average over ocean cells in each bin.

    ``reduce=np.nanmean`` for columns whose below-bottom levels are NaN
    (FESOM's variable-depth columns); the MPAS z-star path has full columns.
    """
    lat_deg = np.asarray(lat_cell_rad) * 180.0 / np.pi
    edges = np.linspace(-90.0, 90.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nlev = T_3d.shape[1]
    T_zm = np.full((n_bins, nlev), np.nan)
    is_ocean = np.asarray(land_mask) > 0.5
    for i in range(n_bins):
        in_bin = is_ocean & (lat_deg >= edges[i]) & (lat_deg < edges[i + 1])
        if not in_bin.any():
            continue
        sub = T_3d[in_bin, :]
        if reduce is np.nanmean:
            # explicit count so an all-NaN (below-bottom) level stays NaN without a warning
            n = np.isfinite(sub).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                T_zm[i, :] = np.where(n > 0, np.nansum(sub, axis=0) / n, np.nan)
        else:
            T_zm[i, :] = reduce(sub, axis=0)
    return centers, T_zm


def build_ocean_triangulation(lon_deg, lat_deg, ocean_mask, max_edge_deg=5.0):
    """Delaunay triangulation over cell centres with unusable triangles masked.

    The Delaunay connects ALL points, including ocean cells on opposite sides
    of continents; three filters apply: all 3 vertices ocean, no edge spanning
    the projection seam (>90 deg longitude), no edge longer than
    ``max_edge_deg`` (removes cross-land connections at coastlines).
    """
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_shifted = np.where(lon_deg > 180, lon_deg - 360, lon_deg)
    tri = mtri.Triangulation(lon_shifted, lat_deg)
    ocean_tri = np.all(np.asarray(ocean_mask, bool)[tri.triangles], axis=1)
    tri_lons = lon_shifted[tri.triangles]
    tri_lats = lat_deg[tri.triangles]
    dlon = np.max(tri_lons, axis=1) - np.min(tri_lons, axis=1)
    dlat = np.max(tri_lats, axis=1) - np.min(tri_lats, axis=1)
    seam_tri = dlon > 90.0
    long_tri = (dlon > max_edge_deg) | (dlat > max_edge_deg)
    tri.set_mask(~ocean_tri | seam_tri | long_tri)
    return tri


def plot_native_panel(ax, lon_deg, lat_deg, field, title, cmap, has_cartopy,
                      vmin=None, vmax=None, symmetric=False, point_size=1.2):
    """One panel drawn at the mesh's OWN resolution: one marker per node, no
    interpolation and no binning.

    Rasterising an unstructured mesh onto a regular grid is lossy in both
    directions at once: a 0.25 degree cell averages several nodes together
    where the mesh is fine, and stays empty where the mesh is coarser than the
    cell, which then has to be filled from a neighbour.  Drawing the nodes
    themselves shows the values the model actually carries.  Element polygons
    would be truer still, but a masked triangulation of this mesh's 4.2 million
    elements takes over 25 minutes to build against 8 seconds for the nodes.
    """
    f = np.asarray(field, dtype=np.float64)
    vmin, vmax = _colour_limits(f, vmin, vmax, symmetric)
    kw = {}
    if has_cartopy:
        import cartopy.crs as ccrs
        kw["transform"] = ccrs.PlateCarree()
    tc = ax.scatter(np.asarray(lon_deg), np.asarray(lat_deg), c=f, s=point_size,
                    marker=".", linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax,
                    rasterized=True, **kw)
    return _finish_map_panel(ax, tc, title, has_cartopy)


def raster_mean(lon_deg, lat_deg, field, res_deg, fill_gap_cells=0):
    """Bin point values onto a regular lon-lat grid (cell mean; NaN where no
    point falls). Returns ``(lon_edges, lat_edges, grid[n_lat, n_lon])``.

    Millions of unstructured nodes render in seconds this way, where a
    tripcolor of the mesh's own elements through cartopy takes tens of
    minutes.  ``res_deg`` must divide 360.  Longitude cells shrink toward
    the poles, so a raster finer than the local mesh spacing leaves empty
    cells there; ``fill_gap_cells`` > 0 fills gaps of up to that many cells
    along each latitude row from the nearest filled neighbour (display
    only — the filled cells are copies, not data).
    """
    if not (res_deg > 0) or abs(180.0 / res_deg - round(180.0 / res_deg)) > 1e-9:
        raise ValueError(f"res_deg must be positive and divide 180 (got {res_deg})")
    n_lat = int(round(180.0 / res_deg))
    n_lon = 2 * n_lat
    res = 360.0 / n_lon
    lon = np.mod(np.asarray(lon_deg, dtype=np.float64) + 180.0, 360.0) - 180.0
    lat = np.asarray(lat_deg, dtype=np.float64)
    f = np.asarray(field, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(lon) & np.isfinite(lat) & (lat >= -90.0) & (lat <= 90.0)
    i = np.clip(np.floor((lon[ok] + 180.0) / res).astype(np.int64), 0, n_lon - 1)
    j = np.clip(np.floor((lat[ok] + 90.0) / res).astype(np.int64), 0, n_lat - 1)
    flat = j * n_lon + i
    count = np.bincount(flat, minlength=n_lon * n_lat).astype(np.float64)
    total = np.bincount(flat, weights=f[ok], minlength=n_lon * n_lat)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(count > 0, total / count, np.nan).reshape(n_lat, n_lon)
    for _ in range(int(fill_gap_cells)):
        hole = np.isnan(grid)
        if not hole.any():
            break
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)
        grid = np.where(hole, np.where(np.isfinite(left), left, right), grid)
    return (np.linspace(-180.0, 180.0, n_lon + 1),
            np.linspace(-90.0, 90.0, n_lat + 1), grid)


def _colour_limits(f, vmin, vmax, symmetric):
    """1-99 percentile limits of the finite values unless given; symmetric about 0 on request."""
    if vmin is None:
        vmin = float(np.nanpercentile(f, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(f, 99))
    if symmetric:
        vm = max(abs(vmin), abs(vmax))
        vmin, vmax = -vm, vm
    return vmin, vmax


def _finish_map_panel(ax, tc, title, has_cartopy):
    if has_cartopy:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", linewidth=0.3)
        ax.coastlines(linewidth=0.3, color="0.4")
        ax.set_global()
    ax.set_title(title, fontsize=12)
    plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)
    return tc


def plot_raster_panel(ax, lon_edges, lat_edges, grid, title, cmap, has_cartopy,
                      vmin=None, vmax=None, symmetric=False):
    """One pcolormesh panel of a ``raster_mean`` grid."""
    vmin, vmax = _colour_limits(grid, vmin, vmax, symmetric)
    kw = {}
    if has_cartopy:
        import cartopy.crs as ccrs
        kw["transform"] = ccrs.PlateCarree()
    tc = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(grid), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="flat", rasterized=True, **kw)
    return _finish_map_panel(ax, tc, title, has_cartopy)


def plot_panel(ax, tri, field, ocean_mask, title, cmap, has_cartopy,
               vmin=None, vmax=None, symmetric=False):
    """One tripcolor panel; colour limits from the 1-99 percentiles of ocean values."""
    f = np.where(np.asarray(ocean_mask, bool), np.asarray(field, dtype=np.float64), np.nan)
    vmin, vmax = _colour_limits(f, vmin, vmax, symmetric)
    if has_cartopy:
        import cartopy.crs as ccrs
        tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), rasterized=True)
    else:
        tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                          rasterized=True)
    return _finish_map_panel(ax, tc, title, has_cartopy)


def _open_figure(has_cartopy):
    if has_cartopy:
        import cartopy.crs as ccrs
        proj = ccrs.Robinson(central_longitude=200)
        return plt.subplots(2, 3, figsize=(24, 12), subplot_kw={"projection": proj})
    return plt.subplots(2, 3, figsize=(22, 12))


def _section_axis(fig, axes, has_cartopy):
    ax_zm = axes[1, 0]
    if has_cartopy:
        # Remove the cartopy projection for this panel — it's a section
        ax_zm.remove()
        ax_zm = fig.add_subplot(2, 3, 4)
    return ax_zm


def _draw_zonal_section(ax_zm, lat_centers, depth_m, t_zm):
    lat2, z2 = np.meshgrid(lat_centers, depth_m, indexing="ij")
    pcm = ax_zm.pcolormesh(lat2, z2, t_zm, cmap="RdYlBu_r", shading="auto")
    ax_zm.invert_yaxis()
    ax_zm.set_xlabel("Latitude")
    ax_zm.set_ylabel("Depth (m)")
    ax_zm.set_title("Zonal-mean T (C)", fontsize=12)
    plt.colorbar(pcm, ax=ax_zm, shrink=0.7, pad=0.02)


def fesom_wet_columns(temp_levels, nlevels_nod2d, n_layers):
    """``[nod2, n_layers]`` copy of a FESOM ``[nz, nod2]`` level array with the
    padding row dropped and every below-bottom layer set to NaN.

    A node with ``nlevels_nod2D = n`` holds ``n-1`` wet layers (rows ``0..n-2``);
    the store carries ``nz = n_layers + 1`` rows, the last being padding.
    """
    temp_levels = np.asarray(temp_levels)
    if temp_levels.shape[0] < n_layers:
        raise ValueError(f"level array has {temp_levels.shape[0]} rows < {n_layers} layers")
    cols = temp_levels[:n_layers].T.astype(np.float64, copy=True)   # [nod2, n_layers]
    nlev = np.asarray(nlevels_nod2d).astype(np.int64)            # unsigned input must not wrap at n-1
    below = np.arange(n_layers)[None, :] >= (nlev[:, None] - 1)
    cols[below] = np.nan
    return cols


def plot_fesom_store(store_path, mesh_dir, output_dir, res_deg=0.25, fill_gap_cells=3,
                     native=True, point_size=1.2, dpi=200):
    """6-panel PNG from a fesom_jax monthly (``<YYYY>_<MM>``) or daily
    (``day_<YYYY>_<DOY>``) canonical-global ushow zarr store.

    Layout: Surface speed | SSH | SST  //  Zonal-mean T (monthly) or T at
    100 m (daily) | SSS | sea-ice concentration.  Maps are node means on a
    ``res_deg`` lon-lat raster (every node is ocean on a FESOM mesh);
    below-bottom levels (``nlevels_nod2D``) are excluded from the zonal mean.
    """
    import zarr
    try:
        import cartopy  # noqa: F401
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False
    store_path = Path(store_path)
    mesh_dir = Path(mesh_dir)
    g = zarr.open_group(str(store_path), mode="r")
    attrs = dict(g.attrs)
    label = attrs.get("calendar_month") or attrs.get("calendar_date") or store_path.name
    lon = np.asarray(g["lon"][:])
    lat = np.asarray(g["lat"][:])
    n_nod = lon.size
    ocean = np.ones(n_nod, dtype=bool)
    nlev_nod = np.load(mesh_dir / "nlevels_nod2D.npy")
    if nlev_nod.shape != (n_nod,):
        raise ValueError(f"mesh {mesh_dir} has {nlev_nod.shape[0]} nodes, store has {n_nod}")

    def panel(ax, field, title, cmap, **kw):
        if native:
            return plot_native_panel(ax, lon, lat, field, title, cmap,
                                     _has_cartopy, point_size=point_size, **kw)
        lon_e, lat_e, grid = raster_mean(lon, lat, field, res_deg, fill_gap_cells=fill_gap_cells)
        return plot_raster_panel(ax, lon_e, lat_e, grid, title, cmap, _has_cartopy, **kw)

    def surf(name):
        return np.asarray(g[name][0], dtype=np.float64)

    monthly = "temp" in g
    ssh = surf("ssh")
    a_ice = surf("a_ice")
    if monthly:
        temp = np.asarray(g["temp"][0], dtype=np.float64)          # [nz, nod2]
        sst = temp[0]
        sss = np.asarray(g["salt"][0, 0], dtype=np.float64)
        speed = np.hypot(np.asarray(g["u"][0, 0]), np.asarray(g["v"][0, 0]))
        kind = f"monthly mean, {attrs.get('n_samples', '?')} samples"
    else:
        sst = surf("sst")
        sss = surf("sss")
        speed = np.hypot(surf("usurf"), surf("vsurf"))
        temp100 = surf("temp100")
        kind = f"daily mean, {attrs.get('n_samples', '?')} samples"
    print(f"Loading {store_path} ({label}, {kind}, {n_nod} nodes)")
    print(f"  SSH: [{ssh.min():.3f}, {ssh.max():.3f}] m  "
          f"SST: [{sst.min():.1f}, {sst.max():.1f}] C  SSS: [{sss.min():.2f}, {sss.max():.2f}]  "
          f"|U|_sfc max {speed.max():.3f} m/s  a_ice max {a_ice.max():.2f}")
    if not (np.isfinite(ssh).all() and np.isfinite(sst).all()):
        print("  WARNING: non-finite values in ssh/sst")

    fig, axes = _open_figure(_has_cartopy)
    panel(axes[0, 0], speed, "Surface speed [m/s]", "magma",
          vmin=0, vmax=max(0.1, float(np.nanpercentile(speed, 99))))
    panel(axes[0, 1], ssh, "SSH [m]", "RdBu_r", symmetric=True)
    panel(axes[0, 2], sst, "SST [C]", "RdYlBu_r")
    if monthly:
        z_mid = np.load(mesh_dir / "Z.npy")                          # layer mid-depths, negative
        t_cols = fesom_wet_columns(temp, nlev_nod, z_mid.size)
        ax_zm = _section_axis(fig, axes, _has_cartopy)
        lat_c, t_zm = _zonal_mean_T(t_cols, np.deg2rad(lat), ocean, n_bins=72, reduce=np.nanmean)
        _draw_zonal_section(ax_zm, lat_c, -np.asarray(z_mid, dtype=np.float64), t_zm)
    else:
        panel(axes[1, 0], temp100, "T at 100 m [C]", "RdYlBu_r")
    panel(axes[1, 1], sss, "SSS [PSU]", "YlGnBu")
    panel(axes[1, 2], a_ice, "Sea-ice concentration", "Blues_r", vmin=0, vmax=1)
    _how = ("native mesh, one point per node" if native
            else f"{res_deg} deg raster")
    fig.suptitle(f"FESOM2-JAX {mesh_dir.name} — {label} ({kind}; {n_nod} nodes, "
                 f"{_how})",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = Path(output_dir) if output_dir is not None else store_path.parent.parent / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"snapshot_{store_path.name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_snapshot(restart_path, mesh, z_coord, output_dir=None):
    """Generate a 6-panel diagnostic PNG from a restart file."""
    try:
        import cartopy  # noqa: F401
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

    tri = build_ocean_triangulation(lon, lat, mask_np)

    def ocean_field(f):
        return np.where(mask_np, f, np.nan)

    fig, axes = _open_figure(_has_cartopy)

    # Row 0: Surface speed | SSH | SST
    spd_sfc = speed[:, 0]
    plot_panel(axes[0, 0], tri, spd_sfc, mask_np, "Surface speed [m/s]", "magma", _has_cartopy,
               vmin=0, vmax=max(0.1, float(np.nanpercentile(
                   ocean_field(spd_sfc), 99))))
    plot_panel(axes[0, 1], tri, eta, mask_np, "SSH [m]", "RdBu_r", _has_cartopy, symmetric=True)
    plot_panel(axes[0, 2], tri, sst, mask_np, "SST [C]", "RdYlBu_r", _has_cartopy)

    # Row 1: Zonal-mean T section | SSS | Deep T
    ax_zm = _section_axis(fig, axes, _has_cartopy)
    lat_centers, T_zm = _zonal_mean_T(T, mesh.latCell, land_mask, n_bins=72)
    _draw_zonal_section(ax_zm, lat_centers, -z_full, T_zm)

    # SSS
    plot_panel(axes[1, 1], tri, sss, mask_np, "SSS [PSU]", "YlGnBu", _has_cartopy)
    # Deep T
    plot_panel(axes[1, 2], tri, T_deep, mask_np,
               f"T at level {deep_lev} [~{int(-z_full[deep_lev])} m] (C)",
               "RdYlBu_r", _has_cartopy)

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
    p.add_argument("restarts", type=Path, nargs="*",
                   help="Path(s) to MPAS restart_dayXXXXXX.npz")
    p.add_argument("--fesom-store", type=Path, nargs="+", default=None,
                   help="fesom_jax monthly (<YYYY>_<MM>) or daily (day_<YYYY>_<DOY>) zarr store(s)")
    p.add_argument("--mesh-dir", type=Path, default=None,
                   help="fesom_jax mesh directory (nlevels_nod2D.npy, Z.npy); "
                        "required with --fesom-store")
    p.add_argument("--res-deg", type=float, default=0.25,
                   help="lon-lat raster resolution for --fesom-store maps [deg] (default 0.25)")
    p.add_argument("--raster", action="store_false", dest="native", default=True,
                   help=("Bin --fesom-store fields onto a regular lon-lat grid instead of "
                         "drawing the mesh's own nodes. Lossy: it averages nodes together "
                         "where the mesh is finer than the cell and fills empty cells from "
                         "a neighbour where it is coarser. Native is the default."))
    p.add_argument("--point-size", type=float, default=1.2,
                   help="Marker area for native rendering [pt^2] (default 1.2)")
    p.add_argument("--dpi", type=int, default=200,
                   help="Output resolution (default 200; native rendering needs it)")
    p.add_argument("--fill-gap-cells", type=int, default=3,
                   help="fill empty raster cells from up to N neighbours along the row "
                        "(display only; high-latitude cells are narrower than the mesh)")
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

    if args.fesom_store:
        if args.mesh_dir is None:
            p.error("--fesom-store requires --mesh-dir")
        for store in args.fesom_store:
            plot_fesom_store(store, args.mesh_dir, args.output_dir, res_deg=args.res_deg,
                             fill_gap_cells=args.fill_gap_cells, native=args.native,
                             point_size=args.point_size, dpi=args.dpi)
    if not args.restarts:
        if not args.fesom_store:
            p.error("give MPAS restart file(s) or --fesom-store")
        return

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
