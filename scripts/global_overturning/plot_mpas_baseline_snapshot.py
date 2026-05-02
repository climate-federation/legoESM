#!/usr/bin/env python
"""Render visual snapshots of the MPAS global overturning baseline run.

Loads a single restart NPZ from the MPAS baseline and produces a
multi-panel PNG covering surface dynamics (SSH, SST, surface speed),
sub-surface temperature structure (mid-depth, bottom), and a zonal-
mean temperature section.  Used to visually verify that the year-1
spinup is qualitatively correct before launching a longer run.

Usage:
    python scripts/global_overturning/plot_mpas_baseline_snapshot.py \
        [restart_day000365.npz]

Default: ``results/ocean/global_overturning_mpas_baseline/restart_day000365.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocean_test_matrix.regridding import _bin_to_latlon  # noqa: E402


def _reconstruct_cell_speed(u_edges: np.ndarray, mesh) -> np.ndarray:
    """Cell-centered |U| from edge-normal velocities.

    Uses the Perot reconstruction available in ``init_mpas``.
    Returns ``|U_cell|`` shape (nCells, nlev).
    """
    import jax.numpy as jnp
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edges), mesh)
    speed = np.sqrt(np.asarray(u_east) ** 2 + np.asarray(v_north) ** 2)
    return speed


def _zonal_mean_T(T_3d: np.ndarray, lat_cell_rad: np.ndarray,
                  land_mask: np.ndarray, n_bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Bin T by latitude and average over ocean cells in each bin.

    Returns
    -------
    bin_centers_deg : (n_bins,)
    T_zm : (n_bins, nlev)  — NaN where bin has no ocean cells.
    """
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


def main():
    if len(sys.argv) > 1:
        restart_path = Path(sys.argv[1])
    else:
        restart_path = Path(
            "results/ocean/global_overturning_mpas_baseline/"
            "restart_day000365.npz")

    if not restart_path.exists():
        print(f"Restart not found: {restart_path}", file=sys.stderr)
        sys.exit(1)

    npz = np.load(restart_path, allow_pickle=False)
    day = float(npz["time_days"])
    print(f"Loading {restart_path.name} (sim day {day:.0f}, year {day/365:.2f})")
    sub_level = int(npz.get("mpas_subdivision_level", 4))

    # Rebuild the mesh deterministically.
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    mesh = create_voronoi_mesh(sub_level)
    z_coord = create_ocean_z_star(
        n_levels=20, H_max=4000.0, dz_surface=10.0, dz_deep=500.0)

    eta = np.asarray(npz["eta"])
    T = np.asarray(npz["T"])         # (nCells, nlev)
    u = np.asarray(npz["u"])         # (nEdges, nlev)
    land_mask = np.asarray(npz["land_mask"])

    nlev = T.shape[1]
    print(f"  nCells={T.shape[0]}, nEdges={u.shape[0]}, nlev={nlev}")
    print(f"  eta range: [{eta.min():.3f}, {eta.max():.3f}] m")
    print(f"  T sfc range: [{T[:,0].min():.2f}, {T[:,0].max():.2f}] °C")
    print(f"  T bot range: [{T[:,-1].min():.2f}, {T[:,-1].max():.2f}] °C")

    # Cell-centered speed
    speed = _reconstruct_cell_speed(u, mesh)
    speed_sfc = speed[:, 0]
    print(f"  surface |U| range: [{speed_sfc.min():.3f}, "
          f"{speed_sfc.max():.3f}] m/s")

    # Choose mid-depth level (~half the column)
    mid_lev = nlev // 2
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    print(f"  mid level k={mid_lev} at depth {-z_full[mid_lev]:.0f} m, "
          f"bottom k={nlev-1} at depth {-z_full[-1]:.0f} m")

    # ---- Figure: 6-panel ----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.20)

    # Voronoi maps share lon/lat extent; panels use _plot_voronoi_field
    panels = [
        ("eta", eta,        "SSH (m)",                        "RdBu_r",  True),
        ("SST", T[:, 0],    "SST [k=0, ~5 m] (°C)",           "RdYlBu_r", False),
        ("|U_sfc|", speed_sfc, "Surface speed (m/s)",           "magma",   False),
        ("T_mid", T[:, mid_lev],
            f"T [k={mid_lev}, ~{int(-z_full[mid_lev])} m] (°C)", "RdYlBu_r", False),
        ("T_bot", T[:, -1], f"T [k={nlev-1}, ~{int(-z_full[-1])} m] (°C)",
            "RdYlBu_r", False),
        # Last panel is zonal-mean T section, drawn separately
    ]

    # Bin source cells to a regular lat-lon grid for visualization.
    # ico4 has ~4° spacing; binning to 1° (181×360) gives a denser
    # nearest-neighbor render that looks natural.
    lon_deg = np.asarray(mesh.lonCell) * 180.0 / np.pi
    # _bin_to_latlon expects [-180, 180] convention; lonCell may be in [0, 2π]
    lon_deg = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)
    lat_deg = np.asarray(mesh.latCell) * 180.0 / np.pi
    n_lat_bin, n_lon_bin = 181, 360
    bin_target_lat = np.linspace(-90.0, 90.0, n_lat_bin)
    bin_target_lon = np.linspace(-180.0, 180.0, n_lon_bin)
    is_ocean = land_mask > 0.5

    # Pre-bin land mask so we can grey out land in the rendered image.
    land_mask_bin = _bin_to_latlon(
        land_mask.astype(np.float64), lon_deg, lat_deg,
        target_lat=bin_target_lat, target_lon=bin_target_lon)
    land_pixel = ~(land_mask_bin > 0.5)

    for i, (key, vals, title, cmap, diverging) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ocean_vals = vals[is_ocean]
        if diverging:
            vmax = float(np.nanmax(np.abs(ocean_vals)))
            vmin = -vmax
        else:
            vmin = float(np.nanmin(ocean_vals))
            vmax = float(np.nanmax(ocean_vals))
        # Bin only over ocean source points (KDTree masks out land).
        binned = _bin_to_latlon(
            vals, lon_deg, lat_deg,
            target_lat=bin_target_lat, target_lon=bin_target_lon,
            ocean_mask=is_ocean)
        # Grey out land pixels in the destination image.
        binned_masked = np.ma.masked_where(land_pixel, binned)
        im = ax.imshow(binned_masked, origin="lower",
                       extent=[-180, 180, -90, 90],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")
        # Light grey wherever masked (land)
        ax.set_facecolor("#d9d9d9")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["180°W", "90°W", "0°", "90°E", "180°E"])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_yticklabels(["90°S", "45°S", "0°", "45°N", "90°N"])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    # Last panel: zonal-mean T section
    ax = fig.add_subplot(gs[1, 2])
    lat_centers, T_zm = _zonal_mean_T(T, mesh.latCell, land_mask)
    depth = -z_full  # positive m
    LAT, Z = np.meshgrid(lat_centers, depth, indexing="ij")
    pcm = ax.pcolormesh(LAT, Z, T_zm, cmap="RdYlBu_r", shading="auto")
    ax.invert_yaxis()
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Zonal-mean T (°C)")
    plt.colorbar(pcm, ax=ax, fraction=0.04)

    fig.suptitle(
        f"MPAS global overturning baseline — sim day {day:.0f} "
        f"(year {day/365:.2f}), ico{sub_level} ({T.shape[0]} cells)",
        y=0.99, fontsize=12)

    out = restart_path.parent / f"snapshot_day{int(round(day)):06d}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
