"""Nice plots for MPAS MBR2 20-year flat-bottom run.

Surface maps (tripcolor + cartopy) and zonal-mean section plots of
T, S, density, zonal velocity, and meridional overturning.

Usage:
    python scripts/tmp/_plot_mpas_sections.py <restart_file>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm import constants


# Grid and vertical config (must match run)
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0


def load_data(restart_path):
    """Load restart and reconstruct grid + vertical coordinate."""
    d = np.load(restart_path)
    subdivision = int(d["subdivision"])
    mesh = create_voronoi_mesh(subdivision)
    z_coord = create_ocean_z_star(n_levels=N_LEVELS, H_max=H_MAX,
                                   dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)

    # Cell centers in degrees
    lat_c = np.degrees(np.asarray(mesh.latCell))
    lon_c = np.degrees(np.asarray(mesh.lonCell))
    # Shift to 0-360
    lon_c = lon_c % 360

    # Vertical midpoints
    z_face = np.asarray(z_coord.z_half_ref)  # (nlev+1,) interfaces
    z_mid = np.asarray(z_coord.z_full_ref)   # (nlev,) midpoints
    dz = np.asarray(z_coord.dz_ref)          # (nlev,)

    mask = np.asarray(d["land_mask"]) > 0.5
    H_bathy = np.asarray(d["H_bathy"])

    return d, mesh, lat_c, lon_c, z_mid, dz, z_face, mask, H_bathy


def cell_to_edge_u(mesh, u_edge, mask_cell):
    """Project edge-normal velocities to cell-center zonal velocity.

    For each cell, average the edge-normal velocities projected onto
    the zonal direction.
    """
    n_cells = len(mask_cell)
    nlev = u_edge.shape[1]
    cells_on_edge = np.asarray(mesh.cellsOnEdge)  # (2, n_edges)
    n_edges = cells_on_edge.shape[1]

    # For Voronoi, edge normal points from cell0 to cell1
    lat_c = np.asarray(mesh.latCell)
    lon_c = np.asarray(mesh.lonCell)

    u_zonal = np.zeros((n_cells, nlev))
    count = np.zeros(n_cells)

    for e in range(n_edges):
        c0, c1 = cells_on_edge[0, e], cells_on_edge[1, e]
        if c0 < 0 or c1 < 0 or c0 >= n_cells or c1 >= n_cells:
            continue
        if not mask_cell[c0] and not mask_cell[c1]:
            continue

        # Edge normal direction (from c0 to c1)
        dlat = lat_c[c1] - lat_c[c0]
        dlon = (lon_c[c1] - lon_c[c0])
        # Wrap dlon
        dlon = dlon - 2*np.pi * np.round(dlon / (2*np.pi))
        cos_lat = np.cos(0.5*(lat_c[c0] + lat_c[c1]))
        dx = dlon * cos_lat
        dy = dlat
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-12:
            continue
        # Unit normal
        nx = dx / mag
        # Zonal projection of edge-normal velocity
        u_proj = u_edge[e, :] * nx

        if mask_cell[c0]:
            u_zonal[c0] += u_proj
            count[c0] += 1
        if mask_cell[c1]:
            u_zonal[c1] += u_proj
            count[c1] += 1

    count = np.maximum(count, 1)
    u_zonal = u_zonal / count[:, None]
    return u_zonal


def zonal_mean_section(lat_c, field, mask, lat_bins, dz=None):
    """Compute zonal mean of a (n_cells, nlev) field in latitude bins."""
    nlev = field.shape[1]
    n_bins = len(lat_bins) - 1
    result = np.full((n_bins, nlev), np.nan)

    for i in range(n_bins):
        idx = np.where((lat_c >= lat_bins[i]) & (lat_c < lat_bins[i+1]) & mask)[0]
        if len(idx) > 0:
            result[i, :] = np.mean(field[idx, :], axis=0)

    return result


def compute_density(T, S):
    """Linear EOS for plotting via the model's canonical helper.

    Imports :func:`legoesm.ocean.eos.linear_eos` so the plotter cannot
    drift from the production EOS (CLAUDE.md: plotters NOT exempt from
    the "no re-derivation" rule).
    """
    from legoesm.ocean.eos import linear_eos
    return np.asarray(linear_eos(T, S))


def main():
    if len(sys.argv) < 2:
        print("Usage: python _plot_mpas_sections.py <restart_file>")
        sys.exit(1)

    restart_path = sys.argv[1]
    d, mesh, lat_c, lon_c, z_mid, dz, z_face, mask, H_bathy = load_data(restart_path)
    day = float(d["time_days"])

    T = np.asarray(d["T"])
    S = np.asarray(d["S"])
    eta = np.asarray(d["eta"])
    u_edge = np.asarray(d["u"])

    print(f"Day {day:.0f}: {mask.sum()} ocean cells")
    print(f"  T range: {T[mask].min():.2f} to {T[mask].max():.2f} °C")
    print(f"  S range: {S[mask].min():.4f} to {S[mask].max():.4f} PSU")
    print(f"  eta range: {eta[mask].min():.3f} to {eta[mask].max():.3f} m")

    # Compute cell-center zonal velocity
    print("  Computing cell-center zonal velocity...")
    u_zonal = cell_to_edge_u(mesh, u_edge, mask)

    # Compute density
    rho = compute_density(T, S)

    # Latitude bins for zonal mean
    lat_bins = np.linspace(-80, 80, 81)  # 2-degree bins
    lat_mid = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    # Zonal means
    print("  Computing zonal means...")
    T_zm = zonal_mean_section(lat_c, T, mask, lat_bins)
    S_zm = zonal_mean_section(lat_c, S, mask, lat_bins)
    rho_zm = zonal_mean_section(lat_c, rho, mask, lat_bins)
    u_zm = zonal_mean_section(lat_c, u_zonal, mask, lat_bins)

    # --- Figure 1: Surface maps ---
    print("  Making surface maps...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10),
                                subplot_kw={"projection": ccrs.Robinson()} if HAS_CARTOPY else {})

    # Triangulation for tripcolor
    tri = mtri.Triangulation(lon_c, lat_c)
    # Mask triangles that span the dateline or include land
    triangles = tri.triangles
    lon_verts = lon_c[triangles]
    bad = np.any(np.abs(np.diff(lon_verts, axis=1)) > 180, axis=1)
    # Also mask triangles with any land vertex
    land_verts = ~mask[triangles]
    bad |= np.any(land_verts, axis=1)
    tri.set_mask(bad)

    fields = [
        (T[:, 0], "SST [°C]", "RdYlBu_r", None),
        (S[:, 0], "SSS [PSU]", "viridis", None),
        (eta, "SSH [m]", "RdBu_r", (-1.5, 1.5)),
        (np.sqrt(u_zonal[:, 0]**2) * mask, "Surface |u| [m/s]", "magma", (0, 0.5)),
    ]

    for ax, (field, title, cmap, vlim) in zip(axes1.flat, fields):
        kwargs = dict(cmap=cmap, shading="flat")
        if vlim:
            kwargs["vmin"], kwargs["vmax"] = vlim
        if HAS_CARTOPY:
            ax.set_global()
            kwargs["transform"] = ccrs.PlateCarree()
            tc = ax.tripcolor(tri, field, **kwargs)
            plt.colorbar(tc, ax=ax, shrink=0.7, label=title)
        else:
            tc = ax.tripcolor(tri, field, **kwargs)
            plt.colorbar(tc, ax=ax, label=title)
        ax.set_title(title)

    fig1.suptitle(f"MPAS MBR2 flat-bottom — Day {day:.0f} ({day/365:.1f} yr)", fontsize=14)
    fig1.tight_layout()
    outdir = Path(restart_path).parent.parent / "snapshots"
    out1 = outdir / f"surface_maps_day{int(day):06d}.png"
    fig1.savefig(out1, dpi=150)
    print(f"  Saved: {out1}")
    plt.close(fig1)

    # --- Figure 2: Zonal-mean sections ---
    print("  Making zonal-mean sections...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

    sections = [
        (T_zm, "Zonal-mean T [°C]", "RdYlBu_r", None),
        (S_zm, "Zonal-mean S [PSU]", "viridis", None),
        (rho_zm - 1000, "Zonal-mean σ₀ [kg/m³]", "RdYlBu_r", None),
        (u_zm, "Zonal-mean u [m/s]", "RdBu_r", (-0.15, 0.15)),
    ]

    for ax, (field, title, cmap, vlim) in zip(axes2.flat, sections):
        kwargs = dict(cmap=cmap)
        if vlim:
            kwargs["vmin"], kwargs["vmax"] = vlim
        cf = ax.pcolormesh(lat_mid, z_mid, field.T, shading="nearest", **kwargs)
        plt.colorbar(cf, ax=ax, label=title)
        ax.set_xlabel("Latitude [°]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(title)
        ax.set_ylim(z_mid[-1], 0)
        ax.set_xlim(-80, 80)

    fig2.suptitle(f"MPAS MBR2 flat-bottom — Day {day:.0f} ({day/365:.1f} yr)", fontsize=14)
    fig2.tight_layout()
    out2 = outdir / f"zonal_sections_day{int(day):06d}.png"
    fig2.savefig(out2, dpi=150)
    print(f"  Saved: {out2}")
    plt.close(fig2)

    # --- Figure 3: Drake Passage section (longitude ~290°E = 70°W) ---
    print("  Making Drake Passage section...")
    drake_lon = 290.0
    drake_width = 5.0  # degrees
    drake_idx = np.where((np.abs(lon_c - drake_lon) < drake_width) & mask)[0]
    drake_lat = lat_c[drake_idx]
    sort_idx = np.argsort(drake_lat)
    drake_idx = drake_idx[sort_idx]
    drake_lat = drake_lat[sort_idx]

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

    drake_fields = [
        (T[drake_idx, :], "T [°C]", "RdYlBu_r", None),
        (rho[drake_idx, :] - 1000, "σ₀ [kg/m³]", "RdYlBu_r", None),
        (u_zonal[drake_idx, :], "u [m/s]", "RdBu_r", (-0.3, 0.3)),
    ]

    for ax, (field, title, cmap, vlim) in zip(axes3, drake_fields):
        kwargs = dict(cmap=cmap, shading="nearest")
        if vlim:
            kwargs["vmin"], kwargs["vmax"] = vlim
        cf = ax.pcolormesh(drake_lat, z_mid, field.T, **kwargs)
        plt.colorbar(cf, ax=ax, label=title)
        ax.set_xlabel("Latitude [°]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"Drake Passage ({drake_lon-drake_width}°-{drake_lon+drake_width}°E)\n{title}")
        ax.set_ylim(z_mid[-1], 0)

    fig3.suptitle(f"MPAS MBR2 — Drake section, Day {day:.0f} ({day/365:.1f} yr)", fontsize=14)
    fig3.tight_layout()
    out3 = outdir / f"drake_section_day{int(day):06d}.png"
    fig3.savefig(out3, dpi=150)
    print(f"  Saved: {out3}")
    plt.close(fig3)

    # --- Figure 4: Atlantic section (longitude ~330°E = 30°W) ---
    print("  Making Atlantic section...")
    atl_lon = 330.0
    atl_width = 5.0
    atl_idx = np.where((np.abs(lon_c - atl_lon) < atl_width) & mask)[0]
    atl_lat = lat_c[atl_idx]
    sort_idx = np.argsort(atl_lat)
    atl_idx = atl_idx[sort_idx]
    atl_lat = atl_lat[sort_idx]

    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))

    atl_fields = [
        (T[atl_idx, :], "T [°C]", "RdYlBu_r", None),
        (rho[atl_idx, :] - 1000, "σ₀ [kg/m³]", "RdYlBu_r", None),
        (u_zonal[atl_idx, :], "u [m/s]", "RdBu_r", (-0.15, 0.15)),
    ]

    for ax, (field, title, cmap, vlim) in zip(axes4, atl_fields):
        kwargs = dict(cmap=cmap, shading="nearest")
        if vlim:
            kwargs["vmin"], kwargs["vmax"] = vlim
        cf = ax.pcolormesh(atl_lat, z_mid, field.T, **kwargs)
        plt.colorbar(cf, ax=ax, label=title)
        ax.set_xlabel("Latitude [°]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"Atlantic ({atl_lon-atl_width}°-{atl_lon+atl_width}°E)\n{title}")
        ax.set_ylim(z_mid[-1], 0)

    fig4.suptitle(f"MPAS MBR2 — Atlantic section, Day {day:.0f} ({day/365:.1f} yr)", fontsize=14)
    fig4.tight_layout()
    out4 = outdir / f"atlantic_section_day{int(day):06d}.png"
    fig4.savefig(out4, dpi=150)
    print(f"  Saved: {out4}")
    plt.close(fig4)

    # --- Figure 5: Time evolution (snapshots at different years) ---
    print("  Making time evolution of zonal-mean T...")
    restart_dir = Path(restart_path).parent
    years = [1, 5, 10, 20]
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))

    for ax, yr in zip(axes5.flat, years):
        target_day = yr * 365
        fname = restart_dir / f"restart_day{target_day:06d}.npz"
        if not fname.exists():
            # Find closest
            candidates = sorted(restart_dir.glob("restart_day*.npz"))
            best = min(candidates, key=lambda f: abs(int(f.stem.replace("restart_day","")) - target_day))
            fname = best
        dd = np.load(fname)
        actual_day = float(dd["time_days"])
        T_yr = np.asarray(dd["T"])
        T_zm_yr = zonal_mean_section(lat_c, T_yr, mask, lat_bins)
        cf = ax.pcolormesh(lat_mid, z_mid, T_zm_yr.T, cmap="RdYlBu_r",
                          shading="nearest", vmin=0, vmax=25)
        plt.colorbar(cf, ax=ax, label="T [°C]")
        ax.set_title(f"Year {actual_day/365:.1f}")
        ax.set_xlabel("Latitude [°]")
        ax.set_ylabel("Depth [m]")
        ax.set_ylim(z_mid[-1], 0)
        ax.set_xlim(-80, 80)

    fig5.suptitle("MPAS MBR2 — Zonal-mean T evolution", fontsize=14)
    fig5.tight_layout()
    out5 = outdir / "T_evolution_zonal_mean.png"
    fig5.savefig(out5, dpi=150)
    print(f"  Saved: {out5}")
    plt.close(fig5)

    print("\nDone!")


if __name__ == "__main__":
    main()
