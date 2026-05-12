"""Zonal and meridional cross-section plots of T, u, and speed.

Shows vertical structure: thermocline, equatorial undercurrent, ACC depth extent,
and any polar convective activity.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_plot_sections.py <restart.npz>
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

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import reconstruct_cell_velocity
from legoesm.ocean.vertical import create_ocean_z_star


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

    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0,
                                  dz_surface=20.0, dz_deep=500.0)
    z_full = np.asarray(z_coord.z_full_ref)

    ocean_mask = data.get("land_mask", np.ones(len(lon)))
    mask = ocean_mask > 0.5

    T = data["T"]
    S = data["S"]
    u_edge = data["u"]
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    u_east = np.asarray(u_east)
    v_north = np.asarray(v_north)
    speed = np.sqrt(u_east**2 + v_north**2)

    # Helper: extract a latitude or longitude section by finding nearest cells
    def zonal_section(target_lat, lat_band=3.0):
        """Average cells within ±lat_band of target_lat, binned by longitude."""
        in_band = mask & (np.abs(lat - target_lat) < lat_band)
        lon_bins = np.linspace(0, 360, 181)
        lon_c = 0.5 * (lon_bins[:-1] + lon_bins[1:])
        nlev = T.shape[1]
        T_sec = np.full((len(lon_c), nlev), np.nan)
        u_sec = np.full((len(lon_c), nlev), np.nan)
        spd_sec = np.full((len(lon_c), nlev), np.nan)
        for i in range(len(lon_c)):
            in_bin = in_band & (lon >= lon_bins[i]) & (lon < lon_bins[i+1])
            if in_bin.sum() > 0:
                T_sec[i] = T[in_bin].mean(axis=0)
                u_sec[i] = u_east[in_bin].mean(axis=0)
                spd_sec[i] = speed[in_bin].mean(axis=0)
        return lon_c, T_sec, u_sec, spd_sec

    def merid_section(target_lon, lon_band=5.0):
        """Average cells within ±lon_band of target_lon, binned by latitude."""
        # Handle wrapping
        lon_diff = np.abs(lon - target_lon)
        lon_diff = np.minimum(lon_diff, 360 - lon_diff)
        in_band = mask & (lon_diff < lon_band)
        lat_bins = np.linspace(-80, 80, 161)
        lat_c = 0.5 * (lat_bins[:-1] + lat_bins[1:])
        nlev = T.shape[1]
        T_sec = np.full((len(lat_c), nlev), np.nan)
        v_sec = np.full((len(lat_c), nlev), np.nan)
        spd_sec = np.full((len(lat_c), nlev), np.nan)
        for i in range(len(lat_c)):
            in_bin = in_band & (lat >= lat_bins[i]) & (lat < lat_bins[i+1])
            if in_bin.sum() > 0:
                T_sec[i] = T[in_bin].mean(axis=0)
                v_sec[i] = v_north[in_bin].mean(axis=0)
                spd_sec[i] = speed[in_bin].mean(axis=0)
        return lat_c, T_sec, v_sec, spd_sec

    fig, axes = plt.subplots(4, 3, figsize=(22, 20))

    # Row 1: Equatorial zonal section (lat=0°)
    lon_c, T_eq, u_eq, spd_eq = zonal_section(0.0, lat_band=3.0)
    LON, Z = np.meshgrid(lon_c, z_full)

    ax = axes[0, 0]
    pc = ax.pcolormesh(LON, Z, T_eq.T, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(pc, ax=ax, label="°C"); ax.set_title("T at equator (±3°)")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-2000, 0)

    ax = axes[0, 1]
    vm = max(0.1, np.nanpercentile(np.abs(u_eq), 99))
    pc = ax.pcolormesh(LON, Z, u_eq.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("u (eastward) at equator")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-2000, 0)

    ax = axes[0, 2]
    pc = ax.pcolormesh(LON, Z, spd_eq.T, cmap="magma", vmin=0, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("Speed at equator")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-2000, 0)

    # Row 2: Atlantic meridional section (lon=330° = 30°W)
    lat_c, T_atl, v_atl, spd_atl = merid_section(330.0, lon_band=10.0)
    LAT, Z2 = np.meshgrid(lat_c, z_full)

    ax = axes[1, 0]
    pc = ax.pcolormesh(LAT, Z2, T_atl.T, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(pc, ax=ax, label="°C"); ax.set_title("T — Atlantic (30°W ±10°)")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    ax = axes[1, 1]
    vm = max(0.01, np.nanpercentile(np.abs(v_atl), 99))
    pc = ax.pcolormesh(LAT, Z2, v_atl.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("v (northward) — Atlantic")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    ax = axes[1, 2]
    pc = ax.pcolormesh(LAT, Z2, spd_atl.T, cmap="magma", vmin=0, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("Speed — Atlantic")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    # Row 3: Pacific meridional section (lon=180°)
    lat_c, T_pac, v_pac, spd_pac = merid_section(180.0, lon_band=10.0)

    ax = axes[2, 0]
    pc = ax.pcolormesh(LAT, Z2, T_pac.T, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(pc, ax=ax, label="°C"); ax.set_title("T — Pacific (180° ±10°)")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    ax = axes[2, 1]
    vm = max(0.01, np.nanpercentile(np.abs(v_pac), 99))
    pc = ax.pcolormesh(LAT, Z2, v_pac.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("v (northward) — Pacific")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    ax = axes[2, 2]
    pc = ax.pcolormesh(LAT, Z2, spd_pac.T, cmap="magma", vmin=0, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("Speed — Pacific")
    ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    # Row 4: Southern Ocean zonal section (lat=-60°)
    lon_c, T_so, u_so, spd_so = zonal_section(-60.0, lat_band=5.0)

    ax = axes[3, 0]
    pc = ax.pcolormesh(LON, Z, T_so.T, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(pc, ax=ax, label="°C"); ax.set_title("T — Southern Ocean (60°S ±5°)")
    ax.set_xlabel("Longitude [°]"); ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    ax = axes[3, 1]
    vm = max(0.01, np.nanpercentile(np.abs(u_so), 99))
    pc = ax.pcolormesh(LON, Z, u_so.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("u (eastward) — Southern Ocean (ACC)")
    ax.set_xlabel("Longitude [°]"); ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    ax = axes[3, 2]
    pc = ax.pcolormesh(LON, Z, spd_so.T, cmap="magma", vmin=0, shading="auto")
    plt.colorbar(pc, ax=ax, label="m/s"); ax.set_title("Speed — Southern Ocean")
    ax.set_xlabel("Longitude [°]"); ax.set_ylabel("Depth [m]"); ax.set_ylim(-5000, 0)

    fig.suptitle(f"MPAS ETOPO — cross-sections, year {year:.1f}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.restart).parent.parent / "snapshots_nice" / f"sections_year{year:.0f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
