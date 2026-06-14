"""Compute Drake Passage transport time series for MPAS vs lat-lon runs.

Drake Passage: longitudinal section across the gap between South America
and Antarctica, ~60°S-68°S, near 65°W (= 295°E).

Usage:
    python scripts/tmp/_drake_timeseries.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import glob

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_mpas import reconstruct_cell_velocity
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm import constants


# Drake Passage band (matches existing diagnostics)
DRAKE_LAT_MIN = -68.0
DRAKE_LAT_MAX = -55.0
DRAKE_LON = 295.0  # near 65°W
DRAKE_LON_BAND = 5.0  # ±5° around DRAKE_LON


def drake_transport_mpas(restart_path, mesh, z_coord):
    """Compute Drake transport [Sv] from MPAS restart."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    u_edge = d["u"]  # (nEdges, nlev)
    H_bathy = d["H_bathy"]

    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    u_east = np.asarray(u_east)

    lat = np.degrees(np.asarray(mesh.latCell))
    lon = np.degrees(np.asarray(mesh.lonCell))

    # Find cells in Drake band
    in_drake = ((lat >= DRAKE_LAT_MIN) & (lat <= DRAKE_LAT_MAX) &
                (np.abs(((lon - DRAKE_LON + 180) % 360) - 180) < DRAKE_LON_BAND))

    if in_drake.sum() == 0:
        return day, 0.0

    # For each cell in Drake band, depth-integrate u_east, weighted by cell width
    # Approximation: bin by latitude, then integrate
    lat_bins = np.linspace(DRAKE_LAT_MIN, DRAKE_LAT_MAX, 30)
    lat_c = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    dy = (DRAKE_LAT_MAX - DRAKE_LAT_MIN) / (len(lat_bins) - 1)
    dy_m = dy * np.pi / 180.0 * constants.R_earth

    dz_ref = np.asarray(z_coord.dz_ref)
    T_Sv = 0.0
    for i in range(len(lat_c)):
        in_bin = in_drake & (lat >= lat_bins[i]) & (lat < lat_bins[i+1])
        if in_bin.sum() == 0:
            continue
        u_bin = u_east[in_bin].mean(axis=0)  # (nlev,)
        H_bin = H_bathy[in_bin].mean()
        # Mask out levels below seafloor
        z_levels = np.cumsum(dz_ref)
        active = z_levels < H_bin
        depth_int = np.sum(u_bin * dz_ref * active)
        T_Sv += depth_int * dy_m
    return day, T_Sv / 1e6


def drake_transport_latlon(restart_path, grid, z_coord):
    """Compute Drake transport [Sv] from lat-lon restart."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    u = d["u"]  # (n_lat, n_lon+1, nlev)
    H_bathy = d["H_bathy"]
    mask = d["land_mask"] > 0.5

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))

    # Find longitude index nearest 295°E
    lon_idx = int(np.argmin(np.abs(lon_deg - DRAKE_LON)))
    # u-face is between cells lon_idx-1 and lon_idx, at lon[lon_idx]
    # Use the face at this longitude

    # Latitude range
    in_lat = (lat_deg >= DRAKE_LAT_MIN) & (lat_deg <= DRAKE_LAT_MAX)
    lat_idx = np.where(in_lat)[0]

    dy = float(grid.dy)  # latitude spacing in meters (constant)
    dz_ref = np.asarray(z_coord.dz_ref)

    T_Sv = 0.0
    for i in lat_idx:
        if not mask[i, lon_idx]:
            continue
        u_col = u[i, lon_idx, :]  # (nlev,)
        H_col = H_bathy[i, lon_idx]
        z_levels = np.cumsum(dz_ref)
        active = z_levels < H_col
        depth_int = np.sum(u_col * dz_ref * active)
        T_Sv += depth_int * dy
    return day, T_Sv / 1e6


def main():
    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0,
                                  dz_surface=20.0, dz_deep=500.0)

    # MPAS
    print("=== MPAS Drake transport ===")
    mesh = create_voronoi_mesh(subdivision_level=5)
    mpas_dir = Path("results/ocean/comparison_mpas_v_latlon/mpas_e4_fp64/restarts")
    mpas_files = sorted(mpas_dir.glob("restart_day*.npz"))
    mpas_data = []
    for f in mpas_files:
        day, T = drake_transport_mpas(f, mesh, z_coord)
        mpas_data.append((day, T))
        if len(mpas_data) % 10 == 1:
            print(f"  day {day:7.1f}: T = {T:+7.2f} Sv")
    mpas_arr = np.array(mpas_data)

    # Lat-lon
    print("\n=== Lat-lon Drake transport ===")
    grid = create_latlon_grid(180, 360)
    ll_dir = Path("results/ocean/comparison_mpas_v_latlon/latlon_e4_fp64/restarts")
    ll_files = sorted(ll_dir.glob("restart_day*.npz"))
    ll_data = []
    for f in ll_files:
        day, T = drake_transport_latlon(f, grid, z_coord)
        ll_data.append((day, T))
        if len(ll_data) % 10 == 1:
            print(f"  day {day:7.1f}: T = {T:+7.2f} Sv")
    ll_arr = np.array(ll_data)

    # K_zeta_bih sensitivity experiment (if available)
    kzb_dir = Path("results/ocean/comparison_mpas_v_latlon/mpas_e5_kzb1e12/restarts")
    kzb_arr = None
    if kzb_dir.exists():
        print("\n=== MPAS K_zeta_bih=1e12 ===")
        kzb_files = sorted(kzb_dir.glob("restart_day*.npz"))
        kzb_data = []
        for f in kzb_files:
            day, T = drake_transport_mpas(f, mesh, z_coord)
            kzb_data.append((day, T))
            if len(kzb_data) % 10 == 1:
                print(f"  day {day:7.1f}: T = {T:+7.2f} Sv")
        if kzb_data:
            kzb_arr = np.array(kzb_data)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(mpas_arr[:, 0]/365.25, mpas_arr[:, 1], "o-",
            color="C0", label="MPAS (K_zeta_bih=1e14)", ms=3, lw=1.5)
    ax.plot(ll_arr[:, 0]/365.25, ll_arr[:, 1], "s-",
            color="C1", label="Lat-lon 1°", ms=3, lw=1.5)
    if kzb_arr is not None:
        ax.plot(kzb_arr[:, 0]/365.25, kzb_arr[:, 1], "d-",
                color="C3", label="MPAS (K_zeta_bih=1e12)", ms=3, lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(150, color="C2", lw=0.8, ls="--", alpha=0.6,
               label="Observed ACC (~150 Sv)")
    ax.set_xlabel("Sim year")
    ax.set_ylabel("Drake transport (Sv)  [eastward = +]")
    ax.set_title("Drake Passage transport — MPAS vs lat-lon")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    out = Path("results/ocean/comparison_mpas_v_latlon/drake_transport_comparison.png")
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")

    # Save data
    save_dict = dict(
        mpas_day=mpas_arr[:, 0], mpas_T_Sv=mpas_arr[:, 1],
        latlon_day=ll_arr[:, 0], latlon_T_Sv=ll_arr[:, 1],
    )
    if kzb_arr is not None:
        save_dict["kzb_day"] = kzb_arr[:, 0]
        save_dict["kzb_T_Sv"] = kzb_arr[:, 1]
    np.savez(out.with_suffix(".npz"), **save_dict)
    print(f"Saved data: {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
