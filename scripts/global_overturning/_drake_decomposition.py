"""Drake transport decomposition: barotropic (depth-mean) vs baroclinic (shear).

Decomposition:
  T_total = T_BT (barotropic transport from depth-mean)
  T_upper = transport in upper 1000m (surface-intensified ACC)
  T_lower = transport below 1000m (deep, mostly barotropic)
  T_total = T_upper + T_lower

A surface-intensified jet has T_upper >> T_lower (baroclinic-like).
A pure barotropic flow has T_upper / T_lower ~ depth_upper / depth_lower.

Usage:
    python scripts/global_overturning/_drake_decomposition.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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


DRAKE_LAT_MIN = -68.0
DRAKE_LAT_MAX = -55.0
DRAKE_LON = 295.0
DRAKE_LON_BAND = 5.0
SPLIT_DEPTH = 1000.0


def decompose_mpas(restart_path, mesh, z_coord):
    """Decompose Drake transport into upper/lower for MPAS."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    u_edge = d["u"]
    H_bathy = d["H_bathy"]

    u_east, _ = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    u_east = np.asarray(u_east)

    lat = np.degrees(np.asarray(mesh.latCell))
    lon = np.degrees(np.asarray(mesh.lonCell))

    in_drake = ((lat >= DRAKE_LAT_MIN) & (lat <= DRAKE_LAT_MAX) &
                (np.abs(((lon - DRAKE_LON + 180) % 360) - 180) < DRAKE_LON_BAND))

    if in_drake.sum() == 0:
        return day, 0.0, 0.0, 0.0

    lat_bins = np.linspace(DRAKE_LAT_MIN, DRAKE_LAT_MAX, 30)
    dy = (DRAKE_LAT_MAX - DRAKE_LAT_MIN) / (len(lat_bins) - 1)
    dy_m = dy * np.pi / 180.0 * constants.R_earth

    dz_ref = np.asarray(z_coord.dz_ref)
    z_levels = np.cumsum(dz_ref)
    upper_mask = z_levels <= SPLIT_DEPTH

    T_upper = 0.0
    T_lower = 0.0
    for i in range(len(lat_bins) - 1):
        in_bin = in_drake & (lat >= lat_bins[i]) & (lat < lat_bins[i+1])
        if in_bin.sum() == 0:
            continue
        u_bin = u_east[in_bin].mean(axis=0)
        H_bin = H_bathy[in_bin].mean()
        active = z_levels < H_bin
        u_active = u_bin * active

        T_upper += np.sum(u_active[upper_mask] * dz_ref[upper_mask]) * dy_m
        T_lower += np.sum(u_active[~upper_mask] * dz_ref[~upper_mask]) * dy_m

    T_total = T_upper + T_lower
    return day, T_total / 1e6, T_upper / 1e6, T_lower / 1e6


def decompose_latlon(restart_path, grid, z_coord):
    """Decompose Drake transport into upper/lower for lat-lon."""
    d = np.load(restart_path)
    day = float(d["time_days"])
    u = d["u"]
    H_bathy = d["H_bathy"]
    mask = d["land_mask"] > 0.5

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))
    lon_idx = int(np.argmin(np.abs(lon_deg - DRAKE_LON)))

    in_lat = (lat_deg >= DRAKE_LAT_MIN) & (lat_deg <= DRAKE_LAT_MAX)
    lat_idx = np.where(in_lat)[0]

    dy = float(grid.dy)
    dz_ref = np.asarray(z_coord.dz_ref)
    z_levels = np.cumsum(dz_ref)
    upper_mask = z_levels <= SPLIT_DEPTH

    T_upper = 0.0
    T_lower = 0.0
    for i in lat_idx:
        if not mask[i, lon_idx]:
            continue
        u_col = u[i, lon_idx, :]
        H_col = H_bathy[i, lon_idx]
        active = z_levels < H_col
        u_active = u_col * active

        T_upper += np.sum(u_active[upper_mask] * dz_ref[upper_mask]) * dy
        T_lower += np.sum(u_active[~upper_mask] * dz_ref[~upper_mask]) * dy

    T_total = T_upper + T_lower
    return day, T_total / 1e6, T_upper / 1e6, T_lower / 1e6


def main():
    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0,
                                  dz_surface=20.0, dz_deep=500.0)

    print("=== MPAS ===")
    mesh = create_voronoi_mesh(subdivision_level=5)
    mpas_dir = Path("results/ocean/comparison_mpas_v_latlon/mpas_e4_fp64/restarts")
    mpas = []
    for f in sorted(mpas_dir.glob("restart_day*.npz")):
        mpas.append(decompose_mpas(f, mesh, z_coord))
    mpas = np.array(mpas)

    print("=== Lat-lon ===")
    grid = create_latlon_grid(180, 360)
    ll_dir = Path("results/ocean/comparison_mpas_v_latlon/latlon_e4_fp64/restarts")
    ll = []
    for f in sorted(ll_dir.glob("restart_day*.npz")):
        ll.append(decompose_latlon(f, grid, z_coord))
    ll = np.array(ll)

    # K_zeta_bih sensitivity (if available)
    kzb_dir = Path("results/ocean/comparison_mpas_v_latlon/mpas_e5_kzb1e12/restarts")
    kzb = None
    if kzb_dir.exists():
        print("\n=== MPAS K_zeta_bih=1e12 ===")
        kzb_list = []
        for f in sorted(kzb_dir.glob("restart_day*.npz")):
            kzb_list.append(decompose_mpas(f, mesh, z_coord))
        if kzb_list:
            kzb = np.array(kzb_list)

    n_panels = 3 if kzb is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4 * n_panels), sharex=True)

    ax = axes[0]
    ax.plot(mpas[:, 0]/365.25, mpas[:, 1], "o-", color="C0",
            label="total", ms=3)
    ax.plot(mpas[:, 0]/365.25, mpas[:, 2], "v--", color="C0",
            label=f"upper (<{SPLIT_DEPTH:.0f}m)", ms=2, alpha=0.6)
    ax.plot(mpas[:, 0]/365.25, mpas[:, 3], "^--", color="C0",
            label=f"lower (>{SPLIT_DEPTH:.0f}m)", ms=2, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Drake transport (Sv)")
    ax.set_title("MPAS (K_zeta_bih=1e14)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(ll[:, 0]/365.25, ll[:, 1], "s-", color="C1",
            label="total", ms=3)
    ax.plot(ll[:, 0]/365.25, ll[:, 2], "v--", color="C1",
            label=f"upper (<{SPLIT_DEPTH:.0f}m)", ms=2, alpha=0.6)
    ax.plot(ll[:, 0]/365.25, ll[:, 3], "^--", color="C1",
            label=f"lower (>{SPLIT_DEPTH:.0f}m)", ms=2, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Drake transport (Sv)")
    ax.set_title("Lat-lon 1°")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    if kzb is not None:
        ax = axes[2]
        ax.plot(kzb[:, 0]/365.25, kzb[:, 1], "d-", color="C3",
                label="total", ms=3)
        ax.plot(kzb[:, 0]/365.25, kzb[:, 2], "v--", color="C3",
                label=f"upper (<{SPLIT_DEPTH:.0f}m)", ms=2, alpha=0.6)
        ax.plot(kzb[:, 0]/365.25, kzb[:, 3], "^--", color="C3",
                label=f"lower (>{SPLIT_DEPTH:.0f}m)", ms=2, alpha=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Drake transport (Sv)")
        ax.set_title("MPAS (K_zeta_bih=1e12)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Sim year")
    plt.tight_layout()
    out = Path("results/ocean/comparison_mpas_v_latlon/drake_decomposition.png")
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")

    # Summary stats
    print(f"\n=== Mean transport (year > 1) ===")
    mask_m = mpas[:, 0] > 365
    mask_l = ll[:, 0] > 365
    if mask_m.sum() > 0:
        print(f"MPAS   total={mpas[mask_m, 1].mean():+6.2f}  "
              f"upper={mpas[mask_m, 2].mean():+6.2f}  "
              f"lower={mpas[mask_m, 3].mean():+6.2f}  Sv")
        print(f"       upper/total = {mpas[mask_m, 2].mean()/mpas[mask_m, 1].mean()*100:.1f}%")
    if mask_l.sum() > 0:
        print(f"LatLon total={ll[mask_l, 1].mean():+6.2f}  "
              f"upper={ll[mask_l, 2].mean():+6.2f}  "
              f"lower={ll[mask_l, 3].mean():+6.2f}  Sv")
        print(f"       upper/total = {ll[mask_l, 2].mean()/ll[mask_l, 1].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
