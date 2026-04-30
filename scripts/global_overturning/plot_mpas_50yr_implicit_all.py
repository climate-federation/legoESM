#!/usr/bin/env python
"""MPAS analogue of plot_50yr_implicit_progress.py + plot_50yr_implicit_drake.py.

Produces the same seven diagnostic plots that the lat-lon 50-yr run has,
on the MPAS 50-yr implicit-CN run, by binning MPAS cell-centered fields
to a regular lat-lon grid for visualisation and computing Drake-band
diagnostics by geographic latitude selection (-77.5° to -57.5° — matching
lat-lon J_DRAKE = arange(2, 7) on a 36-row 5° grid).

Plots produced (in the run's output directory):

  Progress (analog of plot_50yr_implicit_progress.py):
    - timeseries_progress.png
    - snapshots_progress.png
    - T_zonal_mean_progress.png

  Drake (analog of plot_50yr_implicit_drake.py):
    - drake_transport_timeseries.png
    - drake_zonal_u_evolution.png
    - drake_T_evolution.png
    - drake_u_profiles.png

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/plot_mpas_50yr_implicit_all.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from ocean_test_matrix.regridding import _bin_to_latlon


RUN_DIR = Path("results/ocean/global_overturning_mpas_50yr_implicit_dt600")
OUTPUT_DIR = RUN_DIR

# Drake-band latitudes — match lat-lon J_DRAKE on the 36×72 (5°) grid:
# lat row centres at -87.5, -82.5, -77.5, ..., +87.5 in 5° steps.
# J_DRAKE = arange(2, 7) → lat in [-77.5, -57.5] inclusive.
DRAKE_LAT_MIN_DEG = -80.0   # cell selection bound (loose; lat-lon includes -77.5° row)
DRAKE_LAT_MAX_DEG = -55.0
DRAKE_BIN_CENTRES_DEG = np.array([-77.5, -72.5, -67.5, -62.5, -57.5])
DRAKE_BIN_HALFWIDTH_DEG = 2.5

MPAS_SUBDIVISION_LEVEL = 4
N_LEVELS = 20
H_MAX = 4000.0
DZ_SURFACE = 10.0
DZ_DEEP = 500.0

# Lat-lon binning grid for snapshot maps (matches plot_mpas_baseline_snapshot)
N_LAT_BIN, N_LON_BIN = 181, 360


def _gather_restarts(d: Path):
    pairs = []
    for p in d.glob("restart_day*.npz"):
        day = int(p.stem.removeprefix("restart_day"))
        pairs.append((day, p))
    pairs.sort()
    return pairs


def _zonal_mean_by_lat_bins(field_cell, lat_cell_rad, ocean_mask,
                             bin_centres_deg, bin_halfwidth_deg):
    """Cell-area-unweighted zonal mean, binned by latitude.

    field_cell : (nCells,) or (nCells, nlev)
    Returns : same shape but with first axis replaced by len(bin_centres_deg).
    Bins with no ocean cells return NaN.
    """
    lat_deg = np.asarray(lat_cell_rad) * 180.0 / np.pi
    n_bins = len(bin_centres_deg)
    is_ocean = np.asarray(ocean_mask) > 0.5
    field = np.asarray(field_cell)
    out_shape = (n_bins,) + field.shape[1:]
    out = np.full(out_shape, np.nan)
    for i, c in enumerate(bin_centres_deg):
        in_bin = (
            is_ocean
            & (lat_deg >= c - bin_halfwidth_deg)
            & (lat_deg < c + bin_halfwidth_deg)
        )
        if in_bin.any():
            out[i] = np.mean(field[in_bin], axis=0)
    return out


def _reconstruct_cell_velocity(u_edges, mesh):
    import jax.numpy as jnp
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edges), mesh)
    return np.asarray(u_east), np.asarray(v_north)


def main():
    pairs = _gather_restarts(RUN_DIR)
    if not pairs:
        print(f"No restarts found in {RUN_DIR}.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(pairs)} restarts in {RUN_DIR.name}:")
    for day, p in pairs:
        print(f"  day {day:>6}  ({day/365:5.2f} yr)")

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    mesh = create_voronoi_mesh(MPAS_SUBDIVISION_LEVEL)
    z_coord = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP,
    )
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)  # negative depths
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    depth = -z_full  # positive m for plotting

    # Earth radius (matches grids.voronoi default)
    R_earth = 6.371e6
    dy_drake = R_earth * np.deg2rad(2 * DRAKE_BIN_HALFWIDTH_DEG)  # bin meridional extent (m)

    # Source cell coordinates
    lon_deg = np.asarray(mesh.lonCell) * 180.0 / np.pi
    lon_deg = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)
    lat_deg = np.asarray(mesh.latCell) * 180.0 / np.pi
    lat_rad = np.asarray(mesh.latCell)

    bin_target_lat = np.linspace(-90.0, 90.0, N_LAT_BIN)
    bin_target_lon = np.linspace(-180.0, 180.0, N_LON_BIN)

    # Aggregators
    diags = []
    snapshots = []
    drake_T_Sv = []
    drake_zm_u_list = []  # (n_bins, nlev) per restart, restricted to Drake band
    drake_zm_T_list = []
    drake_u_prof = []     # (nlev,) cosine-weighted mean over Drake band

    # For T zonal mean we use 36 uniform bins (matches lat-lon resolution)
    zm_bin_centres = np.linspace(-87.5, 87.5, 36)
    zm_bin_halfwidth = 2.5

    # Pre-bin land mask once (use first restart's mask — should be identical
    # across restarts since geometry is fixed)
    d0 = np.load(pairs[0][1], allow_pickle=False)
    land_mask = np.asarray(d0["land_mask"])
    land_mask_bin = _bin_to_latlon(
        land_mask.astype(np.float64), lon_deg, lat_deg,
        target_lat=bin_target_lat, target_lon=bin_target_lon)
    land_pixel = ~(land_mask_bin > 0.5)

    for day, p in pairs:
        d = np.load(p, allow_pickle=False)
        year = day / 365.0
        eta = np.asarray(d["eta"])               # (nCells,)
        T = np.asarray(d["T"])                   # (nCells, nlev)
        u_edges = np.asarray(d["u"])             # (nEdges, nlev)
        m = np.asarray(d["land_mask"])           # (nCells,)
        is_ocean = m > 0.5

        # Cell-centered velocity
        u_east, v_north = _reconstruct_cell_velocity(u_edges, mesh)
        speed_sfc = np.sqrt(u_east[:, 0] ** 2 + v_north[:, 0] ** 2)
        speed_3d = np.sqrt(u_east ** 2 + v_north ** 2)

        # Scalar diagnostics
        diags.append({
            "day": day,
            "year": year,
            "max_speed": float(np.max(speed_3d[is_ocean])),
            "mean_sst":  float(np.mean(T[is_ocean, 0])),
            "mean_T":    float(np.mean(T[is_ocean, :])),
            "T_deep":    float(np.mean(T[is_ocean, -1])),
            "mean_eta":  float(np.mean(eta[is_ocean])),
            "max_eta":   float(np.max(np.abs(eta[is_ocean]))),
        })

        # Binned snapshots for surface map plots
        eta_bin = _bin_to_latlon(eta, lon_deg, lat_deg,
                                  target_lat=bin_target_lat,
                                  target_lon=bin_target_lon,
                                  ocean_mask=is_ocean)
        sst_bin = _bin_to_latlon(T[:, 0], lon_deg, lat_deg,
                                  target_lat=bin_target_lat,
                                  target_lon=bin_target_lon,
                                  ocean_mask=is_ocean)
        speed_bin = _bin_to_latlon(speed_sfc, lon_deg, lat_deg,
                                    target_lat=bin_target_lat,
                                    target_lon=bin_target_lon,
                                    ocean_mask=is_ocean)

        # T zonal mean by latitude (36 bins to match lat-lon)
        T_zm = _zonal_mean_by_lat_bins(T, lat_rad, is_ocean,
                                        zm_bin_centres,
                                        zm_bin_halfwidth)  # (36, nlev)

        snapshots.append({
            "day": day, "year": year,
            "eta_bin": np.ma.masked_where(land_pixel, eta_bin),
            "sst_bin": np.ma.masked_where(land_pixel, sst_bin),
            "speed_bin": np.ma.masked_where(land_pixel, speed_bin),
            "T_zonal_mean": T_zm,
        })

        # Drake-band diagnostics
        # Zonal-mean u_east, T over Drake bins (5 bins covering -77.5 to -57.5°)
        zm_u_drake = _zonal_mean_by_lat_bins(
            u_east, lat_rad, is_ocean,
            DRAKE_BIN_CENTRES_DEG, DRAKE_BIN_HALFWIDTH_DEG)   # (5, nlev)
        zm_T_drake = _zonal_mean_by_lat_bins(
            T, lat_rad, is_ocean,
            DRAKE_BIN_CENTRES_DEG, DRAKE_BIN_HALFWIDTH_DEG)
        drake_zm_u_list.append(zm_u_drake)
        drake_zm_T_list.append(zm_T_drake)

        # Drake transport (Sv): for each Drake bin, depth-integrate the zonal-
        # mean u_east, multiply by the bin's meridional extent, sum bins.
        # T_Sv = sum_bin( sum_z(<u>(bin, z) * dz_z) ) * dy_bin / 1e6
        # (zonally-periodic Drake band → ⟨u⟩ * H gives section transport per dy)
        T_per_bin = np.nansum(zm_u_drake * dz[None, :], axis=1) * dy_drake
        drake_T_Sv.append(float(np.nansum(T_per_bin)) / 1e6)

        # Cosine-weighted mean u(z) profile over Drake band
        weights = np.cos(np.deg2rad(DRAKE_BIN_CENTRES_DEG))
        valid = ~np.all(np.isnan(zm_u_drake), axis=1)
        if valid.any():
            u_prof = np.average(zm_u_drake[valid], axis=0,
                                 weights=weights[valid])
        else:
            u_prof = np.full(N_LEVELS, np.nan)
        drake_u_prof.append(u_prof)

    years = np.array([r["year"] for r in diags])
    drake_T_Sv = np.array(drake_T_Sv)

    # ====================================================================
    #  PROGRESS PLOTS
    # ====================================================================

    # ---- timeseries_progress ----
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(years, [r["max_speed"] for r in diags], "o-", color="C0")
    axes[0].set_ylabel("Max speed (m/s)")
    axes[0].set_title(
        "MPAS 50-yr implicit-solver run — progress (one point per restart)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(years, [r["mean_sst"] for r in diags], "o-", color="C3")
    axes[1].set_ylabel("Mean SST (°C)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(years, [r["mean_T"] for r in diags], "o-", label="All-depth")
    axes[2].plot(years, [r["T_deep"] for r in diags], "s-", label="Bottom layer")
    axes[2].set_ylabel("Mean T (°C)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(years, [r["mean_eta"] for r in diags], "o-", color="C2",
                 label="mean η")
    axes[3].plot(years, [r["max_eta"] for r in diags], "^-", color="C4",
                 label="max |η|")
    axes[3].set_ylabel("η (m)")
    axes[3].set_xlabel("Sim year")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "timeseries_progress.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'timeseries_progress.png'}")

    # ---- snapshots_progress ----
    n = len(snapshots)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, s in enumerate(snapshots):
        for j, (key, label, cmap, sym) in enumerate([
            ("eta_bin", "SSH (m)", "RdBu_r", True),
            ("sst_bin", "SST (°C)", "RdYlBu_r", False),
            ("speed_bin", "Surface speed (m/s)", "magma", False),
        ]):
            ax = axes[i, j]
            field = s[key]
            if sym:
                vmax = float(np.nanmax(np.abs(field)))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1.0
                im = ax.imshow(field, origin="lower",
                                extent=[-180, 180, -90, 90],
                                cmap=cmap, vmin=-vmax, vmax=vmax,
                                aspect="auto")
            else:
                im = ax.imshow(field, origin="lower",
                                extent=[-180, 180, -90, 90],
                                cmap=cmap, aspect="auto")
            ax.set_facecolor("#d9d9d9")
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            if i == 0:
                ax.set_title(label)
            ax.set_ylabel(f"Yr {s['year']:.0f}")
    plt.suptitle("MPAS 50-yr implicit run — snapshot evolution", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snapshots_progress.png", dpi=110)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'snapshots_progress.png'}")

    # ---- T_zonal_mean_progress ----
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    T_levels = np.arange(0, 23, 2)  # 0, 2, ..., 22 °C
    for i, s in enumerate(snapshots):
        T_zm = s["T_zonal_mean"]
        im = axes[i].pcolormesh(zm_bin_centres, depth, T_zm.T,
                                 cmap="RdYlBu_r", shading="auto",
                                 vmin=0, vmax=22)
        # Contour overlay
        cs = axes[i].contour(zm_bin_centres, depth, T_zm.T,
                              levels=T_levels, colors="k",
                              linewidths=0.5, alpha=0.6)
        axes[i].clabel(cs, inline=True, fontsize=6, fmt="%g")
        axes[i].invert_yaxis()
        plt.colorbar(im, ax=axes[i], fraction=0.046,
                     label="T (°C)" if i == n - 1 else None)
        axes[i].set_title(f"Yr {s['year']:.0f}")
        axes[i].set_xlabel("Latitude (°)")
        if i == 0:
            axes[i].set_ylabel("Depth (m)")
    plt.suptitle("MPAS zonal-mean T(lat, depth) evolution", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "T_zonal_mean_progress.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'T_zonal_mean_progress.png'}")

    # ====================================================================
    #  DRAKE PLOTS
    # ====================================================================

    # ---- drake_transport_timeseries ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years, drake_T_Sv, "o-", color="C0", ms=7, lw=1.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(-405, color="C3", lw=0.8, ls="--", alpha=0.6,
                label="lat-lon old-broken yr-50 ($-405$ Sv)")
    ax.axhline(150, color="C2", lw=0.8, ls="--", alpha=0.6,
                label="real ACC ($\\sim$+150 Sv)")
    ax.set_xlabel("Sim year")
    ax.set_ylabel("Drake transport (Sv)\neastward = +")
    ax.set_title("Drake Passage transport — MPAS 50-yr implicit run")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_transport_timeseries.png", dpi=140)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_transport_timeseries.png'}")

    # ---- drake_zonal_u_evolution ----
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                              sharey=True)
    axes = np.atleast_2d(axes)
    arr = np.array(drake_zm_u_list)  # (n_restarts, 5, nlev)
    vmax_u = float(np.nanmax(np.abs(arr)))
    if vmax_u == 0 or not np.isfinite(vmax_u):
        vmax_u = 0.1
    # u contour levels: 0.02 m/s spacing covering the range, with 0 highlighted
    u_step = max(0.02, np.round(vmax_u / 8, 2))
    u_levels = np.arange(-vmax_u, vmax_u + u_step / 2, u_step)
    u_levels = u_levels[u_levels != 0]
    for k in range(n):
        i, j = k // cols, k % cols
        ax = axes[i, j]
        cf = ax.pcolormesh(DRAKE_BIN_CENTRES_DEG, depth,
                           drake_zm_u_list[k].T,
                           vmin=-vmax_u, vmax=vmax_u, cmap="RdBu_r",
                           shading="auto")
        cs = ax.contour(DRAKE_BIN_CENTRES_DEG, depth, drake_zm_u_list[k].T,
                         levels=u_levels, colors="k", linewidths=0.4,
                         alpha=0.5)
        ax.contour(DRAKE_BIN_CENTRES_DEG, depth, drake_zm_u_list[k].T,
                    levels=[0], colors="k", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.invert_yaxis()
        ax.set_title(f"Yr {years[k]:.0f}  (T={drake_T_Sv[k]:+.0f} Sv)",
                      fontsize=10)
        ax.set_xlabel("Lat (°)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        plt.colorbar(cf, ax=ax, fraction=0.045)
    for k in range(n, rows * cols):
        i, j = k // cols, k % cols
        axes[i, j].axis("off")
    plt.suptitle(
        f"MPAS Drake-band zonal-mean u(lat, depth) — colour scale ±{vmax_u:.2f} m/s",
        y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_zonal_u_evolution.png", dpi=130,
                bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_zonal_u_evolution.png'}")

    # ---- drake_T_evolution ----
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                              sharey=True)
    axes = np.atleast_2d(axes)
    arrT = np.array(drake_zm_T_list)
    vmax_T = float(np.nanmax(arrT))
    vmin_T = float(np.nanmin(arrT))
    T_levels_drake = np.arange(np.floor(vmin_T), np.ceil(vmax_T) + 1, 1.0)
    for k in range(n):
        i, j = k // cols, k % cols
        ax = axes[i, j]
        cf = ax.pcolormesh(DRAKE_BIN_CENTRES_DEG, depth,
                           drake_zm_T_list[k].T,
                           vmin=vmin_T, vmax=vmax_T, cmap="RdYlBu_r",
                           shading="auto")
        cs = ax.contour(DRAKE_BIN_CENTRES_DEG, depth, drake_zm_T_list[k].T,
                         levels=T_levels_drake, colors="k",
                         linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.invert_yaxis()
        ax.set_title(f"Yr {years[k]:.0f}", fontsize=10)
        ax.set_xlabel("Lat (°)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        plt.colorbar(cf, ax=ax, fraction=0.045)
    for k in range(n, rows * cols):
        i, j = k // cols, k % cols
        axes[i, j].axis("off")
    plt.suptitle("MPAS Drake-band zonal-mean T(lat, depth)", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_T_evolution.png", dpi=130,
                bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_T_evolution.png'}")

    # ---- drake_u_profiles ----
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("plasma")
    for k in range(n):
        color = cmap(k / max(n - 1, 1))
        ax.plot(np.array(drake_u_prof[k]) * 100, z_full, "-o", ms=3,
                color=color, label=f"Yr {years[k]:.0f}")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("u (cm/s)  — eastward = +")
    ax.set_ylabel("Depth (m)")
    ax.set_title("MPAS Drake-band-averaged u(z) — evolution over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_u_profiles.png", dpi=140)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_u_profiles.png'}")

    # ---- summary ----
    print("\n--- Diagnostics summary ---")
    print(f"{'Yr':>5} {'max|U|':>7} {'SST':>6} {'meanT':>6} {'T_deep':>6} "
          f"{'meanη':>9} {'max|η|':>7} {'Drake(Sv)':>10}")
    for r, T_Sv in zip(diags, drake_T_Sv):
        print(f"{r['year']:>5.1f} {r['max_speed']:>7.3f} "
              f"{r['mean_sst']:>6.2f} {r['mean_T']:>6.2f} "
              f"{r['T_deep']:>6.2f} {r['mean_eta']:>+9.3e} "
              f"{r['max_eta']:>7.3f} {T_Sv:>+10.1f}")


if __name__ == "__main__":
    main()
