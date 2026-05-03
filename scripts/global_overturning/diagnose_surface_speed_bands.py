"""Stage 0a diagnostic: surface speed × H_bathy overlay for the
realistic-geometry GO spinup.

Reads the same restarts as ``plot_realistic_geometry_progress.py`` and
produces:

  - speed_bathy_overlay.png:   surface speed (filled), H_bathy contours
                               overlaid, one row per restart year.
  - u_v_components.png:        surface u and v separately (signed
                               colormap), H_bathy contours overlaid.
  - speed_zonal_section.png:   zonal mean of surface speed vs latitude,
                               curves overlaid for each restart year
                               — characterise N-S band amplitude.

Question being answered: do the high-speed bands trace isobaths
(slope-current / cold-start adjustment) or sit at fixed longitudes
(numerical mode pinned to grid lines)?  Are u and v equally banded,
or is one component dominating?

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/diagnose_surface_speed_bands.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry


RUN_DIR = Path("results/ocean/global_overturning_realistic_geometry")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")


def _gather_restarts(d):
    pairs = []
    for p in d.glob("restart_day*.npz"):
        day = int(p.stem.removeprefix("restart_day"))
        pairs.append((day, p))
    pairs.sort()
    return pairs


def main():
    pairs = _gather_restarts(RUN_DIR)
    if not pairs:
        print(f"No restarts found in {RUN_DIR}.")
        return
    print(f"Found {len(pairs)} restarts.")

    # Reconstruct the same H_bathy and ocean_mask as the running script
    cfg = GlobalOverturningConfig(
        use_gm_redi=True,
        bottom_drag_coeff=2.5e-3,
        A_h=1.0e4,
        H_max=5000.0,
        dz_surface=20.0,
        kappa_GM=800.0,
        kappa_Redi=800.0,
    )
    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=cfg.H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True,
        r_factor_max=0.2,
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = np.asarray(H_bathy_jax)
    ocean_mask = np.asarray(ocean_mask_jax) > 0.5

    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    # H_bathy on land masked for contour clarity
    H_for_contour = np.where(ocean_mask, H_bathy, np.nan)
    # Isobath levels — pick depths that give us coastlines + slope + abyss
    isobath_levels = [200, 1000, 2000, 3000, 4000]

    # --- Per-restart fields ---
    snapshots = []
    for day, p in pairs:
        d = np.load(p, allow_pickle=False)
        u = d["u"]; v = d["v"]; mask = d["land_mask"]
        # u is (n_lat, n_lon+1, nlev), v is (n_lat+1, n_lon, nlev)
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])      # cell-center u at surface
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])      # cell-center v at surface
        speed_sfc = np.sqrt(u_c ** 2 + v_c ** 2)
        ocean = mask > 0.5
        snapshots.append({
            "day": day,
            "year": day / 365.0,
            "u": np.where(ocean, u_c, np.nan),
            "v": np.where(ocean, v_c, np.nan),
            "speed": np.where(ocean, speed_sfc, np.nan),
        })

    n = len(snapshots)

    # ============================================================
    # Plot 1: Surface speed with H_bathy contours overlaid
    # ============================================================
    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # Use a robust upper limit (95th percentile) so saturated coastal hot
    # spots don't crush the rest of the field
    speed_p95 = max(np.nanpercentile(s["speed"], 95) for s in snapshots[1:])
    speed_p95 = max(speed_p95, 0.1)

    for i, s in enumerate(snapshots):
        ax = axes[i]
        im = ax.pcolormesh(
            lon, lat, s["speed"], cmap="magma",
            vmin=0, vmax=speed_p95, shading="auto",
        )
        # Overlay isobath contours
        cs = ax.contour(
            lon, lat, H_for_contour, levels=isobath_levels,
            colors="cyan", linewidths=0.5, alpha=0.7,
        )
        ax.clabel(cs, inline=True, fontsize=6, fmt="%dm")
        cb = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
        cb.set_label("Surface speed (m/s)" if i == 0 else "", fontsize=8)
        ax.set_title(
            f"Yr {s['year']:.0f}  |  max |u|+|v| = "
            f"{float(np.nanmax(s['speed'])):.2f} m/s  "
            f"(p95 = {float(np.nanpercentile(s['speed'], 95)):.2f})",
            fontsize=9,
        )
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "Surface speed (filled) with H_bathy isobaths (cyan, m) overlaid\n"
        "Question: do bands trace isobaths or sit at fixed longitudes?",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    out1 = RUN_DIR / "speed_bathy_overlay.png"
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # ============================================================
    # Plot 2: u and v components separately
    # ============================================================
    fig, axes = plt.subplots(n, 2, figsize=(15, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    # Symmetric colorbar from the largest signed component
    uv_max = max(
        max(
            np.nanpercentile(np.abs(s["u"]), 98),
            np.nanpercentile(np.abs(s["v"]), 98),
        )
        for s in snapshots[1:]
    )
    uv_max = max(uv_max, 0.1)

    for i, s in enumerate(snapshots):
        for j, (key, label) in enumerate([("u", "u (zonal)"), ("v", "v (meridional)")]):
            ax = axes[i, j]
            im = ax.pcolormesh(
                lon, lat, s[key], cmap="RdBu_r",
                vmin=-uv_max, vmax=uv_max, shading="auto",
            )
            cs = ax.contour(
                lon, lat, H_for_contour, levels=isobath_levels,
                colors="k", linewidths=0.4, alpha=0.5,
            )
            cb = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
            cb.set_label(f"{label} (m/s)" if i == 0 else "", fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Yr {s['year']:.0f}")
        axes[-1, 0].set_xlabel("Longitude (°)")
        axes[-1, 1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "Surface u (zonal) vs v (meridional) — H_bathy isobaths in black\n"
        "Question: are bands in u, v, or both?",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    out2 = RUN_DIR / "u_v_components.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")

    # ============================================================
    # Plot 3: Zonal-mean and meridional-mean speed structure
    # ============================================================
    fig, (ax_zonal, ax_meridional) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.viridis(np.linspace(0, 1, n))

    # Zonal mean: speed averaged over longitude → speed(lat) per year
    for i, s in enumerate(snapshots):
        speed_zm = np.nanmean(s["speed"], axis=1)
        ax_zonal.plot(
            lat, speed_zm, "o-", color=cmap[i],
            label=f"Yr {s['year']:.0f}", ms=3,
        )
    ax_zonal.set_xlabel("Latitude (°)")
    ax_zonal.set_ylabel("Zonal-mean surface speed (m/s)")
    ax_zonal.set_title("Zonal-mean speed(lat)")
    ax_zonal.grid(alpha=0.3)
    ax_zonal.legend(fontsize=8)

    # Meridional mean: speed averaged over latitude → speed(lon) per year
    for i, s in enumerate(snapshots):
        speed_mm = np.nanmean(s["speed"], axis=0)
        ax_meridional.plot(
            lon, speed_mm, "o-", color=cmap[i],
            label=f"Yr {s['year']:.0f}", ms=3,
        )
    ax_meridional.set_xlabel("Longitude (°)")
    ax_meridional.set_ylabel("Meridional-mean surface speed (m/s)")
    ax_meridional.set_title(
        "Meridional-mean speed(lon)\n"
        "Sharp peaks at fixed lon → numerical mode at grid lines"
    )
    ax_meridional.grid(alpha=0.3)
    ax_meridional.legend(fontsize=8)

    plt.tight_layout()
    out3 = RUN_DIR / "speed_zonal_section.png"
    plt.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out3}")

    # ============================================================
    # Quantitative summary: where in lon space is the meridional-mean
    # speed concentrated, year by year?
    # ============================================================
    print("\n--- Meridional-mean speed concentration (top-3 longitude bins) ---")
    print(f"{'Yr':>5}  {'max':>6}  {'top-3 longitudes (°), peak speed (m/s)':<60}")
    for s in snapshots:
        speed_mm = np.nanmean(s["speed"], axis=0)
        top_idx = np.argsort(speed_mm)[-3:][::-1]
        top_str = ", ".join(
            f"{lon[i]:+6.1f}°→{speed_mm[i]:.2f}" for i in top_idx
        )
        print(f"{s['year']:>5.0f}  {float(np.nanmax(speed_mm)):>6.2f}  {top_str}")

    print("\n--- Zonal-mean speed concentration (top-3 latitude bins) ---")
    print(f"{'Yr':>5}  {'max':>6}  {'top-3 latitudes (°), peak speed (m/s)':<60}")
    for s in snapshots:
        speed_zm = np.nanmean(s["speed"], axis=1)
        top_idx = np.argsort(speed_zm)[-3:][::-1]
        top_str = ", ".join(
            f"{lat[i]:+6.1f}°→{speed_zm[i]:.2f}" for i in top_idx
        )
        print(f"{s['year']:>5.0f}  {float(np.nanmax(speed_zm)):>6.2f}  {top_str}")


if __name__ == "__main__":
    main()
