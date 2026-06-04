#!/usr/bin/env python
"""Phase 0 sanity-check script for the realistic-geometry plan.

Runs the full bathymetry pipeline on a small real ETOPO file and produces
diagnostic plots:

  - bathymetry.png:       H_bathy(lat, lon) on the target lat-lon grid
  - coastlines.png:       binary land mask, with critical-strait centres marked
  - slopes.png:           histogram of |∇H|/H, identifying PGF-hard cells

The ETOPO data is small (139 KB, 1° subsample of NOAA ERDDAP etopo180);
fetched once and cached in ``data/bathymetry/etopo_1deg.nc``.

Usage:
    JAX_ENABLE_X64=1 python scripts/tmp/diagnose_realistic_geometry.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import (
    BathymetryConfig,
    CRITICAL_STRAITS,
    init_ocean_bathymetry,
)


DATA_DIR = Path("data/bathymetry")
ETOPO_FILE = DATA_DIR / "etopo_1deg.nc"
ETOPO_URL = (
    "https://upwell.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
    "altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
)
OUTPUT_DIR = Path("results/realistic_geometry_check")

# Target grid: 72×144 (2.5°), the recommended starting resolution per the plan.
N_LAT = 72
N_LON = 144


def _ensure_etopo():
    """Download ETOPO 1° NetCDF subset from NOAA if not already cached."""
    if ETOPO_FILE.exists():
        print(f"Using cached ETOPO: {ETOPO_FILE} "
              f"({ETOPO_FILE.stat().st_size / 1024:.1f} KB)")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ETOPO 1° subset from NOAA ERDDAP → {ETOPO_FILE}")
    subprocess.run(
        ["curl", "-fsS", "-o", str(ETOPO_FILE), ETOPO_URL],
        check=True,
    )
    print(f"  Cached: {ETOPO_FILE.stat().st_size / 1024:.1f} KB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_etopo()

    grid = create_latlon_grid(N_LAT, N_LON)
    print(f"\nTarget grid: {N_LAT}×{N_LON} (≈{180.0/N_LAT:.2f}° lat, "
          f"{360.0/N_LON:.2f}° lon)")

    cfg = BathymetryConfig(
        source="file",
        path=str(ETOPO_FILE),
        H_max=5500.0,
        H_min=10.0,
        smoothing_passes=2,
        enforce_straits=True,
        strait_width_factor=1.0,
        fill_isolated_basins=True,
        depth_is_negative=True,
    )
    print(f"\nConfig: {cfg}")

    H_bathy, ocean_mask = init_ocean_bathymetry(grid, cfg)
    H_bathy = np.asarray(H_bathy)
    ocean_mask = np.asarray(ocean_mask)

    print(f"\nResult shapes: H_bathy={H_bathy.shape}, ocean_mask={ocean_mask.shape}")
    is_ocean = ocean_mask > 0.5
    n_ocean = int(is_ocean.sum())
    n_total = H_bathy.size
    print(f"Ocean fraction: {n_ocean / n_total:.3f}  "
          f"({n_ocean}/{n_total} cells)")
    if n_ocean > 0:
        H_ocean = H_bathy[is_ocean]
        print(f"Ocean depth: min={H_ocean.min():.0f} m, "
              f"max={H_ocean.max():.0f} m, mean={H_ocean.mean():.0f} m")

    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lon_for_plot = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)
    sort_lon = np.argsort(lon_for_plot)
    lon_sorted = lon_for_plot[sort_lon]

    H_for_plot = np.where(is_ocean, H_bathy, np.nan)[:, sort_lon]
    mask_for_plot = ocean_mask[:, sort_lon]

    # ---- bathymetry.png ----
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(
        lon_sorted, lat_deg, H_for_plot,
        cmap="cmo.deep" if "cmo" in plt.colormaps() else "Blues_r",
        vmin=0, vmax=5500,
        shading="auto",
    )
    ax.set_facecolor("#d9d9d9")
    plt.colorbar(im, ax=ax, label="Ocean depth (m)", fraction=0.025)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["180°W", "90°W", "0°", "90°E", "180°E"])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_yticklabels(["90°S", "45°S", "0°", "45°N", "90°N"])
    ax.set_title(f"ETOPO bathymetry on {N_LAT}×{N_LON} grid "
                  f"(smoothing={cfg.smoothing_passes}, straits enforced)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bathymetry.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'bathymetry.png'}")

    # ---- coastlines.png ----
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(
        lon_sorted, lat_deg, mask_for_plot,
        cmap="Greys", vmin=0, vmax=1, shading="auto",
    )
    ax.set_facecolor("white")
    # Mark each enforced strait centre
    for name, slat, slon, _w, _d in CRITICAL_STRAITS:
        slon_plot = slon if slon <= 180 else slon - 360
        ax.plot(slon_plot, slat, "o", color="C3", ms=5, mec="white", mew=0.8)
        ax.annotate(name, (slon_plot, slat), fontsize=7, color="C3",
                     xytext=(4, 4), textcoords="offset points")
    plt.colorbar(im, ax=ax, label="Ocean (1) / Land (0)", fraction=0.025)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["180°W", "90°W", "0°", "90°E", "180°E"])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_yticklabels(["90°S", "45°S", "0°", "45°N", "90°N"])
    ax.set_title(f"Land-sea mask on {N_LAT}×{N_LON} grid "
                  f"(red dots = enforced critical straits)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coastlines.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'coastlines.png'}")

    # ---- slopes.png ----
    # Beckmann-Haidvogel "r-factor": |H_i+1 - H_i| / max(H_i+1, H_i).
    # Dimensionless; r < 0.2 is the standard z-coord PGF stability bound,
    # r < 0.1 stricter.  Computed as max over the four cell neighbours.
    H_o = np.where(is_ocean, H_bathy, np.nan)
    r_east  = np.abs(H_o - np.roll(H_o, -1, axis=1)) \
              / np.fmax(H_o, np.roll(H_o, -1, axis=1))
    r_west  = np.abs(H_o - np.roll(H_o, +1, axis=1)) \
              / np.fmax(H_o, np.roll(H_o, +1, axis=1))
    r_north = np.zeros_like(H_o); r_south = np.zeros_like(H_o)
    r_north[:-1, :] = np.abs(H_o[:-1, :] - H_o[1:, :]) \
                       / np.fmax(H_o[:-1, :], H_o[1:, :])
    r_south[1:, :]  = np.abs(H_o[1:, :] - H_o[:-1, :]) \
                       / np.fmax(H_o[1:, :], H_o[:-1, :])
    # NaN-safe per-cell maximum across the four neighbours.
    r_factor = np.fmax(np.fmax(r_east, r_west),
                        np.fmax(r_north, r_south))
    r_ocean = r_factor[is_ocean & np.isfinite(r_factor)]
    if r_ocean.size == 0:
        r_ocean = np.array([0.0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Histogram
    axes[0].hist(r_ocean[r_ocean > 0], bins=60,
                  color="C0", edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel(r"r-factor  $|H_i - H_j| / \max(H_i, H_j)$")
    axes[0].set_ylabel("Number of ocean cells")
    axes[0].set_yscale("log")
    p99 = float(np.percentile(r_ocean, 99))
    rmax = float(r_ocean.max())
    axes[0].set_title("r-factor histogram (log y)\n"
                       f"99th percentile = {p99:.3f}, max = {rmax:.3f}  "
                       f"(B-H stable: r < 0.2)")
    axes[0].axvline(0.2, color="C3", linestyle="--", linewidth=1,
                     label="B-H stability bound (0.2)")
    axes[0].axvline(p99, color="C0", linestyle=":", linewidth=1,
                     label=f"99th %ile ({p99:.3f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Map of r-factor
    r_for_plot = np.where(is_ocean, r_factor, np.nan)[:, sort_lon]
    im = axes[1].pcolormesh(lon_sorted, lat_deg, r_for_plot,
                             cmap="hot_r", vmin=0,
                             vmax=max(p99 * 2, 0.2),
                             shading="auto")
    axes[1].set_facecolor("#d9d9d9")
    plt.colorbar(im, ax=axes[1], label=r"$|\nabla H| / H$", fraction=0.03)
    axes[1].set_xlabel("Longitude (°)")
    axes[1].set_ylabel("Latitude (°)")
    axes[1].set_title("Spatial distribution of r-factor\n"
                       "(hottest cells = PGF-hardest)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slopes.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'slopes.png'}")

    print(f"\n--- Summary ---")
    print(f"Ocean fraction:         {n_ocean / n_total:.3f}")
    print(f"Mean ocean depth:       {H_ocean.mean():.0f} m")
    print(f"Max ocean depth:        {H_ocean.max():.0f} m")
    print(f"99th-pct r-factor:      {p99:.4f}  (B-H stable: < 0.2)")
    print(f"Max r-factor:           {rmax:.4f}")
    print(f"Cells with r > 0.2:     "
          f"{int((r_ocean > 0.2).sum())}/{r_ocean.size}")
    print(f"Diagnostic plots:       {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
