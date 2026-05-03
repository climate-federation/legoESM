"""Visualise how MEO r-factor capping reshapes the ETOPO bathymetry.

Loads real ETOPO at 5° (36×72), regrids + Laplacian-smooths +
strait-enforces (the standard pipeline), then applies the
Mellor-Ezer-Oey r-factor cap at r ∈ {None, 0.5, 0.3, 0.2, 0.1}.
Saves a single multi-panel figure showing:

  - top row:    H_bathy(lat, lon) maps with continents outlined
  - bottom row: depth histograms for each MEO setting + ocean
                volume change relative to no-MEO baseline.

Useful for assessing the trade-off: aggressive MEO smooths the
slope-induced PGF errors but eliminates continental shelves and
volume.

Usage:
    JAX_ENABLE_X64=1 python scripts/realistic_geometry_validation/plot_meo_bathymetry_sweep.py
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
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry


GRID = (36, 72)
H_MAX = 5000.0
H_MIN = 50.0
SMOOTHING_PASSES = 5
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")
OUTPUT = Path("results/realistic_geometry_validation/meo_bathymetry_comparison.png")


def load_with_meo(grid, r_max):
    cfg = BathymetryConfig(
        source="file",
        path=str(ETOPO_FILE),
        H_max=H_MAX,
        H_min=H_MIN,
        smoothing_passes=SMOOTHING_PASSES,
        enforce_straits=True,
        fill_isolated_basins=True,
        depth_is_negative=True,
        r_factor_max=r_max,
    )
    H, mask = init_ocean_bathymetry(grid, cfg)
    return np.asarray(H), np.asarray(mask)


def compute_r_factor_max(H, mask):
    """Local r-factor max over wet-wet ocean pairs."""
    H_e = np.roll(H, -1, axis=1)
    H_w = np.roll(H, +1, axis=1)
    H_n = np.zeros_like(H)
    H_n[:-1, :] = H[1:, :]
    H_n[-1, :] = H[-1, :]
    H_s = np.zeros_like(H)
    H_s[1:, :] = H[:-1, :]
    H_s[0, :] = H[0, :]

    mask_e = np.roll(mask, -1, axis=1)
    mask_w = np.roll(mask, +1, axis=1)
    mask_n = np.zeros_like(mask)
    mask_n[:-1, :] = mask[1:, :]
    mask_n[-1, :] = mask[-1, :]
    mask_s = np.zeros_like(mask)
    mask_s[1:, :] = mask[:-1, :]
    mask_s[0, :] = mask[0, :]

    r_pairs = []
    for H_nbr, mask_nbr in [(H_e, mask_e), (H_w, mask_w),
                             (H_n, mask_n), (H_s, mask_s)]:
        wet = (mask > 0.5) & (mask_nbr > 0.5)
        denom = np.fmax(H, H_nbr)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(wet & (denom > 0),
                          np.abs(H - H_nbr) / np.maximum(denom, 1e-10),
                          0.0)
        r_pairs.append(r)
    return np.maximum.reduce(r_pairs)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    grid = create_latlon_grid(*GRID)
    lat = np.asarray(grid.lat) * 180.0 / np.pi
    lon = np.asarray(grid.lon) * 180.0 / np.pi

    settings = [(None, "no MEO"),
                (0.5, "r ≤ 0.5"),
                (0.3, "r ≤ 0.3"),
                (0.2, "r ≤ 0.2"),
                (0.1, "r ≤ 0.1")]

    bathys = {}
    for r, label in settings:
        H, mask = load_with_meo(grid, r)
        r_actual = compute_r_factor_max(H, mask)
        H_min_wet = float(H[mask > 0.5].min()) if (mask > 0.5).any() else 0.0
        # Ocean volume — sum H * area_per_cell.  Using approx grid area
        # (uniform 5° spacing × cos lat).
        area_factor = np.cos(np.radians(lat))[:, None] * np.ones((1, len(lon)))
        ocean_vol = float(((H * area_factor) * (mask > 0.5)).sum())
        bathys[label] = {
            "H": H,
            "mask": mask,
            "r_max": float(r_actual.max()),
            "r_mean": float(r_actual[mask > 0.5].mean()) if (mask > 0.5).any() else 0.0,
            "H_min_wet": H_min_wet,
            "ocean_vol": ocean_vol,
            "label": label,
        }

    baseline_vol = bathys["no MEO"]["ocean_vol"]

    # ---- Plot ----
    n = len(settings)
    fig = plt.figure(figsize=(4 * n, 7.5))
    gs = fig.add_gridspec(2, n, height_ratios=[1.6, 1.0], hspace=0.35, wspace=0.18)

    cmap_bathy = "viridis_r"
    vmin, vmax = 0.0, H_MAX

    for i, (_, label) in enumerate(settings):
        b = bathys[label]
        H = b["H"]
        mask = b["mask"]
        H_plot = np.where(mask > 0.5, H, np.nan)

        ax = fig.add_subplot(gs[0, i])
        im = ax.pcolormesh(lon, lat, H_plot, cmap=cmap_bathy,
                            shading="auto", vmin=vmin, vmax=vmax)
        # Land outline — use mask boundary as a contour
        ax.contour(lon, lat, mask, levels=[0.5], colors="black",
                    linewidths=0.5)
        ax.set_xlabel("Longitude (°)")
        if i == 0:
            ax.set_ylabel("Latitude (°)")
        ax.set_title(
            f"{label}\nH_min={b['H_min_wet']:.0f} m, "
            f"r_max={b['r_max']:.3f}, vol={100*b['ocean_vol']/baseline_vol:.1f}%"
        )
        if i == n - 1:
            cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("H_bathy (m)")

        # Bottom row: depth histogram
        ax2 = fig.add_subplot(gs[1, i])
        H_wet = H[mask > 0.5]
        bins = np.linspace(0, H_MAX, 51)
        ax2.hist(H_wet, bins=bins, color="steelblue", edgecolor="white",
                  linewidth=0.3)
        ax2.set_xlabel("H_bathy (m)")
        if i == 0:
            ax2.set_ylabel("# wet cells")
        ax2.axvline(b["H_min_wet"], color="red", lw=1,
                     label=f"H_min={b['H_min_wet']:.0f}")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_xlim(0, H_MAX)
        ax2.set_yscale("log")

    fig.suptitle(
        f"ETOPO 5° (36×72) bathymetry, smoothing_passes={SMOOTHING_PASSES}, "
        f"H_min={H_MIN}m — MEO r-factor sweep",
        fontsize=11, y=0.995,
    )
    plt.savefig(OUTPUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUTPUT}")

    # Also print a summary table
    print()
    print(f"{'MEO':<10} | {'H_min':>7} | {'r_max':>7} | {'r_mean':>7} | "
          f"{'vol/baseline':>12}")
    print("-" * 60)
    for _, label in settings:
        b = bathys[label]
        print(f"{label:<10} | {b['H_min_wet']:>7.0f} | "
              f"{b['r_max']:>7.3f} | {b['r_mean']:>7.4f} | "
              f"{100*b['ocean_vol']/baseline_vol:>11.2f}%")


if __name__ == "__main__":
    main()
