"""Quick partial-results plot for the cos²(lat) sweep, before the
third run finishes.  Compares the two completed cos²-scaled runs
(A_h_global = 1e5, 2e5) against the existing uniform-A_h=5e4 result
from results/ocean/ah_sweep/run_Ah5e04/restart_yr2.npz.
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
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry


OUTPUT_DIR = Path("results/ocean/cos2lat_sweep")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")


def _load_speed(path):
    d = np.load(path, allow_pickle=False)
    u = d["u"]; v = d["v"]; mask = d["land_mask"]
    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed = np.sqrt(u_c ** 2 + v_c ** 2)
    ocean = mask > 0.5
    return (
        np.where(ocean, speed, np.nan),
        np.where(ocean, u_c, np.nan),
        np.where(ocean, v_c, np.nan),
    )


def main():
    cfg = GlobalOverturningConfig(
        use_gm_redi=True, H_max=5000.0, dz_surface=20.0,
        kappa_GM=800.0, kappa_Redi=800.0,
    )
    grid = create_latlon_grid(36, 72)
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=cfg.H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True, r_factor_max=0.2,
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = np.asarray(H_bathy_jax)
    ocean_mask = np.asarray(ocean_mask_jax) > 0.5
    H_for_contour = np.where(ocean_mask, H_bathy, np.nan)
    isobath_levels = [200, 1000, 2000, 3000, 4000]

    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    runs = [
        ("Uniform A_h=1e4 (Phase 4 default)",
         "results/ocean/ah_sweep/run_Ah1e04/restart_yr2.npz"),
        ("Uniform A_h=5e4 (best uniform)",
         "results/ocean/ah_sweep/run_Ah5e04/restart_yr2.npz"),
        ("cos²(lat) A_h_global=1e5",
         "results/ocean/cos2lat_sweep/run_Ah1e05/restart_yr2.npz"),
        ("cos²(lat) A_h_global=2e5",
         "results/ocean/cos2lat_sweep/run_Ah2e05/restart_yr2.npz"),
        ("cos²(lat) A_h_global=5e5",
         "results/ocean/cos2lat_sweep/run_Ah5e05/restart_yr2.npz"),
    ]

    fields = {}
    for label, path in runs:
        if Path(path).exists():
            sp, uc, vc = _load_speed(path)
            fields[label] = {"speed": sp, "u": uc, "v": vc}
        else:
            print(f"  Missing: {path}")

    if not fields:
        print("Nothing to plot.")
        return

    # ============================================================
    # Plot 1: surface speed × bathymetry — 4 panels
    # ============================================================
    n = len(fields)
    speed_p95 = max(np.nanpercentile(d["speed"], 95) for d in fields.values())
    speed_p95 = max(speed_p95, 0.1)

    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for i, (label, d) in enumerate(fields.items()):
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, d["speed"], cmap="magma",
                            vmin=0, vmax=speed_p95, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="cyan", linewidths=0.4, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label="Surface speed (m/s)" if i == 0 else "")
        max_speed = float(np.nanmax(d["speed"]))
        p95 = float(np.nanpercentile(d["speed"], 95))
        ax.set_title(f"{label}  |  yr 2  |  max = {max_speed:.2f} m/s, p95 = {p95:.2f}",
                      fontsize=10)
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "cos²(lat) A_h sweep (partial) — surface speed at yr 2",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    out1 = OUTPUT_DIR / "partial_speed_comparison.png"
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # ============================================================
    # Plot 2: u (zonal) component — 4 panels — most direct view of 2Δy mode
    # ============================================================
    uv_max = max(
        np.nanpercentile(np.abs(d["u"]), 98) for d in fields.values()
    )
    uv_max = max(uv_max, 0.1)

    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for i, (label, d) in enumerate(fields.items()):
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, d["u"], cmap="RdBu_r",
                            vmin=-uv_max, vmax=uv_max, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="k", linewidths=0.3, alpha=0.5)
        plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label="u (m/s)" if i == 0 else "")
        ax.set_title(f"{label}  |  yr 2", fontsize=10)
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "Surface u (zonal) — does cos²(lat) damp the 2Δy zonal-jet mode?",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    out2 = OUTPUT_DIR / "partial_u_comparison.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")

    # ============================================================
    # Plot 3: zonal-mean and meridional-mean speed — overlay all
    # ============================================================
    fig, (ax_z, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, n))
    for i, (label, d) in enumerate(fields.items()):
        speed = d["speed"]
        ax_z.plot(lat, np.nanmean(speed, axis=1), "o-",
                   color=cmap[i], ms=3, label=label)
        ax_m.plot(lon, np.nanmean(speed, axis=0), "o-",
                   color=cmap[i], ms=3, label=label)
    ax_z.set_xlabel("Latitude (°)")
    ax_z.set_ylabel("Zonal-mean surface speed (m/s)")
    ax_z.set_title("Zonal-mean speed(lat) — cos² compared to uniform A_h")
    ax_z.grid(alpha=0.3)
    ax_z.legend(fontsize=8)
    ax_m.set_xlabel("Longitude (°)")
    ax_m.set_ylabel("Meridional-mean surface speed (m/s)")
    ax_m.set_title("Meridional-mean speed(lon)")
    ax_m.grid(alpha=0.3)
    ax_m.legend(fontsize=8)
    plt.tight_layout()
    out3 = OUTPUT_DIR / "partial_zonal_mean_comparison.png"
    plt.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out3}")

    # ============================================================
    # Numerical summary
    # ============================================================
    print("\n--- cos²(lat) sweep partial summary ---")
    print(f"{'config':<40}  {'max|u|':>8}  {'p95':>6}  {'eq.zm':>6}")
    for label, d in fields.items():
        speed = d["speed"]
        max_u = float(np.nanmax(np.abs(d["u"])))
        p95 = float(np.nanpercentile(speed, 95))
        lat_eq = np.abs(lat) <= 5.0
        eq_zm = float(np.nanmean(speed[lat_eq, :]))
        print(f"{label:<40}  {max_u:>8.3f}  {p95:>6.3f}  {eq_zm:>6.3f}")


if __name__ == "__main__":
    main()
