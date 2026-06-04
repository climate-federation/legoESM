#!/usr/bin/env python
"""Zonal-mean u, v as a function of (lat, depth) for the 50yr global
overturning + GM/Redi run.

The 50yr run did not save 3D snapshots, but it saved restart files every
5 years. We average u, v across those restarts to produce a coarse 50-yr
mean zonal-mean section.

Outputs:
  results/ocean/global_overturning_50yr_gmredi/
    zonal_mean_uv_50yr.png   — single restart (day 18250) and 10-restart mean
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig

RUN_DIR = Path("results/ocean/global_overturning_50yr_gmredi")


def load_restart_uv(path: Path):
    npz = np.load(path, allow_pickle=False)
    return (
        np.asarray(npz["u"], dtype=np.float64),       # (n_lat, n_lon+1, nlev)
        np.asarray(npz["v"], dtype=np.float64),       # (n_lat+1, n_lon, nlev)
        np.asarray(npz["u_mask"], dtype=np.float64),  # (n_lat, n_lon+1)
        np.asarray(npz["v_mask"], dtype=np.float64),  # (n_lat+1, n_lon)
        float(npz["time_days"]),
    )


def zonal_mean_masked(field, mask):
    """Zonal mean of `field` (lat, lon[+1], nlev) using a 2D `mask` (lat, lon[+1]).

    Wet-cell-only average so land doesn't dilute the mean. Returns NaN at
    rows that are entirely dry.
    """
    m = mask[:, :, None]                        # broadcast along level
    num = np.sum(field * m, axis=1)             # (lat, nlev)
    den = np.sum(m, axis=1)                     # (lat, nlev) — same for all levels
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def main():
    restarts = sorted(RUN_DIR.glob("restart_day*.npz"))
    if not restarts:
        raise SystemExit(f"No restart files in {RUN_DIR}")

    # Build coordinate axes ----------------------------------------------------
    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi  # cell centers
    dlat = float(grid.dlat)
    lat_v_deg = np.concatenate([
        [lat_deg[0] - 0.5 * dlat * 180 / np.pi],
        0.5 * (lat_deg[:-1] + lat_deg[1:]),
        [lat_deg[-1] + 0.5 * dlat * 180 / np.pi],
    ])

    # Load all restarts and accumulate -----------------------------------------
    print(f"Found {len(restarts)} restart files")
    sum_u = None
    sum_v = None
    u_mask = None
    v_mask = None
    days = []
    for r in restarts:
        u, v, um, vm, day = load_restart_uv(r)
        if sum_u is None:
            sum_u = np.zeros_like(u)
            sum_v = np.zeros_like(v)
            u_mask = um
            v_mask = vm
        sum_u += u
        sum_v += v
        days.append(day)
        print(f"  {r.name}  day={day:.0f} (yr {day/365:.1f})")
    n = len(restarts)
    mean_u = sum_u / n
    mean_v = sum_v / n

    # Final-restart fields for instantaneous comparison
    u_last, v_last, *_ = load_restart_uv(restarts[-1])

    # Zonal means --------------------------------------------------------------
    # u is on u-faces (lat × n_lon+1): zonal-mean over u-faces, plotted at
    # cell-center latitudes (the lon dim is collapsed; the lat dim is unchanged).
    u_zm_mean = zonal_mean_masked(mean_u, u_mask)              # (n_lat, nlev)
    u_zm_last = zonal_mean_masked(u_last, u_mask)
    # v is on v-faces (lat+1 × n_lon): zonal mean stays at v-face latitudes.
    v_zm_mean = zonal_mean_masked(mean_v, v_mask)              # (n_lat+1, nlev)
    v_zm_last = zonal_mean_masked(v_last, v_mask)

    # Plot ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)

    def plot_panel(ax, lat, field, title, sym=True):
        if sym:
            vmax = np.nanmax(np.abs(field))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            levels = np.linspace(-vmax, vmax, 21)
            cmap = "RdBu_r"
        else:
            levels = 21
            cmap = "viridis"
        cf = ax.contourf(lat, z_full, field.T, levels=levels,
                         cmap=cmap, extend="both")
        cs = ax.contour(lat, z_full, field.T, levels=levels[::2],
                        colors="k", linewidths=0.4)
        try:
            ax.clabel(cs, fmt="%.2f", fontsize=6)
        except Exception:
            pass
        plt.colorbar(cf, ax=ax, fraction=0.04, label="m/s")
        ax.set_title(title)
        ax.set_xlabel("Latitude (deg)")
        ax.set_ylabel("Depth (m)")

    plot_panel(axes[0, 0], lat_deg,   u_zm_last,
               f"Zonal-mean u — restart day {int(days[-1])} (yr {days[-1]/365:.0f})")
    plot_panel(axes[0, 1], lat_deg,   u_zm_mean,
               f"Zonal-mean u — average of {n} restarts")
    plot_panel(axes[1, 0], lat_v_deg, v_zm_last,
               f"Zonal-mean v — restart day {int(days[-1])} (yr {days[-1]/365:.0f})")
    plot_panel(axes[1, 1], lat_v_deg, v_zm_mean,
               f"Zonal-mean v — average of {n} restarts")

    plt.suptitle("Global Overturning + GM/Redi 50yr — zonal-mean velocity sections",
                 y=1.00)
    plt.tight_layout()
    out = RUN_DIR / "zonal_mean_uv_50yr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    print(f"  u amplitude: last={np.nanmax(np.abs(u_zm_last)):.3f} m/s, "
          f"mean={np.nanmax(np.abs(u_zm_mean)):.3f} m/s")
    print(f"  v amplitude: last={np.nanmax(np.abs(v_zm_last)):.3f} m/s, "
          f"mean={np.nanmax(np.abs(v_zm_mean)):.3f} m/s")


if __name__ == "__main__":
    main()
