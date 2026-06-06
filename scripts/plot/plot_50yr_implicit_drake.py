#!/usr/bin/env python
"""Drake-Passage-focused progress plots for the implicit 50-yr re-run.

Pulls all available restarts from the spinup + continuation runs and
produces Drake-band-specific diagnostics:

  - drake_transport_timeseries.png: depth-integrated volume flux through
    the Drake band in Sv, one point per restart.  THE headline metric
    for the original ACC question.
  - drake_zonal_u_evolution.png: zonal-mean u(lat, depth) restricted to
    the Drake band (j=2..6, lat -77.5° to -57.5°), small multiples per
    restart year.
  - drake_T_evolution.png: zonal-mean T(lat, depth) in the Drake band,
    small multiples per restart year.
  - drake_u_profiles.png: band-averaged u(z) profiles, overlaid lines
    by year.

Re-runnable as more restarts land.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig


SPINUP_DIR = Path("results/ocean/global_overturning_implicit_spinup")
CONT_DIR = Path("results/ocean/global_overturning_50yr_implicit")
OUTPUT_DIR = CONT_DIR

J_DRAKE = np.arange(2, 7)            # lat-row indices for Drake band


def _gather_restarts():
    pairs = []
    for d in (SPINUP_DIR, CONT_DIR):
        for p in d.glob("restart_day*.npz"):
            day = int(p.stem.removeprefix("restart_day"))
            pairs.append((day, p))
    pairs.sort()
    seen = {day: p for day, p in pairs}
    return [(day, seen[day]) for day in sorted(seen)]


def main():
    pairs = _gather_restarts()
    if not pairs:
        print("No restarts found.")
        return
    print(f"Plotting Drake-band progress from {len(pairs)} restarts:")
    for day, p in pairs:
        print(f"  yr {day/365:5.2f}  ({p.parent.name})")

    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    lat = np.asarray(grid.lat) * 180 / np.pi
    R = float(grid.radius)
    cos_lat = np.cos(np.clip(np.asarray(grid.lat), -np.pi/2 + 1e-9,
                              np.pi/2 - 1e-9))
    dx_u = cos_lat * R * float(grid.dlon)
    dy = R * float(grid.dlat)
    drake_lats = lat[J_DRAKE]   # [-77.5, -72.5, -67.5, -62.5, -57.5]

    years, drake_T_Sv, zm_u_list, zm_T_list, u_prof_list = [], [], [], [], []

    for day, p in pairs:
        d = np.load(p, allow_pickle=False)
        years.append(day / 365.0)
        u = np.asarray(d["u"], dtype=np.float64)         # (36, 73, 20)
        T = np.asarray(d["T"], dtype=np.float64)         # (36, 72, 20)
        u_mask = np.asarray(d["u_mask"], dtype=np.float64)
        mask = np.asarray(d["land_mask"], dtype=np.float64)

        # Zonal-mean u in Drake-band rows (wet u-faces, depth-resolved)
        m = u_mask[:, :, None]
        den = m.sum(axis=1)
        zm_u = np.where(den > 0,
                        (u * m).sum(axis=1) / np.where(den > 0, den, 1.0),
                        np.nan)                          # (36, 20)
        zm_u_list.append(zm_u[J_DRAKE])                  # (5, 20)

        # Zonal-mean T in Drake-band cells
        T_masked = np.where(mask[:, :, None] > 0.5, T, np.nan)
        zm_T = np.nanmean(T_masked, axis=1)              # (36, 20)
        zm_T_list.append(zm_T[J_DRAKE])

        # Drake transport (Sv) — section transport at any single longitude
        # in the zonally-periodic channel.  T_section = ∫_band ∫_z ⟨u⟩(y,z) dy dz
        # where ⟨u⟩ is the zonal-mean u over wet u-faces.
        T_section_per_row = np.nansum(zm_u * dz[None, :], axis=1) * dy   # (36,)
        T_Sv = float(np.nansum(T_section_per_row[J_DRAKE])) / 1e6
        drake_T_Sv.append(T_Sv)

        # Band-mean u(z) profile (cosine-weighted average over Drake band)
        u_prof = np.average(zm_u[J_DRAKE], axis=0,
                            weights=cos_lat[J_DRAKE])
        u_prof_list.append(u_prof)

    years = np.array(years)
    drake_T_Sv = np.array(drake_T_Sv)

    # ---- Drake transport timeseries ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years, drake_T_Sv, "o-", color="C0", ms=7, lw=1.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(10, color="gray", lw=0.5, ls=":")
    ax.text(10.3, ax.get_ylim()[0], "spinup→cont",
             fontsize=8, va="bottom", color="gray")
    # Reference markers from previous experiments
    ax.axhline(-405, color="C3", lw=0.8, ls="--", alpha=0.6,
                label="old broken-solver yr-50 ($-405$ Sv)")
    ax.axhline(150, color="C2", lw=0.8, ls="--", alpha=0.6,
                label="real ACC ($\\sim$+150 Sv)")
    ax.set_xlabel("Sim year")
    ax.set_ylabel("Drake transport (Sv)\neastward = +")
    ax.set_title("Drake Passage transport — implicit-solver re-run")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_transport_timeseries.png", dpi=140)
    plt.close()
    print(f"\nSaved {OUTPUT_DIR / 'drake_transport_timeseries.png'}")

    # ---- Drake zonal-mean u(lat, depth) evolution ----
    n = len(years)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                              sharey=True)
    axes = np.atleast_2d(axes)
    vmax_u = float(np.nanmax(np.abs(np.array(zm_u_list))))
    if vmax_u == 0 or not np.isfinite(vmax_u):
        vmax_u = 0.1
    levels_u = np.linspace(-vmax_u, vmax_u, 21)
    u_step = max(0.02, np.round(vmax_u / 8, 2))
    u_levels = np.arange(-vmax_u, vmax_u + u_step / 2, u_step)
    u_levels = u_levels[u_levels != 0]
    for k, (yr, zmu) in enumerate(zip(years, zm_u_list)):
        i, j = k // cols, k % cols
        ax = axes[i, j]
        cf = ax.pcolormesh(drake_lats, z_full, zmu.T,
                           vmin=-vmax_u, vmax=vmax_u, cmap="RdBu_r",
                           shading="auto")
        cs = ax.contour(drake_lats, z_full, zmu.T,
                         levels=u_levels, colors="k", linewidths=0.4,
                         alpha=0.5)
        ax.contour(drake_lats, z_full, zmu.T,
                    levels=[0], colors="k", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.set_title(f"Yr {yr:.0f}  (T={drake_T_Sv[k]:+.0f} Sv)",
                      fontsize=10)
        ax.set_xlabel("Lat (°)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        plt.colorbar(cf, ax=ax, fraction=0.045)
    # Hide empty axes if any
    for k in range(n, rows * cols):
        i, j = k // cols, k % cols
        axes[i, j].axis("off")
    plt.suptitle(
        f"Drake-band zonal-mean u(lat, depth) — colour scale ±{vmax_u:.2f} m/s",
        y=1.0,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_zonal_u_evolution.png", dpi=130,
                bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_zonal_u_evolution.png'}")

    # ---- Drake zonal-mean T evolution ----
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                              sharey=True)
    axes = np.atleast_2d(axes)
    vmax_T = float(np.nanmax(np.array(zm_T_list)))
    vmin_T = float(np.nanmin(np.array(zm_T_list)))
    T_levels_drake = np.arange(np.floor(vmin_T), np.ceil(vmax_T) + 1, 1.0)
    for k, (yr, zmT) in enumerate(zip(years, zm_T_list)):
        i, j = k // cols, k % cols
        ax = axes[i, j]
        cf = ax.pcolormesh(drake_lats, z_full, zmT.T,
                           vmin=vmin_T, vmax=vmax_T, cmap="RdYlBu_r",
                           shading="auto")
        cs = ax.contour(drake_lats, z_full, zmT.T,
                         levels=T_levels_drake, colors="k",
                         linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.set_title(f"Yr {yr:.0f}", fontsize=10)
        ax.set_xlabel("Lat (°)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        plt.colorbar(cf, ax=ax, fraction=0.045)
    for k in range(n, rows * cols):
        i, j = k // cols, k % cols
        axes[i, j].axis("off")
    plt.suptitle("Drake-band zonal-mean T(lat, depth)", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_T_evolution.png", dpi=130,
                bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_T_evolution.png'}")

    # ---- Drake-band-mean u(z) profiles overlaid ----
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("plasma")
    for k, (yr, prof) in enumerate(zip(years, u_prof_list)):
        color = cmap(k / max(n - 1, 1))
        ax.plot(prof * 100, z_full, "-o", ms=3, color=color,
                 label=f"Yr {yr:.0f}")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("u (cm/s)  — eastward = +")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Drake-band-averaged u(z) — evolution over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drake_u_profiles.png", dpi=140)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'drake_u_profiles.png'}")

    # Print headline
    print(f"\n--- Drake transport summary ---")
    for k in range(n):
        print(f"  Yr {years[k]:5.1f}  T = {drake_T_Sv[k]:+8.1f} Sv")
    print(f"\nReference: old broken-solver yr-50 was -405 Sv (westward)")
    print(f"Real-world ACC is ~+150 Sv (eastward)")


if __name__ == "__main__":
    main()
