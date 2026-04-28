#!/usr/bin/env python
"""Progress plots for the in-flight implicit-solver 50-yr re-run.

Combines all available restarts from the spinup
(``results/ocean/global_overturning_implicit_spinup/``, years 0–10)
and the continuation
(``results/ocean/global_overturning_50yr_implicit/``, years 15+) into
a unified timeline and produces:

  - timeseries_progress.png: coarse scalar timeseries (max speed, mean
    SST, mean T, mean eta) — one point per restart.
  - snapshots_progress.png: SSH / SST / surface-speed maps at each
    available restart year, stacked vertically.
  - T_zonal_mean_progress.png: T(lat, depth) zonal-mean section per
    restart year, side-by-side panels.

Same layout as the original `global_overturning_50yr_gmredi/` plots
for direct visual comparison.

Re-runnable: pick up new restarts as they land.

Usage:
    python scripts/plot_50yr_implicit_progress.py
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


def _gather_restarts():
    """Collect (day, path) pairs from spinup + continuation, sorted."""
    pairs = []
    for d in (SPINUP_DIR, CONT_DIR):
        for p in d.glob("restart_day*.npz"):
            day = int(p.stem.removeprefix("restart_day"))
            pairs.append((day, p))
    pairs.sort()
    # Deduplicate (continuation rewrites day 3650 if present; keep continuation's)
    seen = {}
    for day, p in pairs:
        seen[day] = p
    return [(day, seen[day]) for day in sorted(seen)]


def _record_diagnostics(d):
    """Extract scalar diagnostics from a restart npz."""
    eta = d["eta"]; T = d["T"]; u = d["u"]; v = d["v"]
    mask = d["land_mask"]
    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed = np.sqrt(u_c**2 + v_c**2)
    ocean = mask > 0.5
    return {
        "max_speed": float(np.max(speed)),
        "mean_sst": float(np.mean(T[:, :, 0][ocean])) if ocean.any() else 0.0,
        "mean_T":   float(np.mean(T[ocean])) if ocean.any() else 0.0,
        "T_deep":   float(np.mean(T[:, :, -1][ocean])) if ocean.any() else 0.0,
        "mean_eta": float(np.mean(eta[ocean])) if ocean.any() else 0.0,
        "max_eta":  float(np.max(np.abs(eta[ocean]))) if ocean.any() else 0.0,
    }


def main():
    pairs = _gather_restarts()
    if not pairs:
        print("No restarts found.")
        return
    print(f"Found {len(pairs)} restarts:")
    for day, p in pairs:
        print(f"  day {day:>6}  ({day/365:5.2f} yr)  -- {p.parent.name}")

    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    diags = []
    snapshots = []
    for day, p in pairs:
        d = np.load(p, allow_pickle=False)
        rec = _record_diagnostics(d)
        rec["day"] = day
        diags.append(rec)
        # Snapshot fields
        eta = d["eta"]; T = d["T"]; u = d["u"]; v = d["v"]
        mask = d["land_mask"]
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        speed_sfc = np.sqrt(u_c**2 + v_c**2)
        ocean = mask > 0.5
        snapshots.append({
            "day": day,
            "year": day / 365,
            "eta": np.where(ocean, eta, np.nan),
            "SST": np.where(ocean, T[:, :, 0], np.nan),
            "speed_sfc": np.where(ocean, speed_sfc, np.nan),
            "T_zonal_mean": np.nanmean(np.where(ocean[:, :, None], T, np.nan),
                                        axis=1),
        })

    # ---- Timeseries ----
    days = np.array([r["day"] for r in diags])
    years = days / 365.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(years, [r["max_speed"] for r in diags], "o-", color="C0")
    axes[0].set_ylabel("Max surface speed (m/s)")
    axes[0].set_title("50-yr implicit-solver re-run — progress (one point per restart)")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(10, color="gray", lw=0.5, ls=":")  # spinup-end marker
    axes[0].text(10, axes[0].get_ylim()[0], "spinup→cont", fontsize=8,
                  va="bottom", color="gray")

    axes[1].plot(years, [r["mean_sst"] for r in diags], "o-", color="C3",
                 label="SST")
    axes[1].set_ylabel("Mean SST (°C)")
    axes[1].grid(alpha=0.3); axes[1].axvline(10, color="gray", lw=0.5, ls=":")

    axes[2].plot(years, [r["mean_T"] for r in diags], "o-", label="All-depth")
    axes[2].plot(years, [r["T_deep"] for r in diags], "s-", label="Bottom layer")
    axes[2].set_ylabel("Mean T (°C)")
    axes[2].legend()
    axes[2].grid(alpha=0.3); axes[2].axvline(10, color="gray", lw=0.5, ls=":")

    axes[3].plot(years, [r["mean_eta"] for r in diags], "o-", color="C2",
                 label="mean η")
    axes[3].plot(years, [r["max_eta"] for r in diags], "^-", color="C4",
                 label="max |η|")
    axes[3].set_ylabel("η (m)")
    axes[3].set_xlabel("Sim year")
    axes[3].legend()
    axes[3].grid(alpha=0.3); axes[3].axvline(10, color="gray", lw=0.5, ls=":")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "timeseries_progress.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'timeseries_progress.png'}")

    # ---- Snapshot maps ----
    n = len(snapshots)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, s in enumerate(snapshots):
        for j, (key, label, cmap, sym) in enumerate([
            ("eta", "SSH (m)", "RdBu_r", True),
            ("SST", "SST (°C)", "RdYlBu_r", False),
            ("speed_sfc", "Surface speed (m/s)", "magma", False),
        ]):
            ax = axes[i, j]
            field = s[key]
            if sym:
                vmax = float(np.nanmax(np.abs(field)))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1.0
                im = ax.pcolormesh(lon, lat, field, cmap=cmap,
                                    vmin=-vmax, vmax=vmax, shading="auto")
            else:
                im = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.046)
            if i == 0:
                ax.set_title(label)
            ax.set_ylabel(f"Yr {s['year']:.0f}")
    plt.suptitle("50-yr implicit-solver re-run — snapshot evolution",
                  y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snapshots_progress.png", dpi=110)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'snapshots_progress.png'}")

    # ---- T zonal-mean evolution ----
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for i, s in enumerate(snapshots):
        T_zm = s["T_zonal_mean"]
        im = axes[i].pcolormesh(lat, z_full, T_zm.T,
                                 cmap="RdYlBu_r", shading="auto",
                                 vmin=0, vmax=22)
        plt.colorbar(im, ax=axes[i], fraction=0.046,
                     label="T (°C)" if i == n - 1 else None)
        axes[i].set_title(f"Yr {s['year']:.0f}")
        axes[i].set_xlabel("Latitude (°)")
        if i == 0:
            axes[i].set_ylabel("Depth (m)")
    plt.suptitle("Zonal-mean T(lat, depth) evolution — implicit solver",
                  y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "T_zonal_mean_progress.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'T_zonal_mean_progress.png'}")

    # Print summary
    print("\n--- Diagnostics summary ---")
    print(f"{'Yr':>5} {'max|u|':>8} {'SST':>7} {'meanT':>7} "
          f"{'T_deep':>7} {'meanη':>9} {'max|η|':>8}")
    for r in diags:
        print(f"{r['day']/365:>5.1f} {r['max_speed']:>8.3f} "
              f"{r['mean_sst']:>7.2f} {r['mean_T']:>7.2f} "
              f"{r['T_deep']:>7.2f} {r['mean_eta']:>+9.3e} "
              f"{r['max_eta']:>8.3f}")


if __name__ == "__main__":
    main()
