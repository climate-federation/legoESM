#!/usr/bin/env python
"""Diagnose where the slow-growth instability lives.

Reads day-10/20/30 restarts and reports:
  - location of the top-N strongest u-cells at each time
  - depth-mean and surface |u| maps with the hotspots overplotted
  - growth factor in |u|(j,i,k) between consecutive snapshots
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results/etopo_3month/latlon/180x360")

N_LAT, N_LON, N_LEV = 180, 360, 20
DZ_SURFACE, DZ_DEEP, H_MAX = 20.0, 500.0, 5000.0


def _build_z():
    sigma = np.linspace(0.0, 1.0, N_LEV + 1)
    A = (DZ_DEEP - DZ_SURFACE) / 2
    B = (DZ_DEEP + DZ_SURFACE) / 2
    dz = B + A * np.tanh((sigma[:-1] - 0.5) * 4)
    dz = dz / dz.sum() * H_MAX
    z_half = np.concatenate(([0.0], -np.cumsum(dz)))
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, z_half, dz


def _make_lonlat():
    lat = np.linspace(-89.5, 89.5, N_LAT)
    lon = np.linspace(0.5, 359.5, N_LON)
    return lat, lon


def _save(fig, name):
    out = RESULTS / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _load(day):
    fn = RESULTS / f"restart_day{int(day):06d}.npz"
    if not fn.exists():
        return None
    return dict(np.load(fn))


def _topN_u(u, N=20):
    """Return list of (k, j, i, |u|) for the N largest |u|.

    u shape is (n_lat, n_lon+1, n_lev) — axis order (j, i, k).
    """
    absu = np.abs(u)
    flat = absu.ravel()
    idx = np.argpartition(flat, -N)[-N:]
    idx = idx[np.argsort(-flat[idx])]
    out = []
    for f in idx:
        j, i, k = np.unravel_index(f, absu.shape)
        out.append((int(k), int(j), int(i), float(absu[j, i, k])))
    return out


def main():
    z_full, z_half, dz = _build_z()
    lat, lon = _make_lonlat()
    print(f"z_full (cell centers, depth m):")
    for k, z in enumerate(z_full):
        print(f"  k={k:2d}  z = {z:8.1f}  dz = {dz[k]:.1f}")
    print()

    states = {}
    for d in [30, 60]:
        st = _load(d)
        if st is not None:
            states[d] = st
    if not states:
        print("No restart files found")
        return 1

    # Per-day: locations of top N u-cells, plus per-level max|u|
    print("=" * 110)
    print("Top-20 |u| cells per snapshot")
    print("=" * 110)
    for d, st in states.items():
        H = st["H_bathy"]
        u = st["u"]
        print(f"\n--- Day {d} ---")
        print(f"  global max|u| = {float(np.max(np.abs(u))):.4f} m/s")
        topn = _topN_u(u, 20)
        print(f"  {'k':>3} {'depth':>7} {'j':>4} {'i':>4} "
              f"{'lat':>7} {'lon':>7} {'H_bathy':>8} {'|u|':>8}")
        for (k, j, i, val) in topn:
            lat_d = float(lat[j])
            i_lon = i % N_LON
            lon_d = float(lon[i_lon])
            H_d = float(H[j, i_lon])
            z_d = float(z_full[k])
            print(f"  {k:>3d} {z_d:>7.1f} {j:>4d} {i:>4d} "
                  f"{lat_d:>7.2f} {lon_d:>7.2f} {H_d:>8.1f} {val:>8.4f}")

        # Per-level max|u|
        print(f"  per-level max|u| and location:")
        print(f"  {'k':>3} {'depth':>7}  {'max|u|':>9}  {'lat':>7} {'lon':>7}")
        for k in range(N_LEV):
            uk = np.abs(u[..., k])
            j, i = np.unravel_index(np.argmax(uk), uk.shape)
            print(f"  {k:>3d} {z_full[k]:>7.1f}  {float(uk[j, i]):>9.4f}  "
                  f"{lat[j]:>7.2f} {lon[i % N_LON]:>7.2f}")

    # ---- Plots: surface speed at each day with hotspots marked ----
    days = sorted(states.keys())
    fig, axes = plt.subplots(len(days), 1, figsize=(11, 4.5 * len(days)))
    if len(days) == 1:
        axes = [axes]
    for ax, d in zip(axes, days):
        st = states[d]
        H = st["H_bathy"]; land = st["land_mask"]
        ocean = land > 0.5
        u = st["u"]; v = st["v"]
        u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
        v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        # depth-max speed (worst case across column)
        speed = np.sqrt(u_c**2 + v_c**2)
        sp_max = np.where(ocean, np.max(speed, axis=-1), np.nan)
        vmax = max(0.1, float(np.nanpercentile(sp_max, 99)))
        im = ax.pcolormesh(lon, lat, sp_max, cmap="magma",
                           shading="auto", vmin=0, vmax=vmax)
        # Mark top-20 |u| cells
        topn = _topN_u(u, 20)
        for (k, j, i, val) in topn:
            ax.scatter(lon[i % N_LON], lat[j], s=60, marker="o",
                        facecolors="none", edgecolors="cyan", linewidths=1.2)
        ax.set_title(f"Depth-max speed at day {d}  (max plotted: {vmax:.2f} m/s)\n"
                     f"cyan circles: top-20 |u| cells")
        ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
        plt.colorbar(im, ax=ax, label="|u| [m/s]", fraction=0.04)
    fig.tight_layout()
    _save(fig, "diag_speed_with_hotspots.png")

    # ---- Growth pattern: where did |u| grow most? ----
    if 30 in states and 60 in states:
        u10 = np.abs(states[30]["u"])
        u30 = np.abs(states[60]["u"])
        # depth-max
        s10 = np.max(u10, axis=-1)
        s30 = np.max(u30, axis=-1)
        # Avoid /0
        ratio = s30 / np.maximum(s10, 1e-3)
        ocean = states[30]["land_mask"] > 0.5
        # speed at u-points; trim to cell-center shape
        s30_c = 0.5 * (s30[:, :-1] + s30[:, 1:])
        ratio_c = 0.5 * (ratio[:, :-1] + ratio[:, 1:])
        s30_c_m = np.where(ocean, s30_c, np.nan)
        ratio_c_m = np.where(ocean, ratio_c, np.nan)

        fig, axes = plt.subplots(2, 1, figsize=(11, 9))

        ax = axes[0]
        im = ax.pcolormesh(lon, lat, s30_c_m, cmap="magma",
                           shading="auto",
                           vmin=0, vmax=float(np.nanpercentile(s30_c_m, 99)))
        ax.set_title(f"Depth-max |u| at day 30  [m/s]")
        ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
        plt.colorbar(im, ax=ax, label="|u| [m/s]", fraction=0.04)

        ax = axes[1]
        im = ax.pcolormesh(lon, lat, ratio_c_m, cmap="hot",
                           shading="auto", vmin=1, vmax=100)
        ax.set_title("Growth factor: depth-max |u|(day 30) / |u|(day 10)")
        ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
        plt.colorbar(im, ax=ax, label="ratio", fraction=0.04)

        fig.tight_layout()
        _save(fig, "diag_growth_pattern.png")

        # Print top-20 growth-factor locations
        print("\n" + "=" * 110)
        print("Top-20 GROWTH FACTOR locations  (depth-max |u| ratio day30/day10)")
        print("=" * 110)
        ratio_full = np.where(np.isfinite(ratio_c), ratio_c, 0.0)
        ratio_full = np.where(ocean, ratio_full, 0.0)
        flat = ratio_full.ravel()
        idx = np.argpartition(flat, -20)[-20:]
        idx = idx[np.argsort(-flat[idx])]
        H30 = states[30]["H_bathy"]
        print(f"  {'j':>4} {'i':>4} {'lat':>7} {'lon':>7} {'H_bathy':>8} "
              f"{'|u|@10':>9} {'|u|@30':>9} {'ratio':>7}")
        for f in idx:
            j, i = np.unravel_index(f, ratio_c.shape)
            print(f"  {int(j):>4d} {int(i):>4d} {lat[j]:>7.2f} {lon[i]:>7.2f} "
                  f"{float(H30[j, i]):>8.1f} "
                  f"{float(s10[j, i]):>9.4f} "
                  f"{float(s30[j, i]):>9.4f} "
                  f"{float(ratio_c[j, i]):>7.1f}×")

    return 0


if __name__ == "__main__":
    sys.exit(main())
