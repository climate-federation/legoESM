"""Meridional wavenumber spectrum of surface u — Mercator grid support.

Computes the same South Pacific spectrum as _jet_spectrum.py but handles
Mercator grids (non-uniform dy) by interpolating u onto a uniform
meridional grid before FFT.

Usage:
    python scripts/global_overturning/_jet_spectrum_mercator.py <exp_dir> [--n-lon 180]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.latlon import create_latlon_grid, create_mercator_grid
from legoesm import constants


# South Pacific box
SP_LAT_MIN, SP_LAT_MAX = -70.0, -10.0
SP_LON_MIN, SP_LON_MAX = 200.0, 280.0


def meridional_spectrum(restart_path, grid):
    """Compute meridional wavenumber spectrum of surface u in South Pacific.

    Handles both regular and Mercator grids by interpolating onto uniform
    meridional spacing when grid.dy is non-scalar.
    """
    d = np.load(restart_path)
    day = float(d["time_days"])
    mask = d["land_mask"] > 0.5
    u = d["u"]  # (n_lat, n_lon+1, nlev)

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))

    # Cell-centered surface u
    u_sfc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])  # (n_lat, n_lon)

    # Select South Pacific
    lat_idx = np.where((lat_deg >= SP_LAT_MIN) & (lat_deg <= SP_LAT_MAX))[0]
    lon_idx = np.where((lon_deg >= SP_LON_MIN) & (lon_deg <= SP_LON_MAX))[0]

    u_sp = u_sfc[np.ix_(lat_idx, lon_idx)]       # (nlat_sp, nlon_sp)
    mask_sp = mask[np.ix_(lat_idx, lon_idx)]
    lat_sp = lat_deg[lat_idx]

    # Zonal mean of u over ocean cells only at each latitude
    n_ocean = mask_sp.sum(axis=1)
    n_ocean_safe = np.maximum(n_ocean, 1)
    u_zm = (u_sp * mask_sp).sum(axis=1) / n_ocean_safe
    u_zm = np.where(n_ocean > 0, u_zm, np.nan)

    # For Mercator: interpolate onto uniform lat grid
    # Use the same number of points as original, uniform spacing
    lat_uniform = np.linspace(lat_sp[0], lat_sp[-1], len(lat_sp))
    dy_uniform = (lat_uniform[1] - lat_uniform[0]) * np.pi / 180.0 * constants.R_earth

    # Fill NaN gaps
    def _fill_nan(arr):
        valid = ~np.isnan(arr)
        if valid.sum() < 2 or valid.all():
            return np.where(np.isnan(arr), 0.0, arr)
        return np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])

    # Interpolate zonal-mean u from grid lats to uniform lats
    u_zm_filled = _fill_nan(u_zm)
    u_zm_uniform = np.interp(lat_uniform, lat_sp, u_zm_filled)

    nlat_sp = len(lat_uniform)
    u_zm_detrend = u_zm_uniform - np.nanmean(u_zm_uniform)
    window = np.hanning(nlat_sp)
    u_windowed = u_zm_detrend * window
    U_hat = np.fft.rfft(u_windowed)
    window_energy = np.sum(window**2)
    power_zm = np.abs(U_hat)**2 / window_energy
    freq = np.fft.rfftfreq(nlat_sp, d=abs(dy_uniform))
    wavelength_km = 1.0 / np.maximum(freq, 1e-20) / 1e3

    # --- Lon-averaged spectrum ---
    power_per_lon = []
    for j in range(len(lon_idx)):
        col = u_sp[:, j]
        col_mask = mask_sp[:, j]
        ocean_frac = col_mask.sum() / len(lat_idx)
        if ocean_frac < 0.8:
            continue
        col_filled = np.where(col_mask, col, np.nan)
        col_filled = _fill_nan(col_filled)
        # Interpolate to uniform lat grid
        col_uniform = np.interp(lat_uniform, lat_sp, col_filled)
        col_detrend = col_uniform - col_uniform.mean()
        col_windowed = col_detrend * window
        C_hat = np.fft.rfft(col_windowed)
        power_per_lon.append(np.abs(C_hat)**2 / window_energy)

    if power_per_lon:
        power_full = np.mean(power_per_lon, axis=0)
    else:
        power_full = power_zm

    return day, freq, wavelength_km, power_zm, power_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dirs", nargs="+",
                   help="Experiment directories under results/ocean/comparison_mpas_v_latlon/")
    p.add_argument("--n-lon", type=int, default=360)
    p.add_argument("--mercator", action="store_true")
    p.add_argument("--lat-max", type=float, default=80.0)
    args = p.parse_args()

    base = Path("results/ocean/comparison_mpas_v_latlon")

    if args.mercator:
        grid = create_mercator_grid(n_lon=args.n_lon, lat_max_deg=args.lat_max)
    else:
        n_lat = 180 * args.n_lon // 360
        grid = create_latlon_grid(n_lat, args.n_lon)

    print(f"Grid: {grid.n_lat}x{grid.n_lon}")

    # Compute dy for reference lines
    # For Mercator, use equatorial dy (largest cell)
    dy_arr = np.asarray(grid.dy) if hasattr(grid.dy, '__len__') and np.asarray(grid.dy).ndim > 0 else None
    if dy_arr is not None and len(dy_arr) > 1:
        # Mercator: use median dy in the South Pacific region
        lat_deg = np.degrees(np.asarray(grid.lat))
        sp_idx = np.where((lat_deg >= SP_LAT_MIN) & (lat_deg <= SP_LAT_MAX))[0]
        dy_ref_km = float(np.median(dy_arr[sp_idx])) / 1e3
    else:
        dy_ref_km = float(np.asarray(grid.dy).flat[0]) / 1e3 if np.asarray(grid.dy).size == 1 else float(np.asarray(grid.dlat)) * constants.R_earth / 1e3

    print(f"Reference dy in S. Pacific: {dy_ref_km:.0f} km")

    all_spectra = {}
    for exp_name in args.exp_dirs:
        d = base / exp_name / "restarts"
        if not d.exists():
            print(f"  {exp_name}: no restarts dir")
            continue
        restarts = sorted(d.glob("restart_day*.npz"))
        if not restarts:
            print(f"  {exp_name}: no restart files")
            continue

        # Pick: day ~30, ~60, ~90, ~120 (or latest)
        targets = [30, 60, 90, 120]
        selected = []
        for t in targets:
            best = None
            best_dist = 1e9
            for f in restarts:
                day_str = f.stem.replace("restart_day", "")
                day = int(day_str)
                if abs(day - t) < best_dist:
                    best_dist = abs(day - t)
                    best = f
            if best and best_dist < 20:
                selected.append(best)
        # Also include latest
        if restarts[-1] not in selected:
            selected.append(restarts[-1])

        spectra = []
        for f in selected:
            day, freq, wl, pwr_zm, pwr_full = meridional_spectrum(f, grid)
            spectra.append((day, freq, wl, pwr_full))
            pk = wl[1 + np.argmax(pwr_full[1:])]
            print(f"  {exp_name} day {day:.0f}: peak wavelength = {pk:.0f} km")
        all_spectra[exp_name] = spectra

    if not all_spectra:
        print("No spectra computed!")
        return

    # Plot
    n_exp = len(all_spectra)
    fig, axes = plt.subplots(n_exp, 1, figsize=(10, 4 * n_exp), squeeze=False)
    colors_time = ["C0", "C1", "C2", "C3", "C4", "C5"]

    for row, (name, spectra) in enumerate(all_spectra.items()):
        ax = axes[row, 0]
        for i, (day, freq, wl, pwr) in enumerate(spectra):
            c = colors_time[i % len(colors_time)]
            ax.loglog(wl[1:], pwr[1:], color=c, label=f"day {day:.0f}", lw=1.5)

        ax.axvline(2 * dy_ref_km, color="r", ls=":", alpha=0.5,
                   label=f"2Δy ≈ {2*dy_ref_km:.0f} km")
        ax.axvline(4 * dy_ref_km, color="orange", ls=":", alpha=0.5,
                   label=f"4Δy ≈ {4*dy_ref_km:.0f} km")
        ax.set_xlim(100, 40000)
        ax.set_ylabel("Power [(m/s)²]")
        ax.set_title(f"{name} — meridional spectrum of u (lon-avg, S. Pacific)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        ax.invert_xaxis()

    axes[-1, 0].set_xlabel("Wavelength (km)")
    plt.tight_layout()

    out = base / "jet_spectrum_mercator_comparison.png"
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
