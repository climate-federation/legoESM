"""Meridional wavenumber spectrum of surface u in the South Pacific.

Computes the 1D power spectrum of zonal-mean surface u as a function
of meridional wavenumber, for selected restarts. This shows which
meridional scales (jet spacing) carry the most energy.

South Pacific region: 60°S-10°S, 150°E-280°E

Usage:
    python scripts/tmp/_jet_spectrum.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.latlon import create_latlon_grid
from legoesm import constants


# South Pacific box (wide latitude range for good spectral resolution)
SP_LAT_MIN, SP_LAT_MAX = -70.0, -10.0
SP_LON_MIN, SP_LON_MAX = 200.0, 280.0


def meridional_spectrum(restart_path, grid):
    """Compute meridional wavenumber spectrum of surface u in South Pacific.

    Land handling:
    - Zonal-mean u: computed from ocean cells only (land excluded)
    - Per-longitude spectra: only longitudes with >80% ocean coverage
      are included. At each such longitude, any remaining land cells
      are set to NaN and then linearly interpolated before windowing.
    - Hann window applied to reduce spectral leakage from box edges.
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

    # Zonal mean of u over ocean cells only at each latitude
    n_ocean = mask_sp.sum(axis=1)
    n_ocean_safe = np.maximum(n_ocean, 1)
    u_zm = (u_sp * mask_sp).sum(axis=1) / n_ocean_safe  # (nlat_sp,)
    # Mark latitudes with no ocean as NaN
    u_zm = np.where(n_ocean > 0, u_zm, np.nan)

    nlat_sp = len(lat_idx)

    # Meridional spacing in meters
    dy = float(np.asarray(grid.lat[1] - grid.lat[0])) * constants.R_earth  # m

    # Helper: interpolate NaN gaps linearly
    def _fill_nan(arr):
        valid = ~np.isnan(arr)
        if valid.sum() < 2 or valid.all():
            return np.where(np.isnan(arr), 0.0, arr)
        return np.interp(np.arange(len(arr)),
                         np.where(valid)[0], arr[valid])

    # --- Spectrum of zonal-mean u ---
    u_zm_filled = _fill_nan(u_zm)
    u_zm_detrend = u_zm_filled - np.nanmean(u_zm_filled)
    window = np.hanning(nlat_sp)
    u_windowed = u_zm_detrend * window
    U_hat = np.fft.rfft(u_windowed)
    # Normalize by window energy for proper power estimate
    window_energy = np.sum(window**2)
    power_zm = np.abs(U_hat)**2 / window_energy
    # Wavenumber [cycles/m] and wavelength [km]
    freq = np.fft.rfftfreq(nlat_sp, d=dy)  # cycles/m
    wavelength_km = 1.0 / np.maximum(freq, 1e-20) / 1e3  # km

    # --- Spectrum of u at each longitude, then average ---
    # Only use longitudes with >80% ocean coverage
    power_per_lon = []
    for j in range(len(lon_idx)):
        col = u_sp[:, j]
        col_mask = mask_sp[:, j]
        ocean_frac = col_mask.sum() / nlat_sp
        if ocean_frac < 0.8:
            continue
        # Interpolate over any remaining land cells
        col_filled = np.where(col_mask, col, np.nan)
        col_filled = _fill_nan(col_filled)
        col_detrend = col_filled - col_filled.mean()
        col_windowed = col_detrend * window
        C_hat = np.fft.rfft(col_windowed)
        power_per_lon.append(np.abs(C_hat)**2 / window_energy)

    if power_per_lon:
        power_full = np.mean(power_per_lon, axis=0)
    else:
        power_full = power_zm

    return day, freq, wavelength_km, power_zm, power_full


def main():
    grid = create_latlon_grid(180, 360)
    base = Path("results/ocean/comparison_mpas_v_latlon")

    # Select specific snapshots at different times for each experiment
    experiments = {}
    for tag in ["F1_flat_lap", "F2_flat_bih", "F3_flat_bih_floor",
                 "F4_flat_bih_Ah1e4", "F5_flat_production", "e4_fp64"]:
        d = base / f"latlon_{tag}" / "restarts"
        if d.exists():
            restarts = sorted(d.glob("restart_day*.npz"))
            # Pick: day ~30, ~90, ~180, ~365 (or latest)
            selected = []
            targets = [30, 90, 180, 365]
            for t in targets:
                best = None
                best_dist = 1e9
                for f in restarts:
                    day_str = f.stem.replace("restart_day", "")
                    day = int(day_str)
                    if abs(day - t) < best_dist:
                        best_dist = abs(day - t)
                        best = f
                if best and best_dist < 30:
                    selected.append(best)
            experiments[tag] = selected

    print(f"Found {len(experiments)} experiments")

    # Compute spectra
    all_spectra = {}
    for name, files in experiments.items():
        print(f"\nProcessing {name}...")
        spectra = []
        for f in files:
            day, freq, wl, pwr_zm, pwr_full = meridional_spectrum(f, grid)
            spectra.append((day, freq, wl, pwr_full))
            print(f"  day {day:.0f}: peak wavelength = "
                  f"{wl[1+np.argmax(pwr_full[1:])]:.0f} km")
        all_spectra[name] = spectra

    # Plot: one panel per experiment, lon-averaged spectrum only
    colors_time = ["C0", "C1", "C2", "C3"]
    labels_exp = {
        "F1_flat_lap": "F1: flat, A_h=1e4, Csmag=0.33",
        "F2_flat_bih": "F2: flat, A_h=0, B_h=1e11 (blew up day 40)",
        "F3_flat_bih_floor": "F3: flat, A_h=1e3, B_h=1e11 (blew up day 49)",
        "F4_flat_bih_Ah1e4": "F4: flat, A_h=1e4, B_h=1e11 (blew up day 64)",
        "F5_flat_production": "F5: flat, A_h=2e5+latscale, B_h=5e9 (blew up day 231)",
        "e4_fp64": "baseline: ETOPO, A_h=1e4, Csmag=0.33",
    }

    n_exp = len(all_spectra)
    fig, axes = plt.subplots(n_exp, 1, figsize=(10, 4 * n_exp))
    if n_exp == 1:
        axes = [axes]

    dy_km = float(np.asarray(grid.lat[1] - grid.lat[0])) * constants.R_earth / 1e3

    for row, (name, spectra) in enumerate(all_spectra.items()):
        ax = axes[row]
        for i, (day, freq, wl, pwr) in enumerate(spectra):
            c = colors_time[i % len(colors_time)]
            ax.loglog(wl[1:], pwr[1:], color=c, label=f"day {day:.0f}", lw=1.5)

        ax.axvline(2 * dy_km, color="r", ls=":", alpha=0.5,
                   label=f"2Δy = {2*dy_km:.0f} km")
        ax.axvline(4 * dy_km, color="orange", ls=":", alpha=0.5,
                   label=f"4Δy = {4*dy_km:.0f} km")
        ax.set_xlim(100, 20000)
        ax.set_ylabel("Power [(m/s)²]")
        ax.set_title(f"{labels_exp.get(name, name)} — meridional spectrum of u "
                     f"(lon-averaged, S. Pacific)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        ax.invert_xaxis()

    axes[-1].set_xlabel("Wavelength (km)")

    plt.tight_layout()
    out = base / "jet_spectrum_south_pacific.png"
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
