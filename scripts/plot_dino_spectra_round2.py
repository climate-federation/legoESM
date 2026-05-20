"""Drake Passage meridional spectra comparison — DINO dissipation sensitivity Round 2.

Reproduces the same layout as ``drake_spectra_comparison.png`` (Round 1):
  - 3 rows × 2 cols: u (left), v (right) at 3 depth levels
  - x-axis: wavelength (km), large wavelengths on right
  - y-axis: power [(m/s)^2]
  - Reference lines for 2dy (Nyquist) and Rossby deformation radius
  - Initial state (yr 100) as blue dashed; experiments as solid colours

Usage:
    python scripts/plot_dino_spectra_round2.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- Config ---
SENS_DIR = Path("results/dino_sensitivity")
OUT_PATH = SENS_DIR / "spectra_round2.png"

EXPERIMENTS = {
    "A_biharm_1e14": {"label": r"B$_h$=1e14",           "color": "C0"},
    "B_smag_lap_1":  {"label": "Smag Lap C=1",           "color": "C1"},
    "C_smag_lap_2":  {"label": "Smag Lap C=2",           "color": "C2"},
    "D_ah_2x":       {"label": r"A$_h$ 2× (floor=30k)",  "color": "C3"},
    "E_biharm_smag": {"label": r"B$_h$+Smag combo",      "color": "C4"},
}

# Depth levels to plot (index into snapshot's 3rd axis)
DEPTH_LEVELS = [
    (0,  "5"),      # k=0  → z ≈ 5 m
    (10, "133"),     # k=10 → z ≈ 133 m
    (35, "3773"),    # k=35 → z ≈ 3773 m
]

# Grid params
N_LON = 50
N_LAT = 198
DLON_DEG = 50.0 / N_LON  # 1.0°
R_KM = 6371.229

# Drake Passage
DRAKE_LAT_S = -65.0
DRAKE_LAT_N = -45.0


def mercator_lats():
    j = np.arange(-(N_LAT // 2) + 0.5, N_LAT // 2 + 0.5)
    return (180.0 / math.pi) * np.arcsin(
        np.tanh(DLON_DEG * math.pi / 180.0 * j))


def drake_indices(lat_deg):
    return np.where((lat_deg >= DRAKE_LAT_S) & (lat_deg <= DRAKE_LAT_N))[0]


def meridional_power_spectrum(field_2d, mean_dy_km):
    """Meridional power spectrum averaged over longitude columns.

    Parameters
    ----------
    field_2d : (n_drake_lat, n_lon) — velocity at one depth
    mean_dy_km : mean meridional grid spacing in km

    Returns
    -------
    wavelength_km : 1D array (positive frequencies, excluding DC),
                    sorted large → small
    power : 1D array, mean |FFT|^2 across columns
    """
    n_lat, n_lon = field_2d.shape
    # Remove meridional mean from each column
    field_anom = field_2d - field_2d.mean(axis=0, keepdims=True)
    # FFT along latitude (axis 0)
    fft_vals = np.fft.rfft(field_anom, axis=0)
    # One-sided power, skip DC
    power = (2.0 / n_lat) * np.abs(fft_vals[1:, :]) ** 2
    # Average over longitude columns
    power_mean = power.mean(axis=1)
    # Wavenumber → wavelength
    freqs = np.arange(1, power_mean.size + 1)
    wavelength_km = (n_lat * mean_dy_km) / freqs  # L_total / k
    return wavelength_km, power_mean


def load_snapshot(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def interp_u_to_T(u_3d, k):
    """u on u-faces (n_lat, n_lon+1, nlev) → T-points at depth k."""
    return 0.5 * (u_3d[:, :-1, k] + u_3d[:, 1:, k])


def interp_v_to_T(v_3d, k):
    """v on v-faces (n_lat+1, n_lon, nlev) → T-points at depth k."""
    return 0.5 * (v_3d[:-1, :, k] + v_3d[1:, :, k])


def main():
    lat_deg = mercator_lats()
    di = drake_indices(lat_deg)
    lat_drake = lat_deg[di]
    n_drake = len(di)

    # Mean dy in Drake band
    dy_km = R_KM * np.cos(np.radians(lat_drake)) * DLON_DEG * np.pi / 180.0
    mean_dy = dy_km.mean()
    two_dy = 2.0 * mean_dy
    print(f"Drake Passage: {n_drake} rows, mean dy = {mean_dy:.1f} km, "
          f"2dy = {two_dy:.0f} km")

    # --- Load initial state (year-100 restart) ---
    restart_path = Path("results/dino_100yr_latlon/snapshots/snapshot_00200.npz")
    if restart_path.exists():
        init_snap = load_snapshot(restart_path)
        init_label = "initial (yr 100)"
    else:
        first_exp = list(EXPERIMENTS.keys())[0]
        init_snap = load_snapshot(
            SENS_DIR / first_exp / "snapshots" / "snapshot_001.npz")
        init_label = "initial (yr 101)"
    land = init_snap["land_mask"] < 0.5  # True = land

    # --- Load final snapshots ---
    exp_data = {}
    for name, info in EXPERIMENTS.items():
        snap_dir = SENS_DIR / name / "snapshots"
        snap_files = sorted(snap_dir.glob("snapshot_*.npz"))
        if not snap_files:
            print(f"  Skipping {name}: no snapshots")
            continue
        snap = load_snapshot(snap_files[-1])
        day = float(snap["time_days"])
        exp_data[name] = {"snap": snap, "day": day, **info}

    # --- Plot: 3 rows (depths) × 2 cols (u, v) ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    fig.suptitle(
        f"DINO — Drake Passage meridional spectra: initial (yr 100) "
        f"vs 5yr with different dissipation\n"
        f"2dy = {two_dy:.0f} km",
        fontsize=14)

    for row, (k_level, z_label) in enumerate(DEPTH_LEVELS):
        for col, (var, interp_fn) in enumerate([
            ("u", interp_u_to_T), ("v", interp_v_to_T)
        ]):
            ax = axes[row, col]

            # Initial state spectrum
            field_init = interp_fn(init_snap[var], k_level)
            # Zero out land
            field_drake = np.where(land[di], 0.0, field_init[di])
            wl0, pw0 = meridional_power_spectrum(field_drake, mean_dy)

            ax.loglog(wl0, pw0, "b--", lw=2.0, alpha=0.8,
                      label=init_label, zorder=10)

            # Each experiment
            for name, ed in exp_data.items():
                snap = ed["snap"]
                field_f = interp_fn(snap[var], k_level)
                field_drake_f = np.where(land[di], 0.0, field_f[di])
                wl_f, pw_f = meridional_power_spectrum(field_drake_f, mean_dy)
                yr = ed["day"] / 365.0
                ax.loglog(wl_f, pw_f, "-", lw=1.5, color=ed["color"],
                          label=f"{ed['label']} (yr {yr:.0f})")

            # Reference lines
            ax.axvline(two_dy, color="red", ls=":", alpha=0.6, lw=1)
            # Rossby deformation radius ~30–50 km at these latitudes
            ax.axvline(300, color="orange", ls="-.", alpha=0.4, lw=1)

            ax.set_xlabel("Wavelength (km)")
            ax.set_ylabel("Power [(m/s)^2]")
            ax.set_title(f"Drake {var} — z = {z_label} m")
            ax.grid(True, alpha=0.2, which="both")
            if row == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
