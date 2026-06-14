"""Side-by-side T(y, z) plot at day 84 for the Phase 6 validation runs.

Top row: T(y, z) zonal mean for baseline / centered Redi-only / triad
Redi-only at day 84 (κ_Redi = 5×10⁴, β_S = 0).

Bottom row: difference from baseline for the two Redi-only runs (so the
triad column should be empty / round-off, while centered shows the
spurious cross-isopycnal mixing the plan was worried about).

Run after ``scripts/validate/validate_triad_redi_120day.py``.

Run with:
    python scripts/plot_triad_redi_day84.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    ("baseline_no_gmredi",         "Baseline (no GM/Redi)"),
    ("redi_only_centered_k5e4",    "Redi-only · centered"),
    ("redi_only_triads_k5e4",      "Redi-only · triads"),
]

DAY_TARGET = 84.0
RESULTS_ROOT = Path("results/triad_phase6_validation")
OUT_PNG = Path("results/triad_phase6_validation/T_yz_day84_comparison.png")


def _load(case: str) -> dict:
    f = RESULTS_ROOT / case / "snapshots_native.npz"
    if not f.exists():
        sys.exit(f"missing {f} — run scripts/validate/validate_triad_redi_120day.py first")
    return dict(np.load(f, allow_pickle=False))


def _pick_day(snap: dict, day: float) -> int:
    times = snap["times_days"]
    idx = int(np.argmin(np.abs(times - day)))
    return idx


def main() -> None:
    snaps = {c: _load(c) for c, _ in CASES}
    times = snaps[CASES[0][0]]["times_days"]
    k_day = _pick_day(snaps[CASES[0][0]], DAY_TARGET)
    actual_day = float(times[k_day])
    print(f"Plotting day {actual_day:.0f} (snapshot index {k_day})")

    # Geometry — pick from the first case (all share the same grid).
    s0 = snaps[CASES[0][0]]
    T_b = s0["T_3d"][k_day]   # (n_lat, n_lon, nlev)
    n_lat, n_lon, nlev = T_b.shape
    print(f"shape = {T_b.shape}")

    # Reconstruct y (latitude) and z (depth) axes from the test setup.
    # Eady channel: lat 16°-34°, with 1-cell wall on each side, n_lat = 22.
    # The vertical axis: try to read from any cross-section auxiliary file
    # if available, otherwise fall back to a uniform spacing.
    lat_deg = np.linspace(16.0, 34.0, n_lat)

    z_coord_path = (RESULTS_ROOT / CASES[0][0] / "results.txt")
    # Quick reconstruction: dz_surface=200, dz_deep=1000, H=5500, nlev=20.
    # Use create_ocean_z_star to get the actual z values.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=5500.0, dz_surface=200.0, dz_deep=1000.0,
    )
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full        # positive downward, in meters

    # Zonal-mean — pick the time-0 land mask (it doesn't change).  Mask
    # shape in the npz is (n_snap, n_lat, n_lon); broadcast with the
    # vertical axis when forming the 3-D weight.
    mask_full = s0.get("land_mask")
    if mask_full is None:
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
    else:
        mask = mask_full[k_day]   # (n_lat, n_lon)
    mask3 = mask[:, :, None]      # (n_lat, n_lon, 1) -> broadcasts over nlev
    mask_yz = (mask[:, :, None] * np.ones((1, 1, nlev))).sum(axis=1)
    safe = np.maximum(mask_yz, 1.0)

    # Compute zonal-mean T(y, z) for each case
    T_yz = {}
    for c, _ in CASES:
        T = snaps[c]["T_3d"][k_day]
        T_zonal = (T * mask3).sum(axis=1) / safe   # (n_lat, nlev), 0 on walls
        T_yz[c] = T_zonal

    # Differences from baseline
    base_yz = T_yz[CASES[0][0]]
    diff_yz = {c: T_yz[c] - base_yz for c, _ in CASES}

    # Figure: 2 rows × 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5),
                             constrained_layout=True,
                             gridspec_kw={"hspace": 0.0, "wspace": 0.0})

    # Row 0: absolute T(y, z)
    vmin = min(float(T_yz[c].min()) for c, _ in CASES)
    vmax = max(float(T_yz[c].max()) for c, _ in CASES)
    for col, (c, label) in enumerate(CASES):
        ax = axes[0, col]
        # Plot T (n_lat, nlev) -> imshow with y as latitude, z as depth
        # Mask out the wall rows for cleanliness.
        T_plot = np.ma.masked_where(mask_yz < 0.5, T_yz[c])
        im = ax.pcolormesh(
            lat_deg, depth, T_plot.T,    # transpose: rows = depth, cols = lat
            vmin=vmin, vmax=vmax, cmap="RdYlBu_r", shading="auto",
        )
        ax.invert_yaxis()
        ax.set_title(f"{label}\n(day {actual_day:.0f})", fontsize=11)
        if col == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Latitude (°N)")
        if col == len(CASES) - 1:
            cb = fig.colorbar(im, ax=axes[0, :].tolist(),
                              label="T (°C)", pad=0.01, shrink=0.85,
                              location="right")

    # Row 1: differences from baseline (col 0 is intentionally blank)
    # Use a symmetric colour scale set by the centered run (the larger
    # divergence) so triads vs centered are on the same scale.
    abs_diff_max = max(
        float(np.abs(diff_yz[c]).max())
        for c, _ in CASES if c != CASES[0][0]
    )
    print(f"Max |T - T_baseline| at day {actual_day:.0f}:")
    for c, _ in CASES:
        print(f"  {c:35s}  {float(np.abs(diff_yz[c]).max()):.4e} K  "
              f"RMS {float(np.sqrt(np.mean(diff_yz[c]**2))):.4e}")

    # Show baseline absolute label in col 0 for symmetry
    ax = axes[1, 0]
    ax.text(0.5, 0.5,
            "(reference)\n\n"
            f"max|ΔT| centered = {float(np.abs(diff_yz[CASES[1][0]]).max()):.2e} K\n"
            f"max|ΔT| triads   = {float(np.abs(diff_yz[CASES[2][0]]).max()):.2e} K\n\n"
            "Triads are at float64\nround-off; centered is\n10⁷× larger.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for col, (c, label) in enumerate(CASES[1:], start=1):
        ax = axes[1, col]
        D = np.ma.masked_where(mask_yz < 0.5, diff_yz[c])
        im = ax.pcolormesh(
            lat_deg, depth, D.T,
            vmin=-abs_diff_max, vmax=abs_diff_max,
            cmap="RdBu_r", shading="auto",
        )
        ax.invert_yaxis()
        ax.set_title(f"Δ from baseline · {label.split('·')[1].strip()}",
                     fontsize=11)
        ax.set_xlabel("Latitude (°N)")
        if col == 1:
            ax.set_ylabel("Depth (m)")
        if col == len(CASES) - 1:
            cb = fig.colorbar(im, ax=axes[1, 1:].tolist(),
                              label="ΔT (K)",
                              pad=0.01, shrink=0.85, location="right")

    fig.suptitle(
        "Phase 6 GM/Redi triad validation — Eady, κ_Redi = 5×10⁴, β_S = 0\n"
        "T(y, z) zonal mean at day 84",
        fontsize=12,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
