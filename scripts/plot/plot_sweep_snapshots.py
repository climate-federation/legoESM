#!/usr/bin/env python
"""Snapshot maps comparing two (or three) dissipation sweep runs."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def latest_restart(run_dir):
    """Return the latest restart file in a run directory."""
    restart_dir = Path(run_dir) / "latlon" / "180x360"
    files = sorted(restart_dir.glob("restart_day*.npz"))
    return files[-1] if files else None


def plot_run_snapshot(ax_row, d, label):
    """Plot SST, SSS, SSH, speed on a row of 4 axes."""
    mask = d["land_mask"] > 0.5
    T = d["T"][:, :, 0]
    S = d["S"][:, :, 0]
    eta = d["eta"]
    u = d["u"]
    v = d["v"]

    nlat, nlon = mask.shape

    # Surface speed on T-grid (handle staggered dims)
    u_sfc = u[:, :, 0] if u.ndim == 3 else u
    v_sfc = v[:, :, 0] if v.ndim == 3 else v
    # Interpolate u (nlat, nlon+1) -> (nlat, nlon) and v (nlat+1, nlon) -> (nlat, nlon)
    if u_sfc.shape[1] == nlon + 1:
        u_t = 0.5 * (u_sfc[:, :-1] + u_sfc[:, 1:])
    else:
        u_t = u_sfc[:, :nlon]
    if v_sfc.shape[0] == nlat + 1:
        v_t = 0.5 * (v_sfc[:-1, :] + v_sfc[1:, :])
    else:
        v_t = v_sfc[:nlat, :]
    speed = np.sqrt(u_t**2 + v_t**2)

    # Lat/lon for plotting
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon)

    fields = [
        (T, "SST [°C]", "RdYlBu_r", -2, 30),
        (S, "SSS [PSU]", "viridis", 30, 37),
        (eta, "SSH [m]", "RdBu_r", -1.5, 1.5),
        (speed, "Surface speed [m/s]", "inferno", 0, 1.0),
    ]

    for ax, (field, title, cmap, vmin, vmax) in zip(ax_row, fields):
        masked = np.where(mask, field, np.nan)
        im = ax.pcolormesh(lons, lats, masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_aspect("auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=9)


def main():
    runs = [
        (r"Baseline: A$_h$=2e5, K$_h$=1e3", "results/jra55_10yr_production"),
        (r"Run C: A$_h$=5e4, eq_boost=5", "results/sweep_C_eqboost5"),
        (r"Run D: A$_h$=3e4, eq_boost=7", "results/sweep_D_eqboost7"),
    ]

    # Filter to runs that have data
    valid_runs = []
    for label, path in runs:
        f = latest_restart(path)
        if f is not None:
            valid_runs.append((label, path, f))

    n = len(valid_runs)
    if n == 0:
        print("No data found!")
        return

    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (label, path, restart_file) in enumerate(valid_runs):
        d = np.load(restart_file)
        day = float(d["time_days"])
        plot_run_snapshot(axes[i], d, label)
        axes[i, 0].set_ylabel(f"{label}\nDay {day:.0f}", fontsize=10, rotation=0,
                              labelpad=120, va="center")

    fig.suptitle("Dissipation Sweep — Latest Snapshots", fontsize=14, y=1.01)
    fig.tight_layout()

    out = "results/sweep_snapshots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
