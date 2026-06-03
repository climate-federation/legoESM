#!/usr/bin/env python
"""Plot standard diagnostics from a completed AMIP run.

Usage:
    python scripts/plot_amip.py results/amip/C16_L40_200d_*/
    python scripts/plot_amip.py results/amip/C16_L40_200d_*/ --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np


def load_timeseries(run_dir: Path) -> dict[str, np.ndarray]:
    """Load timeseries.npz, falling back to incremental chunks."""
    ts_path = run_dir / "timeseries.npz"
    if ts_path.exists():
        return dict(np.load(ts_path))

    # Try incremental chunks
    incr_dir = run_dir / "timeseries_incremental"
    if incr_dir.is_dir():
        chunks = sorted(incr_dir.glob("chunk_*.npz"))
        if chunks:
            merged: dict[str, list[np.ndarray]] = {}
            for c in chunks:
                data = np.load(c)
                for k in data.files:
                    merged.setdefault(k, []).append(data[k])
            return {k: np.concatenate(v) for k, v in merged.items()}

    print(f"ERROR: no timeseries.npz found in {run_dir}", file=sys.stderr)
    sys.exit(1)


def plot_scalar_timeseries(data: dict[str, np.ndarray], ax_grid):
    """Panel 1: key scalar timeseries."""
    days = data["days"]
    panels = [
        ("T_atm", "T_atm [K]"),
        ("T_low", "T_low [K]"),
        ("sst", "SST [K]"),
        ("CWV", "CWV [kg/m$^2$]"),
        ("precip", "Precip [mm/day]"),
        ("max_wind", "Max wind [m/s]"),
    ]
    for ax, (key, label) in zip(ax_grid, panels):
        if key in data and data[key].size > 0:
            ax.plot(days, data[key], "k-", linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    ax_grid[-1].set_xlabel("Day")


def plot_energy_budget(data: dict[str, np.ndarray], ax_grid):
    """Panel 2: TOA/surface radiation and energy budget."""
    days = data["days"]
    rad_keys = [
        ("sw_up_toa", "SW$\\uparrow$ TOA"),
        ("lw_up_toa", "LW$\\uparrow$ TOA"),
        ("sw_net_sfc", "SW net sfc"),
        ("lw_net_sfc", "LW net sfc"),
    ]
    ax_rad = ax_grid[0]
    for key, label in rad_keys:
        if key in data and data[key].size > 0:
            ax_rad.plot(days, data[key], linewidth=0.8, label=label)
    ax_rad.set_ylabel("Flux [W/m$^2$]", fontsize=8)
    ax_rad.legend(fontsize=6, ncol=2)
    ax_rad.set_title("Radiation fluxes", fontsize=9)
    ax_rad.tick_params(labelsize=7)

    # Net TOA
    ax_net = ax_grid[1]
    if "energy_toa_net" in data and data["energy_toa_net"].size > 0:
        ax_net.plot(days, data["energy_toa_net"], "k-", linewidth=0.8)
        ax_net.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_net.set_ylabel("Net TOA [W/m$^2$]", fontsize=8)
    ax_net.set_title("TOA energy imbalance", fontsize=9)
    ax_net.tick_params(labelsize=7)
    ax_net.set_xlabel("Day")


def plot_conservation(data: dict[str, np.ndarray], ax_grid):
    """Panel 3: conservation checks."""
    days = data["days"]
    checks = [
        ("dry_mass_ps", "Sfc pressure [Pa]"),
        ("moisture_residual", "Moisture residual [mm/day]"),
        ("energy_residual", "Energy residual [W/m$^2$]"),
    ]
    for ax, (key, label) in zip(ax_grid, checks):
        if key in data and data[key].size > 0:
            vals = data[key]
            if key == "dry_mass_ps":
                # Show anomaly relative to initial value
                vals = vals - vals[0]
                label = "$\\Delta$ Ps [Pa]"
            ax.plot(days, vals, "k-", linewidth=0.8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    ax_grid[-1].set_xlabel("Day")


def plot_vertical_profiles(data: dict[str, np.ndarray], axes):
    """Panel 4: final-timestep vertical T and qv profiles."""
    sigma = data.get("sigma", np.array([]))
    profiles_T = data.get("profiles_T", np.array([]))
    profiles_qv = data.get("profiles_qv", np.array([]))

    ax_T, ax_q = axes

    if profiles_T.ndim == 2 and profiles_T.shape[0] > 0 and sigma.size > 0:
        ax_T.plot(profiles_T[-1], sigma, "r-", linewidth=1.0, label="Final")
        if profiles_T.shape[0] > 1:
            ax_T.plot(profiles_T[0], sigma, "b--", linewidth=0.8, label="Initial")
            ax_T.legend(fontsize=7)
        ax_T.invert_yaxis()
        ax_T.set_xlabel("T [K]", fontsize=8)
        ax_T.set_ylabel("$\\sigma$", fontsize=8)
        ax_T.set_title("Temperature profile", fontsize=9)
    else:
        ax_T.text(0.5, 0.5, "No profile data", transform=ax_T.transAxes,
                  ha="center", va="center", fontsize=9)
    ax_T.tick_params(labelsize=7)

    if profiles_qv.ndim == 2 and profiles_qv.shape[0] > 0 and sigma.size > 0:
        ax_q.plot(profiles_qv[-1], sigma, "r-", linewidth=1.0, label="Final")
        if profiles_qv.shape[0] > 1:
            ax_q.plot(profiles_qv[0], sigma, "b--", linewidth=0.8, label="Initial")
            ax_q.legend(fontsize=7)
        ax_q.invert_yaxis()
        ax_q.set_xlabel("$q_v$ [g/kg]", fontsize=8)
        ax_q.set_ylabel("$\\sigma$", fontsize=8)
        ax_q.set_title("Humidity profile", fontsize=9)
    else:
        ax_q.text(0.5, 0.5, "No profile data", transform=ax_q.transAxes,
                  ha="center", va="center", fontsize=9)
    ax_q.tick_params(labelsize=7)


def plot_amip(run_dir: str | Path, show: bool = True) -> Path:
    """Generate standard diagnostic plots from a completed AMIP run.

    Returns the path to the saved PNG.
    """
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    data = load_timeseries(run_dir)
    n_days = data["days"][-1] if data["days"].size > 0 else 0

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"AMIP diagnostics — {run_dir.name}  ({n_days:.0f} days)",
                 fontsize=11, fontweight="bold")

    # Layout: 4 rows.  Row 1-3 have 3 columns, row 4 has 2 columns.
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.8, 1, 1])

    # Row 1: scalar timeseries (6 panels in 2 rows x 3 cols — use rows 0-1)
    ax_scalar = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_scalar += [fig.add_subplot(gs[1, c]) for c in range(3)]
    plot_scalar_timeseries(data, ax_scalar)

    # Row 3: energy budget (2 panels)
    ax_energy = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    plot_energy_budget(data, ax_energy)

    # Row 3 col 3: SST timeseries (bonus)
    ax_sst = fig.add_subplot(gs[2, 2])
    if "sst" in data and data["sst"].size > 0:
        ax_sst.plot(data["days"], data["sst"], "k-", linewidth=0.8)
    ax_sst.set_ylabel("SST [K]", fontsize=8)
    ax_sst.set_xlabel("Day")
    ax_sst.set_title("Sea surface temperature", fontsize=9)
    ax_sst.tick_params(labelsize=7)

    # Row 4: conservation (3 panels)
    ax_cons = [fig.add_subplot(gs[3, c]) for c in range(3)]
    plot_conservation(data, ax_cons)

    out_path = run_dir / "diagnostics.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Vertical profiles — separate figure
    if ("profiles_T" in data and data["profiles_T"].size > 0
            and "sigma" in data and data["sigma"].size > 0):
        fig2, (ax_T, ax_q) = plt.subplots(1, 2, figsize=(7, 5),
                                           constrained_layout=True)
        fig2.suptitle(f"Vertical profiles — {run_dir.name}", fontsize=11,
                      fontweight="bold")
        plot_vertical_profiles(data, (ax_T, ax_q))
        prof_path = run_dir / "profiles.png"
        fig2.savefig(prof_path, dpi=150)
        print(f"Saved: {prof_path}")

    if show:
        plt.show()
    plt.close("all")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics from a completed AMIP run")
    parser.add_argument("run_dir", type=str,
                        help="Path to AMIP output directory (containing timeseries.npz)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save PNGs without displaying")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.no_show:
        matplotlib.use("Agg")

    plot_amip(run_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
