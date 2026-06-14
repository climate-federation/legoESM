"""Slide 28 figure: geostrophic adjustment SSH on lat-lon and MPAS at matched resolution.

Drops cubed-sphere; uses MPAS ico4 (~445 km) to be resolution-comparable to
lat-lon 36x72 (~556 km at equator, finer at high lat).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/ocean/slide28_validation/geostrophic_adjustment_full")
LATLON_NPZ = RESULTS / "latlon" / "36x72" / "snapshots_latlon.npz"
MPAS_NPZ = RESULTS / "mpas_ico4" / "ico4" / "snapshots_latlon.npz"
OUT_DIR = RESULTS


def _load_eta(npz_path: Path):
    d = np.load(npz_path)
    eta = np.asarray(d["eta"]).copy()
    if "land_mask" in d.files:
        lm = np.asarray(d["land_mask"])
        # Convention: 1 = ocean, 0 = land. Regridded MPAS produces fractional
        # values; treat any cell with ocean fraction < 0.5 as land.
        ocean = lm > 0.5
        eta = np.where(ocean, eta, np.nan)
        land_mask = ocean[0] if ocean.ndim == 3 else ocean
    else:
        land_mask = None
    return {
        "lat": d["lat"],
        "lon": d["lon"],
        "eta": eta,
        "times_days": d["times_days"],
        "land_mask": land_mask,
    }


def plot_snapshot_pair(out_path: Path, t_idx: int = -1) -> None:
    ll = _load_eta(LATLON_NPZ)
    mp = _load_eta(MPAS_NPZ)
    t_day = float(ll["times_days"][t_idx])

    vmax = max(np.nanmax(np.abs(ll["eta"][t_idx])),
               np.nanmax(np.abs(mp["eta"][t_idx])))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, data, label in [
        (axes[0], ll, "lat-lon C-grid (36×72)"),
        (axes[1], mp, "MPAS Voronoi (ico4)"),
    ]:
        im = ax.pcolormesh(
            data["lon"], data["lat"], data["eta"][t_idx],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, label="SSH η (m)")
    fig.suptitle(
        f"Geostrophic adjustment — SSH at t = {t_day:.0f} d "
        f"(matched ~500 km resolution, same A_h = 10⁴ m²/s)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_evolution_pair(out_path: Path, time_indices=(1, 2, 4, 7, 10)) -> None:
    ll = _load_eta(LATLON_NPZ)
    mp = _load_eta(MPAS_NPZ)

    vmax = max(np.nanmax(np.abs(ll["eta"][list(time_indices)])),
               np.nanmax(np.abs(mp["eta"][list(time_indices)])))

    n = len(time_indices)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.0),
                             sharex=True, sharey=True, constrained_layout=True)
    for j, idx in enumerate(time_indices):
        for i, (data, label) in enumerate([
            (ll, "lat-lon"),
            (mp, "MPAS ico4"),
        ]):
            ax = axes[i, j]
            im = ax.pcolormesh(
                data["lon"], data["lat"], data["eta"][idx],
                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
            )
            if i == 0:
                ax.set_title(f"t = {float(data['times_days'][idx]):.0f} d")
            if j == 0:
                ax.set_ylabel(f"{label}\nLatitude")
            if i == 1:
                ax.set_xlabel("Longitude")
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
    fig.colorbar(im, ax=axes, shrink=0.7, label="SSH η (m)")
    fig.suptitle("Geostrophic adjustment — SSH evolution, lat-lon vs MPAS",
                 fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_timeseries(out_path: Path) -> None:
    """Mean SSH and max-|SSH| time series from native mean_timeseries.csv."""
    import csv

    def load_csv(path):
        cols = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    cols.setdefault(k, []).append(float(v))
        return {k: np.asarray(v) for k, v in cols.items()}

    ll = load_csv(RESULTS / "latlon" / "36x72" / "mean_timeseries.csv")
    mp = load_csv(RESULTS / "mpas_ico4" / "ico4" / "mean_timeseries.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    axes[0].plot(ll["time_days"], ll["mean_eta"], "C0", lw=2,
                 marker="o", ms=4, label="lat-lon 36×72")
    axes[0].plot(mp["time_days"], mp["mean_eta"], "C3", lw=2,
                 marker="s", ms=4, label="MPAS ico4")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Native-grid mean η (m)")
    axes[0].set_title("Mass conservation (machine epsilon)")
    axes[0].set_ylim(-1e-15, 1e-15)
    axes[0].legend()
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    axes[1].plot(ll["time_days"], ll["max_abs_eta"], "C0", lw=2,
                 marker="o", ms=4, label="lat-lon 36×72")
    axes[1].plot(mp["time_days"], mp["max_abs_eta"], "C3", lw=2,
                 marker="s", ms=4, label="MPAS ico4")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("max |η| (m)")
    axes[1].set_title("Peak SSH amplitude")
    axes[1].legend()

    fig.suptitle("Geostrophic adjustment — native-grid diagnostics", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_single_panel(out_path: Path, which: str, t_idx: int = -1) -> None:
    """Single-grid SSH snapshot, sized for the 3-box pptx layout."""
    ll = _load_eta(LATLON_NPZ)
    mp = _load_eta(MPAS_NPZ)
    if which == "latlon":
        data, label = ll, "lat-lon C-grid (36×72)"
    elif which == "mpas":
        data, label = mp, "MPAS Voronoi (ico4, ~445 km)"
    else:
        raise ValueError(which)

    vmax = max(np.nanmax(np.abs(ll["eta"][t_idx])),
               np.nanmax(np.abs(mp["eta"][t_idx])))
    t_day = float(data["times_days"][t_idx])

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.pcolormesh(
        data["lon"], data["lat"], data["eta"][t_idx],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
    )
    ax.set_title(f"{label}\nSSH at t = {t_day:.0f} d")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    fig.colorbar(im, ax=ax, shrink=0.85, label="SSH η (m)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_snapshot_pair(OUT_DIR / "slide28_snapshot_t10d.png")
    plot_evolution_pair(OUT_DIR / "slide28_evolution.png")
    plot_timeseries(OUT_DIR / "slide28_timeseries.png")
    plot_single_panel(OUT_DIR / "slide28_panel_latlon.png", "latlon")
    plot_single_panel(OUT_DIR / "slide28_panel_mpas.png", "mpas")
