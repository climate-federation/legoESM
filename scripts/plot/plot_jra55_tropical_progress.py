#!/usr/bin/env python
"""Progress plots for the tropical-OMIP JRA55-do forced-ocean run.

Reads ``restart_dayXXXXXX.npz`` files written by ``run_omip.py`` when
``--forcing-mode jra55_do_tropical --checkpoint-days N`` is set.

Produces (in the run directory):

  - ``timeseries_progress.png``: scalar timeseries (max |u|, mean SST,
    mean T, mean η, max |η|) — one point per restart.
  - ``snapshots_progress.png``: SSH / SST / surface-speed maps stacked
    vertically, one row per restart.
  - ``moc_progress.png``: meridional overturning streamfunction
    ψ(lat, depth) [Sv].  The headline AMOC diagnostic.
  - ``barotropic_streamfunction_progress.png``: ψ_bt(lat, lon) [Sv].
    Shows wind-driven gyre patterns + ACC develop.

Usage::

    JAX_ENABLE_X64=1 python scripts/plot_jra55_tropical_progress.py \\
        --run-dir results/ocean/jra55_smoke_30d/latlon/180x360

The plotter is intentionally minimal: just enough diagnostics to
answer "is the run producing a real ocean state?" before investing in
the OMIP-protocol-grade diagnostics module (Item 7 of the parent
plan).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm import constants
from legoesm.ocean.diagnostics_streamfunction import (
    barotropic_streamfunction,
    moc_streamfunction,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Progress plots for the tropical-OMIP JRA55-do run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir", type=str, required=True,
        help="Directory containing restart_dayXXXXXX.npz files "
             "(typically ``<output>/latlon/<resolution>/``).",
    )
    p.add_argument(
        "--max-restarts", type=int, default=8,
        help="Cap the number of restarts plotted (chronological); "
             "useful for long runs to keep the figure dense.",
    )
    return p.parse_args(argv)


def _gather_restarts(run_dir: Path):
    pairs = []
    for p in run_dir.glob("restart_day*.npz"):
        day = int(p.stem.removeprefix("restart_day"))
        pairs.append((day, p))
    pairs.sort()
    return pairs


def _record_diagnostics(d):
    """Scalar diagnostics from a single restart npz, mirroring
    ``plot_realistic_geometry_progress._record_diagnostics``."""
    eta = d["eta"]
    T = d["T"]
    u = d["u"]
    v = d["v"]
    mask = d["land_mask"]
    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed = np.sqrt(u_c ** 2 + v_c ** 2)
    ocean = mask > 0.5
    return {
        "max_speed": float(np.max(speed)),
        "mean_sst": float(np.mean(T[:, :, 0][ocean])) if ocean.any() else 0.0,
        "mean_T": float(np.mean(T[ocean])) if ocean.any() else 0.0,
        "T_deep": float(np.mean(T[:, :, -1][ocean])) if ocean.any() else 0.0,
        "mean_eta": float(np.mean(eta[ocean])) if ocean.any() else 0.0,
        "max_eta": float(np.max(np.abs(eta[ocean]))) if ocean.any() else 0.0,
    }


class _GridShim:
    """Minimal shim with the attributes our streamfunction helpers need.

    The real ``LatLonGrid`` lives in ``legoesm.grids.latlon``; rebuilding
    one here would require knowing the run's (n_lat, n_lon, radius)
    a priori. Instead we infer those from the restart and expose them
    via this shim — keeps the plot script free of run-specific config
    coupling.
    """
    def __init__(self, n_lat: int, n_lon: int, radius: float = constants.R_earth):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.radius = radius


def _layer_thickness_from_restart(d):
    """Recover layer thickness from the restart dict.

    The lat-lon C-grid restart stores ``h_partial`` directly when the
    partial-cell coordinate is in use; otherwise we fall back to a
    uniform thickness inferred from ``H_bathy`` and the level count.
    """
    if "h_partial" in d.files:
        return np.asarray(d["h_partial"])
    H = np.asarray(d["H_bathy"]) if "H_bathy" in d.files else None
    T = np.asarray(d["T"])
    n_lat, n_lon, nlev = T.shape
    if H is None:
        # No bathymetry stored — assume uniform 1000 m basin.
        return np.full((n_lat, n_lon, nlev), 1000.0 / nlev)
    return np.broadcast_to(
        (H / nlev)[:, :, None], (n_lat, n_lon, nlev),
    ).copy()


def _plot_timeseries(years, diags, out_path: Path):
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    keys = ["max_speed", "mean_sst", "mean_T", "mean_eta", "max_eta"]
    titles = {
        "max_speed": "Max surface speed [m/s]",
        "mean_sst":  "Mean SST [°C]",
        "mean_T":    "Mean ocean T [°C]",
        "mean_eta":  "Mean η [m]",
        "max_eta":   "Max |η| [m]",
    }
    for ax, k in zip(axes, keys):
        ax.plot(years, [r[k] for r in diags], "o-", color="C0")
        ax.set_ylabel(titles[k])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Simulation year")
    fig.suptitle("JRA55-do tropical-OMIP — scalar progress")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_snapshots(restarts, days, out_path: Path):
    n = len(restarts)
    fig, axes = plt.subplots(n, 3, figsize=(14, max(2.5 * n, 4.0)),
                              squeeze=False)
    for row, (d, day) in enumerate(zip(restarts, days)):
        eta = np.asarray(d["eta"])
        T = np.asarray(d["T"])
        u = np.asarray(d["u"])
        v = np.asarray(d["v"])
        mask = np.asarray(d["land_mask"])
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        speed = np.sqrt(u_c ** 2 + v_c ** 2)
        ocean = mask > 0.5

        for ax, field, title, cmap in [
            (axes[row, 0], np.where(ocean, eta, np.nan), "η [m]", "RdBu_r"),
            (axes[row, 1], np.where(ocean, T[..., 0], np.nan),
             "SST [°C]", "RdYlBu_r"),
            (axes[row, 2], np.where(ocean, speed, np.nan),
             "|u_sfc| [m/s]", "viridis"),
        ]:
            im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap)
            ax.set_title(f"{title}  (day {day})", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("JRA55-do tropical-OMIP — surface snapshots")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_moc(restarts, days, grid, out_path: Path):
    n = len(restarts)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(2.5 * n, 4.0)),
                              squeeze=False)
    for row, (d, day) in enumerate(zip(restarts, days)):
        v = np.asarray(d["v"])
        mask = np.asarray(d["land_mask"])
        eta = np.asarray(d["eta"])
        H_bathy = np.asarray(d["H_bathy"]) if "H_bathy" in d.files else (
            np.full_like(mask, 5000.0)
        )
        h_partial = _layer_thickness_from_restart(d)
        psi = moc_streamfunction(v, h_partial, eta, H_bathy, mask, grid)
        ax = axes[row, 0]
        n_lat_v, nlev = psi.shape
        # Plot lat × −cumulative-depth (depth grows downward).
        lat = np.linspace(-90.0, 90.0, n_lat_v)
        depth_levels = np.arange(nlev)
        im = ax.pcolormesh(
            lat, -depth_levels, psi.T, cmap="RdBu_r",
            vmin=-30, vmax=30, shading="auto",
        )
        ax.set_title(f"MOC ψ [Sv]  (day {day})", fontsize=10)
        ax.set_ylabel("Level (surface=0, top of bar)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[-1, 0].set_xlabel("Latitude [°]")
    fig.suptitle("JRA55-do tropical-OMIP — meridional overturning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bsf(restarts, days, grid, out_path: Path):
    n = len(restarts)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.5 * n, 4.0)),
                              squeeze=False)
    for row, (d, day) in enumerate(zip(restarts, days)):
        u = np.asarray(d["u"])
        mask = np.asarray(d["land_mask"])
        h_partial = _layer_thickness_from_restart(d)
        psi_bt = barotropic_streamfunction(u, h_partial, mask, grid)
        psi_masked = np.where(mask > 0.5, psi_bt, np.nan)
        ax = axes[row, 0]
        im = ax.imshow(
            psi_masked, origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-150, vmax=150,
        )
        ax.set_title(f"BSF ψ_bt [Sv]  (day {day})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("JRA55-do tropical-OMIP — barotropic streamfunction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    pairs = _gather_restarts(run_dir)
    if not pairs:
        print(f"No restart_day*.npz files in {run_dir}; nothing to plot.")
        return 1

    if args.max_restarts > 0 and len(pairs) > args.max_restarts:
        # Subsample chronologically, keeping endpoints.
        idx = np.linspace(0, len(pairs) - 1, args.max_restarts).astype(int)
        pairs = [pairs[i] for i in idx]

    days = [d for d, _ in pairs]
    years = [d / 365.0 for d in days]
    restarts = [np.load(p, allow_pickle=False) for _, p in pairs]
    diags = [_record_diagnostics(d) for d in restarts]

    # Infer grid from the first restart.
    T0 = np.asarray(restarts[0]["T"])
    n_lat, n_lon, _ = T0.shape
    grid = _GridShim(n_lat=n_lat, n_lon=n_lon)
    print(f"Plotting {len(restarts)} restarts on {n_lat}×{n_lon} grid:")
    for day in days:
        print(f"    day {day}")

    _plot_timeseries(years, diags, run_dir / "timeseries_progress.png")
    _plot_snapshots(restarts, days, run_dir / "snapshots_progress.png")
    _plot_moc(restarts, days, grid, run_dir / "moc_progress.png")
    _plot_bsf(restarts, days, grid,
              run_dir / "barotropic_streamfunction_progress.png")
    print(f"Wrote 4 progress plots to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
