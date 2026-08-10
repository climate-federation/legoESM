"""Four-arm lock-exchange benchmark figure (three dycores + the tripole
curvilinear variant of the lat-lon C-grid).

Reads the artifacts the ocean test matrix already writes for the
lock-exchange case (``snapshots_latlon.npz`` + ``mean_timeseries.csv`` per
arm) and builds ONE comparison figure:

  rows 1-4 (one per dycore): SST map at the final day | T lon-depth section
      at day 1 | T lon-depth section at the final day
  bottom row: RPE_rel(t) for all arms overlaid (sorted RPE_mov, positive =
      spurious mixing) + the gate summary.

Run AFTER the matrix (the standard benchmark driver):

    sbatch --array=0-3 scripts/cluster/lock_exchange_dycore_comparison.sbatch
    .venv/bin/python scripts/plot/plot_lock_exchange_benchmark.py

See docs/ocean/experiments/lock_exchange_benchmark.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]

# Fixed arm order + fixed categorical colors (Tol bright subset,
# colorblind-safe; never cycled). 4th field = provenance tag shown on the
# row label: latlon panels are the NATIVE 5-deg field, the other arms are
# the matrix's 1-deg regridded artifact (RPE/gates are always computed on
# NATIVE cells by the matrix, never on these regrids).
ARMS = [
    ("latlon", "36x72", "#4477AA", "native 5deg"),
    ("mpas", "ico3", "#EE6677", "regrid 1deg"),
    ("fesom", "pi", "#228833", "regrid 1deg"),
    ("tripole", "eorca1", "#AA3377", "regrid 1deg,\ncontinents"),
]
_T_CMAP = "viridis"          # sequential, perceptually uniform
_T_RANGE = (5.0, 30.0)       # the IC bounds; the gates hold the run inside


def _arm_dir(root: Path, grid: str, res: str) -> Path:
    return root / grid / "lock_exchange" / grid / res


def _nearest(vals: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(vals) - target)))


def _benchmark_depths(nlev: int) -> np.ndarray:
    """Cell-centre depths [m, positive down] of the ACTUAL benchmark
    coordinate — the same stretched z-star the matrix builds
    (create_ocean_z_star; dz_ref 0.095..1.905 m at nlev=20/H_max=20).
    Uniform ``k+0.5`` indices mislabel the section by up to ~0.9 m."""
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
    z = create_ocean_z_star(n_levels=nlev, H_max=LockExchangeConfig().H_max)
    return -np.asarray(z.z_full_ref, dtype=np.float64)


def _masked_section(T3: np.ndarray, mask2d: np.ndarray | None,
                    it: int, j: int) -> np.ndarray:
    """(nlev, nlon) T section at time ``it``, row ``j``; land columns NaN
    (structured arms pin land tracers at 0 C — plotting them as water is
    the same land-0 artifact the matrix gates had)."""
    sec = np.asarray(T3[it, j, :, :]).T
    if mask2d is not None:
        sec = np.where(np.asarray(mask2d[j])[np.newaxis, :] > 0.5, sec, np.nan)
    return sec


def _load_arm(d: Path):
    z = np.load(d / "snapshots_latlon.npz")
    series: dict[str, list[float]] = {}
    with open(d / "mean_timeseries.csv") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                series.setdefault(k, []).append(float(v))
    return z, series


def build_figure(root: Path, out: Path, section_lat: float = 0.0) -> Path:
    n = len(ARMS)
    fig = plt.figure(figsize=(13, 3.0 * n + 3.2), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1.0] * n + [1.1])

    im = None
    for i, (grid, res, _c, tag) in enumerate(ARMS):
        d = _arm_dir(root, grid, res)
        z, _series = _load_arm(d)
        lat = np.asarray(z["lat"])
        lon = np.asarray(z["lon"])
        t = np.asarray(z["times_days"])
        T3 = np.asarray(z["T_3d"])           # (nt, nlat, nlon, nlev)
        sst = np.asarray(z["SST"])           # (nt, nlat, nlon)
        mask = np.asarray(z["land_mask"]) if "land_mask" in z else None
        j = _nearest(lat, section_lat)
        i_final, i_day1 = len(t) - 1, _nearest(t, 1.0)

        mask2d = None if mask is None else (
            mask if mask.ndim == 2 else mask[0])

        ax = fig.add_subplot(gs[i, 0])
        sst_m = sst[i_final] if mask2d is None else np.where(
            mask2d > 0.5, sst[i_final], np.nan)
        im = ax.pcolormesh(lon, lat, sst_m, cmap=_T_CMAP,
                           vmin=_T_RANGE[0], vmax=_T_RANGE[1], shading="auto")
        ax.set_ylabel(f"{grid} ({tag})\nlat [deg]", fontsize=8)
        if i == 0:
            ax.set_title(f"SST, day {t[i_final]:g}")
        if i == n - 1:
            ax.set_xlabel("lon [deg]")

        nlev = T3.shape[-1]
        depth = _benchmark_depths(nlev)
        # Label the SELECTED latitude, not the requested one (nearest-index
        # on a 5-deg grid picks -2.5 for a 0 request).
        lat_sel = float(lat[j])
        hemi = "N" if lat_sel >= 0 else "S"
        lat_lbl = f"{abs(lat_sel):g}{hemi}"
        for col, it, label in ((1, i_day1, f"T at {lat_lbl}, day 1"),
                               (2, i_final,
                                f"T at {lat_lbl}, day {t[i_final]:g}")):
            axs = fig.add_subplot(gs[i, col])
            sec = _masked_section(T3, mask2d, it, j)
            axs.pcolormesh(lon, depth, sec, cmap=_T_CMAP,
                           vmin=_T_RANGE[0], vmax=_T_RANGE[1], shading="auto")
            axs.invert_yaxis()
            # Per-row title: the selected section latitude differs between
            # the 5-deg latlon artifact (-2.5) and the 1-deg regrids (0).
            axs.set_title(label, fontsize=9)
            if col == 1:
                axs.set_ylabel("depth [m]")
            if i == n - 1:
                axs.set_xlabel("lon [deg]")

    fig.colorbar(im, ax=fig.axes, shrink=0.5, label="T [degC]",
                 location="right", pad=0.01)

    # Bottom: RPE_rel overlay (the spurious-mixing metric) + gate summary.
    axr = fig.add_subplot(gs[n, :2])
    summary_lines = []
    for grid, res, c, _tag in ARMS:
        d = _arm_dir(root, grid, res)
        _z, series = _load_arm(d)
        td, rpe = series["time_days"], series["RPE_rel"]
        axr.plot(td, rpe, color=c, lw=2, label=grid)
        axr.annotate(grid, (td[-1], rpe[-1]), color=c, fontsize=9,
                     xytext=(4, 0), textcoords="offset points")
        status = "?"
        rt = (d / "results.txt")
        if rt.exists():
            for line in rt.read_text().splitlines():
                if line.startswith("status:"):
                    status = line.split(":", 1)[1].strip()
        summary_lines.append(
            f"{grid:8s} {status:4s}  RPE_rel={rpe[-1]:+.3e}  "
            f"T=[{min(series['T_min']):.6f},{max(series['T_max']):.6f}]")
    axr.axhline(0.0, color="0.6", lw=0.8)
    axr.set_xlabel("time [days]")
    axr.set_ylabel("RPE_rel (sorted RPE_mov)")
    axr.set_title("Spurious mixing: sorted-RPE drift, NATIVE cells "
                  "(positive = physical sign; mpas runs tvd, others FCT-class)",
                  fontsize=10)
    axr.legend(loc="upper left", frameon=False, fontsize=9)
    axt = fig.add_subplot(gs[n, 2])
    axt.axis("off")
    axt.text(0.0, 0.95, "Gate summary (whole-run):\n" + "\n".join(summary_lines),
             family="monospace", fontsize=8, va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path,
                    default=_REPO / "results" / "lock_exchange_dycore")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "results" / "lock_exchange_dycore"
                    / "lock_exchange_benchmark.png")
    ap.add_argument("--section-lat", type=float, default=0.0)
    a = ap.parse_args()
    missing = [g for g, r, _c, _t in ARMS
               if not (_arm_dir(a.runs_root, g, r) / "snapshots_latlon.npz").exists()]
    if missing:
        raise SystemExit(
            f"missing arms {missing} under {a.runs_root}; run the matrix "
            f"first (see module docstring).")
    out = build_figure(a.runs_root, a.out, section_lat=a.section_lat)
    print(f"COMPLETED: {out}")


if __name__ == "__main__":
    main()
