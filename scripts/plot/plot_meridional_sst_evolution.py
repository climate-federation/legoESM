"""Plot meridional SST profiles over time for MPAS runs.

Produces a 3-panel figure per run:

  1. Per-mesh-row zonal mean (natural MPAS binning: 107 rows at 0.18° pitch)
  2. Single-longitude cut (nearest cells to a chosen lon_target)
  3. Anomaly from IC (same binning as panel 1)

The 3rd panel highlights where kinks / warm tongues / cold tongues form. The
single-longitude cut exposes any grid-scale wiggles that the zonal mean
smooths out.

Usage
-----
    .venv/bin/python scripts/plot_meridional_sst_evolution.py \
        <run_dir_1> [<run_dir_2> ...] [--out <out_dir>] [--lon-target 5.0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _row_zonal_mean(lat: np.ndarray, sst: np.ndarray,
                    ocean_mask: np.ndarray | None = None,
                    row_tol: float = 0.01,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Average SST within each unique mesh latitude row."""
    row_key = np.round(lat / row_tol).astype(int)
    unique_rows, inverse = np.unique(row_key, return_inverse=True)
    n_rows = len(unique_rows)
    mean = np.full(n_rows, np.nan)
    row_lat = np.full(n_rows, np.nan)
    for r in range(n_rows):
        sel = inverse == r
        if ocean_mask is not None:
            sel = sel & (ocean_mask > 0.5)
        if not sel.any():
            continue
        vals = sst[sel]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            mean[r] = vals.mean()
            row_lat[r] = lat[sel].mean()
    ok = np.isfinite(row_lat)
    order = np.argsort(row_lat[ok])
    return row_lat[ok][order], mean[ok][order]


def _single_lon_cut(lat: np.ndarray, lon: np.ndarray, sst: np.ndarray,
                    lon_target: float,
                    ocean_mask: np.ndarray | None = None,
                    row_tol: float = 0.01,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """For each unique mesh row, pick the cell closest to lon_target."""
    row_key = np.round(lat / row_tol).astype(int)
    unique_rows = np.unique(row_key)
    rows_lat = []
    rows_sst = []
    for r in unique_rows:
        sel = row_key == r
        if ocean_mask is not None:
            sel = sel & (ocean_mask > 0.5)
        if not sel.any():
            continue
        lons_in_row = lon[sel]
        sst_in_row = sst[sel]
        lat_in_row = lat[sel]
        i_best = int(np.argmin(np.abs(lons_in_row - lon_target)))
        if not np.isfinite(sst_in_row[i_best]):
            continue
        rows_lat.append(lat_in_row[i_best])
        rows_sst.append(sst_in_row[i_best])
    rows_lat = np.asarray(rows_lat)
    rows_sst = np.asarray(rows_sst)
    order = np.argsort(rows_lat)
    return rows_lat[order], rows_sst[order]


def plot_run(run_dir: Path, out_dir: Path, lon_target: float) -> None:
    nc_path = run_dir / "snapshots_native.nc"
    if not nc_path.exists():
        print(f"[skip] {run_dir}: snapshots_native.nc missing")
        return
    ds = xr.open_dataset(nc_path)
    lat = np.asarray(ds.latCell.values, dtype=np.float64)
    lon = np.asarray(ds.lonCell.values, dtype=np.float64)
    times_days = np.asarray(ds.time.values, dtype=np.float64)
    sst_series = np.asarray(ds.SST.values, dtype=np.float64)
    lm_series = np.asarray(ds.land_mask.values, dtype=np.float64)

    n_t = len(times_days)
    n_show = min(n_t, 8)
    idx_show = np.linspace(0, n_t - 1, n_show).astype(int)
    cmap = plt.get_cmap("viridis")

    # Precompute row-mean IC
    row_lat0, mean0 = _row_zonal_mean(lat, sst_series[0],
                                       ocean_mask=lm_series[0])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for k, ti in enumerate(idx_show):
        c = cmap(k / max(n_show - 1, 1))
        # Panel 1: row-mean profile
        rlat, rmean = _row_zonal_mean(
            lat, sst_series[ti], ocean_mask=lm_series[ti])
        axes[0].plot(rlat, rmean, color=c, lw=1.2,
                     label=f"t={times_days[ti]:.0f} d")
        # Panel 2: single-longitude cut
        lcut_lat, lcut_sst = _single_lon_cut(
            lat, lon, sst_series[ti], lon_target,
            ocean_mask=lm_series[ti])
        axes[1].plot(lcut_lat, lcut_sst, color=c, lw=1.2,
                     label=f"t={times_days[ti]:.0f} d")
        # Panel 3: anomaly from IC (row-mean)
        if ti != 0:
            # Interpolate onto the IC row grid (rows may drift very slightly
            # if land_mask changes — they don't here, but be safe).
            rmean_on_ic = np.interp(row_lat0, rlat, rmean,
                                    left=np.nan, right=np.nan)
            axes[2].plot(row_lat0, rmean_on_ic - mean0, color=c, lw=1.2,
                         label=f"t={times_days[ti]:.0f} d")

    axes[0].set_title("Row-mean SST (zonal mean per mesh row)")
    axes[0].set_xlabel("Latitude (°)")
    axes[0].set_ylabel("SST (°C)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].set_title(f"SST along lon ≈ {lon_target:.1f}° (single-column cut)")
    axes[1].set_xlabel("Latitude (°)")
    axes[1].set_ylabel("SST (°C)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, frameon=False)

    axes[2].axhline(0.0, color="k", lw=0.5, ls=":")
    axes[2].set_title("Row-mean anomaly from IC")
    axes[2].set_xlabel("Latitude (°)")
    axes[2].set_ylabel("SST − SST(t=0)  (°C)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, frameon=False)

    fig.suptitle(f"{run_dir.name} — meridional SST evolution "
                 f"(row-mean vs single-longitude)", fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_dir.name}__meridional_SST_evolution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")
    ds.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lon-target", type=float, default=5.0,
                    help="Longitude (°) of single-column cut (default: 5.0)")
    args = ap.parse_args()

    if args.out is None:
        parents = {p.resolve().parent for p in args.run_dirs}
        out_dir = (next(iter(parents)) / "ic_diagnostics") \
            if len(parents) == 1 else Path("results/meridional_SST_evolution")
    else:
        out_dir = args.out

    for d in args.run_dirs:
        plot_run(d.resolve(), out_dir, args.lon_target)


if __name__ == "__main__":
    main()
