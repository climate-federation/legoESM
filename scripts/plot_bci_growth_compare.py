"""Compare baroclinic-instability growth / equilibration between MPAS and
lat-lon Eady runs via the `mean_timeseries.csv` diagnostic.

Two regimes per figure:
  - Weak forcing  (U=0.2, 600 d): MPAS U02 vs lat-lon U02
  - Strong forcing (U=0.8, 200 d): MPAS U08 vs lat-lon U08 / SOM+Csmag

For each regime, plots:
  - max_speed(t)   [log-y]: exponential-growth phase slope and saturation level
  - mean_ke(t)     [log-y]: EKE proxy (includes barotropic mean, but the
                             ratio of growth to mean remains informative)
  - max_abs_eta(t)         : surface-pressure amplitude (∝ BCI)
  - mean_T spread          : Tmax_row − Tmin_row over time (slumping)

Usage
-----
    .venv/bin/python scripts/plot_bci_growth_compare.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path("results/ocean/eady_uniform")


def _load_ts(run_dir: Path) -> dict[str, np.ndarray] | None:
    p = run_dir / "mean_timeseries.csv"
    if not p.exists():
        return None
    cols: dict[str, list[float]] = {}
    with p.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            for k, v in row.items():
                if k is None or k.startswith("#"):
                    continue
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _row_T_spread(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute max(row-mean T) − min(row-mean T) at each snapshot time.

    For MPAS this is the natural mesh-row mean; for lat-lon we collapse the
    (lat, lon) SST array along lon.
    """
    nc = run_dir / "snapshots_native.nc"
    if not nc.exists():
        return None
    ds = xr.open_dataset(nc)
    kind = ds.attrs.get("coord_kind", "")
    times = np.asarray(ds.time.values, dtype=np.float64)
    sst = np.asarray(ds.SST.values, dtype=np.float64)   # (n_t, *)
    lm = np.asarray(ds.land_mask.values, dtype=np.float64)
    out = np.empty(len(times))
    if kind == "mpas":
        lat = np.asarray(ds.latCell.values, dtype=np.float64)
        row_key = np.round(lat / 0.01).astype(int)
        urows, inv = np.unique(row_key, return_inverse=True)
        for ti in range(len(times)):
            rmean = np.full(len(urows), np.nan)
            for r in range(len(urows)):
                sel = (inv == r) & (lm[ti] > 0.5)
                if sel.any():
                    vals = sst[ti][sel]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        rmean[r] = vals.mean()
            out[ti] = float(np.nanmax(rmean) - np.nanmin(rmean))
    else:
        # lat-lon: sst shape (n_t, n_lat, n_lon)
        for ti in range(len(times)):
            arr = sst[ti]
            m = lm[ti] > 0.5
            arr = np.where(m, arr, np.nan)
            rmean = np.nanmean(arr, axis=-1)   # zonal mean → lat profile
            rmean = rmean[np.isfinite(rmean)]
            out[ti] = float(rmean.max() - rmean.min()) if rmean.size else np.nan
    ds.close()
    return times, out


def _plot_pair(ax, ts: dict, style: dict, label: str, key: str,
               log: bool = False):
    if ts is None or key not in ts or "time_days" not in ts:
        return
    t = ts["time_days"]
    y = ts[key]
    if log:
        y = np.where(y > 0, y, np.nan)
        ax.semilogy(t, y, label=label, **style)
    else:
        ax.plot(t, y, label=label, **style)


def _plot_regime(axes_row, runs: list[tuple[Path, str, dict]], title: str):
    """Populate a 4-panel row for one regime."""
    # max_speed (log)
    axes_row[0].set_title(f"{title} — max_speed (log)")
    axes_row[0].set_xlabel("time (d)")
    axes_row[0].set_ylabel("max |u| (m/s)")
    for d, label, style in runs:
        ts = _load_ts(d)
        _plot_pair(axes_row[0], ts, style, label, "max_speed", log=True)
    axes_row[0].grid(alpha=0.3, which="both")
    axes_row[0].legend(fontsize=8, frameon=False)

    # mean_ke (log)
    axes_row[1].set_title(f"{title} — mean_ke (log)")
    axes_row[1].set_xlabel("time (d)")
    axes_row[1].set_ylabel("mean KE (m²/s²)")
    for d, label, style in runs:
        ts = _load_ts(d)
        _plot_pair(axes_row[1], ts, style, label, "mean_ke", log=True)
    axes_row[1].grid(alpha=0.3, which="both")
    axes_row[1].legend(fontsize=8, frameon=False)

    # max_abs_eta
    axes_row[2].set_title(f"{title} — max |eta|")
    axes_row[2].set_xlabel("time (d)")
    axes_row[2].set_ylabel("max |η| (m)")
    for d, label, style in runs:
        ts = _load_ts(d)
        _plot_pair(axes_row[2], ts, style, label, "max_abs_eta")
    axes_row[2].grid(alpha=0.3)
    axes_row[2].legend(fontsize=8, frameon=False)

    # row-mean T spread (meridional slumping)
    axes_row[3].set_title(f"{title} — meridional SST spread")
    axes_row[3].set_xlabel("time (d)")
    axes_row[3].set_ylabel("Tmax(row) − Tmin(row)  (°C)")
    for d, label, style in runs:
        r = _row_T_spread(d)
        if r is None:
            continue
        t, spread = r
        axes_row[3].plot(t, spread, label=label, **style)
    axes_row[3].grid(alpha=0.3)
    axes_row[3].legend(fontsize=8, frameon=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=RESULTS / "bci_growth_compare.png")
    args = ap.parse_args()

    weak = [
        (RESULTS / "mpas_channel/20km/tvd_mpas_20km_U02_Bh2.3e11_Cs0.0_600d",
         "MPAS tvd", dict(color="C0", lw=1.5)),
        (RESULTS / "mpas_channel/20km/upwind_mpas_20km_U02_Bh2.3e11_Cs0.0_600d",
         "MPAS upwind", dict(color="C0", lw=1.0, ls="--")),
        (RESULTS / "latlon_channel/100x50/tvd_U02_Bh2.3e11_Cs0.0_600d",
         "lat-lon tvd", dict(color="C3", lw=1.5)),
        (RESULTS / "latlon_channel/100x50/som_U02_Bh2.3e11_Cs0.2_600d",
         "lat-lon SOM+Csmag", dict(color="C2", lw=1.5)),
    ]

    strong = [
        (RESULTS / "mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d",
         "MPAS tvd", dict(color="C0", lw=1.5)),
        (RESULTS / "mpas_channel/20km/upwind_mpas_20km_U08_Bh2.3e11_Cs0.0_200d",
         "MPAS upwind (blew up)", dict(color="C0", lw=1.0, ls="--")),
        (RESULTS / "latlon_channel/100x50/tvd_nosponge_200d",
         "lat-lon tvd", dict(color="C3", lw=1.5)),
        (RESULTS / "latlon_channel/100x50/som_csmag02_nosponge_200d",
         "lat-lon SOM+Csmag", dict(color="C2", lw=1.5)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    _plot_regime(axes[0], weak,   "Weak forcing (U=0.2, 600 d)")
    _plot_regime(axes[1], strong, "Strong forcing (U=0.8, 200 d)")

    fig.suptitle("Baroclinic instability: growth & equilibration — "
                 "MPAS vs lat-lon", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
