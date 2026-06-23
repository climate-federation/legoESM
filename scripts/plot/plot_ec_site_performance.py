#!/usr/bin/env python
"""Plot offline EC-site model-vs-observation performance (GPP / LE / H).

Reads a ``run_ec_site.py`` output NetCDF (``<SITE>_ec_<mode>.nc`` with
``{gpp,le,h}_{mod,obs}`` + ``valid``) and produces, per flux:

* a model-vs-obs density scatter with the 1:1 line and RMSE / bias / r / n,
* the mean diurnal cycle (model vs obs),
* the mean seasonal (monthly) cycle (model vs obs).

The hour-of-day / month needed for the diurnal & seasonal composites come from the
driver's datetime axis (``--driver``), aligned to the prediction by index (the
prediction time coordinate is a plain step index).  A second ``--diagnostic``
prediction can be overlaid on the scatter to contrast diagnostic vs prognostic
(e.g. the sensible-heat phase fix).

Usage
-----
    JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_performance.py \
        --pred diagnostics/ec_site/US-MMS_driver_v2_gapfree.nc_ec_prognostic.nc \
        --driver <DifferBESS>/.../US-MMS_driver_v2_gapfree.nc \
        --out-dir diagnostics/ec_site_perf
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

_FLUXES = [("gpp", "GPP", "µmol CO2 m$^{-2}$ s$^{-1}$"),
           ("le", "LE", "W m$^{-2}$"),
           ("h", "H", "W m$^{-2}$")]


def _mask(pred: xr.Dataset, key: str) -> np.ndarray:
    """Comparison mask: prefer the persisted exact score mask, else valid."""
    base = (pred["score_valid"].values if "score_valid" in pred
            else pred["valid"].values).astype(bool)
    return (base & np.isfinite(pred[f"{key}_mod"].values)
            & np.isfinite(pred[f"{key}_obs"].values))


def _stats(m: np.ndarray, o: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))
    bias = float(np.mean(m - o))
    r = float(np.corrcoef(m, o)[0, 1]) if m.std() > 0 and o.std() > 0 else np.nan
    return dict(n=m.size, rmse=rmse, bias=bias, r=r)


def plot(pred_nc: str, driver_nc: str | None, out_dir: str,
         diag_nc: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    pred = xr.open_dataset(pred_nc)
    n = pred.sizes["time"]
    tag = os.path.basename(pred_nc).replace(".nc", "")

    # hour-of-day + month from the driver datetime axis (aligned by index).
    hour = month = None
    if driver_nc:
        t = pd.DatetimeIndex(xr.open_dataset(driver_nc)["time"].values)
        if len(t) >= n:
            hour = t.hour.to_numpy()[:n]
            month = t.month.to_numpy()[:n]

    diag = xr.open_dataset(diag_nc) if diag_nc else None

    fig, axes = plt.subplots(len(_FLUXES), 3, figsize=(15, 4.2 * len(_FLUXES)))
    summary = {}
    for row, (key, label, unit) in enumerate(_FLUXES):
        mask = _mask(pred, key)
        mod = pred[f"{key}_mod"].values[mask]
        obs = pred[f"{key}_obs"].values[mask]
        st = _stats(mod, obs)
        summary[label] = st

        # --- (a) density scatter vs 1:1 ---
        ax = axes[row, 0]
        ax.hexbin(obs, mod, gridsize=60, mincnt=1, cmap="viridis", bins="log")
        lim = [min(obs.min(), mod.min()), max(obs.max(), mod.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"observed {label} [{unit}]", fontsize=8)
        ax.set_ylabel(f"modelled {label} [{unit}]", fontsize=8)
        ax.set_title(f"{label}: r={st['r']:.3f}  RMSE={st['rmse']:.2f}  "
                     f"bias={st['bias']:+.2f}  n={st['n']}", fontsize=9)

        # --- (b) mean diurnal cycle ---
        ax = axes[row, 1]
        if hour is not None:
            hh = hour[mask]
            dm = [np.mean(mod[hh == k]) if np.any(hh == k) else np.nan for k in range(24)]
            do = [np.mean(obs[hh == k]) if np.any(hh == k) else np.nan for k in range(24)]
            ax.plot(range(24), do, "k-o", ms=3, label="obs")
            ax.plot(range(24), dm, "r-o", ms=3, label="model")
            ax.set_xlabel("hour of day", fontsize=8); ax.legend(fontsize=8)
            ax.set_title(f"{label}: mean diurnal cycle", fontsize=9)
        else:
            ax.set_visible(False)

        # --- (c) mean seasonal (monthly) cycle ---
        ax = axes[row, 2]
        if month is not None:
            mm = month[mask]
            sm = [np.mean(mod[mm == k]) if np.any(mm == k) else np.nan for k in range(1, 13)]
            so = [np.mean(obs[mm == k]) if np.any(mm == k) else np.nan for k in range(1, 13)]
            ax.plot(range(1, 13), so, "k-o", ms=3, label="obs")
            ax.plot(range(1, 13), sm, "r-o", ms=3, label="model")
            ax.set_xlabel("month", fontsize=8); ax.legend(fontsize=8)
            ax.set_title(f"{label}: mean seasonal cycle", fontsize=9)
        else:
            ax.set_visible(False)

    fig.suptitle(f"{tag}  (mode={pred.attrs.get('mode','?')})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = os.path.join(out_dir, f"{tag}_performance.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  plot -> {p}")

    # --- optional diagnostic-vs-prognostic scatter contrast ---
    if diag is not None:
        fig, axes = plt.subplots(2, len(_FLUXES), figsize=(5 * len(_FLUXES), 8.5))
        for col, (key, label, unit) in enumerate(_FLUXES):
            for r_i, (ds_, name) in enumerate([(diag, "diagnostic"), (pred, "prognostic")]):
                ax = axes[r_i, col]
                mk = _mask(ds_, key)
                m, o = ds_[f"{key}_mod"].values[mk], ds_[f"{key}_obs"].values[mk]
                st = _stats(m, o)
                ax.hexbin(o, m, gridsize=50, mincnt=1, cmap="magma", bins="log")
                lim = [min(o.min(), m.min()), max(o.max(), m.max())]
                ax.plot(lim, lim, "w--", lw=1)
                ax.set_title(f"{name} {label}: r={st['r']:.2f} RMSE={st['rmse']:.1f}",
                             fontsize=9)
                ax.set_xlabel(f"obs [{unit}]", fontsize=8)
                ax.set_ylabel(f"model [{unit}]", fontsize=8)
        fig.suptitle("diagnostic (top) vs prognostic (bottom)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        p = os.path.join(out_dir, f"{tag}_diag_vs_prog.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"  plot -> {p}")

    print("\n=== performance summary ===")
    for label, st in summary.items():
        print(f"  {label:3s}  n={st['n']:>7d}  RMSE={st['rmse']:8.3f}  "
              f"bias={st['bias']:+8.3f}  r={st['r']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="run_ec_site output NetCDF")
    ap.add_argument("--driver", default=None,
                    help="driver NetCDF (datetime axis for diurnal/seasonal)")
    ap.add_argument("--diagnostic", default=None,
                    help="optional diagnostic-mode output to contrast on scatter")
    ap.add_argument("--out-dir", default="diagnostics/ec_site_perf")
    args = ap.parse_args()
    plot(args.pred, args.driver, args.out_dir, diag_nc=args.diagnostic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
