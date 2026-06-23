#!/usr/bin/env python
"""Comprehensive offline EC-site evaluation: GPP / LE / H / evaporative fraction.

From a ``run_ec_site.py`` output NetCDF (+ the driver for the datetime axis), make
the standard flux-tower evaluation panels for each variable:

* full time series (monthly means, model vs obs),
* mean seasonal cycle (monthly climatology),
* interannual variability (annual means per year),
* summer (JJA) and winter (DJF) mean diurnal cycles,

and report Pearson r at hourly, monthly, seasonal-cycle and annual scales.

Evaporative fraction EF = LE / (LE + H) is evaluated on daytime, positive-available
-energy steps (so the ratio is well posed).

Usage
-----
    JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_evaluation.py \
        --pred diagnostics/ec_site/US-MMS_..._ec_prognostic.nc \
        --driver <...>/US-MMS_driver_v2_gapfree.nc \
        --out-dir diagnostics/ec_site_eval
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

# evaluated fluxes: (model/obs key, label, unit)
_FLUXES = [("gpp", "GPP", "µmol m$^{-2}$ s$^{-1}$"),
           ("le", "LE", "W m$^{-2}$"),
           ("h", "H", "W m$^{-2}$"),
           ("ef", "EF", "-")]
_EF_AVAIL_FLOOR = 50.0   # W/m2: only form EF where LE+H exceeds this (daytime)


def _r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _load(pred_nc, driver_nc):
    pred = xr.open_dataset(pred_nc)
    n = pred.sizes["time"]
    t = pd.DatetimeIndex(xr.open_dataset(driver_nc)["time"].values)[:n]
    base = (pred["score_valid"].values if "score_valid" in pred
            else pred["valid"].values).astype(bool)

    df = pd.DataFrame(index=t)
    for key in ("gpp", "le", "h"):
        df[f"{key}_mod"] = np.where(base, pred[f"{key}_mod"].values, np.nan)
        df[f"{key}_obs"] = np.where(base, pred[f"{key}_obs"].values, np.nan)
    # evaporative fraction on positive available-energy (daytime) steps only
    for src in ("mod", "obs"):
        le, h = df[f"le_{src}"].values, df[f"h_{src}"].values
        avail = le + h
        ef = np.where(avail > _EF_AVAIL_FLOOR, le / avail, np.nan)
        df[f"ef_{src}"] = np.clip(ef, -0.5, 1.5)
    return pred, df


def evaluate(pred_nc, driver_nc, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    pred, df = _load(pred_nc, driver_nc)
    tag = os.path.basename(pred_nc).replace(".nc", "")
    hour = df.index.hour.to_numpy()
    month = df.index.month.to_numpy()
    summer = np.isin(month, [6, 7, 8])
    winter = np.isin(month, [12, 1, 2])

    fig, axes = plt.subplots(len(_FLUXES), 5, figsize=(22, 3.4 * len(_FLUXES)))
    summary = {}
    for row, (key, label, unit) in enumerate(_FLUXES):
        mod = df[f"{key}_mod"].to_numpy()
        obs = df[f"{key}_obs"].to_numpy()

        # --- col 0: full monthly-mean time series ---
        ax = axes[row, 0]
        mon = df[[f"{key}_mod", f"{key}_obs"]].resample("1MS").mean()
        ax.plot(mon.index, mon[f"{key}_obs"], "k-", lw=0.7, label="obs")
        ax.plot(mon.index, mon[f"{key}_mod"], "r-", lw=0.7, label="model")
        r_mon = _r(mon[f"{key}_mod"].to_numpy(), mon[f"{key}_obs"].to_numpy())
        ax.set_title(f"{label} monthly series  r={r_mon:.2f}", fontsize=8)
        ax.set_ylabel(f"{label}\n[{unit}]", fontsize=8); ax.tick_params(labelsize=6)
        if row == 0:
            ax.legend(fontsize=6, loc="upper right")

        # --- col 1: mean seasonal cycle ---
        ax = axes[row, 1]
        sm = [np.nanmean(mod[month == k]) for k in range(1, 13)]
        so = [np.nanmean(obs[month == k]) for k in range(1, 13)]
        ax.plot(range(1, 13), so, "k-o", ms=3, label="obs")
        ax.plot(range(1, 13), sm, "r-o", ms=3, label="model")
        r_seas = _r(np.array(sm), np.array(so))
        ax.set_title(f"{label} seasonal cycle  r={r_seas:.2f}", fontsize=8)
        ax.set_xlabel("month", fontsize=7); ax.tick_params(labelsize=6)

        # --- col 2: interannual variability (annual means) ---
        ax = axes[row, 2]
        ann = df[[f"{key}_mod", f"{key}_obs"]].resample("1YS").mean()
        yrs = ann.index.year
        ax.plot(yrs, ann[f"{key}_obs"], "k-o", ms=3, label="obs")
        ax.plot(yrs, ann[f"{key}_mod"], "r-o", ms=3, label="model")
        r_iav = _r(ann[f"{key}_mod"].to_numpy(), ann[f"{key}_obs"].to_numpy())
        ax.set_title(f"{label} interannual  r={r_iav:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)

        # --- col 3: summer (JJA) diurnal cycle ---
        ax = axes[row, 3]
        for sel, ls, lab in [(summer, "-", "JJA")]:
            dm = [np.nanmean(mod[sel & (hour == k)]) for k in range(24)]
            do = [np.nanmean(obs[sel & (hour == k)]) for k in range(24)]
            ax.plot(range(24), do, "k-o", ms=2, label="obs")
            ax.plot(range(24), dm, "r-o", ms=2, label="model")
        r_jja = _r(np.array(dm), np.array(do))
        ax.set_title(f"{label} JJA diurnal  r={r_jja:.2f}", fontsize=8)
        ax.set_xlabel("hour", fontsize=7); ax.tick_params(labelsize=6)

        # --- col 4: winter (DJF) diurnal cycle ---
        ax = axes[row, 4]
        dmw = [np.nanmean(mod[winter & (hour == k)]) for k in range(24)]
        dow = [np.nanmean(obs[winter & (hour == k)]) for k in range(24)]
        ax.plot(range(24), dow, "k-o", ms=2, label="obs")
        ax.plot(range(24), dmw, "r-o", ms=2, label="model")
        r_djf = _r(np.array(dmw), np.array(dow))
        ax.set_title(f"{label} DJF diurnal  r={r_djf:.2f}", fontsize=8)
        ax.set_xlabel("hour", fontsize=7); ax.tick_params(labelsize=6)

        r_hr = _r(mod, obs)
        summary[label] = dict(hourly=r_hr, monthly=r_mon, seasonal=r_seas,
                              iav=r_iav, jja_diurnal=r_jja, djf_diurnal=r_djf)

    fig.suptitle(f"{tag}  (mode={pred.attrs.get('mode','?')})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = os.path.join(out_dir, f"{tag}_evaluation.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  plot -> {p}")

    print(f"\n=== r summary ({tag}) ===")
    print(f"  {'flux':4s} {'hourly':>7s} {'monthly':>8s} {'seasonal':>9s} "
          f"{'iav':>6s} {'JJAdiu':>7s} {'DJFdiu':>7s}")
    for label, s in summary.items():
        print(f"  {label:4s} {s['hourly']:7.3f} {s['monthly']:8.3f} "
              f"{s['seasonal']:9.3f} {s['iav']:6.3f} {s['jja_diurnal']:7.3f} "
              f"{s['djf_diurnal']:7.3f}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out-dir", default="diagnostics/ec_site_eval")
    args = ap.parse_args()
    evaluate(args.pred, args.driver, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
