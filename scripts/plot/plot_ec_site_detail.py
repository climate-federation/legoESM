#!/usr/bin/env python
"""Per-site EC time-series detail: monthly time series, mean seasonal cycle (MSC)
and interannual variability (IAV) for GPP / LE / H / EF, overlaying observations,
diagnostic and prognostic model output, annotated with r AND RMSE.

One figure per site (4 fluxes x 3 columns), so the actual dynamics behind the
cross-site bar charts are visible.  The datetime axis comes from the site's
driver NetCDF (aligned to the prediction by index — the cross-site runs use the
full record, so index<->time is 1:1).

Usage
-----
    JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_detail.py \
        --diag-dir diagnostics/ec_site_xsite \
        --prog-dir diagnostics/ec_site_xsite_prog \
        --driver-dir <DifferBESS>/data/sitelevel/nc \
        --out-dir diagnostics/ec_site_detail
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

_FLUXES = [("gpp", "GPP", r"$\mu$mol m$^{-2}$ s$^{-1}$"),
           ("le", "LE", "W m$^{-2}$"),
           ("h", "H", "W m$^{-2}$"),
           ("ef", "EF", "-")]
_EF_FLOOR = 50.0

# Okabe-Ito colour-blind-safe palette.
_C = {"obs": "#000000", "diag": "#0072B2", "prog": "#D55E00"}
_LBL = {"obs": "observed", "diag": "diagnostic", "prog": "prognostic"}


def _set_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3, "lines.markersize": 3.5,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.axisbelow": True,
        "font.family": "DejaVu Sans", "mathtext.default": "regular",
    })


def _series(ds):
    """Return a dict of masked model/obs arrays (gpp/le/h/ef) for one nc."""
    base = (ds["score_valid"].values if "score_valid" in ds
            else ds["valid"].values).astype(bool)
    out = {}
    for key, *_ in _FLUXES:
        if key == "ef":
            lem, leo = out["le"][0], out["le"][1]
            hm, ho = out["h"][0], out["h"][1]
            am, ao = lem + hm, leo + ho
            m = np.where(am > _EF_FLOOR, lem / am, np.nan)
            o = np.where(ao > _EF_FLOOR, leo / ao, np.nan)
            out[key] = (np.clip(m, -0.5, 1.5), np.clip(o, -0.5, 1.5))
        else:
            m = np.where(base, ds[f"{key}_mod"].values, np.nan)
            o = np.where(base, ds[f"{key}_obs"].values, np.nan)
            out[key] = (m, o)
    return out


def _rstat(m, o):
    k = np.isfinite(m) & np.isfinite(o)
    if k.sum() < 5 or np.std(m[k]) == 0 or np.std(o[k]) == 0:
        return np.nan, np.nan
    return (float(np.corrcoef(m[k], o[k])[0, 1]),
            float(np.sqrt(np.mean((m[k] - o[k]) ** 2))))


def plot_site(site, pft, t, diag, prog, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    cols = ["obs", "diag"] + (["prog"] if prog is not None else [])
    fig, axes = plt.subplots(len(_FLUXES), 3, figsize=(13, 2.5 * len(_FLUXES)),
                             constrained_layout=True)
    month = t.month.to_numpy()
    for row, (key, label, unit) in enumerate(_FLUXES):
        df = pd.DataFrame({"obs": diag[key][1], "diag": diag[key][0]}, index=t)
        if prog is not None:
            df["prog"] = prog[key][0]

        def _draw(ax, xs, series, markers=False, robust=False):
            allv = []
            for c in cols:
                if c not in series:
                    continue
                v = np.asarray(series[c], dtype=float)
                allv.append(v)
                ax.plot(xs, v, marker="o" if markers else None,
                        color=_C[c], label=_LBL[c],
                        zorder=3 if c == "obs" else 2,
                        alpha=1.0 if c == "obs" else 0.9)
            if robust and allv:
                a = np.concatenate(allv); a = a[np.isfinite(a)]
                if a.size > 5:
                    lo, hi = np.percentile(a, [1, 99])
                    pad = 0.08 * (hi - lo + 1e-9)
                    ax.set_ylim(lo - pad, hi + pad)

        # --- col 0: monthly time series ---
        ax = axes[row, 0]
        mon = df.resample("1MS").mean()
        _draw(ax, mon.index, {c: mon[c] for c in cols}, robust=True)
        ax.set_ylabel(f"{label}\n({unit})")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=max(1, (t[-1].year - t[0].year)//6 + 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        rd, ed = _rstat(diag[key][0], df["obs"].values)
        sub = f"$r$={rd:.2f}, RMSE={ed:.2g}"
        if prog is not None:
            rp, ep = _rstat(prog[key][0], df["obs"].values)
            sub = (f"diag $r$={rd:.2f} RMSE={ed:.2g}   "
                   f"prog $r$={rp:.2f} RMSE={ep:.2g}")
        ax.set_title(sub, fontsize=7.5)
        if row == 0:
            ax.legend(loc="upper right", ncol=len(cols), columnspacing=0.8,
                      handlelength=1.2)
        if row == len(_FLUXES) - 1:
            ax.set_xlabel("year")

        # --- col 1: mean seasonal cycle ---
        ax = axes[row, 1]
        msc = {c: [np.nanmean(df[c].values[month == k]) for k in range(1, 13)]
               for c in cols}
        _draw(ax, np.arange(1, 13), msc, markers=True)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_title("mean seasonal cycle")
        if row == len(_FLUXES) - 1:
            ax.set_xlabel("month")

        # --- col 2: interannual variability (annual means) ---
        ax = axes[row, 2]
        adf = df.resample("1YS").mean()
        yrs = adf.index.year.to_numpy()
        _draw(ax, yrs, {c: adf[c].to_numpy() for c in cols}, markers=True)
        rd_i, ed_i = _rstat(adf["diag"].to_numpy(), adf["obs"].to_numpy())
        ti = f"interannual   diag $r$={rd_i:.2f}"
        if prog is not None:
            rp_i, _ = _rstat(adf["prog"].to_numpy(), adf["obs"].to_numpy())
            ti += f"  prog $r$={rp_i:.2f}"
        ax.set_title(ti, fontsize=7.5)
        if row == len(_FLUXES) - 1:
            ax.set_xlabel("year")

    fig.suptitle(f"{site}  ({pft})   —   monthly time series  |  mean seasonal "
                 f"cycle  |  interannual variability", fontsize=11, y=1.02)
    p = os.path.join(out_dir, f"{site}_detail.png")
    fig.savefig(p); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", default="diagnostics/ec_site_xsite")
    ap.add_argument("--prog-dir", default="diagnostics/ec_site_xsite_prog")
    ap.add_argument("--driver-dir", required=True,
                    help="dir with <SITE>_driver_v2.nc (for the datetime axis)")
    ap.add_argument("--out-dir", default="diagnostics/ec_site_detail")
    ap.add_argument("--sites", nargs="*", default=None,
                    help="optional subset of site ids")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _set_pub_style()

    diag_files = sorted(glob.glob(os.path.join(args.diag_dir, "*_ec_*.nc")))
    done = []
    for dpath in diag_files:
        dds = xr.open_dataset(dpath)
        site = str(dds.attrs.get("site", "")).replace("_driver_v2.nc", "")
        if args.sites and site not in args.sites:
            continue
        pft = str(dds.attrs.get("pft", "?"))
        drv = os.path.join(args.driver_dir, f"{site}_driver_v2.nc")
        if not os.path.isfile(drv):
            print(f"  skip {site}: no driver"); continue
        t = pd.DatetimeIndex(xr.open_dataset(drv)["time"].values)
        n = min(len(t), dds.sizes["time"])
        t = t[:n]
        diag = {k: (m[:n], o[:n]) for k, (m, o) in _series(dds).items()}
        ppath = os.path.join(args.prog_dir, os.path.basename(dpath).replace(
            "_ec_diagnostic.nc", "_ec_prognostic.nc"))
        prog = None
        if os.path.isfile(ppath):
            pds = xr.open_dataset(ppath)
            np_ = min(n, pds.sizes["time"])
            prog = {k: (m[:np_], o[:np_]) for k, (m, o) in _series(pds).items()}
            if np_ != n:   # length mismatch -> drop prog (rare)
                prog = None
        try:
            p = plot_site(site, pft, t, diag, prog, args.out_dir)
            done.append(site); print(f"  plot -> {p}")
        except Exception as e:  # noqa: BLE001 -- per-site robustness
            print(f"  FAILED {site}: {e}")
    print(f"\n{len(done)} site detail figures -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
