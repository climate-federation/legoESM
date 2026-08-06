"""Pooled multi-site summary: big-leaf vs multi-layer canopy vs EC fluxes.

Given a list of (site, bigleaf.nc, multilayer.nc, driver.nc, start, n), pools the
scored diurnal-composite points across sites and draws model-vs-obs scatter for
LE, H, GPP (one panel each) with a 1:1 line, plus a per-site midday Bowen-ratio
bar. Obs reference = closure-corrected where available (LE/H), raw for GPP.

Usage: edit the RUNS list below, then `python plot_canopy_summary.py <out.png>`.
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants

_SECONDS_PER_DAY = 86400.0
_MIDDAY = (10, 15)
_C_BIG, _C_ML = "#E69F00", "#0072B2"


def _diurnal(x, hours, nbin=24):
    o = np.full(nbin, np.nan)
    for h in range(nbin):
        v = x[hours == h]
        v = v[np.isfinite(v)]
        if v.size:
            o[h] = v.mean()
    return o


def _load(site, big_nc, ml_nc, driver_nc, start, n):
    big, ml = xr.open_dataset(big_nc), xr.open_dataset(ml_nc)
    drv = xr.open_dataset(driver_nc).isel(time=slice(start, start + n))
    t = drv["time"].values.astype("datetime64[h]")
    hours = t.astype(int) % 24
    valid = np.asarray(big["score_valid"]).astype(bool)
    def msk(x):
        x = np.asarray(x, float).copy(); x[~valid] = np.nan; return x
    LEc = msk(np.asarray(drv["ET_CORR"]) * constants.L_v / _SECONDS_PER_DAY)
    Hc = msk(np.asarray(drv["H_CORR"]))
    corr = np.isfinite(LEc).any()
    obs = {"LE": (LEc if corr else msk(big["le_obs"])),
           "H": (Hc if corr else msk(big["h_obs"])),
           "GPP": msk(big["gpp_obs"])}
    out = {"site": site, "hours": hours, "corr": corr, "obs": obs, "msk": msk,
           "big": big, "ml": ml}
    return out


def main(runs, out_png):
    fluxes = [("LE", "le_mod", "Latent heat [W m$^{-2}$]"),
              ("H", "h_mod", "Sensible heat [W m$^{-2}$]"),
              ("GPP", "gpp_mod", "GPP [$\\mu$mol m$^{-2}$ s$^{-1}$]")]
    data = [_load(*r) for r in runs]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})

    for ax, (name, mk, lab) in zip(axes[:3], fluxes):
        allo, allb, allm = [], [], []
        for d in data:
            o = _diurnal(d["obs"][name], d["hours"])
            b = _diurnal(d["msk"](d["big"][mk]), d["hours"])
            m = _diurnal(d["msk"](d["ml"][mk]), d["hours"])
            allo.append(o); allb.append(b); allm.append(m)
        o = np.concatenate(allo); b = np.concatenate(allb); m = np.concatenate(allm)
        ax.scatter(o, b, s=16, c=_C_BIG, alpha=0.7, label="Big-leaf", edgecolors="none")
        ax.scatter(o, m, s=16, c=_C_ML, alpha=0.7, label="Multi-layer", edgecolors="none")
        lim = np.nanmax([np.nanmax(o), np.nanmax(b), np.nanmax(m)]) * 1.05
        lo = min(0, np.nanmin(o))
        ax.plot([lo, lim], [lo, lim], "k-", lw=0.8, alpha=0.5)
        def rmse(mod):
            k = np.isfinite(mod) & np.isfinite(o)
            return np.sqrt(np.mean((mod[k] - o[k]) ** 2))
        ax.set_title(f"{name}\nRMSE big {rmse(b):.1f} / ml {rmse(m):.1f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Observed"); ax.set_ylabel("Modelled")
        ax.set_xlim(lo, lim); ax.set_ylim(lo, lim)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    # per-site midday Bowen bars
    axB = axes[3]
    sites = [d["site"] for d in data]
    xb = np.arange(len(sites)); w = 0.27
    def midbowen(d, arr_h, arr_le):
        mid = (d["hours"] >= _MIDDAY[0]) & (d["hours"] < _MIDDAY[1])
        return np.nanmean(d["msk"](arr_h)[mid]) / np.nanmean(d["msk"](arr_le)[mid])
    bo = [np.nanmean(d["obs"]["H"][(d["hours"] >= 10) & (d["hours"] < 15)])
          / np.nanmean(d["obs"]["LE"][(d["hours"] >= 10) & (d["hours"] < 15)]) for d in data]
    bb = [midbowen(d, d["big"]["h_mod"], d["big"]["le_mod"]) for d in data]
    bm = [midbowen(d, d["ml"]["h_mod"], d["ml"]["le_mod"]) for d in data]
    axB.bar(xb - w, bo, w, color="#444444", label="Obs")
    axB.bar(xb, bb, w, color=_C_BIG, label="Big-leaf")
    axB.bar(xb + w, bm, w, color=_C_ML, label="Multi-layer")
    axB.set_xticks(xb); axB.set_xticklabels(sites, rotation=20, ha="right", fontsize=8)
    axB.set_ylabel("Midday Bowen ratio H/LE")
    axB.set_title("Energy partition", fontsize=10, fontweight="bold")
    axB.legend(fontsize=8); axB.grid(alpha=0.25, axis="y")

    fig.suptitle("Big-leaf vs multi-layer canopy vs eddy-covariance — pooled "
                 f"({len(sites)} sites, closure-corrected LE/H)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160)
    print("wrote", out_png, "sites:", sites)


if __name__ == "__main__":
    import json
    runs = json.loads(sys.argv[2])   # list of [site,big,ml,driver,start,n]
    main(runs, sys.argv[1])
