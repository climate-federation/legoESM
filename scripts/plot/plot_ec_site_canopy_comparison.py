"""Paper-ready big-leaf vs multi-layer canopy comparison against EC-tower fluxes.

Reads two run_ec_site.py outputs for the SAME site/window (only --canopy differs)
plus the driver (for closure-corrected ET_CORR/H_CORR), and draws diurnal
composites of LE, H, GPP with:
  - observations: raw EC (points) and closure-corrected band (shaded)
  - Big-leaf (two-leaf) model line
  - Multi-layer (CLM-ML) model line
plus a skill panel (diurnal-RMSE + midday Bowen ratio) vs the corrected obs.

Usage:
  python plot_canopy_comparison.py <bigleaf.nc> <multilayer.nc> <driver.nc> \
      <start> <n> <out.png> [site_label]
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

# Palette (colour-blind safe): obs grey, big-leaf orange, multi-layer blue.
_C_OBS = "#444444"
_C_BIG = "#E69F00"     # big-leaf (two-leaf)
_C_ML = "#0072B2"      # multi-layer (CLM-ML)


def _diurnal(x, hours, nbin=24):
    m = np.full(nbin, np.nan)
    s = np.full(nbin, np.nan)
    for h in range(nbin):
        v = x[hours == h]
        v = v[np.isfinite(v)]
        if v.size:
            m[h], s[h] = v.mean(), v.std()
    return m, s


def _drmse(mod, obs, hours):
    a, _ = _diurnal(mod, hours)
    b, _ = _diurnal(obs, hours)
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[k] - b[k]) ** 2)))


def main(big_nc, ml_nc, driver_nc, start, n, out_png, site="US-MMS"):
    big = xr.open_dataset(big_nc)
    ml = xr.open_dataset(ml_nc)
    drv = xr.open_dataset(driver_nc).isel(time=slice(start, start + n))
    t = drv["time"].values.astype("datetime64[h]")
    hours = t.astype(int) % 24

    valid = np.asarray(big["score_valid"]).astype(bool)
    if not np.array_equal(valid, np.asarray(ml["score_valid"]).astype(bool)):
        raise SystemExit("arms scored on different masks — not controlled")

    def msk(x):
        x = np.asarray(x, dtype=float).copy()
        x[~valid] = np.nan
        return x

    LEc = msk(np.asarray(drv["ET_CORR"]) * constants.L_v / _SECONDS_PER_DAY)
    Hc = msk(np.asarray(drv["H_CORR"]))
    have_corr = np.isfinite(LEc).any()

    fluxes = [
        ("LE", "Latent heat  [W m$^{-2}$]", "le_mod", "le_obs", LEc),
        ("H", "Sensible heat  [W m$^{-2}$]", "h_mod", "h_obs", Hc),
        ("GPP", "GPP  [$\\mu$mol m$^{-2}$ s$^{-1}$]", "gpp_mod", "gpp_obs", None),
    ]
    hh = np.arange(24)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})

    skill = {}   # flux -> (rmse_big, rmse_ml, ref_name)
    for ax, (name, ylab, mk, ok, corr) in zip(axes[:3], fluxes):
        o_raw, _ = _diurnal(msk(big[ok]), hours)
        # obs: raw as points, corrected as shaded band (raw..corrected)
        ax.plot(hh, o_raw, "o", ms=3.5, color=_C_OBS, label="Obs (raw EC)", zorder=5)
        ref = msk(big[ok])
        if corr is not None and np.isfinite(corr).any():
            o_cor, _ = _diurnal(corr, hours)
            ax.fill_between(hh, o_raw, o_cor, color=_C_OBS, alpha=0.15,
                            label="Obs closure band")
            ax.plot(hh, o_cor, "--", lw=1.2, color=_C_OBS, alpha=0.8)
            ref = corr
        for ds, c, lab in ((big, _C_BIG, "Big-leaf"), (ml, _C_ML, "Multi-layer")):
            m, _ = _diurnal(msk(ds[mk]), hours)
            ax.plot(hh, m, "-", lw=2.2, color=c, label=lab)
        skill[name] = (_drmse(msk(big[mk]), ref, hours),
                       _drmse(msk(ml[mk]), ref, hours),
                       "corrected" if (corr is not None and np.isfinite(corr).any())
                       else "raw")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(ylab)
        ax.set_title(name, fontweight="bold")
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # skill panel: grouped diurnal-RMSE bars + Bowen text
    axS = axes[3]
    names = [f[0] for f in fluxes]
    xb = np.arange(len(names))
    w = 0.36
    rb = [skill[k][0] for k in names]
    rm = [skill[k][1] for k in names]
    axS.bar(xb - w / 2, rb, w, color=_C_BIG, label="Big-leaf")
    axS.bar(xb + w / 2, rm, w, color=_C_ML, label="Multi-layer")
    axS.set_xticks(xb)
    axS.set_xticklabels(names)
    axS.set_ylabel(f"Diurnal RMSE (vs {skill['LE'][2]} obs)")
    axS.set_title("Skill", fontweight="bold")
    axS.legend(fontsize=8, loc="upper right")
    axS.grid(alpha=0.25, axis="y")

    mid = (hours >= _MIDDAY[0]) & (hours < _MIDDAY[1])
    def bowen(ds):
        return (np.nanmean(msk(ds["h_mod"])[mid])
                / np.nanmean(msk(ds["le_mod"])[mid]))
    if have_corr:
        bo = np.nanmean(Hc[mid]) / np.nanmean(LEc[mid])
    else:
        bo = np.nanmean(msk(big["h_obs"])[mid]) / np.nanmean(msk(big["le_obs"])[mid])
    txt = (f"Midday Bowen H/LE\nObs {bo:.2f}\n"
           f"Big-leaf {bowen(big):.2f}\nMulti-layer {bowen(ml):.2f}")
    axS.text(0.03, 0.97, txt, transform=axS.transAxes, va="top", ha="left",
             fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    d0, d1 = str(t[0])[:10], str(t[-1])[:10]
    fig.suptitle(f"{site}  —  big-leaf vs multi-layer canopy vs eddy-covariance "
                 f"({d0} to {d1}, {int(valid.sum())} scored steps)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    print("wrote", out_png)
    for k in names:
        print(f"  {k:3s} diurnal-RMSE  big-leaf {skill[k][0]:6.2f}  "
              f"multi-layer {skill[k][1]:6.2f}  (vs {skill[k][2]} obs)")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]), int(a[5]), a[6],
         a[7] if len(a) > 7 else "US-MMS")
