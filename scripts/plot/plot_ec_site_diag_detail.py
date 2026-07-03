#!/usr/bin/env python
"""Per-site mechanistic diagnostic for the free-running EC-site runs.

For each site, overlays observed vs baseline vs strong model for the soil STATE
(top ~5 cm T and moisture) and the surface fluxes (LE, H, GPP), on a daily-mean
time axis, with NaN-revert steps marked.  Reveals the *structure* of the error
(seasonal bias, spikes, drift, instability) rather than a single skill number.

    baseline = canopy Kelvin-h_r only, uniform K
    strong   = S_top**5 evap resistance + K(z)=exp(-z/0.2 m) + roots 2 m
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "diagnostics/ec_site_fix_xsite/baseline"
STR = "diagnostics/ec_site_fix_xsite/strong"
OUT = "diagnostics/ec_site_diag_detail"
DRVDIR = "/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc"
os.makedirs(OUT, exist_ok=True)

# rows: (var, model_key, obs_key, label, daily)
_ROWS = [
    ("TS",  "ts_mod",  "ts_obs",  "soil T ~5 cm [K]", False),
    ("SWC", "swc_mod", "swc_obs", r"soil moisture ~5 cm [m$^3$m$^{-3}$]", False),
    ("LE",  "le_mod",  "le_obs",  "LE [W m$^{-2}$]", True),
    ("H",   "h_mod",   "h_obs",   "H [W m$^{-2}$]", True),
    ("GPP", "gpp_mod", "gpp_obs", r"GPP [$\mu$mol m$^{-2}$s$^{-1}$]", True),
]
_C = {"obs": "#000000", "base": "#0072B2", "strong": "#D55E00"}


def _rstat(m, o, v):
    k = v & np.isfinite(m) & np.isfinite(o)
    if k.sum() < 10 or np.std(m[k]) == 0 or np.std(o[k]) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.corrcoef(m[k], o[k])[0, 1]),
            float(np.sqrt(np.mean((m[k] - o[k]) ** 2))),
            float(np.mean(m[k] - o[k])))


def _time_axis(site, n):
    """Best-year datetime axis from the driver (steps align 1:1 to the run)."""
    # the run used --select-best-year; recover the same window by matching length
    drv = os.path.join(DRVDIR, f"{site}_driver_v2_gapfree.nc")
    if not os.path.isfile(drv):
        return pd.RangeIndex(n)
    t = pd.DatetimeIndex(xr.open_dataset(drv)["time"].values)
    # the run is a contiguous n-step slice; find the calendar year of length n
    yrs = t.year.to_numpy()
    for y in np.unique(yrs):
        idx = np.nonzero(yrs == y)[0]
        if len(idx) == n:
            return t[idx]
    return pd.RangeIndex(n)


def main() -> int:
    files = sorted(glob.glob(os.path.join(STR, "*_ec_prognostic.nc")))
    summary = []
    for sf in files:
        site = xr.open_dataset(sf).attrs.get("site", "?").replace("_driver_v2_gapfree.nc", "")
        bf = os.path.join(BASE, os.path.basename(sf))
        if not os.path.isfile(bf):
            continue
        ds_s, ds_b = xr.open_dataset(sf), xr.open_dataset(bf)
        pft = ds_s.attrs.get("pft", "?")
        n = ds_s.sizes["time"]
        t = _time_axis(site, n)
        sv = (ds_s.score_valid.values if "score_valid" in ds_s else ds_s.valid.values).astype(bool)
        soilf = (ds_s.soil_filled.values == 0) if "soil_filled" in ds_s else np.ones(n, bool)
        rev = ds_s.reverted.values.astype(bool) if "reverted" in ds_s else np.zeros(n, bool)

        fig, axes = plt.subplots(len(_ROWS), 1, figsize=(12, 2.1 * len(_ROWS)), sharex=True)
        for ax, (name, mk, ok, ylab, daily) in zip(axes, _ROWS):
            vmask = sv & (soilf if name in ("TS", "SWC") else np.ones(n, bool))
            obs = np.where(vmask, ds_s[ok].values, np.nan)
            mb = np.where(sv, ds_b[mk].values, np.nan)
            ms = np.where(sv, ds_s[mk].values, np.nan)
            if daily:
                df = pd.DataFrame({"o": obs, "b": mb, "s": ms}, index=pd.DatetimeIndex(t)
                                  if not isinstance(t, pd.RangeIndex) else None)
                if not isinstance(t, pd.RangeIndex):
                    df = df.resample("1D").mean()
                    xo, xs_ = df.index, df.index
                    obs, mb, ms = df["o"].values, df["b"].values, df["s"].values
                else:
                    xo = xs_ = np.arange(n)
            else:
                xo = xs_ = (t if not isinstance(t, pd.RangeIndex) else np.arange(n))
            ax.plot(xo, obs, color=_C["obs"], lw=1.2, label="obs", zorder=3)
            ax.plot(xs_, mb, color=_C["base"], lw=0.9, alpha=0.8, label="baseline")
            ax.plot(xs_, ms, color=_C["strong"], lw=1.0, alpha=0.9, label="strong")
            rb, eb, bb = _rstat(ds_b[mk].values, ds_s[ok].values, vmask)
            rs, es, bs = _rstat(ds_s[mk].values, ds_s[ok].values, vmask)
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(f"{name}:  base r={rb:.2f} bias={bb:+.2g}   |   "
                         f"strong r={rs:.2f} bias={bs:+.2g}", fontsize=8.5, loc="left")
            ax.grid(alpha=0.25)
        # mark reverts on the top axis
        if rev.any() and not isinstance(t, pd.RangeIndex):
            for xr_ in np.asarray(t)[rev]:
                axes[0].axvline(xr_, color="red", lw=0.6, alpha=0.5)
            axes[0].set_title(axes[0].get_title() + f"   [{int(rev.sum())} NaN-reverts (red)]", fontsize=8.5)
        axes[0].legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
        fig.suptitle(f"{site}  ({pft})  —  free-running soil state + fluxes  (obs vs baseline vs strong)",
                     fontsize=11, y=0.995)
        fig.tight_layout()
        p = os.path.join(OUT, f"{site}_diag.png")
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
        summary.append((site, pft, int(rev.sum())))
        print(f"  plot -> {p}   ({pft}, reverts={int(rev.sum())})")
    print(f"\n{len(summary)} site diagnostics -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
