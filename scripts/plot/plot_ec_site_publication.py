"""Publication figures for the offline eddy-covariance site validation.

Reads the per-site NetCDFs written by scripts/tmp/_diag_canopy_run.py and makes
three figures:

  *_energy.png    rows = sites; friction velocity and the turbulent energy fluxes
                  (sensible + latent heat) across the mean diurnal and seasonal
                  cycles.
  *_carbon.png    rows = sites; gross primary productivity, the latent-heat
                  partition into transpiration and soil evaporation, and the
                  soil-moisture time series at several depths.
  *_summary.png   pooled model-observation scatter and year-to-year means for
                  latent heat, sensible heat and gross primary productivity.

Usage:
    python scripts/plot/plot_ec_site_publication.py <val_dir> <out_prefix> [corr_json]
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys, glob, json, os
import numpy as np
np.seterr(all="ignore")
import xarray as xr
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

VAL = sys.argv[1]
OUT = sys.argv[2]
CORR = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else {}

# Sites, ordered from tall wet forest to short dry grassland.  The second entry
# is kept only for internal ordering; figures label rows by site ID alone.
SITES = ["US-MMS", "DE-Obe", "US-Ton", "US-Var"]

# Observed shallow soil-moisture sensor depth [cm], site BADM metadata
# (SWC_F_MDS_1): US-MMS 15, DE-Obe 10, AU-How 10; US-Var has no registered depth
# (~2 cm per site reports).
SWC_SENSOR_CM = {"US-MMS": 15.0, "DE-Obe": 10.0, "US-Ton": 2.0, "US-Var": 2.0}

C_LE = "#D55E00"; C_H = "#0072B2"; C_UST = "#6A51A3"; C_GPP = "#009E73"
C_CAN = "#009E73"; C_SOIL = "#E69F00"; C_OBS = "0.25"
PFT_COL = {"US-MMS": "#009E73", "DE-Obe": "#0072B2", "US-Ton": "#D55E00", "US-Var": "#E69F00"}
MONTHS = list("JFMAMJJASOND")

plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.7, "figure.dpi": 140,
                     "font.family": "Inter", "xtick.direction": "in",
                     "ytick.direction": "in"})


def _load(site):
    f = glob.glob(f"{VAL}/{site}.nc") or glob.glob(f"{VAL}/{site}_*.nc")
    return xr.open_dataset(f[0]) if f else None


def _valid(ds):
    """Scoring mask.  Falls back to all-steps when a site has no strict-valid
    steps (e.g. AU-How has no soil-temperature sensor, so the reader's all-inputs-
    observed flag is empty); finite-observation intersection then does the masking.
    Soil temperature is not needed to validate the turbulent fluxes."""
    v = np.asarray(ds.valid).astype(bool)
    return v if v.sum() > 0 else np.ones_like(v)


def _obs_le(ds):
    """Observed latent heat, RAW eddy covariance (FLUXNET LE, not closure-adjusted).

    Raw eddy-covariance fluxes typically under-close the surface energy budget;
    the model closes energy exactly, so the honest primary target is the raw
    measurement, with the closure-corrected value shown as an uncertainty band
    (see :func:`_obs_le_corr`).  Falls back to the corrected field if a driver
    lacks the raw one."""
    return np.asarray(ds["le_obs"] if "le_obs" in ds.data_vars else ds["le_obs_corr"])


def _obs_h(ds):
    """Observed sensible heat, RAW eddy covariance (FLUXNET H, not closure-adjusted)."""
    return np.asarray(ds["h_obs"] if "h_obs" in ds.data_vars else ds["h_obs_corr"])


def _obs_le_corr(ds):
    """Energy-balance-closure-corrected observed latent heat (FLUXNET ET_CORR):
    the other edge of the closure-uncertainty band around the raw measurement."""
    return np.asarray(ds["le_obs_corr"] if "le_obs_corr" in ds.data_vars else ds["le_obs"])


def _obs_h_corr(ds):
    """Closure-corrected observed sensible heat (FLUXNET H_CORR); band edge."""
    return np.asarray(ds["h_obs_corr"] if "h_obs_corr" in ds.data_vars else ds["h_obs"])


def _daily(a, v, dt, mf=0.3):
    spd = int(round(86400 / dt)); n = (len(a) // spd) * spd
    a = np.where(v, a, np.nan)[:n].reshape(-1, spd)
    return np.where(np.mean(np.isfinite(a), 1) >= mf, np.nanmean(a, 1), np.nan)


def _r2(m, o):
    g = np.isfinite(m) & np.isfinite(o)
    if g.sum() < 20 or m[g].std() == 0 or o[g].std() == 0:
        return np.nan
    return np.corrcoef(m[g], o[g])[0, 1] ** 2


def _monthly(mod, obs, months):
    df = pd.DataFrame({"m": mod, "o": obs, "mon": months}).groupby("mon").mean()
    return df.index.values, df.m.values, df.o.values


def _diurnal(a, v, t, dt, season):
    hr = t.hour.values; out = np.full(24, np.nan)
    for h in range(24):
        m = v & season & (hr == h)
        if m.sum() > 10:
            out[h] = np.nanmean(np.where(m, a, np.nan))
    return out


def _r2box(a, val):
    a.text(0.04, 0.95, f"R$^2$={val:.2f}", transform=a.transAxes, va="top", fontsize=8,
           bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.8", alpha=0.85))


def _row_label(a, site):
    a.annotate(site, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
               va="center", ha="center", fontsize=10, fontweight="bold")


def _month_axis(a, bottom):
    a.set_xticks(range(1, 13)); a.set_xlim(0.5, 12.5)
    a.set_xticklabels(MONTHS if bottom else [])
    if bottom:
        a.set_xlabel("Month", fontsize=8)


def _hour_axis(a, bottom):
    a.set_xlim(0, 23); a.set_xticks([0, 6, 12, 18])
    if bottom:
        a.set_xlabel("Hour of day", fontsize=8)


# ---------------------------------------------------------------------------
# Figure 1: friction velocity + turbulent energy fluxes
# ---------------------------------------------------------------------------
def fig_energy():
    fig, ax = plt.subplots(len(SITES), 4, figsize=(14, 11))
    for r, site in enumerate(SITES):
        ds = _load(site)
        if ds is None:
            for c in range(4): ax[r, c].set_visible(False)
            continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds)
        t = pd.DatetimeIndex(ds.time.values); jja = np.isin(t.month.values, [6, 7, 8])
        bot = r == len(SITES) - 1
        um, uo = np.asarray(ds.ustar_mod), np.asarray(ds.ustar_obs)
        hm, ho, ho_c = np.asarray(ds.h_mod), _obs_h(ds), _obs_h_corr(ds)
        lm, lo, lo_c = np.asarray(ds.le_mod), _obs_le(ds), _obs_le_corr(ds)

        # col0: u* mean diurnal (JJA)
        a = ax[r, 0]
        a.plot(range(24), _diurnal(uo, v, t, dt, jja), "o-", color=C_OBS, ms=2.5, lw=1.2, mfc="white")
        a.plot(range(24), _diurnal(um, v, t, dt, jja), "-", color=C_UST, lw=1.6)
        _r2box(a, _r2(_daily(um, v, dt), _daily(uo, v, dt)))
        a.set_ylabel("Friction velocity (m s$^{-1}$)", fontsize=8); _hour_axis(a, bot)
        a.set_ylim(0.0, 0.8)
        _row_label(a, site)
        if r == 0:
            a.legend(handles=[Line2D([], [], color=C_UST, lw=1.6, label="Model"),
                              Line2D([], [], color=C_OBS, marker="o", ls="", mfc="white",
                                     label="Observed")],
                     frameon=False, fontsize=7, loc="upper right")

        # col1: u* seasonal
        a = ax[r, 1]
        mo, mm, moo = _monthly(_daily(um, v, dt), _daily(uo, v, dt),
                               t[::int(round(86400 / dt))][:len(_daily(um, v, dt))].month)
        a.plot(mo, moo, "o-", color=C_OBS, ms=2.5, lw=1.2, mfc="white")
        a.plot(mo, mm, "-", color=C_UST, lw=1.6)
        a.set_ylabel("Friction velocity (m s$^{-1}$)", fontsize=8); _month_axis(a, bot)
        a.set_ylim(0.0, 0.8)

        # col2: sensible + latent heat, mean diurnal (JJA).  Markers = raw eddy-
        # covariance observations; the shaded band spans up to the energy-balance-
        # closure-corrected value (the closure-uncertainty envelope).
        a = ax[r, 2]
        hrs = range(24)
        for om, oc, col in [(lo, lo_c, C_LE), (ho, ho_c, C_H)]:
            od, ocd = _diurnal(om, v, t, dt, jja), _diurnal(oc, v, t, dt, jja)
            a.fill_between(hrs, od, ocd, color=col, alpha=0.18, lw=0)
            a.plot(hrs, od, "o", color=col, ms=2.5, mfc="white")
        a.plot(hrs, _diurnal(lm, v, t, dt, jja), "-", color=C_LE, lw=1.6)
        a.plot(hrs, _diurnal(hm, v, t, dt, jja), "-", color=C_H, lw=1.6)
        a.set_ylabel("Heat flux (W m$^{-2}$)", fontsize=8); _hour_axis(a, bot)
        if r == 0:
            a.legend(handles=[Line2D([], [], color=C_LE, lw=1.6, label="Latent heat"),
                              Line2D([], [], color=C_H, lw=1.6, label="Sensible heat"),
                              Line2D([], [], color="0.4", marker="o", ls="", mfc="white",
                                     label="Observed"),
                              Patch(facecolor="0.4", alpha=0.25,
                                    label="Closure correction range")],
                     frameon=False, fontsize=7, loc="upper left")

        # col3: sensible + latent heat, seasonal (monthly), same raw markers +
        # closure-correction band.
        a = ax[r, 3]
        td = t[::int(round(86400 / dt))]

        def _mon(arr):
            dd = _daily(arr, v, dt)
            mo, mm, _ = _monthly(dd, dd, td[:len(dd)].month)
            return mo, mm
        for om, oc, col in [(lo, lo_c, C_LE), (ho, ho_c, C_H)]:
            mo, raw = _mon(om); _, cor = _mon(oc)
            a.fill_between(mo, raw, cor, color=col, alpha=0.18, lw=0)
            a.plot(mo, raw, "o", color=col, ms=2.5, mfc="white")
        for arr, col in [(lm, C_LE), (hm, C_H)]:
            mo, mm = _mon(arr); a.plot(mo, mm, "-", color=col, lw=1.6)
        a.set_ylabel("Heat flux (W m$^{-2}$)", fontsize=8); _month_axis(a, bot)

        for c in range(4): ax[r, c].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}_energy.png", bbox_inches="tight", dpi=200)
    print("saved", f"{OUT}_energy.png")


# ---------------------------------------------------------------------------
# Figure 2: productivity, latent-heat partition, soil moisture
# ---------------------------------------------------------------------------
def fig_carbon():
    fig, ax = plt.subplots(len(SITES), 3, figsize=(13, 11))
    for r, site in enumerate(SITES):
        ds = _load(site)
        if ds is None:
            for c in range(3): ax[r, c].set_visible(False)
            continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds)
        t = pd.DatetimeIndex(ds.time.values); td = t[::int(round(86400 / dt))]
        bot = r == len(SITES) - 1

        # col0: GPP seasonal
        a = ax[r, 0]
        gm = _daily(np.asarray(ds.gpp_mod), v, dt); go = _daily(np.asarray(ds.gpp_obs), v, dt)
        mo, mm, moo = _monthly(gm, go, td[:len(gm)].month)
        a.plot(mo, moo, "o-", color=C_OBS, ms=2.5, lw=1.2, mfc="white", label="Observed")
        a.plot(mo, mm, "-", color=C_GPP, lw=1.7, label="Model")
        _r2box(a, _r2(gm, go))
        a.set_ylabel("GPP (µmol m$^{-2}$ s$^{-1}$)", fontsize=8); _month_axis(a, bot)
        _row_label(a, site)
        if r == 0: a.legend(frameon=False, fontsize=7, loc="upper right")

        # col1: latent-heat partition (transpiration vs soil evaporation)
        a = ax[r, 1]
        lec = _daily(np.asarray(ds.le_canopy), v, dt); les = _daily(np.asarray(ds.le_soil), v, dt)
        leo = _daily(_obs_le(ds), v, dt); leo_c = _daily(_obs_le_corr(ds), v, dt)
        n = min(len(lec), len(td)); mon = td.month[:n]
        df = pd.DataFrame({"c": lec[:n], "s": les[:n], "o": leo[:n],
                           "oc": leo_c[:n], "m": mon}).groupby("m").mean()
        m = df.index.values
        a.bar(m, df.c, color=C_CAN, width=0.85, label="Transpiration")
        a.bar(m, df.s, bottom=df.c, color=C_SOIL, width=0.85, label="Soil evaporation")
        # Observed total: raw marker, whisker up to the closure-corrected value.
        a.vlines(m, np.minimum(df.o, df.oc), np.maximum(df.o, df.oc),
                 color=C_OBS, lw=1.0, alpha=0.6)
        a.plot(m, df.o, "o", color=C_OBS, ms=3, mfc="white", label="Observed total")
        a.set_ylabel("Latent heat (W m$^{-2}$)", fontsize=8); _month_axis(a, bot)
        if r == 0: a.legend(frameon=False, fontsize=7, loc="upper right")

        # col2: soil moisture at sensor depth + deeper layers
        a = ax[r, 2]
        z = np.asarray(ds.z) * 100.0; theta = np.asarray(ds.theta_prof)
        vall = np.ones_like(v, dtype=bool)
        sensor = SWC_SENSOR_CM.get(site, 5.0)
        for i_d, (dt_cm, col) in enumerate(zip([sensor, 30, 100], ["#2166ac", "#74add1", "#c6dbef"])):
            k = int(np.argmin(np.abs(z - dt_cm))); thd = _daily(theta[:, k], vall, dt)
            a.plot(np.arange(len(thd)), thd, color=col, lw=1.5 if i_d == 0 else 1.0,
                   label=f"{z[k]:.0f} cm")
        swo = _daily(np.asarray(ds.swc_obs), vall, dt)
        a.plot(np.arange(len(swo)), swo, color="k", lw=1.1, ls="--", label=f"Observed, {sensor:.0f} cm")
        a.set_ylabel("Soil moisture (m$^3$ m$^{-3}$)", fontsize=8)
        if bot: a.set_xlabel("Day of record", fontsize=8)
        if r == 0: a.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=2)

        for c in range(3): ax[r, c].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}_carbon.png", bbox_inches="tight", dpi=200)
    print("saved", f"{OUT}_carbon.png")


# ---------------------------------------------------------------------------
# Figure 3: pooled skill + year-to-year means (latent heat, sensible heat, GPP)
# ---------------------------------------------------------------------------
def _nse(m, o):
    g = np.isfinite(m) & np.isfinite(o); m, o = m[g], o[g]
    if len(m) < 20: return dict(r2=np.nan, nse=np.nan, bias=np.nan, n=0)
    return dict(r2=np.corrcoef(m, o)[0, 1] ** 2,
                nse=1 - np.sum((m - o) ** 2) / np.sum((o - o.mean()) ** 2),
                bias=np.mean(m - o), n=len(m))


def fig_summary():
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.33, wspace=0.28)
    data = {}
    for site in SITES:
        ds = _load(site)
        if ds is None: continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds)
        data[site] = dict(
            yr=pd.DatetimeIndex(ds.time.values)[::int(round(86400 / dt))].year,
            LE=(_daily(np.asarray(ds.le_mod), v, dt), _daily(_obs_le(ds), v, dt)),
            H=(_daily(np.asarray(ds.h_mod), v, dt), _daily(_obs_h(ds), v, dt)),
            GPP=(_daily(np.asarray(ds.gpp_mod), v, dt), _daily(np.asarray(ds.gpp_obs), v, dt)),
            # closure-corrected observation (band edge) for the energy fluxes only
            LE_c=_daily(_obs_le_corr(ds), v, dt), H_c=_daily(_obs_h_corr(ds), v, dt))
    VARS = [("LE", "Latent heat", "W m$^{-2}$"), ("H", "Sensible heat", "W m$^{-2}$"),
            ("GPP", "GPP", "µmol m$^{-2}$ s$^{-1}$")]
    for i, (key, lab, unit) in enumerate(VARS):
        a = fig.add_subplot(gs[0, i])
        M, O, C = [], [], []
        for s in data:
            m, o = data[s][key]; g = np.isfinite(m) & np.isfinite(o)
            M.append(m[g]); O.append(o[g]); C += [PFT_COL[s]] * int(g.sum())
        M, O = np.concatenate(M), np.concatenate(O)
        a.scatter(O, M, s=6, c=C, alpha=0.4, edgecolors="none", rasterized=True)
        lo, hi = np.nanpercentile(np.r_[O, M], 1), np.nanpercentile(np.r_[O, M], 99)
        a.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
        # Skill against the raw measurement (primary); for the energy fluxes also
        # report the Nash-Sutcliffe score against the closure-corrected value so
        # the reader sees the whole closure-uncertainty range.
        st = _nse(M, O)
        txt = f"R$^2$={st['r2']:.2f}\nNSE={st['nse']:.2f}\nbias={st['bias']:+.2g}"
        if key in ("LE", "H"):
            Mc, Oc = [], []
            for s in data:
                mc, oc = data[s][key][0], data[s][f"{key}_c"]
                g = np.isfinite(mc) & np.isfinite(oc); Mc.append(mc[g]); Oc.append(oc[g])
            stc = _nse(np.concatenate(Mc), np.concatenate(Oc))
            txt += f"\nNSE$_{{corr}}$={stc['nse']:.2f}"
        a.text(0.05, 0.95, txt, transform=a.transAxes, va="top", fontsize=8.5,
               bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.9))
        a.set_xlabel(f"Observed {lab} ({unit})"); a.set_ylabel(f"Modelled {lab} ({unit})")
        a.annotate(f"({chr(97 + i)})", xy=(0, 1.02), xycoords="axes fraction",
                   ha="left", va="bottom", fontweight="bold", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
    yrs = sorted({int(y) for s in data for y in np.unique(data[s]["yr"])})
    for j, (key, lab, unit) in enumerate(VARS):
        a = fig.add_subplot(gs[1, j])
        for s in data:
            m, o = data[s][key]; yr = data[s]["yr"][:len(m)].astype(int)
            oc = data[s][f"{key}_c"][:len(yr)] if key in ("LE", "H") else o[:len(yr)]
            grp = pd.DataFrame({"m": m[:len(yr)], "o": o[:len(yr)],
                                "oc": oc, "yr": yr}).groupby("yr")
            df = grp.mean()
            # Only plot an annual mean when the year has adequate valid daily
            # coverage (>=150 days), so a sparse or corrupt observation year does
            # not appear as a spurious annual value.
            ok = (grp["o"].count() >= 150) & (grp["m"].count() >= 150)
            df = df[ok.reindex(df.index).fillna(False).values]
            xs = df.index.values.astype(int)
            # Closure-uncertainty band between raw and corrected annual observation
            if key in ("LE", "H"):
                a.fill_between(xs, df.o.values, df.oc.values, color=PFT_COL[s],
                               alpha=0.15, lw=0)
            a.plot(xs, df.o.values, "o--", color=PFT_COL[s], ms=5, lw=1, alpha=0.7, mfc="white")
            a.plot(xs, df.m.values, "s-", color=PFT_COL[s], ms=5, lw=1.6, label=s)
        a.set_xlabel("Year"); a.set_ylabel(f"Annual mean {lab} ({unit})")
        a.set_xticks(yrs); a.set_xlim(yrs[0] - 0.5, yrs[-1] + 0.5)
        a.annotate(f"({chr(100 + j)})", xy=(0, 1.02), xycoords="axes fraction",
                   ha="left", va="bottom", fontweight="bold", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        if j == 0:
            a.plot([], [], "s-", color="0.35", label="Model")
            a.plot([], [], "o--", color="0.35", mfc="white", label="Observed")
            a.legend(frameon=False, fontsize=7.5, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUT}_summary.png", bbox_inches="tight", dpi=200)
    print("saved", f"{OUT}_summary.png")


if __name__ == "__main__":
    fig_energy()
    fig_carbon()
    fig_summary()
