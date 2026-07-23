"""Publication figures for the offline eddy-covariance site validation.

Reads the per-site NetCDFs written by scripts/run/run_ec_site_evaluation.py (the
checked-in example set lives in scripts/validate/ec_site_example_data/) and makes
three figures:

  *_energy.png    rows = sites; the latent and sensible heat fluxes across the
                  mean summer daily cycle and the seasonal cycle.
  *_carbon.png    rows = sites; gross primary productivity, the latent-heat
                  partition into transpiration and soil evaporation, and the
                  soil-moisture time series at several depths.
  *_summary.png   pooled model-observation scatter (top) and per-site year-to-year
                  means (bottom, one panel per site on its own year axis).

Usage:
    python scripts/plot/plot_ec_site_publication.py <val_dir> <out_prefix> [corr_json]

See docs/land/ec_site_evaluation_runbook.md.
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

# Sites spanning climate regimes: mesic temperate broadleaf forest (US-MMS),
# montane needleleaf forest (DE-Obe), Mediterranean oak savanna (US-Ton), and
# humid temperate deciduous forest (DE-Hai).  Figures label rows by site ID alone.
SITES = ["US-MMS", "DE-Obe", "US-Ton", "DE-Hai"]

# Observed shallow soil-moisture sensor depth [cm], site BADM metadata
# (SWC_F_MDS_1): US-MMS 15, DE-Obe 10, US-Ton 2, DE-Hai ~8 (Hainich).
SWC_SENSOR_CM = {"US-MMS": 15.0, "DE-Obe": 10.0, "US-Ton": 2.0, "DE-Hai": 8.0}

C_LE = "#D55E00"; C_H = "#0072B2"; C_UST = "#6A51A3"; C_GPP = "#009E73"
C_CAN = "#009E73"; C_SOIL = "#E69F00"; C_OBS = "0.25"
PFT_COL = {"US-MMS": "#009E73", "DE-Obe": "#0072B2", "US-Ton": "#D55E00", "DE-Hai": "#CC79A7"}
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


def _pair(m, o, v):
    """Co-sampling mask: timesteps where the run is valid AND both the model and
    the observation are finite. Model and obs must be compared on the SAME
    timesteps, otherwise an obs gap (which the model never has) makes a model-obs
    discrepancy indistinguishable from a temporal-sampling artifact."""
    return np.asarray(v, bool) & np.isfinite(np.asarray(m)) & np.isfinite(np.asarray(o))


# --- stratified, temporally balanced annual mean --------------------------------
# A raw mean over a year's available days over-weights whatever season is best
# sampled. We stratify by calendar month and weight the months EQUALLY (mean of
# monthly means), which reduces to the naive mean when coverage is balanced and
# removes the seasonal-sampling bias when it is not. A year is reported only if the
# seasonal cycle is adequately covered.
_STRAT_MONTHS_MIN = 8     # months (of 12) that must be present in a year
_STRAT_DAYS_MIN = 5       # co-sampled days a month needs to count as "present"


def _annual_balanced(daily_m, daily_o, daily_oc, months):
    """Per-year month-stratified means of co-sampled daily (model, obs, obs_corr).

    Returns (years, m, o, oc, imbalance) where imbalance is max/min of the per-month
    day counts (1.0 = perfectly balanced), reported so we know when stratification
    actually changed anything. Years failing the seasonal-coverage gate are dropped.
    """
    n = min(len(daily_m), len(months))
    df = pd.DataFrame({"m": daily_m[:n], "o": daily_o[:n], "oc": daily_oc[:n],
                       "yr": pd.DatetimeIndex(months[:n]).year,
                       "mon": pd.DatetimeIndex(months[:n]).month}).dropna(subset=["m", "o"])
    yrs, mm, oo, cc, imb = [], [], [], [], []
    for y, g in df.groupby("yr"):
        counts = g.groupby("mon").size()
        present = counts[counts >= _STRAT_DAYS_MIN]
        if len(present) < _STRAT_MONTHS_MIN:
            continue                                   # seasonal cycle under-sampled
        by_month = g[g["mon"].isin(present.index)].groupby("mon").mean(numeric_only=True)
        yrs.append(int(y))
        mm.append(by_month["m"].mean()); oo.append(by_month["o"].mean())
        cc.append(by_month["oc"].mean())
        imb.append(float(present.max() / present.min()))
    return (np.array(yrs), np.array(mm), np.array(oo), np.array(cc),
            float(np.nanmax(imb)) if imb else np.nan)


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
    fig, ax = plt.subplots(len(SITES), 2, figsize=(8.5, 11))
    for r, site in enumerate(SITES):
        ds = _load(site)
        if ds is None:
            for c in range(2): ax[r, c].set_visible(False)
            continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds)
        t = pd.DatetimeIndex(ds.time.values); jja = np.isin(t.month.values, [6, 7, 8])
        bot = r == len(SITES) - 1
        hm, ho, ho_c = np.asarray(ds.h_mod), _obs_h(ds), _obs_h_corr(ds)
        lm, lo, lo_c = np.asarray(ds.le_mod), _obs_le(ds), _obs_le_corr(ds)

        # col0: sensible + latent heat, mean summer daily cycle.  Markers = raw eddy-
        # covariance observations; the shaded band spans up to the energy-balance-
        # closure-corrected value (the closure-uncertainty envelope).
        a = ax[r, 0]
        _row_label(a, site)
        hrs = range(24)
        # model and obs composited on the SAME (co-sampled) timesteps
        for mmod, om, oc, col in [(lm, lo, lo_c, C_LE), (hm, ho, ho_c, C_H)]:
            b = _pair(mmod, om, v)
            od, ocd = _diurnal(om, b, t, dt, jja), _diurnal(oc, b, t, dt, jja)
            a.fill_between(hrs, od, ocd, color=col, alpha=0.18, lw=0)
            a.plot(hrs, od, "o", color=col, ms=2.5, mfc="white")
            a.plot(hrs, _diurnal(mmod, b, t, dt, jja), "-", color=col, lw=1.6)
        a.set_ylabel("Heat flux (W m$^{-2}$)", fontsize=8); _hour_axis(a, bot)
        if r == 0:
            a.legend(handles=[Line2D([], [], color=C_LE, lw=1.6, label="Latent heat"),
                              Line2D([], [], color=C_H, lw=1.6, label="Sensible heat"),
                              Line2D([], [], color="0.4", marker="o", ls="", mfc="white",
                                     label="Observed"),
                              Patch(facecolor="0.4", alpha=0.25,
                                    label="Closure correction range")],
                     frameon=False, fontsize=7, loc="upper left")

        # col1: sensible + latent heat, seasonal (monthly), same raw markers +
        # closure-correction band.
        a = ax[r, 1]
        td = t[::int(round(86400 / dt))]

        def _mon(arr, b):
            dd = _daily(arr, b, dt)
            mo, mm, _ = _monthly(dd, dd, td[:len(dd)].month)
            return mo, mm
        # co-sample each pair, then composite the monthly cycle on shared timesteps
        for mmod, om, oc, col in [(lm, lo, lo_c, C_LE), (hm, ho, ho_c, C_H)]:
            b = _pair(mmod, om, v)
            mo, raw = _mon(om, b); _, cor = _mon(oc, b); _, mln = _mon(mmod, b)
            a.fill_between(mo, raw, cor, color=col, alpha=0.18, lw=0)
            a.plot(mo, raw, "o", color=col, ms=2.5, mfc="white")
            a.plot(mo, mln, "-", color=col, lw=1.6)
        a.set_ylabel("Heat flux (W m$^{-2}$)", fontsize=8); _month_axis(a, bot)

        for c in range(2): ax[r, c].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}_energy.png", bbox_inches="tight", dpi=300)
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

        # col0: GPP seasonal (model and obs co-sampled)
        a = ax[r, 0]
        gm_, go_ = np.asarray(ds.gpp_mod), np.asarray(ds.gpp_obs)
        bG = _pair(gm_, go_, v)
        gm = _daily(gm_, bG, dt); go = _daily(go_, bG, dt)
        mo, mm, moo = _monthly(gm, go, td[:len(gm)].month)
        a.plot(mo, moo, "o-", color=C_OBS, ms=2.5, lw=1.2, mfc="white", label="Observed")
        a.plot(mo, mm, "-", color=C_GPP, lw=1.7, label="Model")
        _r2box(a, _r2(gm, go))
        a.set_ylabel("GPP (µmol m$^{-2}$ s$^{-1}$)", fontsize=8); _month_axis(a, bot)
        _row_label(a, site)
        if r == 0: a.legend(frameon=False, fontsize=7, loc="upper right")

        # col1: latent-heat partition (transpiration vs soil evaporation), model
        # partition and obs total co-sampled on shared timesteps
        a = ax[r, 1]
        lm_, lo_ = np.asarray(ds.le_mod), _obs_le(ds); bLE = _pair(lm_, lo_, v)
        lec = _daily(np.asarray(ds.le_canopy), bLE, dt); les = _daily(np.asarray(ds.le_soil), bLE, dt)
        leo = _daily(lo_, bLE, dt); leo_c = _daily(_obs_le_corr(ds), bLE, dt)
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
    fig.savefig(f"{OUT}_carbon.png", bbox_inches="tight", dpi=300)
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
    fig = plt.figure(figsize=(14, 8.8))
    # 12-col grid so the top row holds 3 scatter panels (width 4) and the bottom
    # row holds ONE short interannual panel PER SITE (width 3) — each with its own
    # year axis, so sites with different coverage do not appear on a disjoint
    # shared timeline.
    gs = fig.add_gridspec(2, 12, height_ratios=[1.15, 1.0], hspace=0.5, wspace=2.2)
    data = {}
    for site in SITES:
        ds = _load(site)
        if ds is None: continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds); spd = int(round(86400 / dt))
        lm, lo = np.asarray(ds.le_mod), _obs_le(ds)
        hm, ho = np.asarray(ds.h_mod), _obs_h(ds)
        gm, go = np.asarray(ds.gpp_mod), np.asarray(ds.gpp_obs)
        bLE, bH, bG = _pair(lm, lo, v), _pair(hm, ho, v), _pair(gm, go, v)
        data[site] = dict(
            day=pd.DatetimeIndex(ds.time.values)[::spd].values,   # daily stamps
            # each model-obs pair aggregated on the SAME (co-sampled) timesteps
            LE=(_daily(lm, bLE, dt), _daily(lo, bLE, dt)),
            H=(_daily(hm, bH, dt), _daily(ho, bH, dt)),
            GPP=(_daily(gm, bG, dt), _daily(go, bG, dt)),
            # closure-corrected obs on the same LE/H paired timesteps
            LE_c=_daily(_obs_le_corr(ds), bLE, dt), H_c=_daily(_obs_h_corr(ds), bH, dt))
    VARS = [("LE", "Latent heat", "W m$^{-2}$"), ("H", "Sensible heat", "W m$^{-2}$"),
            ("GPP", "GPP", "µmol m$^{-2}$ s$^{-1}$")]
    for i, (key, lab, unit) in enumerate(VARS):
        a = fig.add_subplot(gs[0, 4 * i:4 * i + 4])
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
        st = _nse(M, O); stc_nse = np.nan
        txt = f"R$^2$={st['r2']:.2f}\nNSE={st['nse']:.2f}\nbias={st['bias']:+.2g}"
        if key in ("LE", "H"):
            Mc, Oc = [], []
            for s in data:
                mc, oc = data[s][key][0], data[s][f"{key}_c"]
                g = np.isfinite(mc) & np.isfinite(oc); Mc.append(mc[g]); Oc.append(oc[g])
            stc_nse = _nse(np.concatenate(Mc), np.concatenate(Oc))["nse"]
            txt += f"\nNSE$_{{corr}}$={stc_nse:.2f}"
        print(f"  pooled {key:3s}: NSE_raw={st['nse']:.3f}  NSE_corr={stc_nse:.3f}  "
              f"R2={st['r2']:.3f}  n={st['n']}  (co-sampled daily)")
        a.text(0.05, 0.95, txt, transform=a.transAxes, va="top", fontsize=8.5,
               bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.9))
        a.set_xlabel(f"Observed {lab} ({unit})"); a.set_ylabel(f"Modelled {lab} ({unit})")
        a.annotate(f"({chr(97 + i)})", xy=(0, 1.02), xycoords="axes fraction",
                   ha="left", va="bottom", fontweight="bold", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
    # --- bottom row: interannual means, ONE short panel per site (own year axis).
    # Co-sampled (model and obs on the same timesteps) and month-stratified so a
    # season-heavy sample does not bias the annual value. ---
    imbalance = {}
    def _annual(s, key):
        dm, do = data[s][key]
        doc = data[s][f"{key}_c"] if key in ("LE", "H") else do
        yrs, m, o, oc, imb = _annual_balanced(dm, do, doc, data[s]["day"])
        imbalance[(s, key)] = imb
        return pd.DataFrame({"m": m, "o": o, "oc": oc}, index=yrs)

    for j, s in enumerate(data):
        a = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        for key, col in [("LE", C_LE), ("H", C_H)]:      # energy fluxes, left axis
            df = _annual(s, key); xs = df.index.values.astype(int)
            a.fill_between(xs, df.o.values, df.oc.values, color=col, alpha=0.15, lw=0)
            a.plot(xs, df.o.values, "o--", color=col, ms=4, lw=1, alpha=0.7, mfc="white")
            a.plot(xs, df.m.values, "s-", color=col, ms=4, lw=1.6)
        a2 = a.twinx()                                   # GPP on the right axis
        dg = _annual(s, "GPP"); xg = dg.index.values.astype(int)
        a2.plot(xg, dg.o.values, "o--", color=C_GPP, ms=4, lw=1, alpha=0.7, mfc="white")
        a2.plot(xg, dg.m.values, "s-", color=C_GPP, ms=4, lw=1.6)
        a2.tick_params(axis="y", labelcolor=C_GPP, labelsize=7)
        a2.spines[["top"]].set_visible(False)
        xs_all = _annual(s, "LE").index.values.astype(int)
        if len(xs_all):
            a.set_xticks(xs_all); a.set_xlim(xs_all.min() - 0.5, xs_all.max() + 0.5)
        a.tick_params(axis="both", labelsize=7)
        a.set_title(s, fontsize=9.5, fontweight="bold")
        a.set_xlabel("Year", fontsize=8)
        if j == 0:
            a.set_ylabel("Latent, sensible heat (W m$^{-2}$)", fontsize=8)
        if j == len(data) - 1:
            a2.set_ylabel("GPP (µmol m$^{-2}$ s$^{-1}$)", fontsize=8, color=C_GPP)
        a.annotate(f"({chr(100 + j)})", xy=(0, 1.10), xycoords="axes fraction",
                   ha="left", va="bottom", fontweight="bold", fontsize=9)
        a.spines[["top"]].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], color=C_LE, lw=1.6, label="Latent heat"),
        Line2D([], [], color=C_H, lw=1.6, label="Sensible heat"),
        Line2D([], [], color=C_GPP, lw=1.6, label="GPP (right axis)"),
        Line2D([], [], color="0.35", marker="s", ls="-", label="Model"),
        Line2D([], [], color="0.35", marker="o", ls="--", mfc="white", label="Observed")],
        frameon=False, fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    im = [v for v in imbalance.values() if np.isfinite(v)]
    print(f"  interannual month-imbalance (max/min monthly day-count): "
          f"median={np.median(im):.2f} worst={np.max(im):.2f}  "
          f"(1.0 = perfectly balanced; stratification neutralizes it)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}_summary.png", bbox_inches="tight", dpi=300)
    print("saved", f"{OUT}_summary.png")


if __name__ == "__main__":
    fig_energy()
    fig_carbon()
    fig_summary()
