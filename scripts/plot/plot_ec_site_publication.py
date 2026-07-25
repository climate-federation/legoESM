"""Publication figure for the offline eddy-covariance site validation.

Reads the per-site NetCDFs written by scripts/run/run_ec_site_evaluation.py (the
checked-in example set lives in scripts/validate/ec_site_example_data/) and makes
one combined figure:

  *_combined.png  top row = pooled model-observation skill scatter (latent heat,
                  sensible heat, GPP); then one row per site with the energy
                  diurnal + seasonal cycles, the GPP seasonal cycle, the latent-
                  heat partition, and soil moisture.  Panels are labelled (a),
                  (b), ...  Model and observations are co-sampled (compared only on
                  timesteps where both are available).  Interannual-variability
                  panels are intentionally omitted: four years per site is too few
                  to validate interannual variability.

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


def _r2(m, o):
    g = np.isfinite(m) & np.isfinite(o)
    if g.sum() < 20 or m[g].std() == 0 or o[g].std() == 0:
        return np.nan
    return np.corrcoef(m[g], o[g])[0, 1] ** 2


def _nse(m, o):
    g = np.isfinite(m) & np.isfinite(o); m, o = m[g], o[g]
    if len(m) < 20: return dict(r2=np.nan, nse=np.nan, bias=np.nan, n=0)
    return dict(r2=np.corrcoef(m, o)[0, 1] ** 2,
                nse=1 - np.sum((m - o) ** 2) / np.sum((o - o.mean()) ** 2),
                bias=np.mean(m - o), n=len(m))


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
    a.annotate(site, xy=(-0.34, 0.5), xycoords="axes fraction", rotation=90,
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
# One combined figure: pooled skill (top) + per-site process panels
# ---------------------------------------------------------------------------
def fig_combined():
    fig = plt.figure(figsize=(17, 18.5))
    # 15-col grid: the top row holds 3 scatter panels (width 5); each site row
    # holds 5 process panels (width 3): energy diurnal, energy seasonal, GPP,
    # latent-heat partition, soil moisture.
    gs = fig.add_gridspec(5, 15, hspace=0.62, wspace=1.75,
                          height_ratios=[1.3, 1, 1, 1, 1])
    labs = iter(f"({c})" for c in "abcdefghijklmnopqrstuvwxyz")

    def _tag(a):
        a.annotate(next(labs), xy=(0, 1.05), xycoords="axes fraction",
                   ha="left", va="bottom", fontweight="bold", fontsize=10)

    # ---- Row 0: pooled model-obs skill scatter (co-sampled daily) ----
    data = {}
    for site in SITES:
        ds = _load(site)
        if ds is None: continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds)
        lm, lo = np.asarray(ds.le_mod), _obs_le(ds)
        hm, ho = np.asarray(ds.h_mod), _obs_h(ds)
        gm, go = np.asarray(ds.gpp_mod), np.asarray(ds.gpp_obs)
        bLE, bH, bG = _pair(lm, lo, v), _pair(hm, ho, v), _pair(gm, go, v)
        data[site] = dict(
            LE=(_daily(lm, bLE, dt), _daily(lo, bLE, dt)),
            H=(_daily(hm, bH, dt), _daily(ho, bH, dt)),
            GPP=(_daily(gm, bG, dt), _daily(go, bG, dt)),
            LE_c=_daily(_obs_le_corr(ds), bLE, dt), H_c=_daily(_obs_h_corr(ds), bH, dt))
    for i, (key, lab, unit) in enumerate(
            [("LE", "Latent heat", "W m$^{-2}$"), ("H", "Sensible heat", "W m$^{-2}$"),
             ("GPP", "GPP", "µmol m$^{-2}$ s$^{-1}$")]):
        a = fig.add_subplot(gs[0, 5 * i:5 * i + 5])
        M, O, C = [], [], []
        for s in data:
            m, o = data[s][key]; g = np.isfinite(m) & np.isfinite(o)
            M.append(m[g]); O.append(o[g]); C += [PFT_COL[s]] * int(g.sum())
        M, O = np.concatenate(M), np.concatenate(O)
        a.scatter(O, M, s=6, c=C, alpha=0.4, edgecolors="none", rasterized=True)
        lo_, hi_ = np.nanpercentile(np.r_[O, M], 1), np.nanpercentile(np.r_[O, M], 99)
        a.plot([lo_, hi_], [lo_, hi_], "--", color="0.4", lw=1)
        a.set_xlim(lo_, hi_); a.set_ylim(lo_, hi_); a.set_aspect("equal")
        # skill vs raw (primary); for the energy fluxes also vs closure-corrected
        st = _nse(M, O); stc = np.nan
        txt = f"R$^2$={st['r2']:.2f}\nNSE={st['nse']:.2f}\nbias={st['bias']:+.2g}"
        if key in ("LE", "H"):
            Mc, Oc = [], []
            for s in data:
                mc, oc = data[s][key][0], data[s][f"{key}_c"]
                g = np.isfinite(mc) & np.isfinite(oc); Mc.append(mc[g]); Oc.append(oc[g])
            stc = _nse(np.concatenate(Mc), np.concatenate(Oc))["nse"]
            txt += f"\nNSE$_{{corr}}$={stc:.2f}"
        print(f"  pooled {key:3s}: NSE_raw={st['nse']:.3f} NSE_corr={stc:.3f} "
              f"bias={st['bias']:+.2g} n={st['n']} (co-sampled)")
        a.text(0.05, 0.95, txt, transform=a.transAxes, va="top", fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.9))
        a.set_xlabel(f"Observed {lab} ({unit})", fontsize=8)
        a.set_ylabel(f"Modelled {lab} ({unit})", fontsize=8)
        a.spines[["top", "right"]].set_visible(False); _tag(a)

    # ---- Rows 1-4: per-site process panels ----
    for r, site in enumerate(SITES):
        ds = _load(site)
        if ds is None: continue
        dt = float(ds.attrs["dt_s"]); v = _valid(ds); gr = r + 1
        t = pd.DatetimeIndex(ds.time.values); jja = np.isin(t.month.values, [6, 7, 8])
        td = t[::int(round(86400 / dt))]; bot = r == len(SITES) - 1
        hm, ho, ho_c = np.asarray(ds.h_mod), _obs_h(ds), _obs_h_corr(ds)
        lm, lo, lo_c = np.asarray(ds.le_mod), _obs_le(ds), _obs_le_corr(ds)

        # energy — mean summer daily cycle (model and obs co-sampled)
        a = fig.add_subplot(gs[gr, 0:3]); _row_label(a, site); hrs = range(24)
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
                              Patch(facecolor="0.4", alpha=0.25, label="Closure range")],
                     frameon=False, fontsize=6.5, loc="upper left")
        a.spines[["top", "right"]].set_visible(False); _tag(a)

        # energy — seasonal cycle
        a = fig.add_subplot(gs[gr, 3:6])

        def _mon(arr, b):
            dd = _daily(arr, b, dt)
            mo, mm, _ = _monthly(dd, dd, td[:len(dd)].month)
            return mo, mm
        for mmod, om, oc, col in [(lm, lo, lo_c, C_LE), (hm, ho, ho_c, C_H)]:
            b = _pair(mmod, om, v)
            mo, raw = _mon(om, b); _, cor = _mon(oc, b); _, mln = _mon(mmod, b)
            a.fill_between(mo, raw, cor, color=col, alpha=0.18, lw=0)
            a.plot(mo, raw, "o", color=col, ms=2.5, mfc="white")
            a.plot(mo, mln, "-", color=col, lw=1.6)
        a.set_ylabel("Heat flux (W m$^{-2}$)", fontsize=8); _month_axis(a, bot)
        a.spines[["top", "right"]].set_visible(False); _tag(a)

        # GPP seasonal cycle
        a = fig.add_subplot(gs[gr, 6:9])
        gm_, go_ = np.asarray(ds.gpp_mod), np.asarray(ds.gpp_obs); bG = _pair(gm_, go_, v)
        gm = _daily(gm_, bG, dt); go = _daily(go_, bG, dt)
        mo, mm, moo = _monthly(gm, go, td[:len(gm)].month)
        a.plot(mo, moo, "o-", color=C_OBS, ms=2.5, lw=1.2, mfc="white", label="Observed")
        a.plot(mo, mm, "-", color=C_GPP, lw=1.7, label="Model")
        _r2box(a, _r2(gm, go))
        a.set_ylabel("GPP (µmol m$^{-2}$ s$^{-1}$)", fontsize=8); _month_axis(a, bot)
        if r == 0: a.legend(frameon=False, fontsize=6.5, loc="upper right")
        a.spines[["top", "right"]].set_visible(False); _tag(a)

        # latent-heat partition (transpiration vs soil evaporation)
        a = fig.add_subplot(gs[gr, 9:12])
        lmf, lof = np.asarray(ds.le_mod), _obs_le(ds); bLE = _pair(lmf, lof, v)
        lec = _daily(np.asarray(ds.le_canopy), bLE, dt)
        les = _daily(np.asarray(ds.le_soil), bLE, dt)
        leo = _daily(lof, bLE, dt); leo_c = _daily(_obs_le_corr(ds), bLE, dt)
        n = min(len(lec), len(td)); mon = td.month[:n]
        df = pd.DataFrame({"c": lec[:n], "s": les[:n], "o": leo[:n],
                           "oc": leo_c[:n], "m": mon}).groupby("m").mean()
        mx = df.index.values
        a.bar(mx, df.c, color=C_CAN, width=0.85, label="Transpiration")
        a.bar(mx, df.s, bottom=df.c, color=C_SOIL, width=0.85, label="Soil evaporation")
        a.vlines(mx, np.minimum(df.o, df.oc), np.maximum(df.o, df.oc),
                 color=C_OBS, lw=1.0, alpha=0.6)
        a.plot(mx, df.o, "o", color=C_OBS, ms=3, mfc="white", label="Observed total")
        a.set_ylabel("Latent heat (W m$^{-2}$)", fontsize=8); _month_axis(a, bot)
        if r == 0: a.legend(frameon=False, fontsize=6.5, loc="upper right")
        a.spines[["top", "right"]].set_visible(False); _tag(a)

        # soil moisture at sensor depth + deeper layers
        a = fig.add_subplot(gs[gr, 12:15])
        z = np.asarray(ds.z) * 100.0; theta = np.asarray(ds.theta_prof)
        vall = np.ones_like(v, dtype=bool); sensor = SWC_SENSOR_CM.get(site, 5.0)
        for i_d, (dcm, col) in enumerate(zip([sensor, 30, 100],
                                             ["#2166ac", "#74add1", "#c6dbef"])):
            k = int(np.argmin(np.abs(z - dcm))); thd = _daily(theta[:, k], vall, dt)
            a.plot(np.arange(len(thd)), thd, color=col, lw=1.5 if i_d == 0 else 1.0,
                   label=f"{z[k]:.0f} cm")
        swo = _daily(np.asarray(ds.swc_obs), vall, dt)
        a.plot(np.arange(len(swo)), swo, color="k", lw=1.1, ls="--",
               label=f"Obs {sensor:.0f} cm")
        a.set_ylabel("Soil moisture (m$^3$ m$^{-3}$)", fontsize=8)
        if bot: a.set_xlabel("Day of record", fontsize=8)
        if r == 0: a.legend(frameon=False, fontsize=6.2, loc="upper right", ncol=2)
        a.spines[["top", "right"]].set_visible(False); _tag(a)

    fig.savefig(f"{OUT}_combined.png", bbox_inches="tight", dpi=300)
    print("saved", f"{OUT}_combined.png")


if __name__ == "__main__":
    fig_combined()
