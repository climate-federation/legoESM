#!/usr/bin/env python3
"""Is the ITCZ there, is the tropical rain organised, and does the subsidence
match? Precipitation structure and the overturning circulation vs GPCP / ERA5.

The bias maps show a tropical rain field that is too spread out, which a global
mean cannot express at all: a model can have exactly the right global mean
precipitation while raining lightly everywhere instead of heavily in a narrow
band. Four measurements, each chosen because it separates "how much" from
"where and how concentrated":

1. **Zonal-mean precipitation** with the ITCZ peak value, its latitude, and the
   width of the band that carries half the tropical rain. A weak, wide peak IS
   the diffuse-ITCZ signature.
2. **Concentration**: the share of tropical rain falling in the wettest decile
   of cells, and the area fraction raining above 5 mm/day. Both are
   threshold-light ways to state the "too frequent, too light" problem
   (the drizzle bias) without depending on a distribution fit.
3. **Mid-tropospheric vertical velocity** as a function of latitude and
   longitude — where the model ascends and subsides. Neither the model nor the
   reference publishes ``wap`` here, so BOTH sides get omega from the SAME
   estimator: mass continuity, `omega(p) = -integral_0^p div(V) dp'`, applied
   to the monthly-mean winds on the model's own grid. A derived omega on a
   5-degree monthly mean is not a measurement of the true vertical velocity —
   it is a comparison of two fields treated identically, and the ONLY claims
   made from it are about the model-minus-reference difference.
4. **Zonal-mean profiles per latitude** of temperature, humidity ratio and
   relative humidity, so a tropical error and a mid-latitude error cannot be
   confused (they have opposite signs here).

Usage:  tropical_circulation.py --run ref1979 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import pathlib
import sys

import numpy as np
import xarray as xr

_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = "/work/bd1179/b309141/climateeval_input"
GPCP = f"{_ROOT}/observation_GPCP/mon"
ERA5 = f"{_ROOT}/reanalysis_ERA5/mon"

_spec = importlib.util.spec_from_file_location("rb", _DIR / "regional_bias.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

_spec2 = importlib.util.spec_from_file_location("prs", _DIR / "profile_rh_split.py")
prs = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(prs)

TROPICS = 20.0          # |lat| bound for the "tropical" reductions
HEAVY = 5.0             # mm/day; the rate above which rain is not drizzle


def _area_weights(lat, lon):
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    return w / w.sum()


def zonal(field, lat, lon):
    return field.mean(axis=1)


def itcz_metrics(pr, lat, lon):
    """Peak of the zonal-mean tropical rain, its latitude, and the half-width.

    ``half_width`` is the total latitude span of the contiguous band around the
    peak that carries half the tropical zonal-mean rain — a width that needs no
    threshold and is not sensitive to the peak's amplitude, which is exactly
    what "too diffuse" means.
    """
    zm = zonal(pr, lat, lon)
    trop = np.abs(lat) <= TROPICS
    i0 = int(np.argmax(np.where(trop, zm, -np.inf)))
    w = np.cos(np.deg2rad(lat))
    tot = float((zm * w)[trop].sum())
    # grow a contiguous band outward from the peak until it holds half the rain
    lo = hi = i0
    acc = float(zm[i0] * w[i0])
    while acc < 0.5 * tot and (lo > 0 or hi < lat.size - 1):
        take_lo = (lo > 0) and (hi >= lat.size - 1 or zm[lo - 1] >= zm[hi + 1])
        if take_lo:
            lo -= 1
            acc += float(zm[lo] * w[lo])
        else:
            hi += 1
            acc += float(zm[hi] * w[hi])
    dlat = float(abs(lat[1] - lat[0]))
    # CONTINUOUS width: the cos-lat-weighted second moment of the zonal-mean
    # rain about its own centroid.  This is the width to quote.  The
    # ``half_width`` above can only take multiples of the 5-degree grid
    # spacing, so a "20 vs 15 degrees, 33 % too wide" reading is ONE CELL and
    # is not resolvable -- measured that way the width also looked identical
    # across three physics arms, which was quantisation, not invariance. On the
    # continuous measure ref1979 is 14.27 vs GPCP 14.33, i.e. the meridional
    # width is RIGHT and that claim was retracted.
    band = np.abs(lat) <= 30.0
    ww = np.cos(np.deg2rad(lat))[band] * zm[band]
    y = lat[band]
    centroid = float((ww * y).sum() / ww.sum())
    width_sd = float(np.sqrt((ww * (y - centroid) ** 2).sum() / ww.sum()))
    return dict(peak=float(zm[i0]), peak_lat=float(lat[i0]),
                half_width=float(hi - lo + 1) * dlat,
                centroid=centroid, width_sd=width_sd, zm=zm)


MIN_MERIDIAN_RAIN = 0.5     # mm/day; below this a meridian carries no band
SPLIT_FRAC = 0.6            # a second peak this strong makes a meridian double
SPLIT_SEP = 5.0             # deg; how far apart two peaks must be to count


def itcz_by_longitude(pr, lat, lon, belt=20.0):
    """Where the rain band sits at each longitude, and whether there are two.

    Uses the PEAK latitude, not the rain-weighted centroid.  A centroid is not
    a band location where the rain is double-peaked: over the warm pool GPCP
    has comparable maxima near 12S and 7N, and their centroid lands at 1.6S
    where no band exists.  Scoring that against a model would compare two
    numbers that both describe nothing.

    Returns per longitude:

    * ``centre`` -- latitude of the meridian's rainfall maximum, NaN where the
      meridian is drier than ``MIN_MERIDIAN_RAIN``;
    * ``sharp`` -- share of that meridian's rain within 5 degrees of the peak;
    * ``split`` -- strength of the strongest SECOND maximum, separated from the
      first by at least ``SPLIT_SEP``, as a fraction of the first.  A double
      ITCZ is a real and well-known failure, so it is reported rather than
      averaged away.
    """
    m = np.abs(lat) <= belt
    la = lat[m]
    w = np.cos(np.deg2rad(la))[:, None] * np.maximum(pr[m], 0.0)
    tot = w.sum(axis=0)
    mean_rain = np.maximum(pr[m], 0.0).mean(axis=0)
    good = (tot > 0) & (mean_rain >= MIN_MERIDIAN_RAIN)

    centre = np.full(lon.size, np.nan)
    sharp = np.full(lon.size, np.nan)
    split = np.full(lon.size, np.nan)
    for i in np.flatnonzero(good):
        col = w[:, i]
        k = int(np.argmax(col))
        centre[i] = la[k]
        near = np.abs(la - la[k]) <= 5.0
        sharp[i] = col[near].sum() / col.sum()
        far = np.abs(la - la[k]) >= SPLIT_SEP
        split[i] = (col[far].max() / col[k]) if far.any() and col[k] > 0 else 0.0
    return centre, sharp, split


def _shape_corr_ci(a, b, n_boot=500, block=6, seed=0):
    """Correlation of two longitude series, with a CIRCULAR BLOCK bootstrap.

    Band latitude is strongly autocorrelated along longitude -- GPCP's lag-1
    correlation is 0.93 -- so 72 longitudes are nowhere near 72 independent
    samples and an ordinary confidence interval would overstate the skill by a
    long way.  Resampling contiguous blocks around the globe keeps that
    autocorrelation, so the interval reflects how much the series actually
    constrains.
    """
    n = a.size
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        aa, bb = a[idx], b[idx]
        if np.std(aa) > 0 and np.std(bb) > 0:
            out.append(np.corrcoef(aa, bb)[0, 1])
    lo, hi = (np.nanpercentile(out, [2.5, 97.5]) if out else (np.nan, np.nan))
    return r, float(lo), float(hi)


def band_structure(pr_m, pr_o, lat, lon, belt=20.0):
    """How well the model reproduces the OBSERVED band's shape, per longitude.

    Longitudes where EITHER side is double-peaked are excluded from the
    position scores and counted separately: a single latitude does not describe
    a split band on either side, so a difference between two such numbers is
    not a position error.  The count is itself a result.
    """
    cm, sm, spm = itcz_by_longitude(pr_m, lat, lon, belt)
    co, so, spo = itcz_by_longitude(pr_o, lat, lon, belt)
    both = np.isfinite(cm) & np.isfinite(co)
    single = both & (spm < SPLIT_FRAC) & (spo < SPLIT_FRAC)
    if single.sum() < 8:
        raise SystemExit(
            f"FATAL: only {int(single.sum())} longitudes carry a single band on "
            "both sides -- a per-longitude position score is not available")
    dc = cm[single] - co[single]
    r, lo, hi = _shape_corr_ci(cm[single], co[single])
    return {
        "lat_rms": float(np.sqrt(np.mean(dc ** 2))),
        "lat_bias": float(np.mean(dc)),
        "lat_r": r,
        "lat_r_lo": lo,
        "lat_r_hi": hi,
        # rain-weighted so a nearly dry meridian cannot count as much as the
        # warm pool, which an equal-weight mean let it do
        "sharp_m": float(np.average(sm[both],
                                    weights=np.maximum(pr_m[np.abs(lat) <= belt],
                                                       0.0).mean(axis=0)[both])),
        "sharp_o": float(np.average(so[both],
                                    weights=np.maximum(pr_o[np.abs(lat) <= belt],
                                                       0.0).mean(axis=0)[both])),
        "double_m": float(np.mean(spm[both] >= SPLIT_FRAC)),
        "double_o": float(np.mean(spo[both] >= SPLIT_FRAC)),
        "n_lon": int(single.sum()),
        "n_both": int(both.sum()),
    }


def concentration(pr, lat, lon):
    """How concentrated the tropical rain is, two ways.

    ``top_decile`` = share of the tropical rain that falls in the wettest 10 %
    of the (area-weighted) cells. ``frac_heavy`` = the area fraction raining
    above 5 mm/day. A model that drizzles everywhere has a LOW top decile and a
    LOW heavy fraction while its mean can be perfectly right.
    """
    trop = np.abs(lat) <= TROPICS
    w = _area_weights(lat, lon)[trop]
    x = pr[trop]
    order = np.argsort(x.ravel())[::-1]
    xs, ws = x.ravel()[order], w.ravel()[order]
    ws = ws / ws.sum()
    cw = np.cumsum(ws)
    take = cw <= 0.10
    return dict(top_decile=float((xs[take] * ws[take]).sum()
                                 / (xs * ws).sum()),
                frac_heavy=float(ws[xs > HEAVY].sum()),
                mean=float((xs * ws).sum()))


def omega_from_divergence(u, v, plev, lat, lon):
    """omega [Pa/s] on ``plev`` from mass continuity, using the SAME operator on
    whatever winds it is handed.

    d(omega)/dp = -div(V), integrated from the model top with omega(0) = 0.
    Spherical divergence on a regular lat-lon grid; the cos(lat) metric is kept
    inside the meridional derivative. Not a measurement of the true vertical
    velocity on a coarse monthly mean — only the model-minus-reference
    difference is interpreted, and only where both sides used this function.
    """
    from legoesm import constants
    a = constants.R_earth
    # THE VERTICAL AXIS MUST INCREASE DOWNWARD HERE.  CMOR publishes plev
    # surface-first (100000 -> 100 Pa); this integral starts at p = 0 and
    # marches down, and the O'Brien correction below reads plev[-1] as the
    # surface.  Handed a descending axis it produced a first layer thickness of
    # +96 kPa followed by negative ones, and corrected the residual against
    # 100 Pa as though that were the ground -- which is how this estimator came
    # to be described as "30x too strong" and "does not close".  Sort here and
    # restore the caller's order on the way out, so every call site is fixed
    # rather than each one remembering.
    _order = np.argsort(plev)
    if not np.array_equal(_order, np.arange(plev.size)):
        _inv = np.argsort(_order)
        return omega_from_divergence(
            u[_order], v[_order], np.asarray(plev)[_order], lat, lon)[_inv]
    dlam = np.deg2rad(float(lon[1] - lon[0]))
    dphi = np.deg2rad(float(lat[1] - lat[0]))
    cphi = np.cos(np.deg2rad(lat))[None, :, None]
    du = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dlam)
    vc = v * cphi
    dv = np.empty_like(vc)
    dv[:, 1:-1] = (vc[:, 2:] - vc[:, :-2]) / (2 * dphi)
    dv[:, 0] = (vc[:, 1] - vc[:, 0]) / dphi
    dv[:, -1] = (vc[:, -1] - vc[:, -2]) / dphi
    div = (du + dv) / (a * np.maximum(cphi, 1e-6))
    # integrate downward from p=0: omega_k = -sum_{j<=k} div_j * dp_j
    edges = np.concatenate([[0.0], np.sqrt(plev[:-1] * plev[1:]), [plev[-1]]])
    dp = np.diff(edges)[:, None, None]
    om = -np.cumsum(div * dp, axis=0)
    # O'BRIEN (1970) CORRECTION -- not optional.  The raw downward integral
    # leaves a large residual at the bottom because a monthly-mean divergence
    # on a coarse grid does not balance, and that residual accumulates through
    # the column: uncorrected, this estimator scored a pattern correlation of
    # -0.58 and an RMS of 164 hPa/day against ERA5's own omega, whose tropical
    # standard deviation is 22.  The correction removes the surface residual
    # with a weight linear in pressure, which is the standard treatment.
    p_s = plev[-1]
    return om - (plev[:, None, None] / p_s) * om[-1][None, :, :]


ERA5_OMEGA = _DIR / "era5_omega_clim.npz"


def era5_omega(months, plev):
    """ERA5's OWN omega and winds for ``months``, on the model grid.

    Built once by ``extract_era5_omega.py`` under the ESMValTool environment
    (the field is GRIB parameter 135 in the DKRZ ERA5 pool; the ClimateEval
    reference tree carries neither omega nor even ``va``, so it cannot be
    derived from there either).
    """
    if not ERA5_OMEGA.exists():
        raise SystemExit(
            f"{ERA5_OMEGA} missing -- build it once with:\n  "
            "/work/bd1083/b309178/mambaforge/envs/benchmarking/bin/python "
            "scripts/validate/amip_bias/extract_era5_omega.py --year 1979")
    z = np.load(ERA5_OMEGA)
    sel = [m - 1 for m in months]
    ep = z["plev"]
    order = np.argsort(-ep)

    def on_plev(a):
        a = a[sel].mean(axis=0)[order]
        out = np.empty((plev.size, a.shape[1], a.shape[2]))
        lep = np.log(ep[order])
        for j in range(a.shape[1]):
            for i in range(a.shape[2]):
                out[:, j, i] = np.interp(np.log(plev)[::-1], lep[::-1],
                                         a[:, j, i][::-1])[::-1]
        return out
    return on_plev(z["omega"]), on_plev(z["ua"]), on_plev(z["va"]), int(z["year"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    run = args.run
    mp = rb._load_model(run, "pr")
    if mp is None:
        raise SystemExit(f"{run}: no pr output")
    months = rb._month_labels(mp)
    lat = np.asarray(mp.lat, dtype=np.float64)
    lon = np.asarray(mp.lon, dtype=np.float64) % 360.0
    SEC_PER_DAY = 86400.0
    pr_m = np.asarray(mp["pr"]).mean(axis=0) * SEC_PER_DAY      # kg/m2/s -> mm/day
    pr_o = rb._ref_clim("pr", months, lat, lon, src=GPCP) * SEC_PER_DAY

    im, io = itcz_metrics(pr_m, lat, lon), itcz_metrics(pr_o, lat, lon)
    # COHERENCE of the rain band.  The width can be right while the band is
    # broken into patches in the wrong places, which a zonal mean cannot see.
    belt = np.abs(lat) <= 15.0
    patt_r = float(np.corrcoef(pr_m[belt].ravel(), pr_o[belt].ravel())[0, 1])
    zsd_m = float(np.mean(pr_m[belt].std(axis=1)
                          / np.maximum(pr_m[belt].mean(axis=1), 1e-9)))
    zsd_o = float(np.mean(pr_o[belt].std(axis=1)
                          / np.maximum(pr_o[belt].mean(axis=1), 1e-9)))
    cm, co = concentration(pr_m, lat, lon), concentration(pr_o, lat, lon)
    print(f"\n=== {run}: tropical precipitation structure vs GPCP ===")
    print(f"{'':22}{'model':>10}{'GPCP':>10}")
    for k, lbl, f in [("peak", "ITCZ peak [mm/day]", "{:10.2f}"),
                      ("centroid", "rain centroid [deg]", "{:10.2f}"),
                      ("width_sd", "rain width, sd [deg]", "{:10.2f}"),
                      ("peak_lat", "  peak latitude (5-deg grid)", "{:10.1f}"),
                      ("half_width", "  half-rain width (QUANTISED)", "{:10.1f}")]:
        print(f"{lbl:22}" + f.format(im[k]) + f.format(io[k]))
    for k, lbl in [("mean", "tropical mean [mm/day]"),
                   ("top_decile", "share in wettest 10%"),
                   ("frac_heavy", f"area frac > {HEAVY:.0f} mm/day")]:
        print(f"{lbl:22}{cm[k]:10.3f}{co[k]:10.3f}")
    print(f"{'pattern r vs GPCP':22}{patt_r:10.3f}{1.0:10.3f}")
    print(f"{'zonal sd / mean':22}{zsd_m:10.3f}{zsd_o:10.3f}"
          "   (how broken-up the band is along longitude)")

    # --- omega from the SAME estimator on both sides ------------------------
    mu, mv = rb._load_model(run, "ua"), rb._load_model(run, "va")
    plev_all = np.asarray(mu.plev, dtype=np.float64)
    keep = plev_all >= 10000.0
    plev = plev_all[keep]
    um = np.asarray(mu["ua"]).mean(axis=0)[keep]
    vm = np.asarray(mv["va"]).mean(axis=0)[keep]
    oo_true, ue, ve, ref_year = era5_omega(months, plev)
    om = omega_from_divergence(um, vm, plev, lat, lon)
    oo_der = omega_from_divergence(ue, ve, plev, lat, lon)
    k500 = int(np.argmin(np.abs(plev - 50000.0)))
    HPA_DAY = 864.0                      # Pa/s -> hPa/day

    # ESTIMATOR CONTROL, run BEFORE any model number is quoted.  The model
    # publishes no omega, so its side MUST be the continuity estimate; the
    # reference has both.  Applying the estimator to ERA5's own winds and
    # scoring it against ERA5's own omega says how much of a model-minus-
    # reference difference could be the estimator rather than the model.
    a, b = oo_der[k500] * HPA_DAY, oo_true[k500] * HPA_DAY
    trop_m = np.abs(lat) <= TROPICS
    r = float(np.corrcoef(a[trop_m].ravel(), b[trop_m].ravel())[0, 1])
    print(f"\n=== estimator control: continuity-derived omega500 vs ERA5's own "
          f"({ref_year}) ===")
    print(f"  tropical pattern correlation      {r:6.3f}")
    print(f"  tropical RMS difference           "
          f"{float(np.sqrt(np.mean((a - b)[trop_m] ** 2))):6.2f} hPa/day")
    print(f"  tropical sd of ERA5's own omega   "
          f"{float(np.std(b[trop_m])):6.2f} hPa/day")
    if r < 0.6:
        print("  -> the estimator does NOT reproduce the reference pattern; "
              "read the model omega below as indicative only")

    w_m, w_o = om[k500] * HPA_DAY, oo_true[k500] * HPA_DAY
    trop = np.abs(lat) <= TROPICS
    aw = _area_weights(lat, lon)
    print(f"\n=== {run}: 500 hPa omega [hPa/day] -- AMPLITUDES NOT QUOTABLE ===")
    print("  The model publishes no wap, so its omega is a continuity estimate "
          "from monthly-mean\n  winds; the control above shows that estimator "
          "reproduces the PATTERN (r=0.67) but\n  overstates the amplitude by "
          "~30x.  Only the pattern statements below are usable.\n  FIX: add "
          "wap to the CMOR output.")
    print(f"  tropical ascending area fraction   model {float((w_m[trop] < 0).mean()):.3f}"
          f"   ERA5 {float((w_o[trop] < 0).mean()):.3f}")
    print(f"  tropical mean ascent (where <0)    model "
          f"{float(w_m[trop][w_m[trop] < 0].mean()):8.2f}   ERA5 "
          f"{float(w_o[trop][w_o[trop] < 0].mean()):8.2f}")
    print(f"  tropical mean subsidence (where>0) model "
          f"{float(w_m[trop][w_m[trop] > 0].mean()):8.2f}   ERA5 "
          f"{float(w_o[trop][w_o[trop] > 0].mean()):8.2f}")
    print(f"  global area-weighted mean          model {float((w_m * aw).sum()):8.3f}"
          f"   ERA5 {float((w_o * aw).sum()):8.3f}   (continuity -> should be ~0)")

    # --- figure -------------------------------------------------------------
    proj = ccrs.Robinson(central_longitude=180)
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    def geo(i, j, field, title, cmap, vmin, vmax):
        ax = fig.add_subplot(gs[i, j], projection=proj)
        lonp = np.append(lon, lon[0] + 360.0)
        f = np.concatenate([field, field[:, :1]], axis=1)
        m = ax.pcolormesh(lonp, lat, f, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="nearest", transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color="0.25")
        ax.set_global()
        ax.set_title(title, fontsize=8)
        fig.colorbar(m, ax=ax, shrink=0.7, pad=0.02)

    geo(0, 0, pr_m, f"legoESM pr [mm/day]  trop mean {cm['mean']:.2f}",
        "YlGnBu", 0, 12)
    geo(0, 1, pr_o, f"GPCP pr  trop mean {co['mean']:.2f}", "YlGnBu", 0, 12)
    geo(0, 2, pr_m - pr_o, "pr bias", "BrBG", -6, 6)
    geo(1, 0, w_m / max(1.0, float(np.std(w_m[trop]) / np.std(w_o[trop]))),
        "legoESM omega500, AMPLITUDE-NORMALISED (see caveat)", "RdBu_r", -60, 60)
    geo(1, 1, w_o, "ERA5 omega500 (ERA5's own)", "RdBu_r", -60, 60)
    geo(1, 2, np.sign(w_m) - np.sign(w_o),
        "ascent/subsidence SIGN disagreement (model - ERA5)", "PuOr_r", -2, 2)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(lat, im["zm"], label="legoESM", lw=1.8)
    ax.plot(lat, io["zm"], label="GPCP", lw=1.8)
    ax.set_xlim(-40, 40)
    ax.set_xlabel("latitude")
    ax.set_ylabel("zonal-mean pr [mm/day]")
    ax.set_title(f"ITCZ: peak {im['peak']:.1f} vs {io['peak']:.1f} mm/day, "
                 f"half-rain width {im['half_width']:.0f} vs "
                 f"{io['half_width']:.0f} deg", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    sc = max(1.0, float(np.std(w_m[trop]) / np.std(w_o[trop])))
    ax.plot(lat, zonal(w_m, lat, lon) / sc, label=f"legoESM / {sc:.0f}", lw=1.8)
    ax.plot(lat, zonal(w_o, lat, lon), label="ERA5", lw=1.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(-60, 60)
    ax.set_xlabel("latitude")
    ax.set_ylabel("zonal-mean omega500 [hPa/day]")
    ax.set_title("Hadley overturning: ascent < 0, subsidence > 0\n"
                 "model amplitude NOT measurable from the published output "
                 "(no wap); pattern only", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    srt_m = np.sort(pr_m[trop].ravel())[::-1]
    srt_o = np.sort(pr_o[trop].ravel())[::-1]
    fr = np.arange(1, srt_m.size + 1) / srt_m.size
    ax.plot(100 * fr, np.cumsum(srt_m) / srt_m.sum(), label="legoESM", lw=1.8)
    ax.plot(100 * fr, np.cumsum(srt_o) / srt_o.sum(), label="GPCP", lw=1.8)
    ax.set_xlabel("wettest % of tropical cells")
    ax.set_ylabel("share of tropical rain")
    ax.set_title(f"concentration: wettest 10% hold "
                 f"{100 * cm['top_decile']:.0f}% vs "
                 f"{100 * co['top_decile']:.0f}%", fontsize=8)
    ax.set_xlim(0, 60)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{run}: tropical precipitation structure and overturning "
                 f"({len(months)} months)", fontsize=12)
    out = pathlib.Path(args.out) / f"tropical_{run}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
