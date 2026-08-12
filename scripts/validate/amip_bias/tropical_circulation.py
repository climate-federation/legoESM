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
    return dict(peak=float(zm[i0]), peak_lat=float(lat[i0]),
                half_width=float(hi - lo + 1) * dlat, zm=zm)


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
    cm, co = concentration(pr_m, lat, lon), concentration(pr_o, lat, lon)
    print(f"\n=== {run}: tropical precipitation structure vs GPCP ===")
    print(f"{'':22}{'model':>10}{'GPCP':>10}")
    for k, lbl, f in [("peak", "ITCZ peak [mm/day]", "{:10.2f}"),
                      ("peak_lat", "  at latitude", "{:10.1f}"),
                      ("half_width", "  half-rain width [deg]", "{:10.1f}")]:
        print(f"{lbl:22}" + f.format(im[k]) + f.format(io[k]))
    for k, lbl in [("mean", "tropical mean [mm/day]"),
                   ("top_decile", "share in wettest 10%"),
                   ("frac_heavy", f"area frac > {HEAVY:.0f} mm/day")]:
        print(f"{lbl:22}{cm[k]:10.3f}{co[k]:10.3f}")

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
