#!/usr/bin/env python3
"""Is the model too CLOUDY, or too HUMID?  Split the cover excess into the two.

The campaign's largest radiative error is low-cloud cover: +27 % absolute in
the northern trades and +15..21 % over the stratocumulus decks, with cloud
water and cloud ice both near observed.  Two very different faults produce
that, and they have different fixes:

* the cloud-fraction closure makes too much AREA per unit humidity, or
* the humidity itself is too high and the closure is faithful.

This probe separates them with an exact identity.  Let ``C_obs(rh)`` be the
observed cover-versus-humidity curve, measured by binning ESACCI-CLOUD cover
against ERA5 humidity on the model's own grid.  Then, per region,

    <C_mod> - <C_obs>  =  [<C_mod> - <C_obs(rh_mod)>]  +  [<C_obs(rh_mod)> - <C_obs>]
                                 CLOSURE part                   HUMIDITY part

The first term compares the two at the SAME humidity: it is the cover the
model adds beyond what the observations show at that humidity.  The second
holds the observed curve fixed and moves only the humidity distribution: it is
the cover the humidity bias alone buys.  They sum exactly.

WHAT THIS IS NOT.  It is NOT the cloud scheme's instantaneous closure.  Both
sides are monthly means, and ``profile_rh_split.py`` already established that
running an RH-threshold operator on monthly-mean RH is not a valid estimator of
monthly-mean cover -- on ERA5's own monthly means it returns 9 % where 63 % is
observed, because the scheme is sharply nonlinear near its threshold.  What is
compared here is the EMERGENT monthly-mean relation, model against
observation, with the same aggregation applied to both sides.  A gap in the
closure term therefore means "at this monthly-mean humidity the model carries
more cover than the real atmosphere does", which can arise either from the
closure itself or from the model having too little sub-monthly humidity
variability.  Those two are NOT separated here and this probe does not claim
to.  Distinguishing them needs sub-monthly cover and humidity, which the CMOR
day table does not carry.

Usage: cloud_rh_relation.py <run> [<run> ...] [--level 70000]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bm = _load("bias_maps")
prs = _load("profile_rh_split")
rb = bm.rb

# Humidity bins.  Wide enough that every bin the regions populate carries
# enough cells for a mean, fine enough to resolve the threshold region the
# closure is sensitive to.
RH_EDGES = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0])
MIN_CELLS = 20          # a bin below this is not quoted, and says so
REGIONS = ("ITCZ 10S-10N", "trades 10-30N", "trades 10-30S", "Sc Peru",
           "Sc Namibia", "Sc California", "SO stormtrack", "GLOBAL")


def observed_curve(rh_obs, cover_obs, weights):
    """Cover as a function of humidity, from the OBSERVATIONS alone.

    Returned as ``(centres, cover, counts)`` over ``RH_EDGES``.  Weighted by
    cell area so a polar cell does not count as much as an equatorial one --
    the curve is a property of the atmosphere, not of the grid.
    """
    idx = np.clip(np.digitize(rh_obs.ravel(), RH_EDGES) - 1, 0, len(RH_EDGES) - 2)
    w = weights.ravel()
    c = cover_obs.ravel()
    n = len(RH_EDGES) - 1
    num = np.bincount(idx, weights=w * c, minlength=n)
    den = np.bincount(idx, weights=w, minlength=n)
    cnt = np.bincount(idx, minlength=n)
    curve = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    return curve, cnt


def evaluate_curve(curve, counts, rh):
    """Observed cover expected at humidity ``rh``, or NaN where unmeasured.

    A bin the observations barely populate cannot state what cover belongs at
    that humidity, so it returns NaN rather than a number resting on three
    cells.  The caller reports how much of the domain that cost.
    """
    idx = np.clip(np.digitize(rh, RH_EDGES) - 1, 0, len(RH_EDGES) - 2)
    out = curve[idx]
    return np.where(counts[idx] >= MIN_CELLS, out, np.nan)


def run(name, level_pa):
    md_clt = rb._load_model(name, "clt")
    md_ta = rb._load_model(name, "ta")
    md_hus = rb._load_model(name, "hus")
    if md_clt is None or md_ta is None or md_hus is None:
        raise SystemExit(f"FATAL: {name} does not publish clt / ta / hus")

    lat = np.asarray(md_clt.lat, dtype=np.float64)
    lon = np.asarray(md_clt.lon, dtype=np.float64)
    months = rb._month_labels(md_clt)

    plev = np.asarray(md_ta.plev, dtype=np.float64)
    k = int(np.argmin(np.abs(plev - level_pa)))
    if abs(plev[k] - level_pa) > 1.0:
        print(f"  note: nearest published level to {level_pa/100:.0f} hPa is "
              f"{plev[k]/100:.0f} hPa; using that")

    T_m = np.asarray(md_ta["ta"]).mean(axis=0)[k]
    w_m = np.asarray(md_hus["hus"]).mean(axis=0)[k]   # already a mixing ratio
    rh_m = prs._rh(T_m[None], w_m[None], plev[k:k + 1])[0]
    cover_m = np.asarray(md_clt["clt"]).mean(axis=0)

    T_e = prs._era5_on_model("ta", months, plev[k:k + 1], lat, lon)[0]
    q_e = prs._era5_on_model("hus", months, plev[k:k + 1], lat, lon)[0]
    rh_e = prs._rh(T_e[None], prs._to_mixing_ratio(q_e)[None], plev[k:k + 1])[0]

    cover_o = rb._ref_clim("clt", months, lat, lon, src=bm.ESACCI)
    if cover_o is None:
        raise SystemExit("FATAL: no ESACCI clt reference")
    if np.nanmax(cover_o) <= 1.5:
        cover_o = cover_o * 100.0

    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], cover_m.shape)
    curve, counts = observed_curve(rh_e, cover_o, area)

    print(f"\n=== {name}: cover excess split at {plev[k]/100:.0f} hPa ===")
    print("  observed cover-vs-humidity curve (ESACCI cover, ERA5 humidity, "
          "model grid, area weighted):")
    for i in range(len(RH_EDGES) - 1):
        flag = "" if counts[i] >= MIN_CELLS else "   (too few cells to use)"
        val = f"{curve[i]:6.1f}" if np.isfinite(curve[i]) else "   n/a"
        print(f"    RH {RH_EDGES[i]:.1f}-{RH_EDGES[i+1]:.1f}: cover {val} %"
              f"   n={counts[i]:5d}{flag}")

    pred_m = evaluate_curve(curve, counts, rh_m)     # obs curve at MODEL humidity
    pred_o = evaluate_curve(curve, counts, rh_e)     # obs curve at OBS humidity

    n_broken = []
    print(f"\n  {'region':<16}{'d_cover':>9}{'closure':>9}{'humidity':>10}"
          f"{'RH_mod':>8}{'RH_obs':>8}{'unusable':>10}")
    for reg in REGIONS:
        box = rb.REGIONS[reg]
        ok = np.isfinite(pred_m) & np.isfinite(pred_o)
        frac_bad = 1.0 - rb.region_mean(ok.astype(float), lat, lon, box)
        if frac_bad > 0.25:
            print(f"  {reg:<16}{'--':>9}{'--':>9}{'--':>10}{'':>8}{'':>8}"
                  f"{frac_bad:9.0%}  SKIPPED: the observed curve does not "
                  "cover this region's humidity range")
            continue
        gm = rb.region_mean(cover_m, lat, lon, box, ok)
        go = rb.region_mean(cover_o, lat, lon, box, ok)
        closure = gm - rb.region_mean(pred_m, lat, lon, box, ok)
        humid = rb.region_mean(pred_m, lat, lon, box, ok) - \
            rb.region_mean(pred_o, lat, lon, box, ok)
        # pred_o is the obs curve at obs humidity, which is the observed cover
        # up to the binning residual; that residual is folded into `closure`
        # only if we compare against `go`, so report the identity's own sum and
        # the raw difference side by side rather than asserting they match.
        resid = (closure + humid) - (gm - go)
        # The identity is only an identity if the observed CURVE reproduces the
        # observed cover in this region. Where it does not -- stratocumulus is
        # the obvious case, its cover set by the boundary layer and not by the
        # humidity above it -- the two terms do not sum to the difference they
        # claim to split, and neither is interpretable.
        bad = abs(resid) > 0.25 * max(abs(gm - go), 1.0)
        mark = "  <-- DOES NOT CLOSE, both terms meaningless" if bad else ""
        print(f"  {reg:<16}{gm - go:9.2f}{closure:9.2f}{humid:10.2f}"
              f"{rb.region_mean(rh_m, lat, lon, box, ok):8.3f}"
              f"{rb.region_mean(rh_e, lat, lon, box, ok):8.3f}"
              f"{frac_bad:9.0%}{mark}")
        if bad:
            n_broken.append(reg)
    print("\n  d_cover = model - ESACCI [% absolute].  closure = cover the "
          "model adds\n  at the OBSERVED humidity.  humidity = cover the "
          "observed curve gives for\n  the model's humidity instead of the "
          "observed one.  The two must SUM to\n  d_cover; where they do not, "
          "the observed curve does not describe that region\n  and the split "
          "is not available there.")
    if n_broken:
        raise SystemExit(
            "FATAL: the cover-vs-humidity split does not close in "
            f"{', '.join(n_broken)}.\n"
            "A single global cover-versus-humidity curve cannot represent "
            "these regions, and\nthe total-column cover this probe reads is "
            "not the layer cover the humidity at\none level would govern. "
            "The split needs the model's 3-D cloud fraction (`cl`)\n"
            "published per layer; until then no number from this probe is "
            "usable.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--level", type=float, default=70000.0,
                    help="pressure level [Pa] at which humidity is read "
                         "(default 70000, the trade-cumulus / Sc layer)")
    args = ap.parse_args(argv)
    for name in args.runs:
        run(name, args.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
