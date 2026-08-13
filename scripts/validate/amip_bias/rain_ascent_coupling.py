#!/usr/bin/env python3
"""Does the rain sit where the air rises?  Rain, ascent, and how tightly they lock.

The rain-band diagnostics say WHERE the model rains and how hard.  They cannot
say whether the rain is anchored to the circulation.  Two very different faults
give the same weak pattern correlation against observed rainfall:

* the circulation is in the wrong place, and the rain faithfully follows it; or
* the circulation is right and the convection fires in the wrong columns.

This separates them.  Per run it reports the rain-band centroid, the ascent
centroid, and the correlation between rainfall and 500 hPa ascent across the
tropics -- for the model and for the observations, through the SAME reduction.

THE ASCENT IS AN ESTIMATE ON BOTH SIDES.  The runs scored here predate the
model publishing its own vertical motion, so ascent is reconstructed from
monthly-mean winds by the continuity operator in ``tropical_circulation``.  Its
amplitude is not quotable -- against ERA5's own vertical motion it is roughly
30x too strong.  Its PATTERN is what a correlation uses, and the same operator
is applied to the model's winds and to ERA5's winds, so the model-minus-
observation comparison is like for like.  ERA5's TRUE vertical motion is
reported alongside as a sensitivity, so the reader can see how much of any gap
is the estimator rather than the model.

Usage: rain_ascent_coupling.py <run> [<run> ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent
BELT = 30.0          # deg; the tropics over which the coupling is measured
LEVEL = 50000.0      # Pa; mid-tropospheric ascent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load("tropical_circulation")
bm = _load("bias_maps")
rb = bm.rb


def _centroid(field, lat, lon, weight_positive=True):
    """Area-weighted latitude centroid of a field's positive part."""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(field)
    f = np.maximum(field, 0.0) if weight_positive else field
    tot = float((f * w).sum())
    if tot <= 0:
        return np.nan
    return float((f * w * lat[:, None]).sum() / tot)


def _corr(a, b, lat, mask):
    """Area-weighted correlation of two fields over ``mask``."""
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)[mask]
    x, y = a[mask], b[mask]
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    cx, cy = x - xm, y - ym
    den = np.sqrt(np.average(cx * cx, weights=w) * np.average(cy * cy, weights=w))
    return float(np.average(cx * cy, weights=w) / den) if den > 0 else np.nan


def measure(name):
    md_pr = rb._load_model(name, "pr")
    md_u = rb._load_model(name, "ua")
    md_v = rb._load_model(name, "va")
    if md_pr is None or md_u is None or md_v is None:
        raise SystemExit(f"FATAL: {name} does not publish pr / ua / va")

    lat = np.asarray(md_pr.lat, dtype=np.float64)
    lon = np.asarray(md_pr.lon, dtype=np.float64)
    months = rb._month_labels(md_pr)

    plev_all = np.asarray(md_u.plev, dtype=np.float64)
    keep = plev_all >= 10000.0
    plev = plev_all[keep]
    k = int(np.argmin(np.abs(plev - LEVEL)))

    pr_m = np.asarray(md_pr["pr"]).mean(axis=0) * 86400.0        # mm/day
    u_m = np.asarray(md_u["ua"]).mean(axis=0)[keep]
    v_m = np.asarray(md_v["va"]).mean(axis=0)[keep]
    om_m = tc.omega_from_divergence(u_m, v_m, plev, lat, lon)[k]

    pr_o = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP)
    if pr_o is None:
        raise SystemExit("FATAL: no GPCP reference")
    om_true, ue, ve, ref_year = tc.era5_omega(months, plev)
    om_o = tc.omega_from_divergence(ue, ve, plev, lat, lon)[k]   # SAME operator

    belt = np.abs(lat) <= BELT
    mask = np.broadcast_to(belt[:, None], pr_m.shape)

    out = {
        "run": name,
        "ref_year": ref_year,
        "rain_centroid_m": _centroid(pr_m[belt], lat[belt], lon),
        "rain_centroid_o": _centroid(pr_o[belt], lat[belt], lon),
        # ascent is NEGATIVE omega, so the centroid weights -omega
        "asc_centroid_m": _centroid(-om_m[belt], lat[belt], lon),
        "asc_centroid_o": _centroid(-om_o[belt], lat[belt], lon),
        "asc_centroid_true": _centroid(-om_true[k][belt], lat[belt], lon),
        "coupling_m": _corr(pr_m, -om_m, lat, mask),
        "coupling_o": _corr(pr_o, -om_o, lat, mask),
        "coupling_true": _corr(pr_o, -om_true[k], lat, mask),
        "rain_r": _corr(pr_m, pr_o, lat, mask),
        "asc_r": _corr(-om_m, -om_o, lat, mask),
    }
    return out, (lat, lon, pr_m, pr_o, om_m, om_o)


def figure(name, packed, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    lat, lon, pr_m, pr_o, om_m, om_o = packed
    belt = np.abs(lat) <= BELT
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 6.2), subplot_kw={"projection": ccrs.PlateCarree()})
    # ascent shown as -omega in hPa/day, on each field's OWN scale because the
    # estimator's amplitude is not comparable between the two
    sc = 864.0
    panels = [
        (pr_m, "model rain [mm/day]", "YlGnBu", 0, 12),
        (pr_o, "GPCP rain [mm/day]", "YlGnBu", 0, 12),
        (-om_m * sc, "model ascent (estimate, arbitrary scale)", "RdBu_r",
         -np.nanpercentile(np.abs(om_m[belt] * sc), 95),
         np.nanpercentile(np.abs(om_m[belt] * sc), 95)),
        (-om_o * sc, "ERA5 ascent (same estimator, arbitrary scale)", "RdBu_r",
         -np.nanpercentile(np.abs(om_o[belt] * sc), 95),
         np.nanpercentile(np.abs(om_o[belt] * sc), 95)),
    ]
    for ax, (fld, title, cmap, lo, hi) in zip(axes.ravel(), panels):
        m = bm._panel(ax, lon, lat, fld, title, cmap, lo, hi, "")
        ax.set_extent([-180, 180, -BELT, BELT], crs=ccrs.PlateCarree())
        fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    fig.suptitle(f"{name}: rain and ascent, tropics", fontsize=11)
    fig.tight_layout()
    path = pathlib.Path(out_dir) / f"rain_ascent_{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="/work/bd1083/b309178/diffESM/legoesm_pg/"
                                     "amip_runs/_tools/maps")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    for name in args.runs:
        row, packed = measure(name)
        rows.append(row)
        if not args.no_figures:
            print(f"  wrote {figure(name, packed, args.out)}")

    print(f"\n=== rain and ascent, |lat| <= {BELT:.0f}, "
          f"{LEVEL/100:.0f} hPa ===")
    print(f"{'run':<12}{'rain lat':>10}{'ascent lat':>12}{'rain-ascent':>13}"
          f"{'rain r':>9}{'ascent r':>10}")
    for r in rows:
        print(f"{r['run']:<12}{r['rain_centroid_m']:10.2f}"
              f"{r['asc_centroid_m']:12.2f}{r['coupling_m']:13.2f}"
              f"{r['rain_r']:9.2f}{r['asc_r']:10.2f}")
    r0 = rows[0]
    print(f"{'OBSERVED':<12}{r0['rain_centroid_o']:10.2f}"
          f"{r0['asc_centroid_o']:12.2f}{r0['coupling_o']:13.2f}"
          f"{1.0:9.2f}{1.0:10.2f}")
    print(f"\n  rain lat / ascent lat = area-weighted latitude of tropical "
          f"rainfall and of\n  mid-tropospheric ascent.  rain-ascent = "
          f"correlation between the two fields\n  within the belt: how tightly "
          f"the rain is anchored to the rising motion.\n  rain r / ascent r = "
          f"pattern correlation with the observed field.")
    print(f"\n  ESTIMATOR SENSITIVITY: with ERA5's TRUE vertical motion instead "
          f"of the same\n  estimator, the observed rain-ascent coupling is "
          f"{r0['coupling_true']:.2f} (against "
          f"{r0['coupling_o']:.2f}) and the\n  observed ascent latitude is "
          f"{r0['asc_centroid_true']:.2f} (against "
          f"{r0['asc_centroid_o']:.2f}).  A model-minus-observed\n  gap smaller "
          f"than that difference is the estimator talking, not the model.\n"
          f"  ERA5 reference year {r0['ref_year']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
