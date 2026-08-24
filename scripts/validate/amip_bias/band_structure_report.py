#!/usr/bin/env python3
"""Does the rain band have the OBSERVED SHAPE, longitude by longitude?

A single rain-band centroid can be right while the band is wrong everywhere:
too far north over the Pacific and too far south over the Atlantic averages to
the correct latitude and reproduces none of the structure.  The real band bends
-- it splits over the warm pool and runs southeast across the South Pacific --
and a model that misses that is a poor proxy for mean rainfall whatever its
zonal mean says.

This scores the shape:

* **latitude error** of the band at each longitude, as a root-mean-square and
  as a mean offset.  The RMS is the structural number; the offset is the part a
  uniform shift would fix.
* **shape correlation**: does the band bend the same way with longitude.
* **sharpness**: the share of each meridian's rain falling within 5 degrees of
  its own centre, model against observed.  A smear and a tight band can share a
  centroid; this is what separates them.
* **along-longitude spread** of the tropical rain, the existing measure of how
  broken up the band is.

Usage: band_structure_report.py <run> [<run> ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent
BELT = 20.0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load("tropical_circulation")
bm = _load("bias_maps")
rb = bm.rb


def measure(name):
    md = rb._load_model(name, "pr")
    if md is None:
        raise SystemExit(f"FATAL: {name} publishes no pr")
    lat = np.asarray(md.lat, dtype=np.float64)
    lon = np.asarray(md.lon, dtype=np.float64)
    months = rb._month_labels(md)
    pr_m = np.asarray(md["pr"]).mean(axis=0) * 86400.0
    pr_o = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP)
    if pr_o is None:
        raise SystemExit("FATAL: no GPCP reference")
    pr_o = pr_o * 86400.0
    if float(np.nanmax(pr_o)) > 500.0:
        raise SystemExit("FATAL: GPCP appears to be in mm/day already")

    row = {"run": name}
    row.update(tc.band_structure(pr_m, pr_o, lat, lon, BELT))
    zm = tc.concentration(pr_m, lat, lon)
    zo = tc.concentration(pr_o, lat, lon)
    for k in ("mean", "frac_heavy"):
        row[f"{k}_m"], row[f"{k}_o"] = zm[k], zo[k]
    return row, (lat, lon, pr_m, pr_o)


def figure(name, packed, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat, lon, pr_m, pr_o = packed
    cm, sm, _ = tc.itcz_by_longitude(pr_m, lat, lon, BELT)
    co, so, _ = tc.itcz_by_longitude(pr_o, lat, lon, BELT)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(lon, co, "k-", lw=2, label="GPCP")
    ax[0].plot(lon, cm, "C3-", lw=2, label=name)
    ax[0].set_ylabel("band latitude [deg]")
    ax[0].axhline(0, color="0.7", lw=0.6)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{name}: where the rain band sits, longitude by longitude")
    ax[1].plot(lon, so, "k-", lw=2)
    ax[1].plot(lon, sm, "C3-", lw=2)
    ax[1].set_ylabel("share of rain within 5 deg of the band")
    ax[1].set_xlabel("longitude [deg]")
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    path = pathlib.Path(out_dir) / f"band_structure_{name}.png"
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
            print(f"  wrote {figure(row['run'], packed, args.out)}")

    print(f"\n=== rain-band SHAPE, |lat| <= {BELT:.0f}, per longitude ===")
    print(f"{'run':<12}{'lat rms':>9}{'lat bias':>10}{'shape r (95% CI)':>22}"
          f"{'sharpness':>11}{'double':>8}{'heavy frac':>12}{'mean':>8}")
    for r in rows:
        ci = f"{r['lat_r']:.2f} [{r['lat_r_lo']:.2f},{r['lat_r_hi']:.2f}]"
        print(f"{r['run']:<12}{r['lat_rms']:9.2f}{r['lat_bias']:10.2f}{ci:>22}"
              f"{r['sharp_m']:11.2f}{r['double_m']:8.2f}"
              f"{r['frac_heavy_m']:12.3f}{r['mean_m']:8.2f}")
    r0 = rows[0]
    print(f"{'OBSERVED':<12}{'--':>9}{'--':>10}{'-- (identity)':>22}"
          f"{r0['sharp_o']:11.2f}{r0['double_o']:8.2f}"
          f"{r0['frac_heavy_o']:12.3f}{r0['mean_o']:8.2f}")
    print(f"\n  Band position = latitude of each meridian's rainfall PEAK, not "
          f"its centroid: a\n  centroid is not a band location where the rain "
          f"is double-peaked.\n  lat rms  = position error [deg]; the "
          f"structural number.\n  lat bias = the part a uniform shift would "
          f"remove.\n  shape r  = does the band bend with longitude as the "
          f"observed one does, with a\n             CIRCULAR BLOCK bootstrap "
          f"interval -- band latitude is strongly\n             correlated "
          f"along longitude, so 72 longitudes are far fewer than 72\n"
          f"             independent samples and a naive interval would "
          f"overstate the skill.\n             An interval spanning zero means "
          f"NO demonstrated shape skill.\n  sharpness = share of a meridian's "
          f"rain within 5 deg of its peak, rain weighted.\n  double = fraction "
          f"of longitudes carrying a second band at least "
          f"{tc.SPLIT_FRAC:.0%}\n           as strong as the first.\n"
          f"  Position scored on {r0['n_lon']} of {r0['n_both']} rainy "
          f"longitudes; the rest are double-banded\n  on one side or the "
          f"other, where a single latitude describes neither.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
