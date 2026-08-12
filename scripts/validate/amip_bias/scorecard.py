#!/usr/bin/env python3
"""One command that scores an AMIP run the same way every time.

The campaign's diagnostics were accumulating as separate probes, each run by
hand on whichever arm was interesting. That is how a comparison silently drifts:
one arm gets the regional table, another gets the maps, and the two are never
the same reduction. This runs ALL of them, in one place, and prints a single
headline table so arms are comparable by construction.

    scorecard.py --run ref1979 [--ctl w3_ctl] [--out DIR]

What it produces, per run:

* `bias_maps_<run>.png`   model / observation / bias for albedo, rsut, rlut,
                          clt, tas, prw against the reference each ESMValTool
                          suite declares (CERES-EBAF, ESACCI-CLOUD, ERA5).
* `ta_section_<run>.png`  pressure-latitude temperature bias, which a surface
                          map cannot show — the bias reverses sign with height.
* `tropical_<run>.png`    rain-band structure and the overturning.
* a HEADLINE TABLE combining the numbers that have actually decided something
  in this campaign:

  - TOA: rsut, rlut and the imbalance, ALWAYS with the components, because a
    knob that improves the imbalance by making the two errors cancel is not a
    fix.
  - Rain band: centroid and continuous width (NOT the grid-quantised
    half-width, which cannot resolve less than one 5-degree cell), the pattern
    correlation with GPCP, and the tropical mean.
  - Grid-scale roughness of pr / clwvi / clt against the observation's own,
    the measurement that located the convective on/off noise.
  - Temperature bias at 700, 500 and 150 hPa, the three levels where the
    profile changes sign.

With `--ctl`, the regional arm-minus-control table is printed too.
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
tc = _load("tropical_circulation")
prs = _load("profile_rh_split")
rb = bm.rb

ROUGH_VARS = (("pr", tc.GPCP), ("clwvi", bm.ESACCI), ("clt", bm.ESACCI))
TA_LEVELS = (70000.0, 50000.0, 15000.0)


def headline(run):
    """The numbers that have decided something, in one row per run."""
    out = {"run": run}

    lat = lon = months = None
    for var in ("rsut", "rlut", "clt"):
        got = bm._model_clim(run, var)
        if got is None:
            continue
        fld, lat, lon, months = got
        src = bm.SPEC[var][0]
        ref = bm._ref_clim(var, months, lat, lon, src)
        out[var] = bm._gm(fld, lat, lon) - bm._gm(ref, lat, lon) if ref is not None \
            else np.nan
    out["TOA"] = -(out.get("rsut", np.nan) + out.get("rlut", np.nan))

    mp = rb._load_model(run, "pr")
    if mp is not None:
        months = rb._month_labels(mp)
        lat = np.asarray(mp.lat, dtype=np.float64)
        lon = np.asarray(mp.lon, dtype=np.float64) % 360.0
        pr_m = np.asarray(mp["pr"]).mean(axis=0) * 86400.0
        pr_o = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP) * 86400.0
        im, io = tc.itcz_metrics(pr_m, lat, lon), tc.itcz_metrics(pr_o, lat, lon)
        belt = np.abs(lat) <= 15.0
        out["rain_centroid"] = im["centroid"] - io["centroid"]
        out["rain_width"] = im["width_sd"] - io["width_sd"]
        out["rain_r"] = float(np.corrcoef(pr_m[belt].ravel(),
                                          pr_o[belt].ravel())[0, 1])
        out["pr_mean"] = tc.concentration(pr_m, lat, lon)["mean"] \
            - tc.concentration(pr_o, lat, lon)["mean"]

    for var, src in ROUGH_VARS:
        md = rb._load_model(run, var)
        if md is None or lat is None:
            continue
        ref = rb._ref_clim(var, months, lat, lon, src=src)
        if ref is None:
            continue
        f = np.asarray(md[var]).mean(axis=0)
        if var == "clt" and np.nanmax(ref) <= 1.5:
            ref = ref * 100.0
        out[f"rough_{var}"] = (bm.grid_scale_residual(f)
                               / bm.grid_scale_residual(ref))

    mt = rb._load_model(run, "ta")
    if mt is not None:
        plev = np.asarray(mt.plev, dtype=np.float64)
        keep = plev >= prs.P_FLOOR
        pl = plev[keep]
        Tm = np.asarray(mt["ta"]).mean(axis=0)[keep]
        Te = prs._era5_on_model("ta", months, pl, lat, lon)
        ps_m = np.asarray(rb._load_model(run, "ps")["ps"]).min(axis=0)
        ps_e = np.min([rb.bin_to_model(*prs._era5_surface_pressure([m]),
                                       lat, lon, label="ERA5 ps")
                       for m in sorted(set(months))], axis=0)
        below = pl[:, None, None] > np.minimum(ps_m, ps_e)[None, :, :]
        box = rb.REGIONS["GLOBAL"]
        for target in TA_LEVELS:
            k = int(np.argmin(np.abs(pl - target)))
            out[f"ta{int(target / 100)}"] = rb.region_mean(
                Tm[k] - Te[k], lat, lon, box, ~below[k])
    return out


_COLS = [("rsut", "rsut", "{:8.2f}"), ("rlut", "rlut", "{:8.2f}"),
         ("TOA", "netTOA", "{:8.2f}"), ("clt", "clt", "{:8.2f}"),
         ("rain_r", "rain r", "{:8.3f}"), ("rain_width", "d_width", "{:8.2f}"),
         ("rain_centroid", "d_cent", "{:8.2f}"), ("pr_mean", "d_pr", "{:8.2f}"),
         ("rough_pr", "rgh_pr", "{:8.2f}"), ("rough_clwvi", "rgh_lwp", "{:8.2f}"),
         ("rough_clt", "rgh_clt", "{:8.2f}"), ("ta700", "ta700", "{:8.2f}"),
         ("ta500", "ta500", "{:8.2f}"), ("ta150", "ta150", "{:8.2f}")]


def print_headline(rows):
    print("\n=== HEADLINE (model - observation, except rain r and the "
          "roughness ratios) ===")
    print(f"{'run':<14}" + "".join(f"{h:>8}" for _, h, _ in _COLS))
    for r in rows:
        line = f"{r['run']:<14}"
        for key, _, f in _COLS:
            v = r.get(key, np.nan)
            line += f.format(v) if np.isfinite(v) else f"{'-':>8}"
        print(line)
    print("netTOA = -(rsut bias + rlut bias): positive means the model keeps "
          "too much energy.\nrain r = pattern correlation with GPCP over "
          "|lat|<=15.  d_width/d_cent = continuous\nrain-band width and "
          "centroid vs GPCP [deg].  rgh_* = grid-scale roughness / the\n"
          "observation's own (1.0 = as smooth as observed).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--ctl", default=None)
    ap.add_argument("--out", default="/work/bd1083/b309178/diffESM/legoesm_pg/"
                                     "amip_runs/_tools/maps")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    for run in args.run:
        if not args.no_figures:
            print(f"{run}: figures")
            try:
                bm.maps(run, [v for v in bm.SPEC], args.out)
                bm.ta_section(run, args.out)
                tc.main(["--run", run, "--out", args.out])
            except SystemExit as exc:
                print(f"  {run}: {exc}")
        rows.append(headline(run))
    print_headline(rows)

    if args.ctl:
        print(f"\n=== regional deltas vs {args.ctl} ===")
        rb.main([r for r in args.run if r != args.ctl], ctl=args.ctl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
