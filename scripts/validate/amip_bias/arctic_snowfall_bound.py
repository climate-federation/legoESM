#!/usr/bin/env python3
"""Arctic-land snowfall check (#2172): model precipitation vs ERA5 snowfall, same days.

The MPAS AMIP lane publishes no snowfall (no ``prsn``), so the model side here
is TOTAL precipitation (day/pr), which is an UPPER BOUND on the snow the land
received.  If the model's total precipitation over Arctic land is already
below ERA5's snowfall, the snowpack is supply-limited whatever the phase split.
This is a bound, not the snowfall budget itself.

ERA5: daily sums of snowfall (param 144) and total precipitation (228) from the
Levante pool (``sf/fc/1D``, m of water per day), staged by cdo to regular
Gaussian netCDF, the SAME calendar days as the run's day table, binned onto the
model grid with ``regional_bias.bin_to_model``.  Region and land mask are
``regional_bias``'s "Arctic 60-90N" box and its sftlf > 50 % land mask; cos-lat.

Self-checks printed before the numbers:
  * ERA5 pool tp (window mean) vs the climateeval ERA5 monthly pr of that month
    (unit/scale check of the pool reader), global and Arctic land;
  * model day/pr window mean vs the run's Amon pr.

Usage: arctic_snowfall_bound.py --era5-dir DIR --era5-mon-pr FILE --out DIR RUN [RUN ...]
DIR holds sf_YYYY-MM.nc and tp_YYYY-MM.nc (daily records).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regional_bias as rb  # noqa: E402
from desert_daynight import dates_of, model_day  # noqa: E402

M_PER_DAY_TO_MM_PER_DAY = 1000.0
KG_M2_S_TO_MM_PER_DAY = 86400.0


def era5_daily(path, dates, mlat, mlon):
    """ERA5 daily sums [mm/day] binned to the model grid, one per date."""
    e = xr.open_dataset(path)
    v = next(iter(e.data_vars))
    have = [str(t)[:10] for t in e.time.values]
    if len(set(have)) != len(have):
        raise SystemExit(f"FATAL: {path} holds duplicate dates")
    lat = np.asarray(e.lat, float); lon = np.asarray(e.lon, float) % 360.0
    out = []
    for s in dates:
        if s not in have:
            raise SystemExit(f"FATAL: {path} has no record for {s}")
        a = np.asarray(e[v].isel(time=have.index(s)), float)
        if a.ndim == 3:
            a = a[0]
        la = lat
        if la[0] > la[-1]:
            la = la[::-1]; a = a[::-1]
        out.append(rb.bin_to_model(a * M_PER_DAY_TO_MM_PER_DAY, la, lon, mlat, mlon,
                                   label=f"ERA5 {path} {s}"))
    return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--era5-mon-pr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    box = rb.REGIONS["Arctic 60-90N"]
    res = {}
    for run in a.runs:
        d, pr = model_day(run, "pr")
        mlat = np.asarray(d.lat, float); mlon = np.asarray(d.lon, float) % 360.0
        dates = dates_of(d)
        if len(set(dates)) != len(dates):
            raise SystemExit(f"FATAL: {run} day table repeats a date")
        land = rb._land_mask(run, mlat, mlon)
        if land is None:
            raise SystemExit(f"FATAL: {run} published no fx sftlf; land-only scores impossible")
        ym = dates[0][:7]
        if dates[-1][:7] != ym:
            raise SystemExit(f"FATAL: {run} day window {dates[0]}..{dates[-1]} spans two months")
        sf = era5_daily(f"{a.era5_dir}/sf_{ym}.nc", dates, mlat, mlon)
        tp = era5_daily(f"{a.era5_dir}/tp_{ym}.nc", dates, mlat, mlon)
        prd = pr * KG_M2_S_TO_MM_PER_DAY
        am = np.asarray(rb._load_model(run, "pr")["pr"], float).mean(0) * KG_M2_S_TO_MM_PER_DAY
        mon = xr.open_dataset(a.era5_mon_pr)
        mon = mon.sel(time=[t for t in mon.time.values if str(t)[:7] == ym])
        if mon.sizes["time"] != 1:
            raise SystemExit(f"FATAL: ERA5 monthly pr has {mon.sizes['time']} records for {ym}")
        la = np.asarray(mon.lat, float); lo = np.asarray(mon.lon, float) % 360.0
        ma = np.asarray(mon["pr"], float)[0] * KG_M2_S_TO_MM_PER_DAY
        if la[0] > la[-1]:
            la = la[::-1]; ma = ma[::-1]
        emon = rb.bin_to_model(ma, la, lo, mlat, mlon, label="ERA5 mon pr")
        G = rb.REGIONS["GLOBAL"]
        r = lambda f, b=box, v=land: rb.region_mean(f, mlat, mlon, b, v)  # noqa: E731
        print(f"\n=== {run}: window {dates[0]} .. {dates[-1]} ({len(dates)} days); "
              f"Arctic 60-90N land (sftlf>{rb.LAND_PCT:.0f}%), cos-lat, mm/day ===")
        print(f"self-check ERA5 pool tp vs climateeval monthly pr: global {r(tp.mean(0), G, None):.3f} vs "
              f"{r(emon, G, None):.3f}; Arctic land {r(tp.mean(0)):.3f} vs {r(emon):.3f}")
        print(f"self-check model day/pr vs Amon pr: Arctic land {r(prd.mean(0)):.3f} vs {r(am):.3f}")
        row = dict(model_pr=r(prd.mean(0)), era5_tp=r(tp.mean(0)), era5_sf=r(sf.mean(0)))
        row["daily"] = (np.array([r(x) for x in prd]), np.array([r(x) for x in sf]),
                        np.array([r(x) for x in tp]))
        print(f"  model total precipitation (upper bound on snowfall) {row['model_pr']:7.3f}")
        print(f"  ERA5 total precipitation                            {row['era5_tp']:7.3f}")
        print(f"  ERA5 snowfall                                       {row['era5_sf']:7.3f}")
        print(f"  model pr / ERA5 sf                                  {row['model_pr'] / row['era5_sf']:7.3f}")
        res[run] = row
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/arctic_snowfall_bound.npy", res, allow_pickle=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oi = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
    fig, axs = plt.subplots(1, len(a.runs), figsize=(4.5 * len(a.runs), 4), sharey=True)
    for ax, run in zip(np.atleast_1d(axs), a.runs):
        p, s, t = res[run]["daily"]
        x = np.arange(1, p.size + 1)
        ax.plot(x, p, color=oi[0], label="model total precip (bound)")
        ax.plot(x, t, color=oi[1], label="ERA5 total precip")
        ax.plot(x, s, color=oi[3], ls="--", label="ERA5 snowfall")
        ax.set_title(run); ax.set_xlabel("day of window")
    np.atleast_1d(axs)[0].set_ylabel("Arctic 60-90N land mean [mm/day]")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/arctic_snowfall_bound.png", dpi=120)
    print("fig", f"{a.out}/arctic_snowfall_bound.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
