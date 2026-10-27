#!/usr/bin/env python3
"""Desert 2 m temperature split into day (daily max) and night (daily min) vs ERA5 (#2105).

Model: the run's CMOR ``day`` table (tasmax / tasmin / tas).  The model's
extremes are taken over HOURLY samples of tas that were first regridded to the
CMOR 5-degree grid (``feed_daily_extremes_native`` -> ``add_extremes_2d``).

ERA5: hourly 2 m temperature (param 167) of the SAME days, from the Levante
pool, staged to regular-Gaussian netCDF by cdo (one file per day, 24 hours),
taken at 01:00 .. 24:00 UTC of each day as the model samples it (see
``era5_days``).
Each hour is binned onto the model grid with ``regional_bias.bin_to_model``
FIRST and the daily max / min / mean are taken AFTERWARDS, so both sides use
the same order of operations (max of box means, hourly sampling).  ClimateEval
derives ERA5 tasmax/tasmin the same way (daily max/min of hourly tas).

Desert boxes, land mask (sftlf > 50 %) and cos-lat region means are those of
``regional_bias.py`` unchanged.

Self-checks printed before any number:
  * model daily-mean tas averaged over the window vs the run's Amon tas;
  * ERA5 hourly-derived window mean vs the ERA5 1979 monthly file the earlier
    #2105 profile comment used (``--era5-mon``), per desert.

Usage: desert_daynight.py --era5-day-dir DIR --era5-mon FILE --out DIR RUN [RUN ...]
(LEGOESM_AMIP_RUNS = folder holding the runs).  DIR holds 2t_YYYY-MM-DD.nc.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regional_bias as rb  # noqa: E402

DESERTS = ["Sahara 18-30N", "Arabia 18-30N", "Kalahari", "Australia int",
           "Gobi/Taklamakan"]


def daily_stats(hourly):
    """(24, ...) hourly samples of ONE day -> (max, min, mean).  Refuses a day
    that does not carry exactly 24 finite hourly samples."""
    h = np.asarray(hourly, float)
    if h.shape[0] != 24 or not np.isfinite(h).all():
        raise SystemExit(f"FATAL: ERA5 day has shape {h.shape} / non-finite values")
    return h.max(0), h.min(0), h.mean(0)


def model_day(run, var):
    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/day/{var}_day_*.nc"))
    if len(fs) != 1:
        raise SystemExit(f"FATAL: {run} has {len(fs)} day/{var} files")
    d = xr.open_dataset(fs[0], decode_times=False)
    return d, np.asarray(d[var], float)


def dates_of(d):
    """Calendar dates (noleap, from the file's own units) of each daily record."""
    y0 = int(d.time.attrs["units"].split("since")[1].strip()[:4])
    cum = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    out = []
    for t in np.asarray(d.time, float):
        doy = int(t % 365.0)
        m = int(np.searchsorted(cum, doy, side="right"))
        out.append(f"{y0 + int(t // 365.0):04d}-{m:02d}-{doy - cum[m - 1] + 1:02d}")
    return out


def next_date(s):
    return str(np.datetime64(s) + np.timedelta64(1, "D"))


def era5_days(day_dir, dates, mlat, mlon):
    """ERA5 hourly 2t of each model day, as the MODEL samples it: the model
    records the state at the END of each hour (01:00 .. 24:00 UTC) and files
    it under the hour it covers (``model_driver``: ``sample_bin =
    (floor(day*24) - 0.5)/24``), so a model day is 01:00 of that day through
    00:00 of the next.  Needs the file of the day after the window too."""
    mx, mn, me = [], [], []
    for s in dates:
        a = xr.open_dataset(f"{day_dir}/2t_{s}.nc")
        b = xr.open_dataset(f"{day_dir}/2t_{next_date(s)}.nc")
        e = xr.concat([a.isel(time=slice(1, 24)), b.isel(time=slice(0, 1))], dim="time")
        want = np.array([np.datetime64(f"{s}T00:00") + np.timedelta64(h, "h") for h in range(1, 25)])
        if not np.array_equal(e.time.values.astype("datetime64[m]"), want.astype("datetime64[m]")):
            raise SystemExit(f"FATAL: ERA5 hours for model day {s} are not 01:00 {s} .. 00:00 next day")
        v = next(iter(e.data_vars))
        lat = np.asarray(e.lat, float); lon = np.asarray(e.lon, float) % 360.0
        arr = np.asarray(e[v], float)
        if arr.ndim == 4:
            arr = arr[:, 0]
        if lat[0] > lat[-1]:
            lat = lat[::-1]; arr = arr[:, ::-1]
        binned = np.stack([rb.bin_to_model(x, lat, lon, mlat, mlon, label=f"ERA5 2t {s}")
                           for x in arr])
        a_, b_, c_ = daily_stats(binned)
        mx.append(a_); mn.append(b_); me.append(c_)
    return np.stack(mx), np.stack(mn), np.stack(me)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--era5-day-dir", required=True)
    ap.add_argument("--era5-mon", required=True, help="ERA5 1979 monthly tas netCDF")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {}
    for run in a.runs:
        dx, tmax = model_day(run, "tasmax")
        dn, tmin = model_day(run, "tasmin")
        dm, tmean = model_day(run, "tas")
        mlat = np.asarray(dx.lat, float); mlon = np.asarray(dx.lon, float) % 360.0
        dates = dates_of(dx)
        for o in (dn, dm):
            if not (np.array_equal(np.asarray(o.lat), np.asarray(dx.lat))
                    and np.array_equal(np.asarray(o.lon), np.asarray(dx.lon))):
                raise SystemExit(f"FATAL: {run} tasmax / tasmin / tas day tables are on different grids")
        if dates_of(dn) != dates or dates_of(dm) != dates:
            raise SystemExit(f"FATAL: {run} tasmax / tasmin / tas day tables cover different dates")
        if len(set(dates)) != len(dates):
            raise SystemExit(f"FATAL: {run} day table repeats a date")
        land = rb._land_mask(run, mlat, mlon)
        if land is None:
            raise SystemExit(f"FATAL: {run} published no fx sftlf; land-only scores impossible")
        if dates[0][:7] != dates[-1][:7]:
            raise SystemExit(f"FATAL: {run} day window {dates[0]}..{dates[-1]} spans two months; "
                             "the monthly self-checks would compare different windows")
        for f in (tmax, tmin, tmean):
            if not np.isfinite(f).all():
                raise SystemExit(f"FATAL: {run} day table has non-finite values")
        ex, en, em = era5_days(a.era5_day_dir, dates, mlat, mlon)
        print(f"\n=== {run}: window {dates[0]} .. {dates[-1]} ({len(dates)} days), "
              f"land cells sftlf>{rb.LAND_PCT:.0f}%, cos-lat ===")
        # Self-check 1: model daily-mean tas vs the run's Amon tas.
        am = np.asarray(rb._load_model(run, "tas")["tas"], float).mean(0)
        # Self-check 2: ERA5 hourly-derived window mean vs ERA5 1979 monthly file.
        mon = xr.open_dataset(a.era5_mon)
        mon = mon.sel(time=[t for t in mon.time.values
                            if str(t)[:7] == dates[0][:7]])
        if mon.sizes["time"] != 1:
            raise SystemExit(f"FATAL: ERA5 monthly file has {mon.sizes['time']} records for {dates[0][:7]}")
        mlat_e = np.asarray(mon.lat, float); mlon_e = np.asarray(mon.lon, float) % 360.0
        marr = np.asarray(mon["tas"], float)[0]
        if mlat_e[0] > mlat_e[-1]:
            mlat_e = mlat_e[::-1]; marr = marr[::-1]
        emon = rb.bin_to_model(marr, mlat_e, mlon_e, mlat, mlon, label="ERA5 mon tas")
        print(f"{'self-check [K]':<18}{'model day-Amon':>15}{'ERA5 hr-mon':>13}")
        for d in DESERTS:
            box = rb.REGIONS[d]
            c1 = rb.region_mean(tmean.mean(0) - am, mlat, mlon, box, land)
            c2 = rb.region_mean(em.mean(0) - emon, mlat, mlon, box, land)
            print(f"{d:<18}{c1:15.3f}{c2:13.3f}")
        print(f"{'model-ERA5 [K]':<18}{'tasmax':>8}{'tasmin':>8}{'tas':>8}"
              f"{'DTR mod':>9}{'DTR E5':>8}{'tas prev-protocol':>19}")
        rows = {}
        for d in DESERTS:
            box = rb.REGIONS[d]
            r = lambda f: rb.region_mean(f, mlat, mlon, box, land)  # noqa: E731
            bx = r((tmax - ex).mean(0)); bn = r((tmin - en).mean(0)); bm = r((tmean - em).mean(0))
            dtr_m = r((tmax - tmin).mean(0)); dtr_e = r((ex - en).mean(0))
            prev = r(am - emon)
            rows[d] = (bx, bn, bm, dtr_m, dtr_e, prev)
            print(f"{d:<18}{bx:8.2f}{bn:8.2f}{bm:8.2f}{dtr_m:9.2f}{dtr_e:8.2f}{prev:19.2f}")
        res[run] = rows
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/desert_daynight.npy", res, allow_pickle=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oi = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x = np.arange(len(DESERTS)); w = 0.8 / len(a.runs)
    for ax, k, name in zip(axs, (0, 1, 2), ("daily max (day)", "daily min (night)", "daily mean")):
        for i, run in enumerate(a.runs):
            ax.bar(x + i * w, [res[run][d][k] for d in DESERTS], w, color=oi[i % 6], label=run)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x + w * (len(a.runs) - 1) / 2)
        ax.set_xticklabels([d.split()[0] for d in DESERTS], rotation=30)
        ax.set_title(f"2 m T {name}: model - ERA5 1979 [K]")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.out}/desert_daynight.png", dpi=120)
    print("fig", f"{a.out}/desert_daynight.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
