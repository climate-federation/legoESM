#!/usr/bin/env python3
"""Score the #1712 pre-registered metric: January zonal-mean u850, 30-70N, vs ERA5.

Metric (pre-registration, issue #1712 comment 6437791330, applied unchanged):
zonal-mean 850 hPa zonal wind averaged over days 1-30 (CMOR daily-mean ua at
850 hPa, all 30 days, bounds 0..30), compared with the ERA5 Jan-1979 zonal mean
on the same CMOR grid, the same regridded reference for every run; error =
cos(lat)-weighted RMSE of the zonal-mean difference over CMOR latitude centres
in [30, 70] N.  Non-finite values are fatal.  Prints numbers only.

Secondary (NOT pre-registered): Jan SLP RMSE 20-90N (daily-mean psl over days
1-30 vs ERA5 Jan-1979 monthly msl, cos-lat weights), and global cos-lat mean
TOA net = rsdt - rsut - rlut from the Amon files.

ERA5 reference on the CMOR grid (repo recipe, as snowheat_ab_score.py):
  cdo -f nc -chname,var131,u -remapcon,<run>/cmor/day/ua_day_*.nc -sellevel,85000 -selmon,1 \
      -setgridtype,regular /pool/data/ERA5/E5/pl/an/1M/131/E5pl00_1M_1979_131.grb e5_u850.nc
  cdo -f nc -chname,var151,msl -remapcon,<run>/cmor/day/psl_day_*.nc -selmon,1 \
      -setgridtype,regular /pool/data/ERA5/E5/sf/an/1M/151/E5sf00_1M_1979_151.grb e5_msl.nc

Usage: zonal_u850_score.py --era5-u e5_u850.nc --era5-msl e5_msl.nc
           --run NAME=DIR [--run ...] [--baseline NAME --arm NAME] [--figure f.png]
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

BAND = (30.0, 70.0)
SLP_BAND = (20.0, 90.0)
N_DAYS = 30
T_UNITS = "days since 1979-01-01"
GRIB_CODE = {"u": 131, "msl": 151}  # ECMWF table 128: u [m s-1], msl [Pa]


def _one(pattern):
    fs = sorted(glob.glob(pattern))
    if len(fs) != 1:
        raise SystemExit(f"FATAL: expected one file for {pattern}, got {fs}")
    return fs[0]


def _finite(a, what):
    a = np.ma.filled(np.ma.asarray(a, dtype=np.float64), np.nan)
    if not np.isfinite(a).all():
        raise SystemExit(f"FATAL: non-finite values in {what}")
    return a


def window_mean_daily(path, var):
    """Mean over the 30 CMOR daily means of days 1-30; returns (lat, lon, field)."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        if d["time"].units != T_UNITS:
            raise SystemExit(f"FATAL: {path} time units {d['time'].units!r} != {T_UNITS!r}")
        if var == "psl" and d[var].units != "Pa":
            raise SystemExit(f"FATAL: {path} psl units {d[var].units!r}")
        tb = np.asarray(d["time_bnds"][:], np.float64)
        expect = np.stack([np.arange(N_DAYS), np.arange(1, N_DAYS + 1)], axis=1)
        if tb.shape != expect.shape or not np.array_equal(tb, expect):
            raise SystemExit(f"FATAL: {path} time_bnds are not days 1-{N_DAYS}")
        x = _finite(d[var][:], path)
        if var == "ua":
            plev = np.asarray(d["plev"][:])
            if plev.size != 1 or plev[0] != 85000.0:
                raise SystemExit(f"FATAL: {path} plev {plev} is not 850 hPa")
            x = x[:, 0]
        return (np.asarray(d["lat"][:], np.float64), np.asarray(d["lon"][:], np.float64),
                x.mean(axis=0))


def read_ref(path, var, lat, lon):
    """ERA5 field already remapped onto the CMOR grid; grid must match exactly."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        if int(d[var].code) != GRIB_CODE[var]:
            raise SystemExit(f"FATAL: {path} {var} GRIB code {d[var].code}")
        t0 = netCDF4.num2date(d["time"][0], d["time"].units)
        if (t0.year, t0.month) != (1979, 1) or d["time"].size != 1:
            raise SystemExit(f"FATAL: {path} is not the single Jan-1979 record ({t0})")
        rlat, rlon = np.asarray(d["lat"][:]), np.asarray(d["lon"][:])
        if not (np.allclose(rlat, lat) and np.allclose(rlon, lon)):
            raise SystemExit(f"FATAL: {path} grid differs from the CMOR grid")
        x = np.squeeze(_finite(d[var][:], path))
    if x.shape != (lat.size, lon.size):
        raise SystemExit(f"FATAL: {path} {var} shape {x.shape}")
    return x


def band_rmse(diff, lat, band):
    """cos(lat)-weighted RMSE of `diff` (lat,) or (lat, lon) over centres in band."""
    m = (lat >= band[0]) & (lat <= band[1])
    if not m.any():
        raise SystemExit(f"FATAL: no latitude centre in {band}")
    d = np.asarray(diff)[m]
    w = np.cos(np.deg2rad(lat[m]))
    w = w if d.ndim == 1 else np.broadcast_to(w[:, None], d.shape)
    return float(np.sqrt(np.sum(w * d * d) / np.sum(w)))


def global_mean(x, lat):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], x.shape)
    return float(np.sum(w * x) / np.sum(w))


def toa_net(run, lat, lon):
    """Global cos-lat mean of rsdt - rsut - rlut, single Jan-1979 Amon record."""
    import netCDF4
    f = {}
    for v in ("rsdt", "rsut", "rlut"):
        with netCDF4.Dataset(_one(f"{run}/cmor/Amon/{v}_Amon_*.nc")) as d:
            x = _finite(d[v][:], v)
            if (x.shape[0] != 1 or d["time"].units != T_UNITS
                    or float(d["time_bnds"][0, 0]) != 0.0  # record is January:
                    or float(d["time_bnds"][0, 1]) > 31.0  # starts Jan 1, ends by Feb 1
                    or not np.array_equal(d["lat"][:], lat)
                    or not np.array_equal(d["lon"][:], lon)):
                raise SystemExit(f"FATAL: {run} {v} Amon record/grid unexpected")
            f[v] = x[0]
    return global_mean(f["rsdt"] - f["rsut"] - f["rlut"], lat)


def verdict(err_base, err_arm):
    """Pre-registered rule: CONFIRM if arm >= 20 % smaller, REFUTE if < 5 % smaller.

    Compared as arm <= 0.8 base / arm > 0.95 base, the literal form of the rule, so
    the boundaries are not lost to rounding in (base - arm) / base."""
    red = (err_base - err_arm) / err_base
    if err_arm <= 0.80 * err_base:
        return red, "CONFIRM"
    if err_arm > 0.95 * err_base:
        return red, "REFUTE"
    return red, "INCONCLUSIVE"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5-u", required=True)
    p.add_argument("--era5-msl", required=True)
    p.add_argument("--run", action="append", required=True, help="NAME=DIR")
    p.add_argument("--baseline")
    p.add_argument("--arm")
    p.add_argument("--figure")
    a = p.parse_args(argv)
    runs = dict(r.split("=", 1) for r in a.run)
    res, ref_zm = {}, None
    for name, run in runs.items():
        lat, lon, u = window_mean_daily(_one(f"{run}/cmor/day/ua_day_*.nc"), "ua")
        plat, plon, slp = window_mean_daily(_one(f"{run}/cmor/day/psl_day_*.nc"), "psl")
        if not (np.array_equal(plat, lat) and np.array_equal(plon, lon)):
            raise SystemExit(f"FATAL: {run} psl and ua grids differ")
        if ref_zm is None:
            ref_zm = read_ref(a.era5_u, "u", lat, lon).mean(axis=1)
            ref_slp = read_ref(a.era5_msl, "msl", lat, lon)
            lat0, lon0 = lat, lon
        elif not (np.array_equal(lat, lat0) and np.array_equal(lon, lon0)):
            raise SystemExit("FATAL: runs on different CMOR grids")
        zm = u.mean(axis=1)
        res[name] = dict(zm=zm, err=band_rmse(zm - ref_zm, lat, BAND),
                         slp=band_rmse((slp - ref_slp) / 100.0, lat, SLP_BAND),
                         toa=toa_net(run, lat, lon))
    m = (lat0 >= BAND[0]) & (lat0 <= BAND[1])
    print("lat      " + " ".join(f"{x:7.1f}" for x in lat0[m]))
    print("ERA5     " + " ".join(f"{x:7.3f}" for x in ref_zm[m]))
    for n, r in res.items():
        print(f"{n[:8]:8s} " + " ".join(f"{x:7.3f}" for x in r["zm"][m]))
    print("\nrun  u850_rmse_30-70N[m/s]  slp_rmse_20-90N[hPa]  toa_net[W/m2]")
    for n, r in res.items():
        print(f"{n}  {r['err']:.6f}  {r['slp']:.4f}  {r['toa']:.4f}")
    if a.baseline and a.arm:
        red, v = verdict(res[a.baseline]["err"], res[a.arm]["err"])
        print(f"\nreduction (base-arm)/base = {red:+.4f}  -> {v}")
    if a.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        oi = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(lat0, ref_zm, color=oi[0], lw=2.2, label="ERA5 Jan 1979")
        for i, (n, r) in enumerate(res.items()):
            ax[0].plot(lat0, r["zm"], color=oi[i + 1], lw=1.6, label=n)
            ax[1].plot(lat0[m], (r["zm"] - ref_zm)[m], color=oi[i + 1], marker="o",
                       label=f"{n}: RMSE {r['err']:.3f}")
        for x in ax:
            x.axvspan(*BAND, color="0.9", zorder=0)
            x.axhline(0, color="0.5", lw=0.8)
            x.set_xlabel("latitude [deg N]")
            x.legend(fontsize=8)
        ax[0].set_ylabel("zonal-mean u850 [m/s]")
        ax[1].set_ylabel("model - ERA5 [m/s]")
        ax[0].set_title("zonal-mean u850, days 1-30 (grey: scored band)")
        ax[1].set_title("error in 30-70N (cos-lat RMSE in legend)")
        fig.tight_layout()
        fig.savefig(a.figure, dpi=130)


if __name__ == "__main__":
    main()
