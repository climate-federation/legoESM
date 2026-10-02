#!/usr/bin/env python3
"""Northern winter land cold bias: model vs ERA5 / HadCRUT5 on the SAME dates.

Window means come from the run's own CMOR accumulator sidecars:
``(sum_end - sum_start) / (count_end - count_start)`` for one calendar month's
bucket, so a window inside a month is exact and needs no partial-file algebra.
Control: the end sidecar's full-month mean must reproduce the published Amon
file (max |diff| < 1e-3) or the probe refuses.  State fields (tas, ts, ta) are
sampled 24x per day (counts are printed); fluxes once per day (interval means).

ERA5 comes from the DKRZ pool (GRIB, N320) converted by ``prep`` with cdo into
regular-grid NetCDF north of 30N.  Daily fc accumulations are J/m2 per day ->
/86400; ECMWF turbulent fluxes are positive DOWN and are flipped to CMOR's UP.
References are evaluated at the model's SAMPLED points by bilinear
interpolation: published labels when the file carries
``legoesm_lat_sampling = labelled_cell_centres``, else linspace(-90, 90, nlat)
(the pre-#1838 pole-to-pole sampling).  Weights cos(sampled lat) x land
(sftlf >= --land-min %).  Model tas is a surface-layer diagnosis at model
orography; the mean orography difference (model - ERA5) is printed, no lapse
correction is applied.

Soil temperature and snow (SWE) come from the daily checkpoints on the MPAS
mesh (00 UTC snapshots), ERA5 interpolated to the cell centres; land cells =
nearest 5-degree sftlf >= --land-min.

Usage:
  nh_winter_land.py prep                                   # cdo, once
  nh_winter_land.py window <run> --start 344 --end 365     # Dec 11-31 2001
  nh_winter_land.py daily <run> [--year 2001]              # daily tas series
  nh_winter_land.py monthly <run>                          # Amon tas by month
  nh_winter_land.py soil <run> --days 345 365              # checkpoints
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import xarray as xr

ROOT = pathlib.Path(os.environ.get(
    "LEGOESM_AMIP_RUNS", "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs"))
E5DIR = pathlib.Path(os.environ.get("NHW_ERA5", "/scratch/b/b381103/nhwinter/era5"))
POOL = pathlib.Path("/pool/data/ERA5/E5")
CDO = os.environ.get("CDO", "/sw/spack-levante/cdo-2.6.0-akkxhz/bin/cdo")
HADCRUT = ("/work/bd1179/b309141/climateeval_input/observation_HadCRUT5/mon/tas/"
           "OBS_HadCRUT5_ground_5.0.1.0-analysis_Amon_tas_185001-202309.nc")
YEAR = 2001
# ERA5 pool code -> (kind, short name).  an = daily-mean analysis, fc = daily
# accumulation [J/m2].
SF = {"167": ("an", "t2m"), "235": ("an", "skt"), "141": ("an", "sd"),
      "139": ("an", "stl1"), "170": ("an", "stl2"), "183": ("an", "stl3"),
      "236": ("an", "stl4"), "146": ("fc", "sshf"), "147": ("fc", "slhf"),
      "169": ("fc", "ssrd"), "175": ("fc", "strd"), "176": ("fc", "ssr"),
      "177": ("fc", "str")}
BOXES = {  # lat0, lat1, lon0, lon1 (deg E, may wrap)
    "land 45-70N": (45.0, 70.0, 0.0, 360.0),
    "Siberia 55-70N 90-140E": (55.0, 70.0, 90.0, 140.0),
    "Canada 50-65N 95-130W": (50.0, 65.0, 230.0, 265.0),
    "Alaska 60-72N 140-170W": (60.0, 72.0, 190.0, 220.0),
}
# ERA5 soil layer mid-depths [m] (0-7, 7-28, 28-100, 100-289 cm).
STL_DEPTH = {"stl1": 0.035, "stl2": 0.175, "stl3": 0.64, "stl4": 1.945}


# ---------------------------------------------------------------- ERA5 prep
def _cdo(args):
    subprocess.run([CDO, "-s", "-O", "-f", "nc4"] + args, check=True)


def prep(months=(11, 12)):
    E5DIR.mkdir(parents=True, exist_ok=True)
    box = "-sellonlatbox,0,360,30,90"
    for code, (kind, name) in SF.items():
        mons = range(1, 13) if code == "167" else months
        for m in mons:
            src = (POOL / "sf" / kind / "1D" / code /
                   f"E5sf{'00' if kind == 'an' else '12'}_1D_{YEAR}-{m:02d}_{code}.grb")
            out = E5DIR / f"{name}_{YEAR}{m:02d}.nc"
            if not out.exists():
                _cdo([box, "-setgridtype,regular", str(src), str(out)])
    for m in months:
        src = POOL / "pl" / "an" / "1D" / "130" / f"E5pl00_1D_{YEAR}-{m:02d}_130.grb"
        out = E5DIR / f"t925_{YEAR}{m:02d}.nc"
        if not out.exists():
            _cdo([box, "-setgridtype,regular", "-sellevel,92500", str(src), str(out)])
    out = E5DIR / f"t2m_mon_{YEAR}.nc"
    if not out.exists():
        _cdo([box, "-setgridtype,regular",
              str(POOL / "sf" / "an" / "1M" / "167" / f"E5sf00_1M_{YEAR}_167.grb"), str(out)])
    out = E5DIR / "orog.nc"
    if not out.exists():
        _cdo([box, "-setgridtype,regular",
              str(POOL / "sf" / "an" / "IV" / "129" / "E5sf00_IV_INVARIANT_129.grb"), str(out)])


def era5(name, months, days=None):
    """(ntime, lat, lon) DataArray for ``name`` over the given months; fc
    fields converted to W/m2 (turbulent fluxes flipped to positive UP)."""
    das = []
    for m in months:
        ds = xr.open_dataset(E5DIR / f"{name}_{YEAR}{m:02d}.nc")
        das.append(ds[max(ds.data_vars, key=lambda v: ds[v].ndim)])   # skip *_bnds
    da = xr.concat(das, "time")
    if da.ndim == 4:
        da = da.isel({da.dims[1]: 0})
    if days is not None:
        d0, d1 = days
        t = da["time"].dt
        keep = np.array([d0 <= dt.date(int(y), int(mo), int(dd)) <= d1
                         for y, mo, dd in zip(t.year.values, t.month.values, t.day.values)])
        da = da.isel(time=np.flatnonzero(keep))
    if SF.get(_code(name), ("an",))[0] == "fc":
        da = da / 86400.0
        if name in ("sshf", "slhf"):
            da = -da
    return da.load()


def _code(name):
    return next((c for c, (_, n) in SF.items() if n == name), None)


def interp(field2d, flat, flon, plat, plon):
    """Bilinear interpolation of a regular (lat, lon) field to points
    (periodic in longitude; latitude may be descending)."""
    from scipy.interpolate import RegularGridInterpolator
    flat = np.asarray(flat, float)
    flon = np.asarray(flon, float) % 360.0
    f = np.asarray(field2d, float)
    if flat[0] > flat[-1]:
        flat, f = flat[::-1], f[::-1]
    o = np.argsort(flon)
    flon, f = flon[o], f[:, o]
    flon = np.concatenate([flon[-1:] - 360.0, flon, flon[:1] + 360.0])
    f = np.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
    fn = RegularGridInterpolator((flat, flon), f, bounds_error=False, fill_value=np.nan)
    return fn(np.stack([np.asarray(plat, float), np.asarray(plon, float) % 360.0], -1))


# ----------------------------------------------------------- model helpers
def sampled_lat(path):
    d = xr.open_dataset(path, decode_times=False)
    lab = np.asarray(d["lat"].values, float)
    if d.attrs.get("legoesm_lat_sampling") == "labelled_cell_centres":
        return lab, "centres"
    return np.linspace(-90.0, 90.0, lab.size), "pole-to-pole"


def amon_file(run, var):
    fs = sorted(glob.glob(str(ROOT / run / "cmor" / "Amon" / f"{var}_Amon_*.nc")))
    if not fs:
        raise SystemExit(f"{run}: no Amon {var}")
    return fs[-1]


def sftlf(run):
    f = glob.glob(str(ROOT / run / "cmor" / "fx" / "sftlf_fx_*.nc"))[0]
    return np.asarray(xr.open_dataset(f)["sftlf"].values, float)


def orog(run):
    f = glob.glob(str(ROOT / run / "cmor" / "fx" / "orog_fx_*.nc"))[0]
    return np.asarray(xr.open_dataset(f)["orog"].values, float)


def box_mask(lat2, lon2, box):
    la0, la1, lo0, lo1 = box
    m = (lat2 >= la0) & (lat2 <= la1)
    if lo1 - lo0 < 360.0:
        m &= ((lon2 - lo0) % 360.0) <= (lo1 - lo0)
    return m


def wmean(x, w):
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    return float((x[ok] * w[ok]).sum() / w[ok].sum())


def _sidecar(run, day):
    z = np.load(ROOT / run / f"cmor_accum_day_{day:04d}.npz", allow_pickle=True)
    man = json.loads(str(z["monthly.__manifest__"]))
    out = {}
    for kind in ("data_2d", "data_3d"):
        for y, m, var, count, key in man[kind]:
            out[(y, m, var)] = (np.asarray(z[f"monthly.{key}"], float), int(count))
    return out


def window(run, start, end, land_min):
    s0, s1 = _sidecar(run, start), _sidecar(run, end)
    # bucket = calendar month of the window's last day (relative year 0 = YEAR)
    ym = (0, (dt.date(YEAR, 1, 1) + dt.timedelta(days=end - 1)).month)
    if (dt.date(YEAR, 1, 1) + dt.timedelta(days=start)).month != ym[1]:
        raise SystemExit(f"{run}: window {start}..{end} crosses a month")
    fields = {}
    for var in ("tas", "ts", "hfss", "hfls", "ps", "ta"):
        a1, c1 = s1[(*ym, var)]
        a0, c0 = s0.get((*ym, var), (np.zeros_like(a1), 0))
        if c1 - c0 <= 0:
            raise SystemExit(f"{run}/{var}: empty window")
        fields[var] = (a1 - a0) / (c1 - c0)
        if var == "ta":
            # the 3-D sidecar stores plev19 TOP-first (measured: its level 9
            # equals the file's level 9, its level 17 the file's level 1); flip
            # to the file's 1000 hPa-first order.  The Amon control checks it.
            a1 = a1[..., ::-1]
            fields[var] = fields[var][..., ::-1]
        fields[var + "__n"] = (c0, c1)
        # control: end sidecar full-month mean == published Amon (month closed)
        import calendar
        nd = calendar.monthrange(YEAR, ym[1])[1]
        if c1 not in (nd, 24 * nd):
            fields[var + "__ctl"] = "skipped (open month)"
            continue
        fields[var + "__ctl"] = "ok"
        f = amon_file(run, var)
        pub = np.asarray(xr.open_dataset(f, decode_times=False)[var].isel(time=-1).values, float)
        full = a1 / c1
        if var == "ta":
            pub = np.moveaxis(pub, 0, -1)
        err = np.nanmax(np.abs(full - pub))
        if not np.allclose(full, pub, rtol=1e-6, atol=1e-3, equal_nan=True):   # float32 file
            raise SystemExit(f"{run}/{var}: sidecar month mean != Amon (max {err:.3g})")
    fields["t925"] = fields.pop("ta")[..., 1]       # plev19 index 1 = 925 hPa
    return fields, ym


# ------------------------------------------------------------------ reports
def _points(run, var="tas"):
    f = amon_file(run, var)
    plat, how = sampled_lat(f)
    lon = np.asarray(xr.open_dataset(f, decode_times=False)["lon"].values, float) % 360.0
    lat2, lon2 = np.meshgrid(plat, lon, indexing="ij")
    return lat2, lon2, how


def cmd_window(a):
    f, ym = window(a.run, a.start, a.end, a.land_min)
    lat2, lon2, how = _points(a.run)
    land = sftlf(a.run) >= a.land_min
    month = ym[1]
    d0 = (dt.date(YEAR, 1, 1) + dt.timedelta(days=a.start))
    d1 = (dt.date(YEAR, 1, 1) + dt.timedelta(days=a.end - 1))
    print(f"# {a.run} days {a.start}->{a.end} = {d0}..{d1} (month {month}); lat sampling {how}; "
          f"tas samples {f['tas__n']} ps {f['ps__n']} hfss {f['hfss__n']}; Amon control {f['tas__ctl']}/{f['ta__ctl']}; land sftlf>={a.land_min}")
    e = {n: era5(n, [month], (d0, d1)).mean("time") for n in
         ("t2m", "skt", "sshf", "slhf", "strd", "ssrd", "str", "ssr")}
    e["t925"] = era5("t925", [month], (d0, d1)).mean("time")
    g = {k: interp(v.values, v[v.dims[0]].values, v[v.dims[1]].values, lat2.ravel(), lon2.ravel()
                   ).reshape(lat2.shape) for k, v in e.items()}
    zo = xr.open_dataset(E5DIR / "orog.nc")
    zv = zo[list(zo.data_vars)[0]].isel(time=0) / 9.80665
    zE = interp(zv.values, zv[zv.dims[0]].values, zv[zv.dims[1]].values,
                lat2.ravel(), lon2.ravel()).reshape(lat2.shape)
    had = xr.open_dataset(HADCRUT)["tas"]
    had = had.sel(time=(had.time.dt.year == YEAR) & (had.time.dt.month == month)).isel(time=0)
    gh = interp(had.values, had.lat.values, had.lon.values, lat2.ravel(), lon2.ravel()
                ).reshape(lat2.shape)
    deep = f["ps"] >= 95000.0
    w0 = np.cos(np.deg2rad(lat2))
    rows = []
    for name, box in BOXES.items():
        w = w0 * (box_mask(lat2, lon2, box) & land)
        wd = w * deep
        r = {
            "box": name, "npts": int((w > 0).sum()),
            "tas": wmean(f["tas"], w), "dtas_E5": wmean(f["tas"] - g["t2m"], w),
            "dtas_Had": wmean(f["tas"] - gh, w), "Had_n": int(np.isfinite(gh[w > 0]).sum()),
            "dts_E5": wmean(f["ts"] - g["skt"], w),
            "tas-ts": wmean(f["tas"] - f["ts"], w), "E5 t2m-skt": wmean(g["t2m"] - g["skt"], w),
            "inv T925-tas": wmean(f["t925"] - f["tas"], wd),
            "E5 inv": wmean(g["t925"] - g["t2m"], wd),
            "hfss": wmean(f["hfss"], w), "E5 hfss": wmean(g["sshf"], w),
            "hfls": wmean(f["hfls"], w), "E5 hfls": wmean(g["slhf"], w),
            "E5 DLW": wmean(g["strd"], w),
            "E5 ULW": wmean(g["strd"] - g["str"], w), "E5 SWnet": wmean(g["ssr"], w),
            "E5 G=Rn-H-LE": wmean(g["ssr"] + g["str"] - g["sshf"] - g["slhf"], w),
            "dz model-E5 [m]": wmean(orog(a.run) - zE, w),
        }
        rows.append(r)
    for r in rows:
        print(" | ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))


def cmd_daily(a):
    f = sorted(glob.glob(str(ROOT / a.run / "cmor" / "day" / "tas_day_*.nc")))[0]
    d = xr.open_dataset(f)
    plat, how = sampled_lat(f)
    lon = np.asarray(d["lon"].values, float) % 360.0
    lat2, lon2 = np.meshgrid(plat, lon, indexing="ij")
    land = sftlf(a.run) >= a.land_min
    t = d["time"].dt
    dates = [dt.date(int(y), int(m), int(dd)) for y, m, dd in
             zip(t.year.values, t.month.values, t.day.values)]
    months = sorted({x.month for x in dates if x.year == YEAR})
    e = era5("t2m", months)
    et = e["time"].dt
    edates = {dt.date(int(y), int(m), int(dd)): i for i, (y, m, dd) in
              enumerate(zip(et.year.values, et.month.values, et.day.values))}
    print(f"# {a.run} daily tas - ERA5 t2m, lat sampling {how}, cell_methods "
          f"{d['tas'].attrs.get('cell_methods')!r}, land>={a.land_min}")
    names = list(BOXES)
    print("date      " + " | ".join(names))
    for i, x in enumerate(dates):
        if x not in edates or (a.every > 1 and i % a.every):
            continue
        ef = e.isel(time=edates[x])
        g = interp(ef.values, ef[ef.dims[0]].values, ef[ef.dims[1]].values,
                   lat2.ravel(), lon2.ravel()).reshape(lat2.shape)
        diff = np.asarray(d["tas"].isel(time=i).values, float) - g
        vals = [wmean(diff, np.cos(np.deg2rad(lat2)) * (box_mask(lat2, lon2, BOXES[n]) & land))
                for n in names]
        print(f"{x} " + " | ".join(f"{v:+6.2f}" for v in vals))


def cmd_monthly(a):
    f = amon_file(a.run, "tas")
    d = xr.open_dataset(f)
    plat, how = sampled_lat(f)
    lon = np.asarray(d["lon"].values, float) % 360.0
    lat2, lon2 = np.meshgrid(plat, lon, indexing="ij")
    land = sftlf(a.run) >= a.land_min
    e = xr.open_dataset(E5DIR / f"t2m_mon_{YEAR}.nc")
    e = e[list(e.data_vars)[0]]
    had = xr.open_dataset(HADCRUT)["tas"]
    had = had.sel(time=had.time.dt.year == YEAR)
    print(f"# {a.run} monthly tas bias, lat sampling {how}; vs ERA5 t2m {YEAR} | vs HadCRUT5 {YEAR}")
    for i in range(d.sizes["time"]):
        m = int(d["time"].dt.month.values[i])
        ef = e.isel(time=m - 1)
        g = interp(ef.values, ef[ef.dims[0]].values, ef[ef.dims[1]].values,
                   lat2.ravel(), lon2.ravel()).reshape(lat2.shape)
        hf = had.isel(time=m - 1)
        gh = interp(hf.values, hf.lat.values, hf.lon.values, lat2.ravel(), lon2.ravel()
                    ).reshape(lat2.shape)
        x = np.asarray(d["tas"].isel(time=i).values, float)
        out = []
        for n, box in BOXES.items():
            w = np.cos(np.deg2rad(lat2)) * (box_mask(lat2, lon2, box) & land)
            out.append(f"{n}: E5 {wmean(x - g, w):+6.2f} Had {wmean(x - gh, w):+6.2f}")
        print(f"{YEAR}-{m:02d} " + " | ".join(out))


def cmd_soil(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cloud_layers import mesh_coords
    exp = json.load(open(ROOT / a.run / "experiment_config.json"))
    lat, lon, area = mesh_coords(exp)
    fx = glob.glob(str(ROOT / a.run / "cmor" / "fx" / "sftlf_fx_*.nc"))[0]
    flat, _ = sampled_lat(fx)
    flon = np.asarray(xr.open_dataset(fx)["lon"].values, float) % 360.0
    fr = sftlf(a.run)
    i = np.abs(flat[None, :] - lat[:, None]).argmin(1)
    j = np.abs(((flon[None, :] - lon[:, None] + 180.0) % 360.0) - 180.0).argmin(1)
    land = fr[i, j] >= a.land_min
    days = list(range(a.days[0], a.days[1] + 1))
    T, S = [], []
    for dd in days:
        z = np.load(ROOT / a.run / f"checkpoint_day_{dd:04d}.npz", allow_pickle=True)
        T.append(np.asarray(z["land_ml_T_soil"], float))
        S.append(np.asarray(z["land_ml_snow_depth"], float))
        dz = np.asarray(z["land_soil_dz"], float)
    T, S = np.mean(T, 0), np.mean(S, 0)
    zc = np.cumsum(dz) - dz / 2
    # checkpoint day N = 00 UTC on date Jan1 + N days -> ERA5 daily means of the
    # dates the snapshots fall on
    d0 = dt.date(YEAR, 1, 1) + dt.timedelta(days=days[0])
    d1 = dt.date(YEAR, 1, 1) + dt.timedelta(days=days[-1])
    months = sorted({d0.month, d1.month} & {11, 12})
    d1 = min(d1, dt.date(YEAR, 12, 31))
    print(f"# {a.run} checkpoints {days[0]}-{days[-1]} ({d0}..{d1}) soil/snow vs ERA5 daily means; "
          f"model layer centres {np.round(zc, 3).tolist()} m; land>={a.land_min}")
    e = {n: era5(n, months, (d0, d1)).mean("time") for n in list(STL_DEPTH) + ["sd", "skt"]}
    g = {k: interp(v.values, v[v.dims[0]].values, v[v.dims[1]].values, lat, lon)
         for k, v in e.items()}
    for name, box in BOXES.items():
        la0, la1, lo0, lo1 = box
        m = land & (lat >= la0) & (lat <= la1)
        if lo1 - lo0 < 360:
            m &= ((lon - lo0) % 360.0) <= (lo1 - lo0)
        w = area * m
        parts = [f"{name} ncell {int(m.sum())}"]
        for k, zd in STL_DEPTH.items():
            tm = np.array([np.interp(zd, zc, T[c]) for c in np.flatnonzero(m)])
            parts.append(f"{k}@{zd}m model {wmean(tm, w[m]):.1f} E5 {wmean(g[k][m], w[m]):.1f}")
        parts.append(f"SWE model {wmean(S[m], w[m]):.0f} E5 {wmean(1000 * g['sd'][m], w[m]):.0f} kg/m2")
        parts.append(f"snow>1kg frac model {wmean((S[m] > 1).astype(float), w[m]):.2f} "
                     f"E5 {wmean((g['sd'][m] > 0.001).astype(float), w[m]):.2f}")
        print(" | ".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    for name in ("window", "daily", "monthly", "soil"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--land-min", type=float, default=50.0)
        if name == "window":
            p.add_argument("--start", type=int, required=True)
            p.add_argument("--end", type=int, required=True)
        if name == "daily":
            p.add_argument("--every", type=int, default=1)
        if name == "soil":
            p.add_argument("--days", type=int, nargs=2, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        prep()
    else:
        {"window": cmd_window, "daily": cmd_daily, "monthly": cmd_monthly,
         "soil": cmd_soil}[a.cmd](a)


if __name__ == "__main__":
    main()
