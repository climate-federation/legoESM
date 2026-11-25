#!/usr/bin/env python
"""Flux-profile tables for Cabauw, ARM Southern Great Plains and ARM North Slope.

Cabauw: lb1 products are validated and not gap-filled, so tower and flux are measured
values. lb1 flux files stop in February 2021; later flux comes from lc1 where the source
index is 2, which equalled the lb1 sonic flux for 99.8 % of records 2013-2021. lc1 latent
heat never equals the measured LEED, so LE is missing after February 2021.
u*/U10 falls to 0.93, 0.78, then 0.61 of its 2010-2017 value from 2019 while U20/U10 stays
flat, so u* from 2019 on is marked suspect; analyses exclude it by default.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import netCDF4
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "..", "data")]
from arm_sgp_downloader import dataset_builder, netcdf_reader, quality_control, time_alignment  # noqa: E402
from flux_gradient import to_half_hour  # noqa: E402
from flux_profile_table import report, vector_speed, write_table  # noqa: E402

from legoesm import constants  # noqa: E402

ROOT = "/work/bb1596/b309178/diffESM/most_revisit"
SONIC_SOURCE_INDEX = 2
CABAUW_SONIC_LOWERED = 1158105600  # 2006-09-13 UTC: 5 m to 3 m
CABAUW_LB1_FLUX_END = "202012"  # lb1 then has only 2021-05 and 06, which lc1 also covers
CABAUW_USTAR_SUSPECT_FROM = 1546300800  # 2019-01-01 UTC
WIND_SPEED = "wspd_vec_mean"


def clock_lag(ds, level=0, max_lag=3):
    """Correlation of the lowest wind with u* at half-hour lags; lag 0 must win."""
    wind = ds.wind.values[:, level].astype(float)
    ustar = ds.ustar.values[:, 0].astype(float)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        a, b = wind[max_lag + lag:wind.size - max_lag + lag], ustar[max_lag:ustar.size - max_lag]
        ok = np.isfinite(a) & np.isfinite(b)
        out[lag] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None
    return out


def wind_components(speed, direction):
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def on_axis(axis, t, x):
    """Rows of x at each time of axis, NaN where absent."""
    out = np.full((axis.size,) + x.shape[1:], np.nan)
    keep = np.isin(t, axis)
    out[np.searchsorted(axis, t[keep])] = x[keep]
    return out


def latest_versions(paths):
    """One file per month, the highest version."""
    best = {}
    for path in sorted(paths):
        version, month = re.search(r"_v([\d.]+)_(\d{6})\.nc$", path).groups()
        if month not in best or tuple(map(int, version.split("."))) > best[month][0]:
            best[month] = (tuple(map(int, version.split("."))), path)
    return [best[m][1] for m in sorted(best)]


def read_cabauw(paths, names):
    """Interval-end seconds since 1970 and arrays, from monthly CESAR files."""
    times, out = [], {k: [] for k in names}
    for path in paths:
        with netCDF4.Dataset(path) as d:
            origin = netCDF4.date2num(netCDF4.num2date(0.0, d["time"].units, only_use_cftime_datetimes=False),
                                      "seconds since 1970-01-01")
            end = origin + np.asarray(d["time_bnds"][:, 1], dtype=float) * 3600.0
            times.append((np.round(end / 600.0) * 600).astype(np.int64))
            for k in names:
                out[k].append(np.ma.filled(np.ma.asarray(d[k][:], dtype=float), np.nan))
    return np.concatenate(times), {k: np.concatenate(v) for k, v in out.items()}


def half_hour(t, x):
    return to_half_hour(t, x, np.ones(x.shape), samples=3)


def stress_mean(t, ustar):
    """Sub-period stresses average; u* itself does not."""
    t30, tau = half_hour(t, ustar**2)
    return t30, np.sqrt(tau)


def cabauw_flux(lb1_paths, lc1_paths):
    t, v = read_cabauw(lb1_paths, ("HSON", "USTAB", "LEED"))
    tc, c = read_cabauw(lc1_paths, ("H", "UST", "IH", "IUST"))
    h_late = np.where(c["IH"] == SONIC_SOURCE_INDEX, c["H"], np.nan)
    u_late = np.where(c["IUST"] == SONIC_SOURCE_INDEX, c["UST"], np.nan)
    t_all = np.concatenate([t, tc])
    return (t_all, np.concatenate([v["HSON"], h_late]), np.concatenate([v["USTAB"], u_late]),
            np.concatenate([v["LEED"], np.full(tc.size, np.nan)]))


def load_cabauw(root=f"{ROOT}/data/cabauw"):
    tower = latest_versions(glob.glob(f"{root}/cesar_tower_meteo_lb1_t10/*.nc"))
    tt, tv = read_cabauw(tower, ("F", "D", "TA"))
    with netCDF4.Dataset(tower[0]) as d:
        z = np.asarray(d["z"][:], dtype=float)
    order = np.argsort(z)
    z = z[order]
    flux_lb1 = [p for p in sorted(glob.glob(f"{root}/cesar_surface_flux_lb1_t10/*.nc"))
                if re.search(r"_(\d{6})\.nc$", p).group(1) <= CABAUW_LB1_FLUX_END]
    flux_lc1 = [p for p in sorted(glob.glob(f"{root}/cesar_surface_flux_lc1_t10/*.nc"))
                if re.search(r"_(\d{6})\.nc$", p).group(1) > CABAUW_LB1_FLUX_END]
    tf, h, ustar, le = cabauw_flux(flux_lb1, flux_lc1)
    tp, pv = read_cabauw(sorted(glob.glob(f"{root}/cesar_surface_meteo_lb1_t10/*.nc")), ("P0",))
    east, north = wind_components(tv["F"][:, order], tv["D"][:, order])
    t30, east = half_hour(tt, east)
    _, north = half_hour(tt, north)
    _, temp = half_hour(tt, tv["TA"][:, order])
    th, h30 = half_hour(tf, h)
    tu, u30 = stress_mean(tf, ustar)
    tl, le30 = half_hour(tf, le)
    tq, p30 = half_hour(tp, pv["P0"] * 100.0)
    axis = np.union1d(t30, th)
    has_wind = np.isfinite(east).any(axis=0)
    wind = on_axis(axis, t30, vector_speed(east, north)[:, has_wind])
    temp = on_axis(axis, t30, temp)
    n = axis.size
    z_f = np.where(axis < CABAUW_SONIC_LOWERED, 5.0, 3.0)[:, None]
    attrs = dict(site="cabauw", latitude=51.97, longitude=4.926, surface="grass", canopy_height_m=0.0,
                 displacement_m=0.0, source="KNMI CESAR lb1 tower, flux, meteo; lc1 flux after 2020-12",
                 licence="CESAR data policy", averaging_minutes=30, flux_kind="mean_of_shorter_periods",
                 flux_method="eddy covariance, 10-min covariances with KNMI low-frequency correction",
                 le_ends="2021-02: lc1 latent heat is not the measured EC flux",
                 ustar_suspect_from_s=CABAUW_USTAR_SUSPECT_FROM)
    return dict(time_end=axis, z_u=np.tile(z[has_wind], (n, 1)), wind=wind, z_t=np.tile(z, (n, 1)), temp=temp, z_f=z_f,
                ustar=on_axis(axis, tu, u30)[:, None], h=on_axis(axis, th, h30)[:, None],
                le=on_axis(axis, tl, le30)[:, None], pressure=on_axis(axis, tq, p30), t_flux=temp[:, :1], attrs=attrs)


def load_sgp(path=f"{ROOT}/data/arm_sgp/unified/sgp_cf_30min.nc"):
    import xarray as xr

    ds = xr.open_dataset(path)
    time = ds.time.values.astype("datetime64[s]").astype(np.int64)
    n = time.size
    wind_z, temp_z = [4.0, 25.0, 60.0], [2.0, 25.0, 60.0]
    level = ds.sel(height=wind_z)
    wind = vector_speed(level.u.values, level.v.values)
    temp = ds.T.sel(height=temp_z).values + constants.T_freeze
    flux_z = [3.0] + wind_z
    stack = lambda surface, levels: np.column_stack([surface, levels])  # noqa: E731
    attrs = dict(site="arm_sgp", latitude=36.605, longitude=-97.485, surface="pasture and cropland",
                 canopy_height_m=0.0, displacement_m=0.0,
                 source="ARM SGP unified 30-minute dataset: flux level 0 ECOR E14, levels 1-3 tower C1",
                 licence="ARM data policy", averaging_minutes=30, flux_kind="native",
                 flux_method="eddy covariance; EBBR excluded because it derives from profiles")
    return dict(time_end=time, z_u=np.tile(wind_z, (n, 1)), wind=wind, z_t=np.tile(temp_z, (n, 1)), temp=temp,
                z_f=np.tile(flux_z, (n, 1)), ustar=stack(ds.ustar.values, level.ustar_level.values),
                h=stack(ds.H_surface.values, level.H.values), le=stack(ds.LE_surface.values, level.LE.values),
                pressure=ds.surface_pressure.values, t_flux=temp[:, [0, 0, 1, 2]], attrs=attrs)


ECOR_E10 = ("nsa30qcecorE10.c1", "nsa30ecorE10.b1", "nsaecorsfE10.b1")
NSA_STREAMS = {
    "nsa30qcecorE10.c1": {"time": "time_is_start", "vars": {"H": ("corrected_sensible_heat_flux", 1.0),
                                                            "LE": ("corrected_latent_heat_flux", 1.0)}},
    "nsa30ecorE10.b1": {"time": "bounds", "vars": {"H": ("h", 1.0), "LE": ("lv_e", 1.0), "ustar": ("ustar", 1.0),
                                                   "p": ("atm_pres", 1000.0)}},
    "nsaecorsfE10.b1": {"time": "bounds", "vars": {"H": ("sensible_heat_flux", 1.0), "LE": ("latent_flux", 1.0),
                                                   "ustar": ("friction_velocity", 1.0), "p": ("air_pressure", 1000.0)}},
}
MINUTES_PER_AVERAGE = 30


def read_arm_tower(path):
    """One-minute end-stamped tower file, with values the file's own flags reject blanked."""
    with netCDF4.Dataset(path) as d:
        t = int(d["base_time"][...]) + np.round(np.asarray(d["time_offset"][:], dtype=float)).astype(np.int64)
        out = {}
        for name in ("temp_mean", WIND_SPEED, "wspd_arith_mean", "wdir_vec_mean"):
            x = np.ma.filled(np.ma.asarray(d[name][:], dtype=float), np.nan)
            x[x <= netcdf_reader.ARM_MISSING] = np.nan
            flags = np.nan_to_num(np.ma.filled(np.ma.asarray(d["qc_" + name][:], dtype=float), np.nan))
            x[netcdf_reader.rejected(d, d["qc_" + name], flags)] = np.nan
            out[name] = x
        # A dead anemometer writes vector speed 0 with a clean flag; only the arithmetic mean shows the gap.
        out[WIND_SPEED][np.isnan(out["wspd_arith_mean"])] = np.nan
        return t, out, np.asarray(d["height"][:], dtype=float)


def nsa_tower(paths):
    parts = [read_arm_tower(p) for p in paths]
    t = np.concatenate([p[0] for p in parts])
    t, first = np.unique(t, return_index=True)
    v = {k: np.concatenate([p[1][k] for p in parts])[first] for k in parts[0][1]}
    one = np.ones(v["temp_mean"].shape)
    t30, temp = to_half_hour(t, v["temp_mean"] + constants.T_freeze, one, samples=MINUTES_PER_AVERAGE)
    east, north = wind_components(v[WIND_SPEED], v["wdir_vec_mean"])
    _, east = to_half_hour(t, east, one, samples=MINUTES_PER_AVERAGE)
    _, north = to_half_hour(t, north, one, samples=MINUTES_PER_AVERAGE)
    return t30, parts[0][2], vector_speed(east, north), temp


def load_nsa(root=f"{ROOT}/data/arm_nsa_bnf"):
    netcdf_reader.STREAMS.update(NSA_STREAMS)
    tables = {}
    for stream in ECOR_E10:
        raw = netcdf_reader.read_stream(root, stream)
        if len(raw):
            tables[stream] = time_alignment.to_grid(quality_control.apply_flags(raw), time_alignment.native_seconds(raw))
    series = {out: dataset_builder.merged(tables, [(s, name) for s in ECOR_E10])
              for out, name in (("h", "H"), ("le", "LE"), ("ustar", "ustar"), ("p", "p"))}
    seconds = {k: (v.index.tz_localize(None).values.astype("datetime64[s]").astype(np.int64), v.to_numpy(float))
               for k, v in series.items()}
    t30, z, wind, temp = nsa_tower(sorted(glob.glob(f"{root}/raw/armlive/nsatwrC1.b1/*.cdf")))
    axis = np.union1d(t30, seconds["h"][0])
    n = axis.size
    temp = on_axis(axis, t30, temp)
    attrs = dict(site="arm_nsa", latitude=71.323, longitude=-156.609, surface="tundra", canopy_height_m=0.0,
                 displacement_m=0.0, source="ARM NSA tower C1 and ECOR E10, 220 m apart; E11 is 10 km away, not used",
                 licence="ARM data policy", averaging_minutes=30, flux_kind="native",
                 flux_method="eddy covariance at 3 m")
    return dict(time_end=axis, z_u=np.tile(z, (n, 1)), wind=on_axis(axis, t30, wind), z_t=np.tile(z, (n, 1)), temp=temp,
                z_f=np.full((n, 1), 3.0), ustar=on_axis(axis, *seconds["ustar"])[:, None],
                h=on_axis(axis, *seconds["h"])[:, None], le=on_axis(axis, *seconds["le"])[:, None],
                pressure=on_axis(axis, *seconds["p"]), t_flux=temp[:, :1], attrs=attrs)


LOADERS = {"cabauw": load_cabauw, "arm_sgp": load_sgp, "arm_nsa": load_nsa}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site", choices=sorted(LOADERS))
    a = ap.parse_args()
    path = f"{ROOT}/data/flux_profile/{a.site}.nc"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_table(path, **LOADERS[a.site]())
    import xarray as xr

    result = report(path)
    ds = xr.open_dataset(path, decode_times=False)
    result["clock_lag_corr"] = clock_lag(ds)
    if "ustar_suspect_from_s" in ds.attrs:
        trusted = path + ".trusted.tmp"
        ds.sel(time=ds.time < ds.attrs["ustar_suspect_from_s"]).to_netcdf(trusted)
        result["before_suspect"] = report(trusted)
        os.remove(trusted)
    with open(f"{ROOT}/out/flux_profile_{a.site}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
