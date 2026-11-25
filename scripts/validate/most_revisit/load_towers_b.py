#!/usr/bin/env python
"""Common flux-profile tables for MOSAiC Met City, SHEBA, FINO1 and Falkenberg.

Stamp conventions, measured or stated by the source: MOSAiC 10-min files stamp the START of
the interval (their 10-min temperature equals the mean of the next ten 1-min values);
SHEBA stamps the start of the hour (shortwave peaks at stamp 11.6 local); FINO1 and
Falkenberg stamp the end. Only source flags are applied.
"""
from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import time

import netCDF4
import numpy as np

import flux_gradient as fg
import flux_profile_table as table
from legoesm import constants

MOSAIC_LEVELS = ("2m", "6m", "10m", "mast")
MOSAIC_PERIOD_S = 600
MOSAIC_GOOD = 0  # GLM and codex: Caution marks tower-sector flow distortion, so it is dropped.
HPA_TO_PA = 100.0
SHEBA_MISSING = (999.0, 9999.0)
SHEBA_DAY_ONE = calendar.timegm((1997, 1, 1, 0, 0, 0))
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
FINO_PERIOD_SAMPLES = 3
FINO_WIND_GOOD = (0, 1)  # flag 2 is 0.3 percent of records, so strict costs nothing
FINO_FLUX_GOOD = (0, 1)  # Mauder and Foken 2004: 2 means discard
STANDARD_SEA_LEVEL_PRESSURE_PA = 101325.0
FALKENBERG_FLUX_HEIGHTS = {"2m": 2.3, "50m": 50.3, "90m": 90.3}  # sensor_height in the files
FALKENBERG_GOOD_FLUX = 3


def _values(variable):
    return np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan)


def _epoch(text):
    return calendar.timegm(time.strptime(text, "%Y-%m-%d %H:%M:%S"))


def _height(variable, t):
    """Measured height per time; the attributes give the day's start, end and last change."""
    if not hasattr(variable, "height_start"):
        return np.full(t.shape, np.nan)
    change = _epoch(variable.height_change_time)
    return np.where(t >= change, float(variable.height_end), float(variable.height_start))


def _mosaic_day(path):
    with netCDF4.Dataset(path) as d:
        t = float(d["base_time"][...]) + _values(d["time"]) + MOSAIC_PERIOD_S
        cols, good = {}, {}

        def put(name, x, ok):
            cols[name], good[name] = x, ok

        for lev in MOSAIC_LEVELS:
            sector = d[f"wind_sector_qc_info_{lev}"][:] == MOSAIC_GOOD
            turb = (d[f"turbulence_{lev}_qc"][:] == MOSAIC_GOOD) & sector
            wind_ok = (d[f"wspd_u_mean_{lev}_qc"][:] == MOSAIC_GOOD) & (d[f"wspd_v_mean_{lev}_qc"][:] == MOSAIC_GOOD) & sector
            put(f"u_{lev}", _values(d[f"wspd_u_mean_{lev}"]), wind_ok)
            put(f"v_{lev}", _values(d[f"wspd_v_mean_{lev}"]), wind_ok)
            put(f"t_{lev}", _values(d[f"temp_{lev}"]) + constants.T_freeze, d[f"temp_{lev}_qc"][:] == MOSAIC_GOOD)
            put(f"stress_{lev}", _values(d[f"ustar_{lev}"]) ** 2, turb)
            put(f"h_{lev}", _values(d[f"Hs_{lev}"]), turb)
            every = np.ones(t.shape, bool)
            put(f"zu_{lev}", _height(d[f"wspd_u_mean_{lev}"], t - MOSAIC_PERIOD_S), every)
            put(f"zt_{lev}", _height(d[f"temp_{lev}"], t - MOSAIC_PERIOD_S), every)
            put(f"zf_{lev}", _height(d[f"Hs_{lev}"], t - MOSAIC_PERIOD_S), every)
        put("le", _values(d["Hl_Webb"]), d["Hl_qc"][:] == MOSAIC_GOOD)
        put("p", _values(d["atmos_pressure_2m"]) * HPA_TO_PA, d["atmos_pressure_2m_qc"][:] == MOSAIC_GOOD)
        lat, lon = float(np.nanmedian(_values(d["lat_tower"]))), float(np.nanmedian(_values(d["lon_tower"])))
    return t, cols, good, lat, lon


def load_mosaic(root):
    days = [_mosaic_day(p) for p in sorted(glob.glob(f"{root}/mosseb.metcity.level3.4.10min.*.nc"))]
    if not days:
        raise FileNotFoundError(root)
    t = np.concatenate([d[0] for d in days])
    half = {}
    for name in days[0][1]:
        x = np.concatenate([d[1][name] for d in days])
        ok = np.concatenate([np.ma.filled(d[2][name], False) for d in days])
        keys, half[name] = fg.to_half_hour(t, x, ok.astype(int), samples=3)
    stack = lambda prefix: np.stack([half[f"{prefix}_{lev}"] for lev in MOSAIC_LEVELS], axis=1)
    le = np.full(stack("h").shape, np.nan)
    le[:, 0] = half["le"]  # the gas analyser sits at the first tower level, 2.35 m
    wind = table.vector_speed(stack("u"), stack("v"))
    return dict(time_end=keys, z_u=stack("zu"), wind=wind, z_t=stack("zt"), temp=stack("t"), z_f=stack("zf"),
                ustar=np.sqrt(stack("stress")), h=stack("h"), le=le, pressure=half["p"], t_flux=stack("t"),
                attrs=dict(site="mosaic_metcity", latitude=float(np.nanmedian([d[3] for d in days])),
                           longitude=float(np.nanmedian([d[4] for d in days])), surface="sea ice",
                           canopy_height_m=0.0, displacement_m=0.0, source="doi:10.18739/A2PV6B83F level 3.4 mosseb",
                           licence="CC0", averaging_minutes=30, flux_kind="mean_of_shorter_periods",
                           flux_method="eddy covariance, spectrum-integrated 10-min sonic fluxes; le is Hl_Webb",
                           note="drifting floe, latitude and longitude are medians; bulk fluxes excluded"))


def load_sheba(profile_path):
    with open(profile_path) as f:
        names = [s.strip() for s in f.readline().split("\t")]
    names = [n for n in names if n]
    x = np.genfromtxt(profile_path, delimiter="\t", skip_header=2, usecols=range(len(names)))
    x[np.isin(x, SHEBA_MISSING)] = np.nan
    col = lambda n: x[:, names.index(n)]
    levels = lambda fmt: np.stack([col(fmt.format(k)) for k in range(1, 6)], axis=1)
    time_end = SHEBA_DAY_ONE + np.round((col("JD") - 1.0) * 24.0).astype(np.int64) * SECONDS_PER_HOUR + SECONDS_PER_HOUR
    sonic_ok = levels("fl{}") == 0
    z = levels("z{}")
    temp = levels("T{}") + constants.T_freeze
    le = np.full(z.shape, np.nan)
    ophir_level = int(np.nanargmin(np.abs(np.nanmedian(z, axis=0) - np.nanmedian(col("zoph")))))
    le[:, ophir_level] = col("hl")
    return dict(time_end=time_end, z_u=z, wind=np.where(sonic_ok, levels("ws{}"), np.nan), z_t=z, temp=temp, z_f=z,
                ustar=np.where(sonic_ok, levels("u*{}"), np.nan), h=np.where(sonic_ok, levels("hs{}"), np.nan),
                le=le, pressure=col("Press") * HPA_TO_PA, t_flux=temp,
                attrs=dict(site="sheba", latitude=float(np.nanmedian(col("lat"))), longitude=float(np.nanmedian(col("lon"))),
                           surface="sea ice", canopy_height_m=0.0, displacement_m=0.0,
                           source="ASFG 3.0 prof_file_all6_ed_hd.txt, Arctic Data Center mirror", licence="see science_metadata.xml",
                           averaging_minutes=60, flux_kind="mean_of_shorter_periods",
                           flux_method="eddy covariance, hourly from 10-min sonic periods; negative u* means upward momentum flux, as in source",
                           note="drifting floe, latitude and longitude are medians; the file gives one height set for wind and temperature; level order is not always ascending"))


def load_fino1(path):
    with netCDF4.Dataset(path) as d:
        order = np.argsort(d["height"][:])
        z = np.asarray(d["height"][:], dtype=float)[order]
        t = (_epoch("2006-02-01 00:10:00") + 60.0 * _values(d["time"]))
        get = lambda k: _values(d[k])[order].T
        speed, direction = get("wind_speed"), np.deg2rad(get("wind_from_direction"))
        wind_ok = np.isin(get("wind_speed_qc"), FINO_WIND_GOOD) & np.isin(get("wind_from_direction_qc"), FINO_WIND_GOOD)
        tau_ok, heat_ok = np.isin(get("qc_Tau"), FINO_FLUX_GOOD), np.isin(get("qc_H"), FINO_FLUX_GOOD)
        ustar, heat, sonic = get("u*"), get("H"), get("sonic_temperature")
        lat, lon, licence = float(d.geospatial_lat_max), float(d.geospatial_lon_max), d.license
    every = np.ones_like(speed, dtype=int)
    half = lambda x, ok: fg.to_half_hour(t, x, ok.astype(int), samples=FINO_PERIOD_SAMPLES)
    keys, east = half(-speed * np.sin(direction), wind_ok)
    _, north = half(-speed * np.cos(direction), wind_ok)
    sonic30 = half(sonic, every)[1]
    n = keys.size
    heights = np.tile(z, (n, 1))
    return dict(time_end=keys, z_u=heights, wind=table.vector_speed(east, north), z_t=heights, temp=sonic30, z_f=heights,
                ustar=np.sqrt(half(ustar**2, tau_ok)[1]), h=half(heat, heat_ok)[1], le=np.full(heights.shape, np.nan),
                pressure=np.full(n, STANDARD_SEA_LEVEL_PRESSURE_PA), t_flux=sonic30,
                attrs=dict(site="fino1", latitude=lat, longitude=lon, surface="sea", canopy_height_m=0.0, displacement_m=0.0,
                           source="doi:10.60609/B9T9-0X55", licence=licence, averaging_minutes=30,
                           flux_kind="mean_of_shorter_periods", flux_method="eddy covariance, EddyPro 7.0.9 10-min sonic fluxes",
                           temp_kind="sonic", pressure_source="assumed constant, none in file",
                           height_reference="Lowest Astronomical Tide, not corrected for sea level"))


def _on_grid(grid, t, x):
    out = np.full((grid.size,) + x.shape[1:], np.nan)
    out[np.searchsorted(grid, t)] = x
    return out


def falkenberg_table(mast, flux):
    """Levels run up the 10 m mast, then the 99 m mast, so 10 m appears twice on different sensors."""
    names = tuple(FALKENBERG_FLUX_HEIGHTS)
    grid = np.unique(np.concatenate([mast[m]["t"] for m in mast] + [flux[f]["t"] for f in names]))
    wind = np.concatenate([_on_grid(grid, mast[m]["t"], mast[m]["wspeed"]) for m in ("low", "high")], axis=1)
    temp = np.concatenate([_on_grid(grid, mast[m]["t"], mast[m]["ta"]) for m in ("low", "high")], axis=1)
    z = np.tile(np.concatenate([mast["low"]["z"], mast["high"]["z"]]), (grid.size, 1))

    def flux_column(key, flag):
        rows = []
        for f in names:
            b = flux[f]
            rows.append(_on_grid(grid, b["t"], np.where(b[flag] == FALKENBERG_GOOD_FLUX, b[key], np.nan) if flag else b[key]))
        return np.stack(rows, axis=1)

    pressure = np.full(grid.size, np.nan)
    for f in names:
        pressure = np.where(np.isfinite(pressure), pressure, _on_grid(grid, flux[f]["t"], flux[f]["pa"]))
    z_f = np.tile([FALKENBERG_FLUX_HEIGHTS[f] for f in names], (grid.size, 1))
    return dict(time_end=grid.astype(np.int64), z_u=z, wind=wind, z_t=z, temp=temp, z_f=z_f,
                ustar=flux_column("ustar", "ustar_flag"), h=flux_column("hfss", "hfss_flag"),
                le=flux_column("hfls", "hfls_flag"), pressure=pressure, t_flux=flux_column("tsonic", None),
                attrs=dict(site="falkenberg_2021", latitude=52.1665, longitude=14.1222, surface="short grass",
                           canopy_height_m=0.0, displacement_m=0.0, source="doi:10.25592/uhhfdm.14516",
                           licence="deposit CC BY; file attributes say registered users, non-commercial", averaging_minutes=30,
                           flux_kind="native", flux_method="eddy covariance, EddyPro 7.0.9",
                           note="cups differ by mast; near-neutral phi_m is known to be low and unexplained"))


def build(site, data_root):
    if site == "mosaic_metcity":
        return load_mosaic(f"{data_root}/mosaic_metcity")
    if site == "sheba":
        return load_sheba(f"{data_root}/mirror_sheba_asfg_5level/prof_file_all6_ed_hd.txt")
    if site == "fino1":
        return load_fino1(f"{data_root}/fino1/fe-a511976f-2af7-4939-ae57-f61c93aa02e6.nc")
    if site == "falkenberg_2021":
        return falkenberg_table(*fg.load_falkenberg(f"{data_root}/falkenberg_2021/x"))
    raise ValueError(f"unknown site {site}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, help="table directory")
    ap.add_argument("--reports", required=True, help="report directory")
    ap.add_argument("sites", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for site in a.sites:
        path = f"{a.out}/{site}.nc"
        table.write_table(path, **build(site, a.data_root))
        with open(f"{a.reports}/flux_profile_{site}.json", "w") as f:
            json.dump(table.report(path), f, indent=1)
        print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
