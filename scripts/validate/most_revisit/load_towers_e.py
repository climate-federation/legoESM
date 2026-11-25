#!/usr/bin/env python
"""Flux-profile tables for Summit (ICECAPS), Netheravon 2023, BLLAST 2011 and LIAISE La Cendrosa 2021,
and boundary-layer depth files for BLLAST, LIAISE and Cardington.

Source flags only. Every level is written; the analysis chooses pairs. Side decisions were put to a
GLM+codex vote (coscientist/vote_lde_result.txt); each is named in the note attribute of its table.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules, also under python -I
import flux_profile_table as table  # noqa: E402
from flux_gradient import SECONDS_PER_AVERAGE, to_half_hour  # noqa: E402
from load_towers_d import sensible_from_sonic  # noqa: E402
from zi_sondes import bulk_ri_height, great_circle_km, theta_v  # noqa: E402

from legoesm import constants  # noqa: E402

HPA_TO_PA = 100.0

SUMMIT_LAT, SUMMIT_LON = 72.572962, -38.470361
SUMMIT_PRESSURE_PA = 67760.0  # ISA at 3267 m; voted (S6), sets density only
SUMMIT_GOOD_FLUX = (1, 2)  # good, good for research (S4)
SUMMIT_GOOD_TOWER = 1
MINUTE = 60

NETH_LAT, NETH_LON = 51.241, -1.777
NETH_PRESSURE_PA = 99890.0  # ISA at 120 m; voted (S10), sets density only
NETH_GOOD = 1
NETH_T = {"1p2m": ("rtd_1p2m_A", "rtd_1p2m_B", "prt_1p2m_C"), "16m": ("rtd_16m_A", "rtd_16m_B", "prt_16m_A")}
NETH_T_Z = {"1p2m": 1.2, "16m": 16.0}
NETH_U_Z = {"2m": 2.0, "16m": 16.0}
NETH_T_FOR_FLUX = {"2m": "1p2m", "16m": "16m"}

BLLAST_LAT, BLLAST_LON = 43.124, 0.362
# Heights from the BLLAST uniform-processing report, appendix I; 15 m is the read-me's nominal level.
BLLAST_T_Z = {"2": 2.0, "15": 15.0, "30": 29.4, "45": 45.6, "60": 61.6}
BLLAST_SONIC_Z = {"30m": 29.3, "45m": 45.8, "60m": 61.4}
BLLAST_MICRO_Z = 1.95
BLLAST_SLOW = {"cr10X": {"T2": 9, "T15": 11, "p": 3}, "cr300030L": {"T30": 5, "T45": 7}, "cr300060L": {"T60": 5}}
# Calibrations from read_me_60m_mast_how-to-process-raw-data_update20160802.txt, subtracted from raw degC.
BLLAST_T_OFFSET_K = {"T2": 0.13, "T15": 0.34, "T30": 0.28, "T45": 0.50, "T60": 0.06}
BLLAST_SAMPLES = {"cr10X": 180, "cr300030L": 1800, "cr300060L": 1800}
BLLAST_MISSING = -9999.0
BLLAST_LIDAR_SITE1_ALT_M = 595.0  # lidar file header altitude, used to convert asl to agl (S15)

LIAISE_LAT, LIAISE_LON = 41.693406, 0.928538
LIAISE_U_Z = [3.0, 10.0, 25.0, 50.0]
LIAISE_T_Z = [0.2, 2.0, 10.0, 25.0, 45.0]
LIAISE_F_Z = [3.0, 25.0, 50.0]
LIAISE_LE = {0: "lhf_1", 2: "lhf_2"}  # lhf_1 at 3 m, lhf_2 at 50 m (long_name); none at 25 m
LIAISE_CANOPY_M, LIAISE_D_M = 0.7, 0.5  # EddyPro metadata of the unified processing, La Cendrosa CNRM

CARD_LAT, CARD_LON = 52.10438, -0.42286


def _f(v):
    return np.ma.filled(np.ma.asarray(v[:]).astype(float), np.nan)


def _nc_time(var):
    """Seconds since 1970 UTC from a CF time variable."""
    t = netCDF4.num2date(var[:], var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return np.array([int(round(x.replace(tzinfo=dt.timezone.utc).timestamp())) for x in np.ravel(t)], np.int64)


def place(axis, t, x):
    """Rows of x at times t onto the sorted axis; missing rows NaN."""
    out = np.full((axis.size,) + np.shape(x)[1:], np.nan)
    i = np.searchsorted(axis, t)
    out[i] = x
    return out


def nearest_finite(z_t, temp, z_f):
    """Temperature at the level nearest z_f among levels finite at that time."""
    dist = np.where(np.isfinite(temp) & np.isfinite(z_t), np.abs(z_t - z_f[:, None]), np.inf)
    k = np.argmin(dist, axis=1)
    out = temp[np.arange(temp.shape[0]), k]
    return np.where(np.isfinite(dist[np.arange(dist.shape[0]), k]), out, np.nan)


# --- Summit ------------------------------------------------------------------------------------

def summit_ustar(tau_u, tau_v, h, w_ts):
    """u* from the source's momentum fluxes and its own density, rho = H / (cp w'Ts') (vote S5)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(w_ts != 0, h / (constants.c_pd * w_ts), np.nan)
        return np.sqrt(np.hypot(tau_u, tau_v) / np.where(rho > 0, rho, np.nan))


def vector_half_hour(t_end, speed, direction, good, samples):
    """Vector-mean speed per half hour from per-sample speed and direction."""
    rad = np.deg2rad(direction)
    keys, u = to_half_hour(t_end, -speed * np.sin(rad), good.astype(int), samples)
    _, v = to_half_hour(t_end, -speed * np.cos(rad), good.astype(int), samples)
    return keys, np.hypot(u, v)


def _summit_month(path, names):
    with netCDF4.Dataset(path) as d:
        out = {k: _f(d[v]) for k, v in names.items()}
        out["t"] = np.rint(_f(d["time"])).astype(np.int64) + MINUTE  # minute starts (vote S1, source code)
    return out


def summit_tower(root):
    w = [_summit_month(p, {"s": "wind_speed", "d": "wind_from_direction", "z": "height_above_snow_surface", "q": "qc_flag"})
         for p in sorted(glob.glob(f"{root}/surface-winds-profile/*/*.nc"))]
    tt = [_summit_month(p, {"T": "air_temperature", "z": "height_above_snow_surface", "q": "qc_flag_surface_temperature"})
          for p in sorted(glob.glob(f"{root}/surface-temperature-profile/*/*.nc"))]
    cat = lambda xs, k: np.concatenate([x[k] for x in xs])
    tw, tT = cat(w, "t"), cat(tt, "t")
    gw, gT = cat(w, "q") == SUMMIT_GOOD_TOWER, cat(tt, "q") == SUMMIT_GOOD_TOWER
    kw, wind = vector_half_hour(tw, cat(w, "s"), cat(w, "d"), gw & np.isfinite(cat(w, "z")), 30)
    _, z_u = to_half_hour(tw, cat(w, "z"), (gw & np.isfinite(cat(w, "s"))).astype(int), 30)
    kT, temp = to_half_hour(tT, cat(tt, "T"), (gT & np.isfinite(cat(tt, "z"))).astype(int), 30)
    _, z_t = to_half_hour(tT, cat(tt, "z"), (gT & np.isfinite(cat(tt, "T"))).astype(int), 30)
    return kw, wind, z_u, kT, temp, z_t


def summit_flux_files(root):
    """flux-1 daily 30-min files, v2 where it exists (vote S3)."""
    best = {}
    for p in sorted(glob.glob(f"{root}/flux-estimates/ace-flux-1_30min/*/*/*.nc")):
        day, ver = re.search(r"_(\d{8})_flux-estimates_30min_v(\d+)\.nc$", p).groups()
        if day not in best or int(ver) > best[day][0]:
            best[day] = (int(ver), p)
    return [best[k][1] for k in sorted(best)]


def summit_flux(root):
    rows = []
    for p in summit_flux_files(root):
        with netCDF4.Dataset(p) as d:
            g = {k: np.isin(_f(d[f"qc_flag_quality_{k}"]), SUMMIT_GOOD_FLUX) for k in ("wu", "wv", "wts")}
            h, wts = _f(d["upward_sensible_heat_flux_in_air"]), _f(d["kinematic_sonic_temperature_flux"])
            ust = summit_ustar(_f(d["momentum_flux_u"]), _f(d["momentum_flux_v"]), h, wts)
            le = _f(d["upward_latent_heat_flux_in_air"]) if "upward_latent_heat_flux_in_air" in d.variables else np.full(h.size, np.nan)
            rows.append(dict(t=np.rint(_f(d["time"])).astype(np.int64) + SECONDS_PER_AVERAGE,  # stamp = run start
                             ustar=np.where(g["wu"] & g["wv"], ust, np.nan), h=np.where(g["wts"], h, np.nan),
                             le=le, z=_f(d["height_above_surface"])))
    x = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    if np.any(x["t"] % SECONDS_PER_AVERAGE):
        raise ValueError("Summit flux run not on the half hour")
    t, first = np.unique(x["t"], return_index=True)
    return t, {k: v[first] for k, v in x.items() if k != "t"}


def load_summit(root):
    kw, wind, z_u, kT, temp, z_t = summit_tower(root)
    tf, f = summit_flux(root)
    axis = np.union1d(np.union1d(kw, kT), tf)
    z_u, wind = place(axis, kw, z_u), place(axis, kw, wind)
    z_t, temp = place(axis, kT, z_t), place(axis, kT, temp)
    z_f = place(axis, tf, f["z"])
    t_flux = nearest_finite(z_t, temp, z_f)
    n = axis.size
    return dict(time_end=axis, z_u=z_u, wind=wind, z_t=z_t, temp=temp, z_f=z_f[:, None],
                ustar=place(axis, tf, f["ustar"])[:, None], h=place(axis, tf, f["h"])[:, None],
                le=place(axis, tf, f["le"])[:, None], pressure=np.full(n, SUMMIT_PRESSURE_PA), t_flux=t_flux[:, None],
                attrs=dict(site="summit", latitude=SUMMIT_LAT, longitude=SUMMIT_LON, surface="Greenland ice sheet snow",
                           canopy_height_m=0.0, displacement_m=0.0,
                           source="ICECAPS-ACE Summit, CEDA c725c6ea1cd84c8792b677aa581bcb5d (winds), surface-temperature-profile, "
                                  "bbd0a00c3d7f42f7bbd7ca69a8a9f4e6 (flux-estimates)",
                           licence="OGL v3", averaging_minutes=30, flux_kind="native",
                           flux_method="eddy covariance, Metek uSonic-3 (ace-flux-1, lowest sonic) with LI-7500",
                           note=("1-min tower stamps are minute starts (source code resample('1min'), vote S1), vector-averaged to 30 min "
                                 "with all 30 minutes present; heights above snow per time from the tower files, flux height from "
                                 "flux-1's own field; flux-1 only (vote S2: flux-2 height field wrong before 2021); v2 files over v1; "
                                 "flux QC quality_wu/wv/wts 1 or 2 (vote S4), skew and kurtosis flags not applied; "
                                 "u* = (|tau|/rho_src)^0.5 with rho_src = H/(cp w'Ts') from the same file (vote S5); H is the source's "
                                 "rho cp w'Ts' (sonic, no humidity correction), source values span only -150..150 W m-2; LE as given, no flag; pressure constant ISA "
                                 "(vote S6); t_flux is the finite tower temperature nearest the flux height at each time")))


# --- Netheravon --------------------------------------------------------------------------------

def a_else_b(a, a_good, b, b_good):
    """Sonic A where its flag is good, else sonic B where good (vote S7)."""
    return np.where(a_good, a, np.where(b_good, b, np.nan))


def mean_of_good(values, goods):
    """Mean over the sensors whose flag is good; NaN when none (vote S8)."""
    x = np.where(np.array(goods), np.array(values), np.nan)
    with np.errstate(invalid="ignore"):
        n = np.isfinite(x).sum(axis=0)
        return np.where(n > 0, np.nansum(x, axis=0) / np.maximum(n, 1), np.nan)


def _neth_day(path):
    with netCDF4.Dataset(path) as d:
        def get(name):
            if name not in d.variables:
                return None, None
            q = d.variables.get("qc_flag_" + name)
            return _f(d[name]), (_f(q) == NETH_GOOD) if q is not None else np.zeros(d.dimensions["time"].size, bool)
        n = d.dimensions["time"].size
        nanv, false = np.full(n, np.nan), np.zeros(n, bool)
        pair = lambda base: [x if x[0] is not None else (nanv, false) for x in (get(base + "_A"), get(base + "_B"))]
        out = {"t": np.rint(_f(d["time_bnds"])[:, 1] * 3600).astype(np.int64)}
        for h in NETH_U_Z:
            (ua, ga), (ub, gb) = pair(f"wind_speed_vector_{h}")
            out["u" + h] = a_else_b(ua, ga, ub, gb)
            (xa, xga), (xb, xgb) = pair(f"uw_covariance_{h}")
            (ya, yga), (yb, ygb) = pair(f"vw_covariance_{h}")
            ok_a, ok_b = xga & yga, xgb & ygb
            out["ust" + h] = a_else_b(np.hypot(xa, ya) ** 0.5, ok_a, np.hypot(xb, yb) ** 0.5, ok_b)
            (wa, wga), (wb, wgb) = pair(f"wt_covariance_{h}")
            out["wt" + h] = a_else_b(wa, wga, wb, wgb)
            q, qg = get(f"wq_covariance_{h}_A")
            out["wq" + h] = np.where(qg, q, np.nan) if q is not None else nanv
        for lev, names in NETH_T.items():
            got = [get("air_temperature_" + s) for s in names]
            got = [g for g in got if g[0] is not None]
            out["T" + lev] = mean_of_good([g[0] for g in got], [g[1] for g in got])
    return out


def load_netheravon(root):
    days = [_neth_day(p) for p in sorted(glob.glob(f"{root}/2023/*/*_30min.nc"))]
    x = {k: np.concatenate([d[k] for d in days]) for k in days[0]}
    t, first = np.unique(x["t"], return_index=True)
    x = {k: v[first] for k, v in x.items()}
    n = t.size
    temp = np.column_stack([x["T" + k] for k in NETH_T]) + constants.T_freeze
    levels = list(NETH_U_Z)
    t_flux = np.column_stack([x["T" + NETH_T_FOR_FLUX[h]] for h in levels]) + constants.T_freeze
    p = np.full(n, NETH_PRESSURE_PA)
    both = [sensible_from_sonic(x["wt" + h], x["wq" + h], t_flux[:, i], p) for i, h in enumerate(levels)]
    return dict(time_end=t, z_u=np.tile(list(NETH_U_Z.values()), (n, 1)), wind=np.column_stack([x["u" + h] for h in levels]),
                z_t=np.tile(list(NETH_T_Z.values()), (n, 1)), temp=temp, z_f=np.tile(list(NETH_U_Z.values()), (n, 1)),
                ustar=np.column_stack([x["ust" + h] for h in levels]), h=np.column_stack([b[0] for b in both]),
                le=np.column_stack([b[1] for b in both]), pressure=p, t_flux=t_flux,
                attrs=dict(site="netheravon", latitude=NETH_LAT, longitude=NETH_LON, surface="grass airfield (WesCon 2023)",
                           canopy_height_m=0.0, displacement_m=0.0,
                           source="Met Office WesCon Netheravon flux tower, CEDA, 30-min files",
                           licence="Non-Commercial Government Licence v2 (file attribute)", averaging_minutes=30, flux_kind="native",
                           flux_method="eddy covariance, two sonics per height, LI-7500 on sonic A",
                           note=("source flag 1 only (vote S9); per quantity sonic A where flag good else sonic B (vote S7); "
                                 "temperature is the mean of the flag-good sensors at each height (vote S8); u* = (uw^2+vw^2)^0.25 "
                                 "of one sonic; H from w'Ts' with the Schotanus term using sonic A's w'q' at that height, NaN without it; "
                                 "pressure constant ISA (vote S10); wind is vector mean")))


# --- BLLAST ------------------------------------------------------------------------------------

def bllast_clock(date, hhmm_s):
    """Seconds since 1970 from logger date (yyyymmdd) and a time string HHMM followed by seconds ("00592.125", "140019")."""
    s = pd.Series(hhmm_s).astype(str).str.strip()
    sec = (s.str[:2].astype(int) * 3600 + s.str[2:4].astype(int) * 60 + s.str[4:].astype(float)).to_numpy()
    day = pd.to_datetime(pd.Series(date).astype(int).astype(str), format="%Y%m%d").to_numpy().astype("datetime64[s]").astype(np.int64)
    return day + np.round(sec).astype(np.int64)


def drop_repeated_stamps(t):
    """Mask of rows whose stamp occurs once; a logger clock step back makes both copies ambiguous."""
    _, inv, cnt = np.unique(t, return_inverse=True, return_counts=True)
    return cnt[inv] == 1


def _bllast_slow(root, logger, cols, samples):
    frames = []
    for p in sorted(glob.glob(f"{root}/raw/TXT/*/{logger}2011*.txt.res")):
        frames.append(pd.read_csv(p, header=None, usecols=[0, 1] + sorted(cols.values()), dtype=str))
    x = pd.concat(frames, ignore_index=True)
    # A few lines are cut short by the logger: date not 8 digits or time shorter than HHMMS; they are dropped.
    x = x[x[0].str.fullmatch(r"2011\d{4}", na=False) & x[1].str.strip().str.fullmatch(r"\d{5,6}(\.\d+)?", na=False)]
    for c in [0] + sorted(cols.values()):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    t = bllast_clock(x[0].to_numpy(), x[1].to_numpy())
    keep = drop_repeated_stamps(t)
    out = {}
    for name, c in cols.items():
        v = x[c].to_numpy(float)
        good = keep & np.isfinite(v) & (v != BLLAST_MISSING)
        keys, out[name] = to_half_hour(t[keep], v[keep], good[keep].astype(int), samples)
    return keys, out


def bllast_calibrated(name, raw_c):
    """Raw logger temperature (degC) with the source's documented calibration offset removed."""
    return raw_c - BLLAST_T_OFFSET_K[name]


def _ecpack(path):
    """EC-pack 30-min file: end time and the columns used."""
    with netCDF4.Dataset(path) as d:
        g = lambda k: np.where(_f(d[k]) == BLLAST_MISSING, np.nan, _f(d[k]))
        day = g("DOY_end")
        t = (np.datetime64("2011-01-01", "s").astype(np.int64) + np.rint((day - 1) * 86400 + g("hour_end") * 3600
                                                                         + g("min_end") * 60)).astype(np.int64)
        return t, {k: g(k) for k in ("Mean(u)", "u_star", "H_sonic", "LvE_sum")}


def load_bllast(tower_root, micro_path):
    temps, p = {}, None
    for logger, cols in BLLAST_SLOW.items():
        keys, out = _bllast_slow(tower_root, logger, cols, BLLAST_SAMPLES[logger])
        for k, v in out.items():
            temps[k] = (keys, v)
    sonic = {h: _ecpack(f"{tower_root}/processed/uniform_processing/{h}/Flux_Files/NetCDF/BLLAST2011_SuperSite1_Valimev{h}_flux30mins.nc")
             for h in BLLAST_SONIC_Z}
    tm, micro = _ecpack(micro_path)
    axis = np.unique(np.concatenate([k for k, _ in temps.values()] + [s[0] for s in sonic.values()] + [tm]))
    n = axis.size
    temp = np.column_stack([bllast_calibrated("T" + k, place(axis, *temps["T" + k])) for k in BLLAST_T_Z]) + constants.T_freeze
    pressure = place(axis, *temps["p"]) * HPA_TO_PA
    wind = np.column_stack([place(axis, sonic[h][0], sonic[h][1]["Mean(u)"]) for h in BLLAST_SONIC_Z])
    flux = [(tm, micro)] + [sonic[h] for h in BLLAST_SONIC_Z]
    z_f = np.tile([BLLAST_MICRO_Z] + list(BLLAST_SONIC_Z.values()), (n, 1))
    z_t = np.tile(list(BLLAST_T_Z.values()), (n, 1))
    col = lambda k: np.column_stack([place(axis, t, f[k]) for t, f in flux])
    t_flux = np.column_stack([nearest_finite(z_t, temp, z_f[:, i]) for i in range(z_f.shape[1])])
    return dict(time_end=axis, z_u=np.tile(list(BLLAST_SONIC_Z.values()), (n, 1)), wind=wind, z_t=z_t, temp=temp, z_f=z_f,
                ustar=col("u_star"), h=col("H_sonic"), le=col("LvE_sum"), pressure=pressure, t_flux=t_flux,
                attrs=dict(site="bllast", latitude=BLLAST_LAT, longitude=BLLAST_LON,
                           surface="Lannemezan plateau, mixed fields, meadow and forest patches; flux at the MicroSite (bare soil, grass, small bushes)",
                           canopy_height_m=np.nan, displacement_m=np.nan,
                           source="BLLAST 2011 60 m Valimev tower (AERIS) slow raw files and uniform EC processing; MicroSite EC",
                           licence="BLLAST data policy (AERIS)", averaging_minutes=30, flux_kind="native",
                           flux_method="eddy covariance, EC-pack uniform processing; lowest flux MicroSite CSAT3 1.95 m; tower sonics 29.3/45.8/61.4 m",
                           note=("wind is the sonic rotated mean (vector) at 29.3/45.8/61.4 m only, cups left out (vote S11); "
                                 "temperatures 2/15/30/45/60 m HMP and logger probes, heights from the uniform-processing report "
                                 "(15 m nominal), calibration offsets of the 2016 raw-data read-me subtracted; raw logger clock taken as UTC (vote S12, checked by the diurnal H peak in the report); "
                                 "rows with a repeated logger stamp dropped, so that half hour is incomplete and NaN; every sample "
                                 "of a half hour required; no QC flags in the EC files, all kept (vote S13); canopy NaN (vote S14); "
                                 "H is EC-pack H_sonic, LE is LvE_sum; whether the uniform processing applied the read-me's Gill 45 m "
                                 "w correction (x1.166 up, x1.289 down) is not stated; tower fluxes carry unscreened spikes "
                                 "(H to 2e10 W m-2), left to the analysis limits; t_flux nearest finite temperature level")))


# --- LIAISE ------------------------------------------------------------------------------------

def load_liaise(root):
    rows = []
    for p in sorted(glob.glob(f"{root}/V3/*_V3.nc")):
        with netCDF4.Dataset(p) as d:
            g = lambda k: _f(d[k])
            n = d.dimensions["time"].size
            rows.append(dict(t=_nc_time(d["time"]),
                             wind=np.column_stack([g(f"ws_{i}") for i in range(1, 5)]),
                             temp=np.column_stack([g(f"ta_{i}") for i in range(1, 6)]),
                             ustar=np.column_stack([g(f"u_star_{i}") for i in range(1, 4)]),
                             h=np.column_stack([g(f"shf_{i}") for i in range(1, 4)]),
                             le=np.column_stack([g(LIAISE_LE[i]) if i in LIAISE_LE else np.full(n, np.nan) for i in range(3)]),
                             p=g("pa")))
    x = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    t, first = np.unique(x["t"], return_index=True)
    x = {k: v[first] for k, v in x.items()}
    n = t.size
    temp = x["temp"] + constants.T_freeze
    z_t, z_f = np.tile(LIAISE_T_Z, (n, 1)), np.tile(LIAISE_F_Z, (n, 1))
    t_flux = np.column_stack([nearest_finite(z_t, temp, z_f[:, i]) for i in range(z_f.shape[1])])
    return dict(time_end=t, z_u=np.tile(LIAISE_U_Z, (n, 1)), wind=x["wind"], z_t=z_t, temp=temp, z_f=z_f,
                ustar=x["ustar"], h=x["h"], le=x["le"], pressure=x["p"] * HPA_TO_PA, t_flux=t_flux,
                attrs=dict(site="liaise_cendrosa", latitude=LIAISE_LAT, longitude=LIAISE_LON,
                           surface="irrigated alfalfa inside dry farmland (LIAISE, Ebro basin)",
                           canopy_height_m=LIAISE_CANOPY_M, displacement_m=LIAISE_D_M,
                           source="LIAISE La Cendrosa CNRM 30-min L2 V3 (AERIS doi 10.25326/320)", licence="LIAISE data policy (AERIS)",
                           averaging_minutes=30, flux_kind="native", flux_method="eddy covariance, CNRM, 3/25/50 m",
                           note=("no QC flags in L2, all times kept (vote S16); canopy 0.7 m and d 0.5 m from the unified-processing "
                                 "EddyPro metadata; the 0.2 m temperature is inside the canopy and is left for the analysis to exclude; "
                                 "irrigated field in dry surroundings: expect local advection, not filtered; LE at 3 and 50 m only; "
                                 "wind is the source's magnitude of wind velocity (scalar or vector not stated); near neutral the 10 m wind reads "
                                 "about 8 % below a log law through 3, 25 and 50 m (n=176), cause unknown; t_flux nearest finite level")))


# --- boundary-layer depth ----------------------------------------------------------------------

def write_obs(path, t, zi, attrs):
    """zi file for profiler or lidar estimates: time (UTC, s) and zi (m above ground)."""
    order = np.argsort(t, kind="stable")
    with netCDF4.Dataset(path, "w") as o:
        o.createDimension("obs", len(t))
        for k, typ, vals, units in (("time", "i8", np.asarray(t, np.int64)[order], "seconds since 1970-01-01 00:00:00 UTC"),
                                    ("zi", "f4", np.asarray(zi, float)[order], "m above ground")):
            v = o.createVariable(k, typ, ("obs",))
            v[:], v.units = vals, units
        for k, v in attrs.items():
            setattr(o, k, v)


def write_sondes(path, launch, zi, dist, moist, attrs):
    """Same schema as the *_sondes.nc files written by zi_sondes.py."""
    with netCDF4.Dataset(path, "w") as o:
        o.createDimension("sonde", len(launch))
        for k, typ, vals, units in (("launch_time", "i8", np.asarray(launch, np.int64), "seconds since 1970-01-01 00:00:00 UTC"),
                                    ("zi", "f4", zi, "m above ground"), ("distance_km", "f4", dist, "km from tower at launch"),
                                    ("theta_v_used", "i1", np.asarray(moist, int), "1 if humidity entered theta_v")):
            v = o.createVariable(k, typ, ("sonde",))
            v[:], v.units = vals, units
        for k, v in attrs.items():
            setattr(o, k, v)


def uhf_zi(text):
    """BLLAST UHF Zi file: day line then HH MM SS Zi(m agl) rows."""
    day = re.search(r"Day:\s*(\d{2}-\w{3}-\d{4})", text).group(1)
    base = int(dt.datetime.strptime(day, "%d-%b-%Y").replace(tzinfo=dt.timezone.utc).timestamp())
    rows = re.findall(r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+([-\d.]+)\s*$", text, re.M)
    return [(base + int(h) * 3600 + int(m) * 60 + int(s), float(z)) for h, m, s, z in rows]


def lidar_zi(text, ground_alt_m):
    rows = re.findall(r"^(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})\s+([-\d.]+)\s*$", text, re.M)
    return [(int(dt.datetime.strptime(f"{a} {b}", "%d/%m/%Y %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()),
             float(z) - ground_alt_m) for a, b, z in rows]


def bllast_zi(root, out_dir):
    uhf = [r for p in sorted(glob.glob(f"{root}/*/zi/*_UHF_Zi.txt")) for r in uhf_zi(open(p, errors="replace").read())]
    write_obs(f"{out_dir}/bllast_uhf.nc", [r[0] for r in uhf], [r[1] for r in uhf],
              dict(site="bllast", source="BLLAST UHF profiler Site 1 (LA/CRA), Zi v1.0", station="Site 1, same supersite as the 60 m tower",
                   method="local maxima of Cn2; mixed or residual layer top, daytime (source)"))
    lid = [r for p in sorted(glob.glob(f"{root}/*/Lidar_LPCA_*.txt")) for r in lidar_zi(open(p, errors="replace").read(), BLLAST_LIDAR_SITE1_ALT_M)]
    write_obs(f"{out_dir}/bllast_lidar.nc", [r[0] for r in lid], [r[1] for r in lid],
              dict(site="bllast", source="BLLAST LPCA ALS300 aerosol lidar, Super site 1", station="Super site 1",
                   method=f"wavelet covariance transform (source), mixed or residual layer top; asl minus header altitude {BLLAST_LIDAR_SITE1_ALT_M} m"))
    return len(uhf), len(lid)


def sonde_profile_zi(z_agl, t_k, p_pa, r, u, v):
    """Seidel 2012 bulk-Ri zi via zi_sondes; humidity used where present."""
    thv = theta_v(t_k, p_pa, r)
    return bulk_ri_height(np.asarray(z_agl, float), np.asarray(thv, float), np.asarray(u, float), np.asarray(v, float)), bool(np.isfinite(r).any())


def cardington_sondes(root, out_dir):
    launch, zi, dist, moist = [], [], [], []
    for p in sorted(glob.glob(f"{root}/*/*.nc")):
        with netCDF4.Dataset(p) as d:
            t = _f(d["time"]) * 3600.0
            z = _f(d["height"])
            ok = np.isfinite(z)
            if not ok.any():
                continue
            z0 = z[ok][0]
            r = _f(d["mixr"]) * 1e-3
            h, m = sonde_profile_zi(z - z0, _f(d["air_temperature"]), _f(d["air_pressure"]) * HPA_TO_PA, r,
                                    _f(d["wind_east"]), _f(d["wind_north"]))
            lat, lon = _f(d["latitude"]), _f(d["longitude"])
            launch.append(int(round(t[ok][0])))
            zi.append(h)
            moist.append(m)
            dist.append(float(great_circle_km(CARD_LAT, CARD_LON, lat[ok][0], lon[ok][0])))
    order = np.argsort(launch, kind="stable")
    take = lambda a: np.asarray(a)[order]
    write_sondes(f"{out_dir}/cardington_sondes.nc", take(launch), take(zi), take(dist), take(moist),
                 dict(site="cardington", source="metoffice cardington radiosondes (CEDA)", station="Cardington",
                      method="Seidel et al. 2012 bulk Ri 0.25, own code (zi_sondes.bulk_ri_height); height above launch level"))
    return int(np.isfinite(zi).sum()), len(zi)


def liaise_sonde(text):
    """UKMO LIAISE sonde text: (launch s, station lat, lon, columns dict)."""
    rel = re.search(r"Balloon release time\s+(\S+ \S+)", text).group(1)
    launch = int(dt.datetime.strptime(rel, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
    lat = float(re.search(r"Station latitude\s+([\d.]+)\s*N", text).group(1))
    lon = float(re.search(r"Station longitude\s+([\d.]+)\s*E", text).group(1))
    lines = text.splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.split()[:2] == ["Time", "Height"])
    names = lines[head].split()
    data = np.array([[float(v) for v in ln.split()] for ln in lines[head + 2:] if ln.strip() and len(ln.split()) == len(names)])
    return launch, lat, lon, {k: data[:, i] for i, k in enumerate(names)}


def liaise_sondes(root, out_dir):
    launch, zi, dist, moist = [], [], [], []
    for p in sorted(glob.glob(f"{root}/*.txt")):
        t, lat, lon, c = liaise_sonde(open(p, errors="replace").read())
        rad = np.deg2rad(c["Dir"])
        h, m = sonde_profile_zi(c["Height"] - c["Height"][0], c["Temp"] + constants.T_freeze, c["Pres"] * HPA_TO_PA,
                                c["MixR"] * 1e-3, -c["Speed"] * np.sin(rad), -c["Speed"] * np.cos(rad))
        launch.append(t)
        zi.append(h)
        moist.append(m)
        dist.append(float(great_circle_km(LIAISE_LAT, LIAISE_LON, lat, lon)))
    order = np.argsort(launch, kind="stable")
    take = lambda a: np.asarray(a)[order]
    write_sondes(f"{out_dir}/liaise_cendrosa_sondes.nc", take(launch), take(zi), take(dist), take(moist),
                 dict(site="liaise_cendrosa", source="LIAISE Els Plans UKMO radiosondes (AERIS)", station="Els Plans (dry site)",
                      method="Seidel et al. 2012 bulk Ri 0.25, own code (zi_sondes.bulk_ri_height); height above launch level"))
    return int(np.isfinite(zi).sum()), len(zi)


def write(name, data, tables, reports):
    path = f"{tables}/{name}.nc"
    table.write_table(path, **data)
    with open(f"{reports}/{name}.json", "w") as fh:
        json.dump(table.report(path), fh, indent=1)
    print("WROTE", path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites", nargs="+", choices=("summit", "netheravon", "bllast", "liaise", "zi"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--zi-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.reports, exist_ok=True)
    r = a.data_root
    for site in a.sites:
        if site == "summit":
            write("summit", load_summit(f"{r}/summit_icecaps"), a.tables, a.reports)
        elif site == "netheravon":
            write("netheravon", load_netheravon(f"{r}/netheravon_woest"), a.tables, a.reports)
        elif site == "bllast":
            write("bllast", load_bllast(f"{r}/bllast_tower",
                                        f"{r}/bllast_microsite/uniformely_processed_data/Flux_Files/NetCDF/BLLAST2011_SuperSite1_MicroSite_flux30mins.nc"),
                  a.tables, a.reports)
        elif site == "liaise":
            write("liaise_cendrosa", load_liaise(f"{r}/liaise_lacendrosa_cnrm"), a.tables, a.reports)
        else:
            print("bllast uhf, lidar", bllast_zi(f"{r}/bllast_zi", a.zi_dir), flush=True)
            print("cardington sondes valid/total", cardington_sondes(f"{r}/cardington_sonde", a.zi_dir), flush=True)
            print("liaise sondes valid/total", liaise_sondes(f"{r}/liaise_elsplans_sondes", a.zi_dir), flush=True)


if __name__ == "__main__":
    main()
