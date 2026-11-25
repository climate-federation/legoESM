#!/usr/bin/env python
"""Flux-profile tables for CHATS, SMEAR II Hyytiala, i-Box Kolsass and METCRAX II.

Choices voted by GLM and codex are named at the point of use. Every level is written;
canopy_height_m tells the analysis which pairs lie above the canopy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import netCDF4
import numpy as np
import pandas as pd

import flux_profile_table as table
from flux_gradient import to_half_hour
from legoesm import constants

HALF_HOUR = 1800

# CHATS README table: 1-based CSV column of the first level, and the measured heights.
CHATS_TEMP_Z = [1.55, 3.11, 4.54, 6.07, 7.65, 9.04, 10.04, 11.12, 13.57, 17.52, 22.55, 28.51]
CHATS_SONIC_Z = [1.54, 3.11, 4.54, 6.07, 7.62, 9.02, 10.03, 11.13, 12.54, 14.09, 18.04, 23.07, 29.03]
CHATS_LE_LEVELS = [0, 2, 4, 6, 9, 11]  # sonics that carry a krypton hygrometer
CHATS_COL = dict(pressure=16, temp=17, wind=53, ustar=79, h=92, le=105)
CHATS_MISSING = 1e35

SMEAR_T = ["T42", "T84", "T168", "T336", "T504", "T672"]
SMEAR_T_Z = [4.2, 8.4, 16.8, 33.6, 50.4, 67.2]
SMEAR_U_Z = [4.2, 8.4, 16.8, 33.6, 50.4, 67.2, 125.0]
SMEAR_CUP = ["WS42", "WS84", "WS168", "WS336", "WS504", "WS672", None]
SMEAR_SONIC = [None, "WSU84", "WSU168", "WSU336", None, "WSU672", "WSU1250"]
SMEAR_CLOCK_AHEAD_OF_UTC_S = 7200
SMEAR_BAD_QC = 2

IBOX_LEVELS = {1: 4.0, 2: 8.7}
IBOX_MINUTES_PER_BLOCK = 30

METCRAX_PRESSURE_PA = 82500.0  # standard atmosphere at about 1700 m; voted, sets density only


def _grid(z, n):
    return np.tile(np.asarray(z, dtype=float), (n, 1))


def _seconds(frame):
    stamp = pd.to_datetime(frame.rename(columns=str.lower)[["year", "month", "day", "hour", "minute", "second"]])
    return (stamp - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy().astype(np.int64)


def load_chats(root):
    paths = sorted(glob.glob(f"{root}/chats_30min_data_*.csv"))
    x = np.vstack([np.loadtxt(p, delimiter=",") for p in paths])
    x[x >= CHATS_MISSING] = np.nan
    col = lambda name, n: x[:, CHATS_COL[name] - 1:CHATS_COL[name] - 1 + n]
    n, nu, nt = x.shape[0], len(CHATS_SONIC_Z), len(CHATS_TEMP_Z)
    stamp = pd.DataFrame(x[:, :6].astype(int), columns=["year", "month", "day", "hour", "minute", "second"])
    le = np.full((n, nu), np.nan)
    le[:, CHATS_LE_LEVELS] = col("le", len(CHATS_LE_LEVELS))
    temp = col("temp", nt)
    nearest = [int(np.argmin(np.abs(np.array(CHATS_TEMP_Z) - z))) for z in CHATS_SONIC_Z]
    attrs = dict(site="chats", latitude=38.488, longitude=-121.846, surface="walnut orchard", canopy_height_m=10.0,
                 displacement_m=np.nan, source="doi:10.5281/zenodo.17831843 (Burns and Bonan)", licence="CC-BY-4.0",
                 averaging_minutes=30, flux_kind="native",
                 flux_method="eddy covariance, 30-min covariance rebuilt from 5-min statistics",
                 note=("canopy 10 m: CHATS dataset description; April-May 2007 only; source fills gaps of up to 4 T/RH "
                       "levels by vertical interpolation; north winds footprint leaves the orchard, not filtered"))
    return dict(time_end=_seconds(stamp) + HALF_HOUR // 2, z_u=_grid(CHATS_SONIC_Z, n), wind=col("wind", nu),
                z_t=_grid(CHATS_TEMP_Z, n), temp=temp, z_f=_grid(CHATS_SONIC_Z, n), ustar=col("ustar", nu),
                h=col("h", nu), le=le, pressure=col("pressure", 1)[:, 0], t_flux=temp[:, nearest], attrs=attrs)


def load_smear(root):
    frame = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{root}/hyy_CHECKED_*.csv"))], ignore_index=True)
    frame.columns = [c.split(".")[-1] for c in frame.columns]
    frame = frame.drop_duplicates(["Year", "Month", "Day", "Hour", "Minute"])
    n = len(frame)
    get = lambda names: np.column_stack([frame[k].to_numpy(float) if k else np.full(n, np.nan) for k in names])
    good = lambda x, qc: np.where(frame[qc].to_numpy(float) == SMEAR_BAD_QC, np.nan, frame[x].to_numpy(float))
    # Stamps open the interval and run on UTC+2 (SmartSMEAR About page; raw minutes and the H phase agree).
    time_end = _seconds(frame) + HALF_HOUR - SMEAR_CLOCK_AHEAD_OF_UTC_S
    cup, sonic = get(SMEAR_CUP), get(SMEAR_SONIC)
    # One instrument family per time, so no pair mixes cups with sonics.
    wind = np.where(np.isfinite(sonic).any(axis=1, keepdims=True), sonic, cup)
    temp = get(SMEAR_T) + constants.T_freeze
    h = np.column_stack([good("H", "Qc_H"), good("H_radtow", "Qc_H_radtow"), frame["H_460"].to_numpy(float)])
    le = np.column_stack([good("LE", "Qc_LE"), good("LE_radtow", "Qc_LE_radtow"), frame["LE_460"].to_numpy(float)])
    ustar = np.column_stack([good("u_star", "Qc_tau"), good("u_star_radtow", "Qc_tau_radtow"),
                             frame["u_star_460"].to_numpy(float)])
    t_flux = np.column_stack([frame["av_t"].to_numpy(float), frame["av_t_radtow"].to_numpy(float),
                              frame["T504"].to_numpy(float)]) + constants.T_freeze
    attrs = dict(site="smear_hyytiala", latitude=61.847, longitude=24.295, surface="Scots pine forest, sown 1962",
                 canopy_height_m=19.0, displacement_m=np.nan,
                 source="SmartSMEAR, quality=CHECKED, 30-min ARITHMETIC aggregation", licence="CC-BY-4.0",
                 averaging_minutes=30, flux_kind="native", flux_method="eddy covariance",
                 note=("canopy voted: one conservative 19 m (about 19 m recent, arXiv 2010.04194; 16.3 m in 2006, "
                       "Boreal Env. Res.), so the 16.8 m level never counts as above canopy; flux levels: mast 23.3 m, "
                       "radiation tower 23 m, mast 46 m (1998-2000); Qc 2 blanked, Qc absent in most years; old T "
                       "profile only, newer Tmm sensors are inconsistent with it per the data owner; cups to 2003 then "
                       "sonics; server aggregation completeness unknown; H_460 has no quality flag; t_flux at 46 m "
                       "is T at 50.4 m"))
    return dict(time_end=time_end, z_u=_grid(SMEAR_U_Z, n), wind=wind, z_t=_grid(SMEAR_T_Z, n), temp=temp,
                z_f=_grid([23.3, 23.0, 46.0], n), ustar=ustar, h=h, le=le,
                pressure=frame["Pamb0"].to_numpy(float) * 100.0, t_flux=t_flux, attrs=attrs)


def _time_seconds(var):
    base = np.datetime64(var.units.split("since ")[1].strip().replace(" ", "T"), "s").astype(np.int64)
    return base + np.round(np.asarray(var[:], dtype=float) * 60).astype(np.int64)


def _values(d, name):
    return np.ma.filled(np.ma.asarray(d[name][:], dtype=float), np.nan)


def _ibox_profiles(root):
    t, temp, u, v, p = [], [], [], [], []
    for path in sorted(glob.glob(f"{root}/low_frequency_data/*.nc")):
        with netCDF4.Dataset(path) as d:
            z_t, z_u = _values(d, "z_TRH"), _values(d, "z_wind")
            speed = _values(d, "wind_speed_tower").T
            speed = np.where(_values(d, "wind_diag_tower").T == 0, speed, np.nan)
            turn = np.deg2rad(_values(d, "wind_dir_tower").T)
            t.append(_time_seconds(d["time"]))
            temp.append(_values(d, "T_tower").T)
            u.append(-speed * np.sin(turn))
            v.append(-speed * np.cos(turn))
            p.append(_values(d, "P"))
    t = np.concatenate(t)
    block = lambda x: to_half_hour(t, x, np.ones_like(x), samples=IBOX_MINUTES_PER_BLOCK)
    t30, temp30 = block(np.concatenate(temp))
    _, u30 = block(np.concatenate(u))
    _, v30 = block(np.concatenate(v))
    _, p30 = block(np.concatenate(p))
    return t30, z_u, table.vector_speed(u30, v30), z_t, temp30, p30


def _ibox_flux(root, level, min_flag):
    out = {k: [] for k in ("t", "ustar", "h", "le", "t_flux")}
    for path in sorted(glob.glob(f"{root}/EC_flux_lvl{level}/*.nc")):
        with netCDF4.Dataset(path) as d:
            out["t"].append(_time_seconds(d["time"]) + HALF_HOUR // 2)
            ok = lambda flag: _values(d, flag) >= min_flag
            out["ustar"].append(np.where(ok(f"qcflag_uw{level}"), _values(d, f"ustr{level}"), np.nan))
            out["h"].append(np.where(ok(f"qcflag_sh{level}"), _values(d, f"h{level}cc"), np.nan))
            le = f"le{level}cw"
            out["le"].append(np.where(ok(f"qcflag_le{level}"), _values(d, le), np.nan) if le in d.variables
                             else np.full(d["time"].size, np.nan))
            out["t_flux"].append(_values(d, f"tair{level}") + constants.T_freeze)
    return {k: np.concatenate(v) for k, v in out.items()}


def load_ibox(root, min_flag):
    t30, z_u, wind30, z_t, temp30, p30 = _ibox_profiles(root)
    flux = {k: _ibox_flux(root, k, min_flag) for k in IBOX_LEVELS}
    time_end = np.union1d(t30, np.concatenate([f["t"] for f in flux.values()]))
    n = time_end.size

    def place(t, x):
        out = np.full((n,) + x.shape[1:], np.nan)
        out[np.searchsorted(time_end, t)] = x
        return out

    stack = lambda key: np.column_stack([place(f["t"], f[key]) for f in flux.values()])
    attrs = dict(site="ibox_kolsass", latitude=47.305341, longitude=11.62219, surface="mixed agricultural land, valley floor",
                 canopy_height_m=np.nan, displacement_m=np.nan, source="doi:10.5281/zenodo.10522708", licence="CC-BY-4.0",
                 averaging_minutes=30, flux_kind="native",
                 flux_method="eddy covariance, double rotation, recursive filter",
                 note=(f"kept qcflag>={min_flag}; canopy height not in the sources, crops change through summer; metadata "
                       "gives a 30 min perturbation scale but also a recursive filter with 200 s time constant, which "
                       "removes low-frequency flux; 16.9 m flux level not in the deposit; 1-min stamps assumed end of "
                       "minute; 8 June to 18 July 2022, steep valley"))
    return dict(time_end=time_end, z_u=_grid(z_u, n), wind=place(t30, wind30), z_t=_grid(z_t, n),
                temp=place(t30, temp30) + constants.T_freeze, z_f=_grid(list(IBOX_LEVELS.values()), n),
                ustar=stack("ustar"), h=stack("h"), le=stack("le"), pressure=place(t30, p30) * 100.0,
                t_flux=stack("t_flux"), attrs=attrs)


def load_metcrax(root):
    with netCDF4.Dataset(f"{root}/MetCrax_unstable_30m.nc") as d:
        # Named US/Mountain, yet unstable hours run 14-24 h on this clock, which is UTC at 111 W.
        start = _time_seconds(d["time"])
        z = _values(d, "heights")
        wind, temp, ustar, wt = (_values(d, k) for k in ("meanU", "meanT", "ustar", "wT"))
    n = start.size
    density = METCRAX_PRESSURE_PA / (constants.R_d * temp)
    attrs = dict(site="metcrax2", latitude=35.0110, longitude=-111.0380, surface="desert plain near Meteor Crater",
                 canopy_height_m=np.nan, displacement_m=np.nan, source="doi:10.5281/zenodo.13222171", licence="CC-BY-4.0",
                 averaging_minutes=30, flux_kind="native", flux_method="eddy covariance, planar fit, linear detrend",
                 note=("selected sample: unstable half hours of October 2013 only, stable data exist only as 1-min "
                       "statistics with 3 complete half hours; stamps taken as interval start, unverified; pressure is "
                       "a constant standard-atmosphere value; temp is sonic temperature and h is sonic w'T' times "
                       "rho cp, so close to a buoyancy flux, with no sensor offset correction; canopy height not "
                       "in the sources"))
    return dict(time_end=start + HALF_HOUR, z_u=_grid(z, n), wind=wind, z_t=_grid(z, n), temp=temp, z_f=_grid(z, n),
                ustar=ustar, h=density * constants.c_pd * wt, le=np.full((n, z.size), np.nan),
                pressure=np.full(n, METCRAX_PRESSURE_PA), t_flux=temp, attrs=attrs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site", choices=("chats", "smear", "ibox", "metcrax"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ibox-min-flag", type=int)
    a = ap.parse_args()
    if a.site == "ibox":
        if a.ibox_min_flag is None:
            ap.error("ibox needs --ibox-min-flag")
        data = load_ibox(a.root, a.ibox_min_flag)
    else:
        data = {"chats": load_chats, "smear": load_smear, "metcrax": load_metcrax}[a.site](a.root)
    os.makedirs(os.path.dirname(a.table), exist_ok=True)
    table.write_table(a.table, **data)
    with open(a.report, "w") as fh:
        json.dump(table.report(a.table), fh, indent=1)
    print("WROTE", a.table, a.report)


if __name__ == "__main__":
    main()
