#!/usr/bin/env python
"""Flux-profile tables for Cardington, ASCOS, AO2018, N-ICE2015 and four ICOS crop and grass sites.

Source flags only. Every level is written; the analysis chooses pairs. Each choice not
fixed by a source file is named in the note attribute of that table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile

import netCDF4
import numpy as np
import pandas as pd

import flux_profile_table as table
from flux_gradient import to_half_hour
from legoesm import constants

HALF_HOUR = 1800
SONIC_HUMIDITY_COEFF = 0.51  # Schotanus et al. 1983: w'Ts' = w'T' + 0.51 T w'q'

CARD_U_Z = [2.0, 10.0, 25.0, 50.0]
CARD_T_Z = {"1p2m": 1.2, "2m": 2.0, "10m": 10.0, "25m": 25.0, "50m": 50.0}
CARD_GOOD = 1
CARD_T_NEAREST = {"2m": "1p2m", "10m": "10m", "25m": "25m", "50m": "50m"}

ASCOS_CSAT_CM = [94, 521, 819]
ASCOS_GILL_CM = [404, 1540]
ASCOS_TOP_CM = 3060
ASCOS_T_CM = [20, 102, 179, 532, 836]
ASCOS_WIND_OK_MAX = 2  # CSAT only: 3 is icing on winds, 4 maintenance
MATLAB_EPOCH_DAYS = 719529

AO_WINDOW_S = 1200
AO_GOOD = (1, 2)
AO_FLUX_Z = {"2m": 2.65, "15m": 15.55}

ICOS_UTC_OFFSET_S = 3600
ICOS_GOOD_SSITC = 1
ICOS_MAX_CANOPY_M = 0.3
ICOS_SITES = (("FR-Gri", "_INTERIM", ["TA_1_1_1", "TA_1_3_1"], [f"WS_1_{i}_1" for i in range(1, 6)]),
              ("FR-Aur", "", ["TA_1_1_1", "TA_1_2_1"], ["WS_1_1_1", "WS_1_2_1"]),
              ("FR-Tou", "", ["TA_1_1_1", "TA_1_2_1"], ["WS_1_1_1", "WS_2_1_1"]),
              ("FR-Lus", "_INTERIM", ["TA_1_1_1", "TA_1_2_1"], ["WS_1_1_1", "WS_1_2_1"]))

# Sensor records end where the data contradict the listed heights; see ICOS_NOTES.
ICOS_END_BEFORE = {"FR-Aur": "202201010000"}
ICOS_NOTES = {
    "FR-Aur": ("; ends 2021: on 2022-01-01 the 4.5 to 2.8 m wind ratio steps from 1.13 to 0.86 with no metadata change, "
               "and stable-time temperature difference changes sign"),
    "FR-Gri": ("; TA_1_2_1 at 2 m left out: near-neutral it reads 0.26 to 0.51 K above the 5 and 1 m sensors, "
               "growing yearly, while those two agree within 0.07 K"),
    "FR-Tou": "; inside Toulouse; near-neutral 2 m reads 0.03 to 0.15 K below 10 m, growing yearly",
}

NICE_U_Z = [2.0, 4.0, 10.0]
NICE_FLUX_Z = 2.2  # (z-d)/L divided by L in the source file, constant


def _v(d, k):
    return np.ma.filled(d[k][:].astype(float), np.nan)


def _grid(z, n):
    return np.tile(np.asarray(z, dtype=float), (n, 1))


def sensible_from_sonic(wt_sonic, wq_density, t, p):
    """H and LE in W m-2 from w'Ts' (K m/s) and the water vapour density flux (kg m-2 s-1)."""
    rho = p / (constants.R_d * t)
    return rho * constants.c_pd * (wt_sonic - SONIC_HUMIDITY_COEFF * t * wq_density / rho), constants.L_v * wq_density


def _card_day(path):
    with netCDF4.Dataset(path) as d:
        def good(name):
            if name not in d.variables:
                return np.full(d.dimensions["time"].size, np.nan)
            return np.where(_v(d, "qc_flag_" + name) == CARD_GOOD, _v(d, name), np.nan)
        out = {"time_end": np.rint(_v(d, "time_bnds")[:, 1] * 3600).astype(np.int64)}
        for z in ("2m", "10m", "25m", "50m"):
            out["u" + z] = good(f"wind_speed_vector_{z}")
            out["uw" + z], out["vw" + z] = good(f"uw_covariance_{z}"), good(f"vw_covariance_{z}")
            out["wt" + z] = good(f"wt_covariance_{z}")
        for z in CARD_T_Z:
            out["t" + z] = good(f"air_temperature_prt_{z}")
        out["wq"] = good("wq_covariance_10m")
        p = good("pressure_barometric_10m")
        out["p"] = np.where(np.isfinite(p), p, good("pressure_barometric"))
    return out


def cardington_arrays(days):
    """Column arrays, sorted, from per-day dictionaries."""
    x = {k: np.concatenate([d[k] for d in days]) for k in days[0]}
    order = np.argsort(x["time_end"], kind="stable")
    return {k: v[order] for k, v in x.items()}


def cardington_tables(x):
    """Split where the 2 m sonic starts: the lowest flux changes from 10 m to 2 m and must not mix."""
    n = x["time_end"].size
    wind = np.column_stack([x["u" + z] for z in ("2m", "10m", "25m", "50m")])
    temp = np.column_stack([x["t" + k] for k in CARD_T_Z]) + constants.T_freeze
    p = x["p"] * 100.0
    levels = ("2m", "10m", "25m", "50m")
    ustar = np.column_stack([np.hypot(x["uw" + z], x["vw" + z]) ** 0.5 for z in levels])
    t_flux = np.column_stack([x["t" + CARD_T_NEAREST[z]] for z in levels]) + constants.T_freeze
    both = [sensible_from_sonic(x["wt" + z], x["wq"], t_flux[:, i], p) for i, z in enumerate(levels)]
    h = np.column_stack([b[0] for b in both])
    le = np.full(h.shape, np.nan)
    le[:, levels.index("10m")] = both[levels.index("10m")][1]
    attrs = dict(latitude=52.10438, longitude=-0.42286, surface="grass airfield", canopy_height_m=0.0, displacement_m=0.0,
                 source="Met Office Cardington 30 min averages, CEDA uuid 1f92b91149704c4bb5048300615a1945",
                 licence="OGL v3 on the CEDA record; file attribute says Non-Commercial Government Licence v2",
                 averaging_minutes=30, flux_kind="native", flux_method="eddy covariance, Gill HS-50 sonic, LI7500 at 10 m",
                 temp_kind="aspirated PRT, RTD set ignored", wind_kind="sonic vector mean",
                 note=("only flag 1 kept, so suspect, object upwind and sonic orientation flags are blanked; "
                       "H from w'Ts' minus 0.51 T w'q' using the 10 m water vapour flux at every level, blank without it; LE only at 10 m; "
                       "that flux is uncorrected open-path covariance, effect on H under 0.5 W m-2; "
                       "t_flux for the 2 m level is the 1.2 m PRT; in every year u* at 2 m is 0.75 to 0.80 of u* at 10 m for U10 above 6 m/s, "
                       "cause unknown, possibly rougher ground upwind of the 10 m footprint; U50/U25 drifts from 1.08 in 2004 to 1.15 in 2024"))
    start = int(np.argmax(np.isfinite(ustar[:, 0])))
    early, late = slice(0, start), slice(start, n)
    tz = _grid(list(CARD_T_Z.values()), n)
    common = dict(z_t=tz, temp=temp, pressure=p, z_u=_grid(CARD_U_Z, n), wind=wind, time_end=x["time_end"])
    def cut(s, cols, site, note):
        out = {k: v[s] for k, v in common.items()}
        out.update(z_f=_grid(np.array(CARD_U_Z)[cols], n)[s], ustar=ustar[s][:, cols], h=h[s][:, cols], le=le[s][:, cols],
                   t_flux=t_flux[s][:, cols], attrs=dict(attrs, site=site, note=attrs["note"] + note))
        return out
    return (cut(early, [1, 2, 3], "cardington_flux10m", "; before the 2 m sonic: lowest flux is 10 m, not surface-near"),
            cut(late, [0, 1, 2, 3], "cardington_flux2m", "; from the first 2 m sonic record"))


def load_cardington(root):
    days = [_card_day(p) for p in sorted(glob.glob(f"{root}/30min-averages/*/*.nc"))]
    return cardington_tables(cardington_arrays(days))


def _half(t, x, ok, samples=3):
    return to_half_hour(t, x, ok.astype(int), samples=samples)


def load_ascos(path):
    with netCDF4.Dataset(path) as d:
        g = lambda k: _v(d, k)
        # Stamps are 10-min interval midpoints on a MATLAB day number; each falls inside its own half hour.
        t = (np.rint((g("time") - MATLAB_EPOCH_DAYS) * 1440) * 60).astype(np.int64)
        ship_ok = g("flag_ship") == 0
        csat_mast, gill_mast = g("flag_mast_csat") == 0, g("flag_mast_gill") == 0
        csat = {c: g(f"flag_csat_{c}_cm") for c in ASCOS_CSAT_CM}
        gill = {c: g(f"flag_gill_{c}_cm") for c in ASCOS_GILL_CM}
        wind_ok = {c: (f <= ASCOS_WIND_OK_MAX) & csat_mast & ship_ok for c, f in csat.items()}
        # Gill flag 2 reads "sonic temperature only", yet its median stress is 100 to 4000 times that of flag 0.
        wind_ok.update({c: (f == 0) & gill_mast & ship_ok for c, f in gill.items()})
        wind_ok[ASCOS_TOP_CM] = ship_ok
        levels = sorted(wind_ok)
        keys, _ = _half(t, g("Pressure"), np.ones(t.size, bool))
        comp = lambda name: np.column_stack([_half(t, g(f"{name}_{c}_cm"), wind_ok[c])[1] for c in levels])
        wind = table.vector_speed(comp("ugeo"), comp("vgeo"))
        temp = np.column_stack([_half(t, g(f"TC_Air_{c}_cm"), ship_ok)[1] for c in ASCOS_T_CM]) + constants.T_freeze
        p = _half(t, g("Pressure"), np.ones(t.size, bool))[1] * 100.0
        flux_cm = sorted(ASCOS_CSAT_CM + ASCOS_GILL_CM)
        stress = lambda c: np.hypot(g(f"uw_{c}_cm"), g(f"vw_{c}_cm"))
        ustar = np.column_stack([np.sqrt(_half(t, stress(c), wind_ok[c])[1]) for c in flux_cm])
        nearest = [int(np.argmin(np.abs(np.array(ASCOS_T_CM) - c))) for c in flux_cm]
        t_flux = temp[:, nearest]
        wq_ok = (g("flag_licor_404_cm") == 0) & (gill[404] == 0) & gill_mast & ship_ok
        wq = _half(t, g("wq_404_cm"), wq_ok)[1]
        rho = p[:, None] / (constants.R_d * t_flux)
        h = np.full(ustar.shape, np.nan)
        for i, c in enumerate(flux_cm):
            if c in ASCOS_CSAT_CM:
                h[:, i] = rho[:, i] * constants.c_pd * _half(t, g(f"wtc_{c}_cm"), (csat[c] == 0) & csat_mast & ship_ok)[1]
            else:
                wts = _half(t, g(f"wts_{c}_cm"), (gill[c] == 0) & gill_mast & ship_ok)[1]
                h[:, i] = rho[:, i] * constants.c_pd * (wts - SONIC_HUMIDITY_COEFF * t_flux[:, i] * wq)
        le = np.full(h.shape, np.nan)
        le[:, flux_cm.index(404)] = rho[:, flux_cm.index(404)] * constants.L_v * wq
        lat, lon = float(np.nanmedian(g("latitude"))), float(np.nanmedian(g("longitude")))
    n = keys.size
    attrs = dict(site="ascos", latitude=lat, longitude=lon, surface="sea ice floe, summer", canopy_height_m=0.0,
                 displacement_m=0.0, source="doi:10.17043/oden-ascos-2008-micrometeorological-1, ascos_Metalley.nc",
                 licence="none stated on the landing page", averaging_minutes=30, flux_kind="mean_of_shorter_periods",
                 flux_method="eddy covariance, three 10-min covariances per half hour; CSAT with finewire, Gill with LI7500",
                 temp_kind="aspirated thermocouples; two Rotronic levels left out, source warns they differ",
                 note=("heights nominal, snow and melt not tracked; ship and through-mast flags blank wind, stress and heat; "
                       "Gill flag 0 only for wind and stress too: flag 2 stress is 100 to 4000 times flag 0; "
                       "CSAT heat from finewire with flag 0; Gill heat from w'Ts' with humidity term from the 4.04 m Licor; "
                       "LE only at 4.04 m, the Licor level; lead sonic left out; 30.6 m is wind only, no icing flag; "
                       "a floe drifting, lat and lon are medians; wind mixes CSAT and Gill sonics with no intercalibration, "
                       "so pairs across types carry offsets: near-neutral phi_m 4.04-5.21 m is 3.2, 5.21-8.19 m is 0"))
    return dict(time_end=keys, z_u=_grid(np.array(levels) / 100.0, n), wind=wind, z_t=_grid(np.array(ASCOS_T_CM) / 100.0, n),
                temp=temp, z_f=_grid(np.array(flux_cm) / 100.0, n), ustar=ustar, h=h, le=le, pressure=p, t_flux=t_flux, attrs=attrs)


def window_mean(t, x, ends, width, samples):
    """Mean of x over (end - width, end] at each end, only where every sample is present."""
    idx = np.searchsorted(ends, t, side="left")
    idx = np.where((idx < ends.size) & (ends[np.minimum(idx, ends.size - 1)] - t < width), idx, -1)
    shape = (ends.size,) + x.shape[1:]
    total, count = np.zeros(shape), np.zeros(shape)
    use = idx >= 0
    good = np.isfinite(x[use])
    np.add.at(total, idx[use], np.where(good, x[use], 0.0))
    np.add.at(count, idx[use], good)
    return np.where(count == samples, total / np.maximum(count, 1), np.nan)


def load_ao2018(root):
    with netCDF4.Dataset(f"{root}/oden-ao-2018-meteorology-ice-station-1.nc") as d:
        tm = np.rint(_v(d, "time")).astype(np.int64)
        tz, uz = _v(d, "tz"), _v(d, "wsz")
        tprof = np.where(_v(d, "tprof_corflag") == 0, _v(d, "tprof"), np.nan) + constants.T_freeze
        uprof = np.where(_v(d, "wsprof_corflag") == 0, _v(d, "wsprof"), np.nan)
    flux = {}
    for name in AO_FLUX_Z:
        with netCDF4.Dataset(f"{root}/oden-ao-2018-turbulent-flux-{name}-ice-station-1.nc") as d:
            mom = np.isin(_v(d, "qc_flag_quality_wu"), AO_GOOD) & np.isin(_v(d, "qc_flag_quality_wv"), AO_GOOD)
            # Stamps sit at :10, :30, :50: midpoints of the 20-min windows, as the minute data sit at :30 s.
            flux[name] = dict(end=np.rint(_v(d, "time") / 600).astype(np.int64) * 600 + AO_WINDOW_S // 2,
                              ustar=np.where(mom, _v(d, "ustar"), np.nan),
                              h=np.where(np.isin(_v(d, "qc_flag_quality_wt"), AO_GOOD), _v(d, "tsflux"), np.nan),
                              rho=_v(d, "airdensity"))
    ends = np.unique(np.concatenate([f["end"] for f in flux.values()]))
    n = ends.size
    # Minute stamps are midpoints; every window holds 20 of them.
    wind = window_mean(tm, uprof, ends, AO_WINDOW_S, AO_WINDOW_S // 60)
    temp = window_mean(tm, tprof, ends, AO_WINDOW_S, AO_WINDOW_S // 60)
    names = list(AO_FLUX_Z)
    place = lambda f, k: _on(ends, f["end"], f[k])
    ustar = np.column_stack([place(flux[k], "ustar") for k in names])
    h = np.column_stack([place(flux[k], "h") for k in names])
    nearest = [int(np.argmin(np.abs(tz - AO_FLUX_Z[k]))) for k in names]
    t_flux = temp[:, nearest]
    pressure = place(flux["2m"], "rho") * constants.R_d * t_flux[:, 0]
    attrs = dict(site="ao2018", latitude=89.0, longitude=40.0, surface="sea ice floe, late summer", canopy_height_m=0.0,
                 displacement_m=0.0, source="doi:10.17043/oden-ao-2018-meteorology-ice-station-1 and turbulent flux 2 m, 15 m",
                 licence="UK Open Government Licence", averaging_minutes=20, flux_kind="native",
                 flux_method="eddy covariance, Gill R3 at 2.65 m, Metek USA-1 at 15.55 m",
                 temp_kind="aspirated thermocouples", wind_kind="cup anemometers, scalar, no profile direction",
                 note=("20-min fluxes do not divide into 30 min, so the table stays at 20 min and the 1-min profiles are "
                       "averaged over the same windows; H is the source sonic heat flux, no humidity term, no LE; "
                       "pressure rebuilt from source air density and the lowest flux temperature; qc 1 and 2 kept; "
                       "corflag 0 only; heights as given, snow not tracked; coordinates approximate, floe drifting"))
    return dict(time_end=ends, z_u=_grid(uz, n), wind=wind, z_t=_grid(tz, n), temp=temp,
                z_f=_grid(list(AO_FLUX_Z.values()), n), ustar=ustar, h=h, le=np.full(h.shape, np.nan),
                pressure=pressure, t_flux=t_flux, attrs=attrs)


def _on(grid, t, x):
    out = np.full((grid.size,) + x.shape[1:], np.nan)
    out[np.searchsorted(grid, t)] = x
    return out


def load_nice(flux_path, met_path):
    with netCDF4.Dataset(met_path) as m:
        t = np.rint(_v(m, "unix_time")).astype(np.int64)
        speed = {z: _v(m, f"wind_speed_{z}m") for z in (2, 4, 10)}
        direction = {z: np.deg2rad(_v(m, f"wind_from_direction_{z}m")) for z in (2, 4, 10)}
        # Flag 1 is wind reconstructed from the ship, not measured on the mast.
        ok = {2: np.ones(t.size, bool), 4: np.ones(t.size, bool),
              10: (_v(m, "wind_speed_10m_flag") == 0) & (_v(m, "wind_from_direction_10m_flag") == 0)}
        keys, _ = _half(t, np.ones(t.size), np.ones(t.size, bool), samples=30)
        comp = lambda f: np.column_stack([_half(t, f(z), ok[z], samples=30)[1] for z in (2, 4, 10)])
        wind = table.vector_speed(comp(lambda z: -speed[z] * np.sin(direction[z])), comp(lambda z: -speed[z] * np.cos(direction[z])))
        temp = np.column_stack([_half(t, _v(m, f"air_temperature_{z}m"), np.ones(t.size, bool), samples=30)[1] for z in (2, 4, 10)])
        p = _half(t, _v(m, "air_pressure_at_sea_level"), np.ones(t.size, bool), samples=30)[1]
    with netCDF4.Dataset(flux_path) as d:
        tf = np.rint(_v(d, "unix_time")).astype(np.int64)
        ustar, down_h, down_le = _v(d, "friction_velocity"), _v(d, "surface_downward_sensible_heat_flux"), _v(d, "surface_downward_latent_heat_flux")
        lat, lon = float(np.nanmedian(_v(d, "latitude"))), float(np.nanmedian(_v(d, "longitude")))
    ends = np.union1d(keys, tf)
    n = ends.size
    onto = lambda x: _on(ends, keys, x)
    flux = lambda x: _on(ends, tf, x)[:, None]
    temp = onto(temp)
    attrs = dict(site="nice2015", latitude=lat, longitude=lon, surface="young sea ice, drifting floes", canopy_height_m=0.0,
                 displacement_m=0.0, source="doi:10.21334/npolar.2015.056a61d1 and N-ICE_sebData_v1",
                 licence="open; N-ICE team asks users to contact Hudson and Cohen and to acknowledge", averaging_minutes=30,
                 flux_kind="native", flux_method="eddy covariance, Campbell CPEC200 CSAT3A, EC155 closed path",
                 temp_kind="HMP155 aspirated, plus or minus 0.3 K each",
                 note=("flux height 2.2 m from zeta/L in the source; source H and LE are downward positive, flipped here; "
                       "flux stamp taken as interval end, lag test cannot separate end from midpoint, start is worse; "
                       "gap-filled variables left out; 10 m ship-reconstructed wind blanked; no flux quality flag in the source; "
                       "heights nominal above the surface; lat and lon are medians"))
    return dict(time_end=ends, z_u=_grid(NICE_U_Z, n), wind=onto(wind), z_t=_grid(NICE_U_Z, n), temp=temp,
                z_f=_grid([NICE_FLUX_Z], n), ustar=flux(ustar), h=-flux(down_h), le=-flux(down_le), pressure=onto(p),
                t_flux=temp[:, :1], attrs=attrs)


def stamp_seconds(stamp):
    """YYYYMMDD or YYYYMMDDHHMM strings to seconds since 1970, on the clock of the stamp."""
    text = pd.Series(stamp).astype(str).str.replace(r"\.0$", "", regex=True).str.ljust(12, "0")
    return (pd.to_datetime(text, format="%Y%m%d%H%M") - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def steps(dates, values, times):
    """Value in force at each time: last entry at or before it, NaN before the first."""
    order = np.argsort(dates, kind="stable")
    dates, values = np.asarray(dates)[order], np.asarray(values, dtype=float)[order]
    idx = np.searchsorted(dates, times, side="right") - 1
    return np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)


def short_canopy(dates, heights, times):
    """Times when canopy height, interpolated between observations, is within the limit."""
    order = np.argsort(dates, kind="stable")
    h = np.interp(times, np.asarray(dates)[order], np.asarray(heights, dtype=float)[order], left=np.nan, right=np.nan)
    return h <= ICOS_MAX_CANOPY_M


def _groups(z, name):
    frame = pd.read_csv(z.open(name), dtype=str)
    return frame.pivot_table(index="GROUP_ID", columns="VARIABLE", values="DATAVALUE", aggfunc="first")


def _icos_site(path, code, suffix, temps, winds):
    z = zipfile.ZipFile(path)
    p = f"ICOSETC_{code}_"
    file = lambda kind: f"{p}{kind}{suffix}_L2.csv"
    info = _groups(z, file("VARINFO_METEOSENS"))
    cols = ["TIMESTAMP_END", "PA_1_1_1"] + temps + winds
    met = pd.read_csv(z.open(file("METEOSENS")), usecols=cols, na_values=[-9999])
    flux = pd.read_csv(z.open(file("FLUXES")), usecols=["TIMESTAMP_END", "H", "LE", "USTAR", "H_SSITC_TEST", "LE_SSITC_TEST", "TAU_SSITC_TEST"],
                       na_values=[-9999])
    x = met.merge(flux, on="TIMESTAMP_END", how="inner")
    # ICOS stamps run on local standard time, UTC+1 at these sites, and close the interval.
    clock = stamp_seconds(x["TIMESTAMP_END"])
    t = (clock - ICOS_UTC_OFFSET_S).astype(np.int64)
    n = t.size

    def heights(names):
        out = np.empty((n, len(names)))
        for i, name in enumerate(names):
            g = info[info["VAR_INFO_VARNAME"] == name]
            out[:, i] = steps(stamp_seconds(g["VAR_INFO_DATE"]), g["VAR_INFO_HEIGHT"].astype(float), clock)
        return out

    inst = _groups(z, file("INST"))
    sonic = inst[inst["INSTOM_MODEL"].str.startswith("SA-", na=False) & inst["INSTOM_HEIGHT"].notna()]
    z_f = steps(stamp_seconds(sonic["INSTOM_DATE"]), sonic["INSTOM_HEIGHT"].astype(float), clock)[:, None]
    anc = _groups(z, file("ANCILLARY"))
    c = anc[anc["HEIGHTC"].notna() & anc["HEIGHTC_STATISTIC"].isin(["Mean", "Single observation"])]
    keep = short_canopy(stamp_seconds(c["HEIGHTC_DATE"]), c["HEIGHTC"].astype(float), clock)
    site = _groups(z, file("SITEINFO"))
    lat, lon = (float(site[k].dropna().iloc[0]) for k in ("LOCATION_LAT", "LOCATION_LONG"))
    desc = str(site["SITE_DESC"].dropna().iloc[0])[:200]
    z_t, z_u = heights(temps), heights(winds)
    up_t, up_u = np.argsort(np.nanmedian(z_t, axis=0)), np.argsort(np.nanmedian(z_u, axis=0))  # ICOS numbers levels top-down
    z_t, z_u = z_t[:, up_t], z_u[:, up_u]
    temp = x[temps].to_numpy(float)[:, up_t] + constants.T_freeze
    nearest = int(np.nanargmin(np.abs(np.nanmedian(z_t, axis=0) - np.nanmedian(z_f))))
    good = lambda name, flag: np.where(x[flag].to_numpy(float) <= ICOS_GOOD_SSITC, x[name].to_numpy(float), np.nan)
    data = dict(time_end=t, z_u=z_u, wind=x[winds].to_numpy(float)[:, up_u], z_t=z_t, temp=temp, z_f=z_f,
                ustar=good("USTAR", "TAU_SSITC_TEST")[:, None], h=good("H", "H_SSITC_TEST")[:, None], le=good("LE", "LE_SSITC_TEST")[:, None],
                pressure=x["PA_1_1_1"].to_numpy(float) * 1000.0, t_flux=temp[:, nearest:nearest + 1])
    if code in ICOS_END_BEFORE:
        keep &= clock < stamp_seconds(ICOS_END_BEFORE[code])[0]
    data = {k: v[keep] for k, v in data.items()}
    data["attrs"] = dict(site="icos_" + code.lower().replace("-", "_"), latitude=lat, longitude=lon, surface=desc,
                         canopy_height_m=ICOS_MAX_CANOPY_M, displacement_m=np.nan, source=f"ICOS ETC L2 archive {code}",
                         licence="CC-BY-4.0", averaging_minutes=30, flux_kind="native", flux_method="eddy covariance, ICOS ETC",
                         quality="INTERIM, not final QC" if suffix else "final L2",
                         note=(f"only times with canopy at most {ICOS_MAX_CANOPY_M} m, height interpolated between visits, so "
                               "no time before the first or after the last visit; canopy_height_m is that limit, not the "
                               "height at each time; heights step with VAR_INFO and INSTOM dates; SSITC 0 and 1 kept; "
                               "wind is scalar 30-min mean from the source; meteo as delivered in L2" + ICOS_NOTES.get(code, "")))
    return data


def load_icos(root):
    for code, suffix, temps, winds in ICOS_SITES:
        tag = "ARCHIVE" + suffix
        data = _icos_site(f"{root}/ICOSETC_{code}_{tag}_L2.zip", code, suffix, temps, winds)
        yield data["attrs"]["site"], data


def write(name, data, tables, reports):
    path = f"{tables}/{name}.nc"
    table.write_table(path, **data)
    with open(f"{reports}/{name}.json", "w") as fh:
        json.dump(table.report(path), fh, indent=1)
    print("WROTE", path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites", nargs="+", choices=("cardington", "ascos", "ao2018", "nice2015", "icos"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--reports", required=True)
    a = ap.parse_args()
    os.makedirs(a.reports, exist_ok=True)
    r = a.data_root
    for site in a.sites:
        if site == "cardington":
            for d in load_cardington(f"{r}/cardington"):
                write(d["attrs"]["site"], d, a.tables, a.reports)
        elif site == "ascos":
            write("ascos", load_ascos(f"{r}/ascos_metalley/ascos_Metalley.nc"), a.tables, a.reports)
        elif site == "ao2018":
            write("ao2018", load_ao2018(f"{r}/ao2018_ice_station"), a.tables, a.reports)
        elif site == "nice2015":
            met = glob.glob(f"{r}/nice2015_met/*.bin")[0]
            write("nice2015", load_nice(f"{r}/nice2015_flux/x/N-ICE_sebData_v1.nc", met), a.tables, a.reports)
        else:
            for code, data in load_icos(f"{r}/icos_etc_profiles"):
                write(code, data, a.tables, a.reports)


if __name__ == "__main__":
    main()
