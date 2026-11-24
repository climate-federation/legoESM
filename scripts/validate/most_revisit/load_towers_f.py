#!/usr/bin/env python
"""Flux-profile tables for CASES-99 and AHATS 2008 from NCAR/EOL ISFF 5-minute statistics.

Source flags only; every level is written and the analysis chooses pairs. Side decisions were put to a
GLM+codex vote (coscientist/newdata/votes.md) and are named in each table's note. HATS 2000 is not loaded:
its flux sonic heights are undocumented (vote V10).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules, also under python -I
import flux_profile_table as table  # noqa: E402
from flux_gradient import to_half_hour  # noqa: E402

from legoesm import constants  # noqa: E402

BLOCKS = 6  # 5-min records per half hour, all required
BLOCK_S = 300
CORRUPT = 1e10  # 5-min means this large are broken fills, not data (vote V3)
HPA_TO_PA = 100.0
PRESSURE_RANGE_HPA = (800.0, 1100.0)  # each station's 30-min mean outside this is blanked before the median (vote W3)
ATI_MIN_SAMPLES = 10  # readme: ATI wind and temperature unusable when u/v/wsamples < 10

# (height m, variable suffix, QC kind); CSAT3 'diag' and AHATS 'ldiag' must be 0, ATI samples >= 10 (vote V2)
CASES_SONICS = [(1.5, "1_5m", "diag"), (5.0, "5m", "diag"), (10.0, "10m", "ati"), (20.0, "20m", "ati"),
                (30.0, "30m", "diag"), (50.0, "50m", "diag"), (55.0, "55m", "ati")]  # 40 m ATI dropped (vote W2)
CASES_T = [(5.0, "T_5m"), (15.0, "T_15m"), (25.0, "T_25m"), (35.0, "T_35m"), (45.0, "T_45m"), (55.0, "T_55m")]
AHATS_SONICS = [(1.55, "1_55m", "ldiag"), (3.30, "3_30m", "ldiag"), (4.24, "4_24m", "ldiag"),
                (5.53, "5_53m", "ldiag"), (7.08, "7_08m", "ldiag"), (8.05, "8_05m", "ldiag")]
AHATS_T = [(1.51, "T_1_51m"), (3.26, "T_3_26m"), (4.23, "T_4_23m"), (5.47, "T_5_47m"), (7.01, "T_7_01m"), (8.01, "T_8_01m")]


def _f(v):
    return np.ma.filled(np.ma.asarray(v[:]).astype(float), np.nan)


def read_days(pattern, names):
    """End-of-block stamps (s since 1970) and {name: values}, corrupt numbers NaN, over all daily files."""
    t, out, n_files = [], {k: [] for k in names}, 0
    for path in sorted(glob.glob(pattern)):
        n_files += 1
        with netCDF4.Dataset(path) as d:
            n = d.dimensions["time"].size
            # time is the 5-min midpoint, seconds after base_time (readme)
            mid = _f(d["time"])
            if not np.all((mid >= 0) & (mid < 86400)):
                raise ValueError(f"{path}: time outside the file's day")
            t.append(float(d["base_time"][...]) + mid + BLOCK_S / 2)
            for k in names:
                x = _f(d[k]) if k in d.variables else None
                out[k].append(None if x is None else np.where(np.abs(x) < CORRUPT, x, np.nan))
                if x is not None and x.shape[0] != n:
                    raise ValueError(f"{path}:{k} has {x.shape[0]} rows, time has {n}")
            out.setdefault("_n", []).append(n)
    if not n_files:
        raise FileNotFoundError(pattern)
    lens = out.pop("_n")
    absent = [k for k, parts in out.items() if all(p is None for p in parts)]
    if absent:
        raise KeyError(f"{pattern}: {absent} in no file")
    for k, parts in out.items():
        shape = next((p.shape[1:] for p in parts if p is not None), ())
        out[k] = np.concatenate([p if p is not None else np.full((m,) + shape, np.nan) for p, m in zip(parts, lens)])
    t = np.rint(np.concatenate(t)).astype(np.int64)
    order = np.argsort(t, kind="stable")
    return t[order], {k: v[order] for k, v in out.items()}


def half(t, x, ok):
    """Half-hour mean of x where every one of the six blocks is good."""
    ok = np.broadcast_to(np.reshape(ok, ok.shape + (1,) * (np.ndim(x) - 1)), np.shape(x))
    return to_half_hour(t, x, ok.astype(int), samples=BLOCKS)


def covariance(t, ok, c, a, b):
    """30-min covariance: mean of the 5-min covariances plus covariance of the 5-min means (vote V1)."""
    keys, mc = half(t, c, ok)
    _, mab = half(t, a * b, ok)
    _, ma = half(t, a, ok)
    _, mb = half(t, b, ok)
    return keys, mc + mab - ma * mb


def sonic_ok(x, s, kind):
    if kind in ("diag", "ldiag"):
        return x[f"{kind}_{s}"] == 0
    if kind == "ati":
        return np.minimum.reduce([x[f"{c}samples_{s}"] for c in "uvw"]) >= ATI_MIN_SAMPLES
    raise ValueError(f"unknown sonic QC kind {kind}")


def sonic_names(s, kind):
    qc = [f"{kind}_{s}"] if kind != "ati" else [f"{c}samples_{s}" for c in "uvw"]
    return [f"{k}_{s}" for k in ("u", "v", "w", "tc", "u_w", "v_w", "w_tc")] + qc


def isff_table(t, x, sonics, temps, pressure, profile_winds=True):
    """Columns for flux_profile_table.write_table from 5-min ISFF statistics; no wind levels if not profile_winds."""
    winds, ustar, wts = [], [], []
    for _, s, kind in sonics:
        ok = sonic_ok(x, s, kind)
        keys, u = half(t, x[f"u_{s}"], ok)
        _, v = half(t, x[f"v_{s}"], ok)
        winds.append(table.vector_speed(u, v))
        uw = covariance(t, ok, x[f"u_w_{s}"], x[f"u_{s}"], x[f"w_{s}"])[1]
        vw = covariance(t, ok, x[f"v_w_{s}"], x[f"v_{s}"], x[f"w_{s}"])[1]
        ustar.append(np.hypot(uw, vw) ** 0.5)
        wts.append(covariance(t, ok, x[f"w_tc_{s}"], x[f"w_{s}"], x[f"tc_{s}"])[1])
    temp = np.column_stack([half(t, x[k], np.ones(t.size, bool))[1] for _, k in temps]) + constants.T_freeze
    _, p = half(t, x[pressure], np.ones(t.size, bool))
    p = np.where((p >= PRESSURE_RANGE_HPA[0]) & (p <= PRESSURE_RANGE_HPA[1]), p, np.nan)
    with np.errstate(all="ignore"):
        p = (np.nanmedian(p, axis=1) if p.ndim == 2 else p) * HPA_TO_PA  # station median (vote V8)
    z_s, z_t = np.array([z for z, _, _ in sonics]), np.array([z for z, _ in temps])
    # nearest thermometer; mean of the two when the sonic sits midway (vote T1)
    t_flux = np.column_stack([temp[:, np.abs(z_t - z) == np.abs(z_t - z).min()].mean(axis=1) for z in z_s])
    wts = np.column_stack(wts)
    # sonic heat flux, no humidity term (vote V6)
    h = p[:, None] / (constants.R_d * t_flux) * constants.c_pd * wts
    n = keys.size
    grid = lambda z: np.tile(z.astype(float), (n, 1))  # noqa: E731
    wind, z_u = (np.column_stack(winds), z_s) if profile_winds else (np.empty((n, 0)), np.empty(0))
    return dict(time_end=keys, z_u=grid(z_u), wind=wind, z_t=grid(z_t), temp=temp,
                z_f=grid(z_s), ustar=np.column_stack(ustar), h=h, le=np.full(h.shape, np.nan), pressure=p, t_flux=t_flux)


def _names(sonics, temps, pressure):
    return [k for _, s, kind in sonics for k in sonic_names(s, kind)] + [k for _, k in temps] + [pressure]


COMMON_NOTE = ("30-min rows from six 5-min ISFF blocks, all six good; fluxes are full 30-min covariances (mean 5-min "
               "covariance plus covariance of 5-min means); 5-min values of magnitude 1e10 or more blanked as broken fills; "
               "H = rho cp w'Ts' with no humidity term, LE not used; wind is the sonic vector mean; t_flux is the nearest "
               "thermometer, or the mean of two equidistant ones; data tilt-corrected by the source (planar fit)")


def load_cases(root):
    t, x = read_days(f"{root}/cases.*.nc", _names(CASES_SONICS, CASES_T, "P"))
    out = isff_table(t, x, CASES_SONICS, CASES_T, "P")
    out["attrs"] = dict(site="cases99", latitude=37.6481, longitude=-96.7360, surface="tallgrass prairie, October",
                        canopy_height_m=0.0, displacement_m=0.0, averaging_minutes=30, flux_kind="native",
                        source="NCAR/EOL ISFF 5-min statistics, CASES-99, EOL dataset 45.953",
                        licence="NSF NCAR/EOL data policy", flux_method="eddy covariance, CSAT3 1.5/5/30/50 m, ATI 10/20/55 m",
                        temp_kind="tower thermometers 5-55 m", wind_kind="sonic vector mean",
                        note=COMMON_NOTE + ("; CSAT3 used only where the 5-min diag is 0 and ATI where u/v/wsamples >= 10 "
                                            "(readme rules); 40 m ATI left out (30-min winds to 57 m/s, u* to 11.6 m/s); props and sonic "
                                            "temperature not used; pressure is the median of the six outlying stations, each kept "
                                            "within 800-1100 hPa; d = 0 (no published value); latitude and longitude approximate "
                                            "(main tower, not in the files)"))
    return out


def load_ahats(root):
    t, x = read_days(f"{root}/isff_*.nc", _names(AHATS_SONICS, AHATS_T, "P_3m"))
    out = isff_table(t, x, AHATS_SONICS, AHATS_T, "P_3m", profile_winds=False)
    out["attrs"] = dict(site="ahats2008", latitude=36.02405, longitude=-119.90002, surface="fallow field, summer",
                        canopy_height_m=0.0, displacement_m=0.0, averaging_minutes=30, flux_kind="native",
                        source="NCAR/EOL ISFF 5-min statistics, AHATS 2008, EOL dataset 109.011, profile tower only",
                        licence="NSF NCAR/EOL data policy", flux_method="eddy covariance, CSAT3 at 6 heights 1.55-8.05 m",
                        temp_kind="tower thermometers 1.51-8.01 m", wind_kind="none: heat only",
                        note=COMMON_NOTE + ("; CSAT3 used only where the 5-min ldiag (fraction of bad samples) is 0; "
                                            "no wind profile (heat only): the profile sonics carry uncorrected zero-wind offsets of 4-30 cm/s "
                                            "and near-neutral phi_m runs 1.9-2.2 and below 0 on two adjacent pairs; "
                                            "pressure kept within 800-1100 hPa; horizontal arrays not used; d = 0 (HATS 2000 found 0.32 m on this field, not applied)"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites", nargs="+", choices=("cases99", "ahats2008"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--reports", required=True)
    a = ap.parse_args()
    os.makedirs(a.reports, exist_ok=True)
    for site in a.sites:
        data = load_cases(f"{a.data_root}/cases99") if site == "cases99" else load_ahats(f"{a.data_root}/ahats")
        path = f"{a.tables}/{site}.nc"
        table.write_table(path, **data)
        with open(f"{a.reports}/{site}.json", "w") as fh:
            json.dump(table.report(path), fh, indent=1)
        print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
