#!/usr/bin/env python3
"""Score the snow-insulation A/B (snowheat_oct/PREREG.md) for one ON/OFF pair.

Region: 45-70N, land fraction >= 0.5.  Native-mesh fields (checkpoints, 00Z
instants, day N = Jan 1 00Z + N days) are weighted by cell area, the native
analogue of cos-lat; CMOR lat-lon fields (daily-mean tas) by areacella over
cells whose centre lies in the band with sftlf >= 50 %.  Window days are the
checkpoints at 00Z of each day in [first, last].  Non-finite values in the
region are fatal.  Prints numbers only; verdicts are read against the bands.

Per arm, window mean of the regional mean:
  T_0.175   soil temperature at 0.175 m, linear between the bracketing nodes
  H_1m      heat of the top 1 m relative to each layer at T_freeze with its own
            water: integral of the land's apparent heat capacity (sensible with
            separate liquid/ice capacities + latent of its freezing curve) from
            T_freeze to T, Gauss-Legendre in T, times dz clipped to 1 m
  F_soil12  upward conductive flux between the two top soil NODES (harmonic-mean
            interface conductivity, as the soil solve); the surface ground heat
            flux itself is not checkpointed
"valid" rows (exploratory, not preregistered; NOT a substitute for run validity):
cells whose ON snow-node T exceeds freezing under snow on any window day are dropped.
  SWE       snow water equivalent
  tas       CMOR daily-mean near-surface air temperature (and vs ERA5 if given)

ERA5 daily-mean t2m on the SAME days, on the CMOR grid, e.g. for Dec 11-31:
  cdo -f nc -remapcon,<cmor tas file> -timmean -seldate,2001-12-11,2001-12-31 \
      -setgridtype,regular /pool/data/ERA5/E5/sf/an/1D/167/E5sf00_1D_2001-12_167.grb e5_dec.nc

Run with the RUN's code on PYTHONPATH, JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1.
Usage: snowheat_ab_score.py <on_run> <off_run> <scratch> --window 2001-12-11 2001-12-31
           [--era5 e5.nc] [--all-days-snow-check] [--figures DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np

ROOT = os.environ.get("LEGOESM_AMIP_RUNS",
                      "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
BAND = (45.0, 70.0)
DEPTH_SOIL_T = 0.175
DEPTH_HEAT = 1.0


def checkpoint_days(start_year, first, last):
    """Checkpoint day indices (00Z, day N = Jan 1 + N) from first to last date, inclusive."""
    jan1 = dt.date(start_year, 1, 1)
    d0, d1 = (dt.date.fromisoformat(s) for s in (first, last))
    if d1 < d0 or d0.year != start_year or d1.year != start_year:
        raise SystemExit(f"FATAL: window {first}..{last} not in {start_year}")
    return list(range((d0 - jan1).days, (d1 - jan1).days + 1))


def at_depth(T, z, depth):
    """(ncol, nz) node values -> (ncol,) at `depth`, linear between bracketing nodes."""
    z = np.asarray(z, np.float64)
    k = int(np.searchsorted(z, depth)) - 1
    if not (0 <= k < z.size - 1) or not (z[k] <= depth <= z[k + 1]):
        raise SystemExit(f"FATAL: depth {depth} m not bracketed by nodes {z.tolist()}")
    f = (depth - z[k]) / (z[k + 1] - z[k])
    return (1.0 - f) * T[:, k] + f * T[:, k + 1]


def overlap(dz, depth):
    """Thickness of each layer (thicknesses dz, from the surface) above `depth`."""
    top = np.concatenate([[0.0], np.cumsum(dz)[:-1]])
    return np.clip(np.minimum(top + dz, depth) - top, 0.0, None)


def heat_from_freezing(capacity, T, T_ref, n=48):
    """Integral of capacity(T') dT' from T_ref to T, elementwise (Gauss-Legendre)."""
    x, wt = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (x + 1.0)
    acc = sum(0.5 * wi * capacity(T_ref + si * (T - T_ref)) for si, wi in zip(s, wt))
    return (T - T_ref) * acc


def flat_config(path):
    """experiment_config.json flattened to dotted keys."""
    import json

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v, f"{pre}{k}.")
            else:
                yield f"{pre}{k}", v
    return dict(walk(json.load(open(path))))


def config_diff(a, b):
    """Keys whose values differ, a missing key counting as different from null."""
    missing = object()
    return {k for k in set(a) | set(b) if a.get(k, missing) != b.get(k, missing)}


def wmean(x, w):
    x, on = np.asarray(x, np.float64), w > 0
    if not on.any():
        raise SystemExit("FATAL: empty region")
    if not np.isfinite(x[on]).all():
        raise SystemExit("FATAL: non-finite value inside region")
    return float(np.sum(np.where(on, x, 0.0) * w) / np.sum(w))


def cmor_tas(run, first, last, era5=None):
    """Window-mean regional tas (and ERA5 on the same grid/mask) from the run's CMOR daily file."""
    import xarray as xr
    d = xr.open_dataset(glob.glob(f"{ROOT}/{run}/cmor/day/tas_day_*.nc")[0], use_cftime=True)
    sft = xr.open_dataset(glob.glob(f"{ROOT}/{run}/cmor/fx/sftlf_*.nc")[0])["sftlf"].values
    area = xr.open_dataset(glob.glob(f"{ROOT}/{run}/cmor/fx/areacella_*.nc")[0])["areacella"].values
    if np.nanmax(sft) <= 1.0:
        raise SystemExit("FATAL: sftlf is not in percent")
    lat = d["lat"].values
    w = area * (sft >= 50.0) * ((lat >= BAND[0]) & (lat <= BAND[1]))[:, None]
    dates = np.array([t.strftime("%Y-%m-%d") for t in d["time"].values])
    d0 = dt.date.fromisoformat(first)
    want = [(d0 + dt.timedelta(i)).isoformat()
            for i in range((dt.date.fromisoformat(last) - d0).days + 1)]
    got = [s for s in dates if first <= s <= last]
    if got != want:
        raise SystemExit(f"FATAL: CMOR days in {first}..{last} are not exactly one per day")
    tas = d["tas"].values[np.isin(dates, want)].mean(0)
    out = {"tas": wmean(tas, w), "tas_cells": int((w > 0).sum())}
    if era5:
        e = xr.open_dataset(era5)
        name = next((n for n in ("var167", "t2m", "2t") if n in e.data_vars), None)
        if name is None or e[name].sizes.get("time", 1) != 1:
            raise SystemExit(f"FATAL: {era5} has no single-time 2 m temperature")
        bnds = np.asarray(e["time_bnds"].values).reshape(-1).astype("datetime64[D]").astype(str)
        if bnds.tolist() != [first, last]:
            raise SystemExit(f"FATAL: {era5} averages {bnds.tolist()}, not {first}..{last}")
        v = np.squeeze(e[name].values)
        if (v.shape != tas.shape or not np.allclose(e["lat"].values, lat)
                or not np.allclose(e["lon"].values, d["lon"].values)):
            raise SystemExit(f"FATAL: ERA5 {v.shape} not on the CMOR grid {tas.shape}")
        out["era5"] = wmean(v, w)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("on_run")
    ap.add_argument("off_run")
    ap.add_argument("scratch")
    ap.add_argument("--window", nargs=2, required=True, metavar=("FIRST", "LAST"))
    ap.add_argument("--era5", default=None)
    ap.add_argument("--figures", default=None,
                    help="write ON-OFF window-mean maps (soil 0.175 m, tas) and a "
                         "zonal-mean depth section of soil T to this directory")
    ap.add_argument("--all-days-snow-check", action="store_true",
                    help="ON run: T_snow finite and <= T_freeze in every checkpoint")
    a = ap.parse_args(argv)
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.land.soil_grid import make_soil_grid
    from legoesm.land.soil_thermal import (compute_apparent_heat_capacity,
                                           compute_thermal_conductivity)
    sys.path.insert(0, os.environ.get(
        "LEGOESM_DIAG_LAND_PROBES",
        "/work/bd1083/b309178/diffESM/legoesm_pg/wt_diag_land/scripts/validate/amip_bias"))
    from land_deployed_lai_stress import build_driver

    d = build_driver(a.off_run, a.scratch)
    c = d.physics.land_ml_cfg
    g = make_soil_grid(c.soil_grid)
    z, dz, dzi = (np.asarray(x, np.float64) for x in (g.z_node, g.dz, g.dz_interface))
    lat = np.rad2deg(np.asarray(d.grid.latCell))
    f_land = np.asarray(d._f_land).reshape(-1)
    area = np.asarray(d.grid.areaCell)
    w = area * ((lat >= BAND[0]) & (lat <= BAND[1]) & (f_land >= 0.5))
    hw = overlap(dz, DEPTH_HEAT)
    print(f"soil nodes {np.round(z, 4).tolist()} m; 0.175 m between "
          f"{z[np.searchsorted(z, DEPTH_SOIL_T) - 1]:.4f} and {z[np.searchsorted(z, DEPTH_SOIL_T)]:.4f}; "
          f"top-1m overlap sums {hw.sum():.3f} m; region cells {int((w > 0).sum())}, "
          f"area {w.sum() / 1e12:.2f} Mkm2")
    pair = {}
    for run in (a.on_run, a.off_run):
        commit = open(f"{ROOT}/{run}/arm.tsv").read().split("\t")[3]
        import hashlib
        first = sorted(glob.glob(f"{ROOT}/{run}/checkpoint_day_*.npz"))[0]
        pair[run] = (commit, first.rsplit("_", 1)[1],
                     hashlib.sha256(open(first, "rb").read()).hexdigest())
    cfg = {r: flat_config(f"{ROOT}/{r}/experiment_config.json") for r in (a.on_run, a.off_run)}
    diff = config_diff(cfg[a.on_run], cfg[a.off_run]) - {"output.output_dir"}
    if (diff != {"land_snow_insulation"} or cfg[a.on_run]["land_snow_insulation"] is not True
            or cfg[a.off_run]["land_snow_insulation"] is not False):
        raise SystemExit(f"FATAL: configs differ in {sorted(diff)} or the ON/OFF switch is wrong")
    if pair[a.on_run] != pair[a.off_run]:
        raise SystemExit(f"FATAL: not a matched pair (commit, restart file): {pair}")
    print(f"pair commit {pair[a.on_run][0][:12]}, restart {pair[a.on_run][1]}")
    days = checkpoint_days(int(d.config.start_year), *a.window)
    # Cells failing the preregistered run-validity rule (snow-layer T above freezing
    # under snow) on ANY window day of the ON run; scored again without them.
    bad = np.zeros(lat.size, bool)
    for day in days:
        ck = np.load(f"{ROOT}/{a.on_run}/checkpoint_day_{day:04d}.npz")
        if "land_ml_T_snow" in ck.files:
            bad |= (ck["land_ml_snow_depth"] > 0) & (ck["land_ml_T_snow"] > constants.T_freeze)
    wv = w * ~bad
    print(f"invalid cells (ON T_snow > T_freeze under snow on a window day): "
          f"{int((bad & (w > 0)).sum())} of {int((w > 0).sum())} region cells, "
          f"{100 * wv.sum() / w.sum():.1f} % of region area kept")
    res, Tcol = {}, {}
    for run in (a.on_run, a.off_run):
        acc = {k: [] for k in ("T_0.175", "H_1m", "F_soil12", "SWE", "T_0.175 valid",
                               "H_1m valid", "F_soil12 valid", "SWE valid")}
        Tsum = 0.0
        for day in days:
            ck = np.load(f"{ROOT}/{run}/checkpoint_day_{day:04d}.npz")
            if abs(float(ck["day"]) - day) > 1e-6:
                raise SystemExit(f"FATAL: {run} checkpoint day {float(ck['day'])} != {day}")
            T, th = np.asarray(ck["land_ml_T_soil"]), np.asarray(ck["land_ml_theta_soil"])
            H = heat_from_freezing(lambda TT: np.asarray(compute_apparent_heat_capacity(
                jnp.asarray(TT), jnp.asarray(th), c.hydraulics, c.thermal)),
                T, constants.T_freeze) @ hw
            k = np.asarray(compute_thermal_conductivity(jnp.asarray(th), c.hydraulics, c.thermal))
            kh = 2.0 * k[:, 0] * k[:, 1] / (k[:, 0] + k[:, 1])
            Tsum = Tsum + T
            acc["T_0.175"].append(wmean(at_depth(T, z, DEPTH_SOIL_T), w))
            acc["H_1m"].append(wmean(H, w))
            acc["F_soil12"].append(wmean(kh * (T[:, 1] - T[:, 0]) / dzi[0], w))
            acc["SWE"].append(wmean(ck["land_ml_snow_depth"], w))
            acc["T_0.175 valid"].append(wmean(at_depth(T, z, DEPTH_SOIL_T), wv))
            acc["H_1m valid"].append(wmean(H, wv))
            acc["F_soil12 valid"].append(wmean(kh * (T[:, 1] - T[:, 0]) / dzi[0], wv))
            acc["SWE valid"].append(wmean(ck["land_ml_snow_depth"], wv))
        res[run] = {k: float(np.mean(v)) for k, v in acc.items()}
        Tcol[run] = Tsum / len(days)
        res[run].update(cmor_tas(run, *a.window, era5=a.era5))
    print(f"window {a.window[0]}..{a.window[1]} = checkpoint days {days[0]}-{days[-1]} "
          f"({len(days)}); tas CMOR cells {res[a.off_run]['tas_cells']}")
    on, off = res[a.on_run], res[a.off_run]
    for key, unit, s in (("T_0.175", "K", 1), ("tas", "K", 1), ("H_1m", "MJ/m2", 1e-6),
                         ("F_soil12", "W/m2", 1), ("SWE", "kg/m2", 1),
                         ("T_0.175 valid", "K", 1), ("H_1m valid", "MJ/m2", 1e-6),
                         ("F_soil12 valid", "W/m2", 1), ("SWE valid", "kg/m2", 1)):
        print(f"{key:14s} ON {on[key] * s:10.3f}  OFF {off[key] * s:10.3f}  "
              f"ON-OFF {(on[key] - off[key]) * s:+8.3f} {unit}")
    print(f"SWE ON/OFF {on['SWE'] / off['SWE']:.3f}")
    if a.era5:
        print(f"tas bias vs ERA5 same days: ON {on['tas'] - on['era5']:+.3f} K, "
              f"OFF {off['tas'] - off['era5']:+.3f} K (ERA5 {off['era5']:.3f} K)")
    if a.figures:
        figures(a, lat, np.rad2deg(np.asarray(d.grid.lonCell)) % 360, f_land, z,
                Tcol[a.on_run] - Tcol[a.off_run])
    if a.all_days_snow_check:
        worst, n, n_hot, swe_hot = -np.inf, 0, 0, 0.0
        is_land = f_land >= 0.5
        # the first file is the copied restart, written before the layer existed
        for f in sorted(glob.glob(f"{ROOT}/{a.on_run}/checkpoint_day_*.npz"))[1:]:
            ck = np.load(f)
            ts, swe = np.asarray(ck["land_ml_T_snow"]), np.asarray(ck["land_ml_snow_depth"])
            if not np.isfinite(ts[is_land]).all():
                raise SystemExit(f"FATAL: non-finite T_snow on land in {f}")
            snow = is_land & (swe > 0)
            if snow.any():
                worst = max(worst, float(ts[snow].max()))
            hot = snow & (ts > constants.T_freeze)
            n_hot += int(hot.sum())
            swe_hot = max(swe_hot, float(swe[hot].max()) if hot.any() else 0.0)
            n += 1
        ck_on = np.load(f"{ROOT}/{a.on_run}/checkpoint_day_{days[-1]:04d}.npz")
        ck_off = np.load(f"{ROOT}/{a.off_run}/checkpoint_day_{days[-1]:04d}.npz")
        hot = (is_land & (ck_on["land_ml_snow_depth"] > 0)
               & (ck_on["land_ml_T_snow"] > constants.T_freeze))
        print(f"day {days[-1]} above-freezing land snow cells {int(hot.sum())}: in scoring "
              f"region {int((hot & (w > 0)).sum())}; lat quartiles "
              f"{np.round(np.percentile(lat[hot], [0, 25, 50, 75, 100]), 1).tolist() if hot.any() else []}; "
              f"SWE median {np.median(ck_on['land_ml_snow_depth'][hot]) if hot.any() else 0:.3g} kg/m2; "
              f"top soil T median ON {np.median(ck_on['land_ml_T_soil'][hot, 0]) if hot.any() else 0:.1f} K, "
              f"OFF same cells {np.median(ck_off['land_ml_T_soil'][hot, 0]) if hot.any() else 0:.1f} K")
        import xarray as xr
        for run in (a.on_run, a.off_run):
            ts = xr.open_dataset(glob.glob(f"{ROOT}/{run}/cmor/Amon/ts_*.nc")[0])["ts"].values
            sft = xr.open_dataset(glob.glob(f"{ROOT}/{run}/cmor/fx/sftlf_*.nc")[0])["sftlf"].values
            print(f"{run} exported surface T (CMOR Amon ts, all land cells, not snow-only) max per month: "
                  f"{[round(float(m[sft >= 50].max()), 1) for m in ts]} K")
        print(f"ON T_snow finite on land (f_land >= 0.5) in {n} checkpoints; max on land "
              f"snow cells {worst - constants.T_freeze:+.4f} K vs freezing; above-freezing "
              f"land snow cell-days {n_hot}, their max SWE {swe_hot:.3g} kg/m2")
        print(f"RUN VALIDITY PROXY (snow node <= freezing under snow): "
              f"{'FAIL' if n_hot else 'PASS'}")


def figures(a, lat, lon, f_land, z, dT):
    """ON-OFF window means: soil 0.175 m map, CMOR tas map, zonal-mean soil T by depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    os.makedirs(a.figures, exist_ok=True)
    tag = f"{a.on_run}-{a.off_run}_{a.window[0]}_{a.window[1]}"
    land = f_land >= 0.5
    d175 = at_depth(dT, z, DEPTH_SOIL_T)
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(lon[land], lat[land], c=d175[land], s=2, cmap="RdBu_r", vmin=-15, vmax=15)
    ax.set(xlim=(0, 360), ylim=(20, 90), xlabel="lon", ylabel="lat",
           title=f"soil T at 0.175 m, ON - OFF [K], {a.window[0]}..{a.window[1]}")
    for y in BAND:
        ax.axhline(y, color="k", lw=0.5)
    fig.colorbar(sc)
    fig.savefig(f"{a.figures}/soil0175_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    tas = {}
    for run in (a.on_run, a.off_run):
        ds = xr.open_dataset(glob.glob(f"{ROOT}/{run}/cmor/day/tas_day_*.nc")[0], use_cftime=True)
        dates = np.array([t.strftime("%Y-%m-%d") for t in ds["time"].values])
        tas[run] = ds["tas"].values[(dates >= a.window[0]) & (dates <= a.window[1])].mean(0)
        glat, glon = ds["lat"].values, ds["lon"].values
    fig, ax = plt.subplots(figsize=(10, 4))
    pc = ax.pcolormesh(glon, glat, tas[a.on_run] - tas[a.off_run], cmap="RdBu_r",
                       vmin=-6, vmax=6, shading="nearest")
    ax.set(ylim=(20, 90), xlabel="lon", ylabel="lat",
           title=f"tas ON - OFF [K], {a.window[0]}..{a.window[1]}")
    for y in BAND:
        ax.axhline(y, color="k", lw=0.5)
    fig.colorbar(pc)
    fig.savefig(f"{a.figures}/tas_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    edges = np.arange(20, 92, 2.0)
    k = np.digitize(lat, edges) - 1
    sec = np.full((edges.size - 1, z.size), np.nan)
    for i in range(edges.size - 1):
        m = land & (k == i)
        if m.any():
            sec[i] = dT[m].mean(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    pc = ax.pcolormesh(0.5 * (edges[:-1] + edges[1:]), z, sec.T, cmap="RdBu_r",
                       vmin=-15, vmax=15, shading="nearest")
    ax.set(ylim=(z[-1], 0), xlabel="lat", ylabel="depth [m]", yscale="symlog",
           title="zonal-mean land soil T, ON - OFF [K]")
    fig.colorbar(pc)
    fig.savefig(f"{a.figures}/soilT_section_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"figures in {a.figures}")


if __name__ == "__main__":
    main()
