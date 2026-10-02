#!/usr/bin/env python3
"""Northern winter land STATE of an MPAS AMIP run against ERA5 at the same instants.

From the run's own checkpoints (instantaneous 00Z state, day N = start_year-01-01
00Z + N days) and ERA5 hourly analyses sampled at the SAME 00Z instants
(extracted to netCDF by ``extract.sh`` next to the ERA5 files, 30-90N, regular
Gaussian), over MPAS land cells (driver land fraction > 0.5 AND ERA5 lsm > 0.5 at
the nearest ERA5 point), weights = cell area x land fraction:

* snow water equivalent (model kg/m2 vs ERA5 sd x 1000) and the shares above
  1 / 10 kg/m2;
* soil temperature interpolated in depth to the ERA5 layer mid-depths
  (0.035, 0.175, 0.64, 1.945 m) vs stl1..stl4;
* the albedo the land EXPORTS and ABSORBS with (snow layered on the soil-colour
  bands, the shipped ``land_albedo`` + ``broadband_albedo`` path, zenith term off)
  vs ERA5 fal, and the deployed LAI/SAI and canopy snow-mask scale;
* top-soil minus 0.175 m soil temperature vs ERA5 skt - stl2 (no model skin is
  checkpointed; the top soil node is the ground temperature the canopy solve uses).

The driver is built for SETUP ONLY (no step) through
``land_deployed_lai_stress.build_driver`` (reused, path in $LEGOESM_DIAG_LAND_PROBES).
Run with the run's code on PYTHONPATH, JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1.
Non-finite values inside a region are fatal.  The probe prints numbers only.

Usage: nh_winter_land_state.py <run> <scratch> <era5_dir> --days 341 364 [--every 1]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.environ.get("LEGOESM_AMIP_RUNS",
                      "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
sys.path.insert(0, os.environ.get(
    "LEGOESM_DIAG_LAND_PROBES",
    "/work/bd1083/b309178/diffESM/legoesm_pg/wt_diag_land/scripts/validate/amip_bias"))
from land_deployed_lai_stress import build_driver  # noqa: E402

ERA5_MID = np.array([0.035, 0.175, 0.64, 1.945])   # m, layers 0-7/7-28/28-100/100-289 cm
REGIONS = {  # name: (lat0, lat1, lon0, lon1) degrees east
    "45-70N": (45, 70, 0, 360), "Siberia": (50, 70, 60, 140),
    "Canada": (50, 70, 220, 300), "Alaska": (60, 70, 190, 220),
}


def wmean(x, w):
    on = w > 0
    x = np.asarray(x, np.float64)
    if not on.any():
        raise SystemExit("FATAL: empty region")
    if not np.isfinite(x[on]).all():
        raise SystemExit("FATAL: non-finite value inside region")
    return float(np.sum(np.where(on, x, 0.0) * w) / np.sum(w))


def nearest_index(e5, lat, lon):
    """Index of the nearest ERA5 point on the regular grid (lat descending ok)."""
    la, lo = e5["lat"].values, e5["lon"].values
    i = np.abs(la[None, :] - lat[:, None]).argmin(1)
    j = np.abs(((lo[None, :] - lon[:, None] + 180) % 360) - 180).argmin(1)
    return i, j


def era5_at(era5_dir, code, dates, i, j):
    d = xr.open_dataset(f"{era5_dir}/e5_{code}.nc")
    v = d[f"var{int(code)}"].squeeze(drop=True)
    path = f"e5_{code}"
    t = v["time"].values.astype("datetime64[D]").astype(str)
    out = []
    for s in dates:
        k = np.flatnonzero(t == s)
        if k.size != 1:
            raise SystemExit(f"FATAL: {path} has {k.size} records for {s}")
        out.append(np.asarray(v.isel(time=int(k[0])).values)[i, j])
    return np.array(out)


def interp_depth(T, z):
    """(ncol, nz) model soil T -> (ncol, 4) at ERA5 mid-depths, linear in depth."""
    return np.stack([np.array([np.interp(zz, z, row) for row in T]) for zz in ERA5_MID], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("scratch")
    ap.add_argument("era5")
    ap.add_argument("--days", nargs=2, type=int, required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--series", action="store_true",
                    help="also print one row per checkpoint day per region")
    ap.add_argument("--surfdata", default=None,
                    help="CLM surfdata (PCT_NATVEG, PCT_NAT_PFT): adds a 45-70N region of "
                         "cells whose nearest surfdata cell is >50%% trees (PFTs 1-8)")
    a = ap.parse_args(argv)
    import jax.numpy as jnp
    from legoesm.land.canopy.radiative_transfer import broadband_albedo
    from legoesm.land.soil_albedo import rewet_soil_bands
    from legoesm.land.soil_grid import make_soil_grid
    from legoesm.surface_albedo import land_albedo

    d = build_driver(a.run, a.scratch)
    p, c = d.physics.land_ml_params, d.physics.land_ml_cfg
    lat = np.rad2deg(np.asarray(d.grid.latCell))
    lon = np.rad2deg(np.asarray(d.grid.lonCell)) % 360
    f_land = np.asarray(d._f_land).reshape(-1)
    area = np.asarray(d.grid.areaCell)
    z = np.asarray(make_soil_grid(c.soil_grid).z_node)
    la = c.land_albedo
    print(f"SAI provided: {getattr(p, 'SAI', None) is not None}")
    print(f"code {d.__class__.__module__}; soil nodes {np.round(z, 3).tolist()} m")
    print(f"snow albedo max/min {la.alpha_snow_max:.3f}/{la.alpha_snow_min:.3f}, "
          f"tau {la.tau_snow_decay / 86400:.1f} d, crit {la.snow_depth_crit:.1f} kg/m2, "
          f"activation {la.snow_age_activation_K} K, zenith factor {la.snow_zenith_factor}")
    sel = (lat > 25)
    e5 = xr.open_dataset(f"{a.era5}/e5_172.nc")
    ii, jj = nearest_index(e5, lat[sel], lon[sel])
    lsm = np.full(lat.size, np.nan)
    lsm[sel] = np.asarray(e5["var172"].isel(time=0).values)[ii, jj]
    # Ice sheets / glaciers: ERA5 caps snow at 10 m w.e.; any point at >= 9.9 m on
    # the first window day is glacier and would dominate an SWE mean -> excluded.
    sd0 = np.full(lat.size, np.nan)
    sd0[sel] = era5_at(a.era5, "141", [(dt.date(int(d.config.start_year), 1, 1)
                                         + dt.timedelta(a.days[0])).isoformat()],
                       ii, jj)[0]
    land = (f_land > 0.5) & (lsm > 0.5) & (sd0 < 9.9)
    print(f"glacier points excluded (ERA5 sd >= 9.9 m): "
          f"{int(((f_land > 0.5) & (lsm > 0.5) & (sd0 >= 9.9) & (lat > 45)).sum())} cells >45N")
    w_reg = {k: area * f_land * (land & (lat >= r[0]) & (lat <= r[1])
                                 & (lon >= r[2]) & (lon <= r[3]))
             for k, r in REGIONS.items()}

    if a.surfdata:
        sd = xr.open_dataset(a.surfdata)
        tree = (sd["PCT_NATVEG"].values / 100.0
                * sd["PCT_NAT_PFT"].values[1:9].sum(0) / 100.0)
        si = np.abs(sd["LATIXY"].values[:, 0][None, :] - lat[:, None]).argmin(1)
        sj = np.abs(((sd["LONGXY"].values[0][None, :] - lon[:, None] + 180) % 360)
                    - 180).argmin(1)
        tf = tree[si, sj]
        w_reg["forest>50%"] = area * f_land * (land & (lat >= 45) & (lat <= 70) & (tf > 0.5))
        w_reg["nonforest<10%"] = area * f_land * (land & (lat >= 45) & (lat <= 70) & (tf < 0.1))
    lai = np.asarray(p.LAI)
    sai = (np.asarray(p.SAI) if getattr(p, "SAI", None) is not None
           else np.zeros_like(lai))
    scl = (np.ones_like(lat) if la.snow_cover_scale is None
           else np.asarray(la.snow_cover_scale).reshape(-1))
    for k, w in w_reg.items():
        print(f"[{k}] cells {int((w > 0).sum())}: LAI {wmean(lai, w):.2f} SAI "
              f"{wmean(sai, w):.2f} share LAI+SAI>2 {wmean((lai + sai) > 2, w):.2f} "
              f"snow-mask scale {wmean(scl, w):.3f}")

    base = dt.date(int(d.config.start_year), 1, 1)
    days = list(range(a.days[0], a.days[1] + 1, a.every))
    dates = [(base + dt.timedelta(n)).isoformat() for n in days]
    E = {v: era5_at(a.era5, p_, dates, ii, jj) for v, p_ in (
        ("sd", "141"), ("stl1", "139"), ("stl2", "170"), ("stl3", "183"),
        ("stl4", "236"), ("skt", "235"), ("fal", "243"), ("t2m", "167"))}
    rows = {k: [] for k in w_reg}
    for n, day in enumerate(days):
        ck = np.load(f"{ROOT}/{a.run}/checkpoint_day_{day:04d}.npz")
        if abs(float(ck["day"]) - day) > 1e-6:
            raise SystemExit(f"FATAL: checkpoint day {float(ck['day'])} != {day}")
        swe, age = np.asarray(ck["land_ml_snow_depth"]), np.asarray(ck["land_ml_snow_age"])
        Ts, th = np.asarray(ck["land_ml_T_soil"]), np.asarray(ck["land_ml_theta_soil"])
        Tz = interp_depth(Ts[sel], z)
        lp = rewet_soil_bands(p, jnp.asarray(th[:, 0]))
        band = lambda b: land_albedo(jnp.asarray(np.deg2rad(lat)), jnp.asarray(swe),
                                     jnp.asarray(age), la, base_albedo=b)
        alb = np.asarray(broadband_albedo(band(lp.ALB_VIS), band(lp.ALB_NIR)))
        alb0 = np.asarray(broadband_albedo(lp.ALB_VIS, lp.ALB_NIR))
        T_air = np.asarray(ck["T"])[:, -1]
        full = lambda x: (lambda o: (o.__setitem__(sel, x), o)[1])(np.full(lat.size, np.nan))
        m = {"swe": swe, "f>1": swe > 1, "f>10": swe > 10,
             "e5swe": full(E["sd"][n] * 1000), "e5f>1": full(E["sd"][n] * 1000 > 1),
             "e5f>10": full(E["sd"][n] * 1000 > 10)}
        for q in range(4):
            m[f"T{q + 1}"] = full(Tz[:, q])
            m[f"e5T{q + 1}"] = full(E[f"stl{q + 1}"][n])
        m.update({"Ttop": Ts[:, 0], "Ttop-T2": full(Ts[sel, 0] - Tz[:, 1]),
                  "e5skt-stl2": full(E["skt"][n] - E["stl2"][n]), "e5skt": full(E["skt"][n]),
                  "Tair1": T_air, "e5t2m": full(E["t2m"][n]),
                  "alb": alb, "alb_nosnow": alb0, "e5fal": full(E["fal"][n])})
        for k, w in w_reg.items():
            rows[k].append({kk: wmean(v, w) for kk, v in m.items()})
    print(f"window days {days[0]}-{days[-1]} ({dates[0]}..{dates[-1]}, 00Z instants, "
          f"every {a.every}); model run {a.run}")
    keys = list(rows[next(iter(rows))][0])
    for k in w_reg:
        mean = {kk: np.mean([r[kk] for r in rows[k]]) for kk in keys}
        print(f"[{k}] SWE {mean['swe']:.1f} vs ERA5 {mean['e5swe']:.1f} kg/m2 | "
              f"share>1 {mean['f>1']:.2f}/{mean['e5f>1']:.2f} share>10 "
              f"{mean['f>10']:.2f}/{mean['e5f>10']:.2f}")
        print(f"[{k}] soil T model-ERA5 at 0.035/0.175/0.64/1.945 m: " + " ".join(
            f"{mean[f'T{q}'] - mean[f'e5T{q}']:+.1f}" for q in range(1, 5))
            + "  (ERA5 " + " ".join(f"{mean[f'e5T{q}']:.1f}" for q in range(1, 5)) + ")")
        print(f"[{k}] top-soil - 0.175 m: model {mean['Ttop-T2']:+.2f} K; ERA5 skt-stl2 "
              f"{mean['e5skt-stl2']:+.2f} K | model top soil {mean['Ttop']:.1f}, ERA5 skt "
              f"{mean['e5skt']:.1f} | lowest air {mean['Tair1']:.1f} vs ERA5 t2m {mean['e5t2m']:.1f}")
        print(f"[{k}] albedo exported {mean['alb']:.3f} (snow-free bands {mean['alb_nosnow']:.3f})"
              f" vs ERA5 fal {mean['e5fal']:.3f}")
        if a.series:
            for day_, r in zip(days, rows[k]):
                print(f"  [{k}] day {day_:3d} SWE {r['swe']:6.1f}/{r['e5swe']:6.1f}  "
                      f"dT 0.035 {r['T1'] - r['e5T1']:+6.1f} 0.64 {r['T3'] - r['e5T3']:+6.1f}"
                      f" 1.945 {r['T4'] - r['e5T4']:+6.1f}  air-t2m "
                      f"{r['Tair1'] - r['e5t2m']:+6.1f}  alb {r['alb']:.2f}/{r['e5fal']:.2f}")
        tr = [r["T1"] - r["e5T1"] for r in rows[k]]
        print(f"[{k}] stl1 bias first/last day {tr[0]:+.1f}/{tr[-1]:+.1f}")


if __name__ == "__main__":
    main()
