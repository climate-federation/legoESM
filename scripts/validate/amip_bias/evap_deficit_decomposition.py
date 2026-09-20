#!/usr/bin/env python3
"""Where the ocean evaporation deficit lives: wind, humidity gradient, or coefficient.

Bulk evaporation is LH = rho * L_v * C_E * |U| * (q_sfc - q_air), so the ratio
of the model's flux to a reference's factors into a wind ratio, a humidity
difference ratio, and a residual that is the transfer coefficient (plus
density, plus everything the factorisation hides).  Codex named this as the
deciding measurement: does the air state explain the deficit, and which
variable?

Reference is MERRA2, which publishes the flux, the 10 m wind, the 2 m humidity
and the skin temperature, so every factor comes from ONE product.  Model side
is the lowest full level (~135 m) and the run's own prescribed SST with the
sea-water factor.  The height mismatch is stated in the output and NOT
corrected away: a log profile makes the model's wind read ~20% higher than a
10 m value would, and its humidity slightly lower; both are in the direction
that makes the model's deficit look SMALLER than it is.

Usage: evap_deficit_decomposition.py <run>
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from surface_humidity_deficit import _prescribed_sst  # noqa: E402

MERRA = "/work/bd1179/b309141/climateeval_input/reanalysis_MERRA2/mon"
BANDS = {"tropical ocean 20S-20N": (-20, 20, 0, 360),
         "trades 10-30N": (10, 30, 0, 360),
         "trades 10-30S": (-30, -10, 0, 360),
         "global": (-90, 90, 0, 360)}


def merra(var, months, mlat, mlon):
    """MERRA2 climatology on the model grid without dask: one file per year,
    per-file monthly means accumulated with equal year weights."""
    fs = sorted(glob.glob(f"{MERRA}/{var}/*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: no MERRA2 {var}")
    acc, n = None, 0
    for f in fs:
        v = xr.open_dataset(f)[var]
        if int(str(np.asarray(v.time)[0])[:4]) < rb.REF_MIN_YEAR:
            continue
        sel = v.groupby("time.month").mean("time")
        have = set(np.asarray(sel["month"]).tolist())
        got = [m for m in months if m in have]
        if not got:
            continue
        a = np.asarray(sel.sel(month=got).mean("month"), dtype=np.float64)
        acc = a if acc is None else acc + a
        n += 1
    if n == 0:
        raise SystemExit(f"FATAL: no MERRA2 {var} years >= {rb.REF_MIN_YEAR}")
    d0 = xr.open_dataset(fs[0])
    rlat = np.asarray(d0["lat"], dtype=np.float64)
    rlon = np.asarray(d0["lon"], dtype=np.float64) % 360.0
    arr = acc / n
    if arr.shape != (rlat.size, rlon.size):
        arr = arr.T
    return np.asarray(rb.bin_to_model(arr, rlat, rlon, mlat, mlon, label=var))


def main(run):
    from legoesm import constants
    from legoesm.thermo import saturation_specific_humidity

    du, dv = rb._load_model(run, "ua"), rb._load_model(run, "va")
    dq = rb._load_model(run, "hus")
    lat, lon = np.asarray(du.lat), np.asarray(du.lon)
    plev = np.asarray(du["plev"], dtype=np.float64)
    k = int(np.argmin(np.abs(plev - 100000.0)))
    months = rb._month_labels(du)
    U_m = np.hypot(np.asarray(du["ua"]).mean(0)[k], np.asarray(dv["va"]).mean(0)[k])
    q_m = np.asarray(dq["hus"]).mean(0)[k]
    ps = np.asarray(rb._load_model(run, "ps")["ps"]).mean(0)
    sst = _prescribed_sst(run, months, lat, lon)
    qs_m = np.asarray(saturation_specific_humidity(sst, ps)) * constants.q_sat_saline_fraction
    lh_m = np.asarray(rb._load_model(run, "hfls")["hfls"]).mean(0)

    U_o = np.hypot(merra("uas", months, lat, lon), merra("vas", months, lat, lon))
    q_o = merra("huss", months, lat, lon)
    ts_o = merra("ts", months, lat, lon)
    ps_o = merra("ps", months, lat, lon)
    qs_o = np.asarray(saturation_specific_humidity(ts_o, ps_o)) * constants.q_sat_saline_fraction
    lh_o = merra("hfls", months, lat, lon)

    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf")
    d = xr.open_dataset(fs[0])
    fl = np.asarray(rb.bin_to_model(np.asarray(d["sftlf"], dtype=np.float64) / 100.0,
                                    np.asarray(d.lat), np.asarray(d.lon) % 360.0,
                                    lat, lon, label="sftlf"))
    ocean = (fl < 0.5) & (ps > 100500.0) & np.isfinite(sst) & np.isfinite(lh_o)

    dq_m, dq_o = qs_m - q_m, qs_o - q_o
    print(f"{run}: {100 * ocean.mean():.1f}% valid ocean columns, months {months}")
    print(f"{'band':<24}{'LH ratio':>9}{'U ratio':>9}{'dq ratio':>9}{'resid':>8}"
          f"{'  dq mdl':>9}{'dq M2':>8}{'q_air mdl':>10}{'q_air M2':>9}")
    for name, box in BANDS.items():
        r = lambda f: rb.region_mean(f, lat, lon, box, valid=ocean)  # noqa: E731
        lh, u, dqr = r(lh_m) / r(lh_o), r(U_m) / r(U_o), r(dq_m) / r(dq_o)
        print(f"{name:<24}{lh:9.2f}{u:9.2f}{dqr:9.2f}{lh / (u * dqr):8.2f}"
              f"{1e3 * r(dq_m):9.2f}{1e3 * r(dq_o):8.2f}{1e3 * r(q_m):10.2f}"
              f"{1e3 * r(q_o):9.2f}")
    print("\nratios = model / MERRA2 of band means; resid = LH / (U * dq), i.e. the "
          "transfer coefficient and\neverything else. dq and q in g/kg. Model wind "
          "and humidity are at ~135 m, MERRA2's at 10 m and 2 m:\nthe log profile "
          "puts the model's U ratio ~1.2 too HIGH and its q_air slightly LOW, "
          "so the\nmodel's true wind excess is larger and its humidity excess "
          "larger still than printed.")


if __name__ == "__main__":
    main(sys.argv[1])
