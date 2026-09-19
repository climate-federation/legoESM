#!/usr/bin/env python3
"""Why the model evaporates too little: the air-sea humidity difference.

Bulk evaporation is proportional to the air-sea humidity difference, and with
the sea surface prescribed that difference is set by how moist the air just
above it is.  Over ocean, where the air temperature sits close to the surface
temperature, the difference scales with (1 - RH) of the near-surface air, so a
small relative-humidity excess is a large fractional cut in evaporation --
0.84 against 0.78 is a quarter of the flux.

This scores 1000 hPa relative humidity over OCEAN against ERA5 on the same
bands as the water budget, and prints the saturation deficit ratio next to the
evaporation ratio the flux diagnostics report, so the two can be compared
directly.  Nothing here replaces a bulk-flux calculation with real winds and a
real exchange coefficient; it isolates the humidity factor alone.

Usage: surface_humidity_deficit.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402

from legoesm.thermo import saturation_specific_humidity  # noqa: E402

BANDS = {"ITCZ 10S-10N": (-10.0, 10.0), "trades 10-30N": (10.0, 30.0),
         "trades 10-30S": (-30.0, -10.0), "NH midlat 30-60N": (30.0, 60.0),
         "SO stormtrack 60-30S": (-60.0, -30.0), "global": (-90.0, 90.0)}
PLEV = 100000.0


def _at_1000(d, var):
    plev = np.asarray(d["plev"], dtype=np.float64)
    k = int(np.argmin(np.abs(plev - PLEV)))
    if abs(plev[k] - PLEV) > 1.0:
        raise SystemExit(f"FATAL: no 1000 hPa level in {var} (closest "
                         f"{plev[k]:.0f} Pa) -- refusing to score a "
                         f"different level against ERA5's")
    return np.asarray(d[var]).mean(axis=0)[k]


def _ref_1000(var, months, mlat, mlon):
    """ERA5 field at 1000 hPa, binned onto the model grid.

    The shared reference loader bins a 2-D field; these are 3-D, so the level
    is selected FIRST and the same binning is then reused rather than
    re-implemented.
    """
    fs = sorted(glob.glob(f"{rb.ERA5}/{var}/*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: no ERA5 {var}")
    d = xr.open_mfdataset(fs, combine="by_coords") if len(fs) > 1 \
        else xr.open_dataset(fs[0])
    v = d[var].sel(time=slice(f"{rb.REF_MIN_YEAR}-01-01", None))
    lev = [c for c in ("plev", "level", "pressure_level") if c in v.dims]
    if not lev:
        raise SystemExit(f"FATAL: ERA5 {var} has no level dimension")
    p = np.asarray(d[lev[0]], dtype=np.float64)
    scale = 1.0 if p.max() > 2000.0 else 100.0     # hPa files exist in the wild
    k = int(np.argmin(np.abs(p * scale - PLEV)))
    if abs(p[k] * scale - PLEV) > 1.0:
        raise SystemExit(f"FATAL: ERA5 {var} has no 1000 hPa level")
    clim = v.isel({lev[0]: k}).groupby("time.month").mean("time") \
            .sel(month=months).mean("month").load()
    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rlon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    arr = np.asarray(clim, dtype=np.float64)
    if arr.shape != (rlat.size, rlon.size):
        arr = arr.T
    return np.asarray(rb.bin_to_model(arr, rlat, rlon, mlat, mlon, label=var))


def _prescribed_sst(run, months, mlat, mlon):
    """The run's OWN prescribed sea surface temperature [K], binned to the
    model grid.  Read from the forcing file named in the run manifest, not a
    default path: this is the boundary condition both the model and the
    reference evaporation see, so it is the one term that is genuinely shared
    and it must come from the run rather than from an assumption."""
    man = json.load(open(f"{rb.ROOT}/{run}/run_manifest.json"))
    cmd = man["run"]["command_line"].split()
    if "--forcing-path" not in cmd:
        raise SystemExit(f"FATAL: {run} manifest names no --forcing-path")
    path = cmd[cmd.index("--forcing-path") + 1]
    off = float(cmd[cmd.index("--sst-offset") + 1]) if "--sst-offset" in cmd else 0.0
    d = xr.open_dataset(path, decode_times=True)
    var = "tosbcs" if "tosbcs" in d else "tos"
    clim = d[var].groupby("time.month").mean("time").sel(month=months).mean("month").load()
    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rlon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    arr = np.asarray(clim, dtype=np.float64)
    if arr.shape != (rlat.size, rlon.size):
        arr = arr.T
    return np.asarray(rb.bin_to_model(arr, rlat, rlon, mlat, mlon,
                                      label="sst", allow_gaps=True)) + off


def _box_mask(mlat, mlon, box):
    la0, la1, _lo0, _lo1 = box
    g = np.meshgrid(mlat, mlon, indexing="ij")[0]
    return (g >= la0) & (g <= la1)


def _ocean_mask(run, mlat, mlon):
    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf")
    d = xr.open_dataset(fs[0])
    frac = np.asarray(d["sftlf"]) / 100.0
    glat, glon = np.asarray(d.lat), np.asarray(d.lon) % 360.0
    i = np.abs(glat[:, None] - mlat[None, :]).argmin(axis=0)
    j = np.abs(((glon[:, None] - mlon[None, :] + 180.0) % 360.0) - 180.0).argmin(axis=0)
    return frac[np.ix_(i, j)] < 0.5


def _report(run, flux_run=None):
    flux_run = flux_run or run
    hus, ta = rb._load_model(run, "hus"), rb._load_model(run, "ta")
    if hus is None or ta is None:
        raise SystemExit(f"FATAL: {run} publishes no hus/ta")
    mlat, mlon = np.asarray(hus.lat), np.asarray(hus.lon)
    months = rb._month_labels(hus)
    q_m, t_m = _at_1000(hus, "hus"), _at_1000(ta, "ta")
    q_o, t_o = _ref_1000("hus", months, mlat, mlon), _ref_1000("ta", months, mlat, mlon)

    rh_m = q_m / saturation_specific_humidity(t_m, PLEV)
    rh_o = q_o / saturation_specific_humidity(t_o, PLEV)
    ocean = _ocean_mask(run, mlat, mlon)
    # 1000 hPa is about 125 m up where the surface is at 1015, and it is
    # UNDERGROUND wherever surface pressure falls below it -- most of the
    # Southern Ocean in the monthly mean.  Both reviewers refused the southern
    # row for that reason, and a reference that extrapolates below its own
    # surface is not a measurement.  Keep only columns where BOTH sides have
    # real air at this level, and print how much of each band survives.
    ps_m = rb._load_model(run, "ps")
    if ps_m is None:
        raise SystemExit(f"FATAL: {run} publishes no ps")
    ps_mod = np.asarray(ps_m["ps"]).mean(axis=0)
    ps_ref = rb._ref_clim("ps", months, mlat, mlon, src=rb.ERA5)
    if ps_ref is None:
        raise SystemExit("FATAL: no ERA5 surface pressure")
    above = ocean & (ps_mod > PLEV + 500.0) & (np.asarray(ps_ref) > PLEV + 500.0)

    ev = rb._load_model(flux_run, "evspsbl")
    evm = np.asarray(ev["evspsbl"]).mean(axis=0) if ev is not None else None
    evo = rb._ref_clim("evspsbl", months, mlat, mlon)

    print(f"\n=== {run}: 1000 hPa air-sea humidity difference over OCEAN "
          f"against the run's own prescribed sea surface ===")
    # RH alone cannot tell a moist bias from a cold one: the same excess
    # appears if the air holds more water or if it is colder at the same
    # water.  The two have opposite consequences for the bulk flux, so the
    # humidity and the temperature are printed separately beside it.
    sst = _prescribed_sst(flux_run, months, mlat, mlon)
    q_sea = saturation_specific_humidity(sst, PLEV) * 0.98   # saline surface
    print(f"{'band':<22}{'q[g/kg]':>9}{'qobs':>8}{'dq':>8}{'dqobs':>8}"
          f"{'ratio':>8}{'E/Eobs':>8}{'implied':>9}{'kept':>8}")
    for name, (lo, hi) in BANDS.items():
        box = (lo, hi, 0, 360)
        a = rb.region_mean(rh_m, mlat, mlon, box, valid=above)
        b = rb.region_mean(rh_o, mlat, mlon, box, valid=above)
        er = np.nan
        if evm is not None and evo is not None:
            er = (rb.region_mean(evm, mlat, mlon, box, valid=above)
                  / rb.region_mean(np.asarray(evo), mlat, mlon, box, valid=above))
        qa = rb.region_mean(q_m, mlat, mlon, box, valid=above) * 1e3
        qb = rb.region_mean(q_o, mlat, mlon, box, valid=above) * 1e3
        dq = rb.region_mean(q_sea - q_m, mlat, mlon, box, valid=above) * 1e3
        dqo = rb.region_mean(q_sea - q_o, mlat, mlon, box, valid=above) * 1e3
        kept = float((above & _box_mask(mlat, mlon, box)).sum()
                     / max(1, (ocean & _box_mask(mlat, mlon, box)).sum()))
        print(f"{name:<22}{qa:9.3f}{qb:8.3f}{dq:8.3f}{dqo:8.3f}"
              f"{dq / dqo:8.3f}{er:8.3f}{er / (dq / dqo):9.3f}{kept:8.2f}")
    print("dq is the real air-sea humidity difference against the run's OWN "
          "prescribed sea surface, which both sides share, so it replaces the "
          "relative-humidity proxy: a proxy on RH cannot tell a moist bias "
          "from a cold one, and the near-surface air here is moister in every "
          "band but warmer only in the tropics.")
    print("implied = the compensation the rest of the bulk formula (wind, "
          "stability, gustiness, exchange coefficient) must be supplying.")
    print("_old_ratio = the model's saturation deficit divided by the reference's, "
          "which is the factor the humidity term alone puts on evaporation. "
          "E/Eobs is what the flux diagnostics actually report: if the two "
          "columns agree, humidity explains the evaporation deficit on its "
          "own and neither the wind nor the exchange coefficient is implicated.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for r in sys.argv[1:]:
        _report(r)
