"""Is an MPAS energy LEAK real, or a diurnal-sampling artefact? (#1354/#1353)

MANDATORY before quoting any number from ``mpas_energy_budget_leak.py``.

The lightweight series samples the energy channels AT a diagnostic step.  This
run subcycles radiation -- RRTMGP is solved every 32 steps (1 h) and the heating
is HELD in between -- so a sampled TOA flux is whatever was last solved, not the
time mean the model actually applied.  A constant offset between the two is
exactly what a leak looks like, so the two must be compared before the leak is
believed.

CMOR Amon output is properly time-accumulated, so its global means are the
applied values.  This compares them against the sampled series over the SAME
window.  Area weights are cos(lat) -- an unweighted mean on a lat-lon grid is
not a global mean.

Numbers only.
"""
import glob
import os
import sys

import numpy as np
import netCDF4 as nc

import argparse

_P = argparse.ArgumentParser(description=__doc__)
_P.add_argument("run_dir", help="run output directory (holds timeseries.npz and cmor/)")
ROOT = _P.parse_args().run_dir


def gmean(var):
    """Area-weighted global mean of a CMOR Amon field [time, lat, lon]."""
    f = glob.glob(f"{ROOT}/cmor/Amon/{var}_Amon_*.nc")
    if not f:
        return None
    with nc.Dataset(f[0]) as ds:
        a = np.asarray(ds.variables[var][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
    w = np.cos(np.deg2rad(lat))[None, :, None]
    w = np.broadcast_to(w, a.shape)
    m = np.isfinite(a)
    return float((a[m] * w[m]).sum() / w[m].sum())


print("CMOR Amon area-weighted global means [W/m^2] (time-accumulated):")
vals = {}
for v in ("rsdt", "rsut", "rlut", "hfss", "hfls"):
    vals[v] = gmean(v)
    print(f"  {v:8s} {vals[v]:10.3f}" if vals[v] is not None else f"  {v:8s}  MISSING")

toa_cmor = vals["rsdt"] - vals["rsut"] - vals["rlut"]
print(f"  toa_net (rsdt - rsut - rlut) = {toa_cmor:.3f}")

print()
print("Sampled series means over the same window (day > 1):")
d = np.load(f"{ROOT}/timeseries.npz", allow_pickle=True)
days = np.asarray(d["days"], dtype=float)
k = days > 1.0
k[0] = False
for v in ("energy_toa_net", "sw_up_toa", "lw_up_toa", "hfss", "hfls",
          "sw_net_sfc", "lw_net_sfc", "energy_dE_dt", "energy_residual"):
    if v in d:
        a = np.asarray(d[v], dtype=float)[k]
        print(f"  {v:16s} {np.nanmean(a):10.3f}")

print()
print("THE COMPARISON (sampled minus CMOR):")
samp_toa = float(np.nanmean(np.asarray(d["energy_toa_net"], dtype=float)[k]))
print(f"  toa_net   sampled {samp_toa:9.3f}  CMOR {toa_cmor:9.3f}  "
      f"diff {samp_toa - toa_cmor:9.3f}")
for v in ("hfss", "hfls"):
    s = float(np.nanmean(np.asarray(d[v], dtype=float)[k]))
    print(f"  {v:8s}  sampled {s:9.3f}  CMOR {vals[v]:9.3f}  "
          f"diff {s - vals[v]:9.3f}")

print()
print("Implied LEAK if the CMOR toa_net is substituted for the sampled one")
print("(everything else held): LEAK = sfc_rad - hfss - hfls - (toa - dE/dt)")
sfc = (np.asarray(d["sw_net_sfc"], dtype=float)
       + np.asarray(d["lw_net_sfc"], dtype=float))[k]
dedt = np.asarray(d["energy_dE_dt"], dtype=float)[k]
sh = np.asarray(d["hfss"], dtype=float)[k]
lh = np.asarray(d["hfls"], dtype=float)[k]
leak_sampled = sfc - sh - lh - (np.asarray(d["energy_toa_net"], dtype=float)[k] - dedt)
leak_cmor_toa = sfc - sh - lh - (toa_cmor - dedt)
print(f"  LEAK with sampled toa_net : {np.nanmean(leak_sampled):8.3f}")
print(f"  LEAK with CMOR    toa_net : {np.nanmean(leak_cmor_toa):8.3f}")
