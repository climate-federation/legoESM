#!/usr/bin/env python3
"""Regional reduction of a per-column process ledger (``budget_ledger_columns.npz``).

The global ledger table hides WHERE a term acts. This reduces the per-column
rows to area-weighted means over the tropics split by surface (fractional
``sftlf`` from the run's own published fx file, nearest-neighbour onto the
MPAS mesh), in kg/m2/day. It also prints the checkpoint's instantaneous
convective rain source (``physstate_conv_precip``, the post-sub-cloud-
evaporation survivor that enters the ``q_r`` tracer) so the sedimentation
rows can be read against the rain the convection scheme actually hands over.

Usage: ledger_regional.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from cloud_layers import mesh_coords  # noqa: E402

SEC_PER_DAY = 86400.0
BOXES = {"ITCZ 10S-10N": (-10.0, 10.0), "trades 10-30N": (10.0, 30.0),
         "trades 10-30S": (-30.0, -10.0), "NH midlat 30-60N": (30.0, 60.0),
         "SO stormtrack 60-30S": (-60.0, -30.0), "global": (-90.0, 90.0)}


def _sftlf_on_mesh(run, lat, lon):
    # A short branch run has not published a monthly stream yet, so its land
    # fraction has to be named EXPLICITLY from the parent it branched off.
    # Falling back to some other run's mask silently would be exactly the
    # hidden choice that makes two tables incomparable.
    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf -- name the run "
                         f"whose land mask to use with --sftlf-from <run>")
    d = xr.open_dataset(fs[0])
    frac = np.asarray(d["sftlf"]) / 100.0
    glat, glon = np.asarray(d.lat), np.asarray(d.lon) % 360.0
    i = np.abs(glat[None, :] - lat[:, None]).argmin(axis=1)
    j = np.abs(((glon[None, :] - lon[:, None] + 180.0) % 360.0) - 180.0).argmin(axis=1)
    return frac[i, j]


def _report(run, sftlf_run=None):
    rundir = f"{rb.ROOT}/{run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat, lon, area = mesh_coords(exp)
    d = np.load(f"{rundir}/budget_ledger_columns.npz", allow_pickle=True)
    rates = np.asarray(d["ledger_rates"])[:, :, 0] * SEC_PER_DAY   # water, kg/m2/day
    procs = [str(p) for p in d["processes"]]
    if not np.all(np.isfinite(rates)):
        raise SystemExit(f"FATAL: non-finite ledger rates in {run}")
    fl = _sftlf_on_mesh(sftlf_run or run, lat, lon)
    cks = sorted(glob.glob(f"{rundir}/checkpoint_day_*.npz"))
    ck = np.load(cks[-1], allow_pickle=True)
    pconv = np.asarray(ck["physstate_conv_precip"]) * SEC_PER_DAY if "physstate_conv_precip" in ck else None
    print(f"\n=== {run}: per-column ledger, day {float(d['day']):.1f}, {int(d['n_steps'])} steps "
          f"[kg/m2/day, area-weighted] ===")
    hdr = f"{'region':<26}" + "".join(f"{p[:12]:>13}" for p in procs) + f"{'conv_src':>13}"
    print(hdr)
    for name, (lo, hi) in BOXES.items():
        box = (lat >= lo) & (lat <= hi)
        for surf, wsurf in (("all", np.ones_like(fl)), ("ocean", 1.0 - fl), ("land", fl)):
            w = area * box * wsurf
            if w.sum() <= 0.0:
                continue
            row = (rates * w[:, None]).sum(0) / w.sum()
            pc = (pconv * w).sum() / w.sum() if pconv is not None else np.nan
            print(f"{name + ' ' + surf:<26}" + "".join(f"{v:13.3f}" for v in row) + f"{pc:13.3f}")
    print("conv_src = checkpoint-instant survivor convective rain source entering q_r "
          "(after IFS downdraft + sub-cloud evaporation).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    argv = sys.argv[1:]
    src = None
    if "--sftlf-from" in argv:
        i = argv.index("--sftlf-from")
        src = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    for r in argv:
        _report(r, sftlf_run=src)
