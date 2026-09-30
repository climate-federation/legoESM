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
not a global mean.  The CMOR monthly mean spans the WHOLE run while the
sampled series drops day <= 1, so a sub-W/m^2 gap is window mismatch, not
signal; this script decides "aliased or not", not tenths.

A missing CMOR variable or a non-finite series sample is FATAL rather than
skipped: a comparison with one side silently dropped is the plausible wrong
answer this script exists to catch.

Numbers only.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import netCDF4 as nc
import numpy as np

_CMOR = ("rsdt", "rsut", "rlut", "hfss", "hfls")
_SAMPLED = ("energy_toa_net", "hfss", "hfls", "sw_net_sfc", "lw_net_sfc",
            "energy_dE_dt")


def gmean(run_dir: pathlib.Path, var: str) -> float:
    """Area-weighted global mean of a CMOR Amon field [time, lat, lon]."""
    f = sorted(glob.glob(f"{run_dir}/cmor/Amon/{var}_Amon_*.nc"))
    if len(f) != 1:
        raise SystemExit(f"{run_dir}/cmor/Amon has {len(f)} {var}_Amon_*.nc "
                         "files; need exactly one (none = the CMOR feed did "
                         "not run; several = which window is undefined).")
    with nc.Dataset(f[0]) as ds:
        a = np.asarray(ds.variables[var][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[None, :, None], a.shape)
    m = np.isfinite(a)
    return float((a[m] * w[m]).sum() / w[m].sum())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=pathlib.Path,
                   help="run output directory (holds timeseries.npz and cmor/)")
    args = p.parse_args(argv)

    print("CMOR Amon area-weighted global means [W/m^2] (time-accumulated):")
    vals = {v: gmean(args.run_dir, v) for v in _CMOR}
    for v, x in vals.items():
        print(f"  {v:8s} {x:10.3f}")
    toa_cmor = vals["rsdt"] - vals["rsut"] - vals["rlut"]
    print(f"  toa_net (rsdt - rsut - rlut) = {toa_cmor:.3f}")

    print()
    print("Sampled series means over the same window (day > 1):")
    d = np.load(args.run_dir / "timeseries.npz", allow_pickle=True)
    missing = [v for v in ("days",) + _SAMPLED if v not in d]
    if missing:
        raise SystemExit(f"timeseries.npz lacks {missing}; nothing to compare.")
    days = np.asarray(d["days"], dtype=float)
    k = days > 1.0
    k[0] = False
    if not k.any():
        raise SystemExit("no samples after day 1 (first sample always dropped);"
                         " nothing to average.")
    s = {}
    for v in _SAMPLED:
        a = np.asarray(d[v], dtype=float)[k]
        if not np.isfinite(a).all():
            raise SystemExit(f"{v} has non-finite samples in the window; "
                             "refusing to average over them.")
        s[v] = a
        print(f"  {v:16s} {a.mean():10.3f}")
    # Provenance, not a gate: this script is the instrument that MEASURES the
    # aliasing, so a snapshot series is its intended input, labelled as such.
    flag = d["energy_flux_interval_mean"] if "energy_flux_interval_mean" in d else None
    if flag is None:
        print("  flux timing: flag absent (series predates it) -- SNAPSHOTS")
    else:
        n_snap = int((np.asarray(flag, dtype=float)[k] < 1.0).sum())
        print(f"  flux timing: {n_snap} of {int(k.sum())} samples are "
              f"SNAPSHOTS, {int(k.sum()) - n_snap} interval means")

    print()
    print("THE COMPARISON (sampled minus CMOR):")
    samp_toa = s["energy_toa_net"].mean()
    print(f"  toa_net   sampled {samp_toa:9.3f}  CMOR {toa_cmor:9.3f}  "
          f"diff {samp_toa - toa_cmor:9.3f}")
    for v in ("hfss", "hfls"):
        print(f"  {v:8s}  sampled {s[v].mean():9.3f}  CMOR {vals[v]:9.3f}  "
              f"diff {s[v].mean() - vals[v]:9.3f}")

    print()
    print("Implied LEAK if the CMOR toa_net is substituted for the sampled one")
    print("(everything else held): LEAK = sfc_rad - hfss - hfls - (toa - dE/dt)")
    base = s["sw_net_sfc"] + s["lw_net_sfc"] - s["hfss"] - s["hfls"]
    leak_sampled = base - (s["energy_toa_net"] - s["energy_dE_dt"])
    leak_cmor_toa = base - (toa_cmor - s["energy_dE_dt"])
    print(f"  LEAK with sampled toa_net : {leak_sampled.mean():8.3f}")
    print(f"  LEAK with CMOR    toa_net : {leak_cmor_toa.mean():8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
