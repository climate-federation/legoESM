#!/usr/bin/env python3
"""Paired arm-minus-control means over a short window, from the CMOR
accumulator sidecars -- the scorer for few-day cloud arms.

A 5-day arm never completes a calendar month, so it publishes no ``Amon``
file.  It does write ``cmor_accum_day_<D>.npz`` at every checkpoint: the
running SUM and sample COUNT of every CMOR field for the month in progress.
Two sidecars D0 < D1 inside the same calendar month therefore give the exact
window mean ``(S(D1) - S(D0)) / (n(D1) - n(D0))`` on the 5-degree CMOR grid,
with no re-integration and no reading of log lines.  Restored through the
accumulator's own ``set_state`` so the layout is never re-implemented here.

The window MUST lie inside one month: a completed month is popped from the
sidecar when it closes, and the probe refuses a pair whose buckets differ.
Both arms of a pair must be scored over the SAME (D0, D1); the control's
window is the same restart-transient the arm carries, so the difference is
the arm's effect (plus code-drift, which the control arm exists to catch).

Usage: window_diff.py --ctl <ctl_run> --d0 80 --d1 85 <arm_run> [<arm_run> ...]
       (prints ctl absolute values, then arm - ctl, per regional_bias region)
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("regional_bias")

FIELDS_2D = ("rsut", "rlut", "rsutcs", "rlutcs", "clt", "clwvi", "clivi", "prw", "pr")


def load_monthly_sums(path):
    """{(year, month): {field: (sum, count)}} from one sidecar, via the
    accumulator's own set_state."""
    from legoesm.diagnostics.monthly_means import SpatialMonthlyAccumulator, _decode_manifest
    with np.load(path, allow_pickle=False) as npz:
        state = {k[len("monthly."):]: npz[k] for k in npz.files if k.startswith("monthly.")}
    if "__manifest__" not in state:
        raise SystemExit(f"FATAL: {path} has no monthly accumulator")
    man = _decode_manifest(state["__manifest__"])
    acc = SpatialMonthlyAccumulator(int(man["nlat"]), int(man["nlon"]), int(man["nlev"]))
    acc.set_state(state)
    return acc._data_2d, (acc.nlat, acc.nlon)


def window_mean(run, d0, d1):
    """Per-field mean over (d0, d1] on the CMOR grid, plus the sample count."""
    s0, dims = load_monthly_sums(f"{rb.ROOT}/{run}/cmor_accum_day_{d0:04d}.npz")
    s1, dims1 = load_monthly_sums(f"{rb.ROOT}/{run}/cmor_accum_day_{d1:04d}.npz")
    if dims != dims1:
        raise SystemExit("FATAL: sidecar grids differ")
    if set(s0) != set(s1) or len(s1) != 1:
        raise SystemExit(f"FATAL: window {d0}..{d1} of {run} crosses a month "
                         f"boundary (buckets {sorted(s0)} vs {sorted(s1)}); "
                         "pick a window inside one calendar month")
    (key,) = s1
    out = {}
    n = None
    for f in FIELDS_2D:
        if f not in s1[key]:
            continue
        a1, c1 = s1[key][f]
        a0, c0 = s0[key].get(f, (np.zeros_like(a1), 0))
        if c1 - c0 <= 0:
            raise SystemExit(f"FATAL: no new samples of {f} in {d0}..{d1} for {run}")
        out[f] = (a1 - a0) / (c1 - c0)
        n = c1 - c0 if n is None else n
    if "rlutcs" in out and "rlut" in out:
        out["cre_lw"] = out["rlutcs"] - out["rlut"]
    if "rsutcs" in out and "rsut" in out:
        out["cre_sw"] = out["rsut"] - out["rsutcs"]
    return out, n, dims


def cmor_grid(dims):
    nlat, nlon = dims
    lat = -90.0 + 180.0 / nlat * (np.arange(nlat) + 0.5)
    lon = 360.0 / nlon * (np.arange(nlon) + 0.5)
    return lat, lon


def table(fields, lat, lon, title):
    cols = [c for c in ("rsut", "rlut", "rsutcs", "rlutcs", "cre_sw", "cre_lw",
                        "clt", "clivi", "clwvi", "prw", "pr") if c in fields]
    print(f"\n=== {title} ===")
    print(f"{'region':18s}" + "".join(f"{c:>9s}" for c in cols))
    for name, box in rb.REGIONS.items():
        vals = []
        for c in cols:
            v = rb.region_mean(fields[c], lat, lon, box)
            if c == "pr":
                v *= 86400.0          # kg/m2/s -> mm/day
            elif c in ("clivi", "clwvi", "prw"):
                v *= 1.0e3 if c != "prw" else 1.0   # kg/m2 -> g/m2 for condensate
            vals.append(v)
        print(f"{name:18s}" + "".join(f"{v:9.2f}" for v in vals))
    print("  units: W/m2; clt %; clivi/clwvi g/m2; prw kg/m2; pr mm/day")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--d0", type=int, required=True)
    ap.add_argument("--d1", type=int, required=True)
    args = ap.parse_args(argv)
    ctl, n_ctl, dims = window_mean(args.ctl, args.d0, args.d1)
    lat, lon = cmor_grid(dims)
    table(ctl, lat, lon, f"{args.ctl} window mean, days {args.d0}..{args.d1} "
                          f"({n_ctl} samples)")
    for arm in args.arms:
        a, n_arm, dims_a = window_mean(arm, args.d0, args.d1)
        if dims_a != dims or n_arm != n_ctl:
            raise SystemExit(f"FATAL: {arm} window has {n_arm} samples vs ctl "
                             f"{n_ctl}, or a different grid -- not a paired window")
        table({k: a[k] - ctl[k] for k in a if k in ctl}, lat, lon,
              f"{arm} - {args.ctl}, days {args.d0}..{args.d1} ({n_arm} samples)")


if __name__ == "__main__":
    main()
