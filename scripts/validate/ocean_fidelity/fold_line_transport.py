#!/usr/bin/env python
"""Gross northward transport across the eORCA1 F-pivot fold line.

The fold line is self-mapped (v(i) = -v(P_V i)), so its NET transport is zero
by construction; the gross northward part (sum of positive cell transports)
is the quantity that shows whether the fold is open.

* ours : snapshot npz of a ``--tripole-strip-north-rows 1 --tripole-fold-pivot
  F`` run with ``--gateway-transports`` (mass_flux_v [m^2/s] x dx_v [m]); the
  fold line is v row ``n_lat`` (the last row).  ``v`` max|.| also printed.
* NEMO : grid_V ``vocetr_eff`` [m^3/s] at output row 330 (= mesh row 330, the
  fold line; output rows = mesh 0..330, cols = mesh 1..360), record ``rec``.

Usage: fold_line_transport.py OURS.npz [NEMO_grid_V.nc [rec]]
"""
from __future__ import annotations

import sys

import numpy as np

FOLD_ROW_NEMO = 330
SV = 1.0e6


def ours(path: str) -> dict:
    d = np.load(path)
    # Window mean (--state-accumulate) when present -- NEMO vocetr_eff is a
    # 5-day mean -- else the instantaneous snapshot flux.
    key = "mass_flux_v_mean" if "mass_flux_v_mean" in d.files else "mass_flux_v"
    F = np.asarray(d[key])                    # (n_lat+1, n_lon, nlev)
    dx = np.asarray(d["dx_v"])                # (n_lat+1, n_lon)
    v = np.asarray(d["v"])
    if not (np.isfinite(F).all() and np.isfinite(v).all()):
        raise SystemExit(f"{path}: non-finite mass_flux_v / v")
    tr = F[-1] * dx[-1][:, None]              # m^3/s per (i, k), fold line
    # cyclic halo columns 0, n-1 duplicate interior ones: exclude
    tr = tr[1:-1].sum(axis=1)                 # per column (depth-summed)
    return dict(field=key, gross_north_Sv=tr[tr > 0].sum() / SV,
                net_Sv=tr.sum() / SV,
                max_abs_v_fold=float(np.abs(v[-1]).max()),
                time_days=float(d["time_days"]) if "time_days" in d else None)


def nemo(path: str, rec: int = 0) -> dict:
    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        tr = np.ma.filled(ds["vocetr_eff"][rec, :, FOLD_ROW_NEMO, :], 0.0).sum(0)
        v = np.ma.filled(ds["vo"][rec, :, FOLD_ROW_NEMO, :], 0.0)
    return dict(gross_north_Sv=tr[tr > 0].sum() / SV, net_Sv=tr.sum() / SV,
                max_abs_v_fold=float(np.abs(v).max()), rec=rec)


if __name__ == "__main__":
    print("ours:", ours(sys.argv[1]))
    if len(sys.argv) > 2:
        print("NEMO:", nemo(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0))
