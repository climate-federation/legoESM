#!/usr/bin/env python
"""Williamson-2 DUO-ORACLE gate — score a legoESM cube W2 run against the
authoritative FV3 duo-grid reference (USER DIRECTIVE 2026-07-16: the FV3
duo-grid IS the oracle).

Reference: Mouallem 2023 (Zenodo 8327578) `atmos_daily.nc`, case2 (W2)
C48 duo, day 5, on the same 181x360 lat-lon grid the matrix exports.
Reference envelope (hord6):  max|v| = 0.0236 m/s, rms(v) = 0.0096 m/s.
For other cube resolutions N the envelope is scaled by (48/N)^2
(second-order truncation scaling) — stated on every report line so the
comparison protocol is explicit.

Usage:
  w2_duo_oracle_gate.py <snapshots_latlon.npz> [--res N] [--tol-factor F]

Exit 0 if the run's day-5 v-error is within F x the (scaled) duo
envelope on BOTH max and rms; exit 1 otherwise.  Default F=1.5.
This is the TARGET gate for the integrated native-duo solver; the
production A-L cube (v_max ~0.54 at C36) is NOT expected to pass — the
gate quantifies the remaining distance to the duo oracle.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ZENODO_BASE = ("/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/"
               "extracted/Code and simulations files")
REF_CASE = "C48.sw.case2.alpha0.duo.hord6"
REF_RES = 48


def reference_stats(case: str = REF_CASE) -> tuple[float, float]:
    """Day-5 max|v| and rms(v) of the duo reference (exact W2 v == 0, so
    the field IS the error).  Read from the published atmos_daily.nc."""
    import netCDF4
    path = os.path.join(ZENODO_BASE, case, "rundir", "atmos_daily.nc")
    d = netCDF4.Dataset(path)
    v = np.asarray(d.variables["vcomp"][-1, 0], dtype=np.float64)
    d.close()
    return float(np.abs(v).max()), float(np.sqrt((v ** 2).mean()))


def run_stats(npz_path: str) -> tuple[float, float]:
    z = np.load(npz_path, allow_pickle=True)
    v = np.asarray(z["v"], dtype=np.float64)
    v_last = v[-1] if v.ndim == 3 else v
    return float(np.abs(v_last).max()), float(np.sqrt((v_last ** 2).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="legoESM W2 cube snapshots_latlon.npz")
    ap.add_argument("--res", type=int, default=36,
                    help="cube resolution N of the run (envelope scales "
                         "by (48/N)^2; default C36)")
    ap.add_argument("--tol-factor", type=float, default=1.5,
                    help="acceptance = F x scaled duo envelope")
    args = ap.parse_args()

    ref_max, ref_rms = reference_stats()
    scale = (REF_RES / args.res) ** 2
    env_max, env_rms = ref_max * scale, ref_rms * scale
    got_max, got_rms = run_stats(args.npz)

    print(f"duo reference ({REF_CASE}, day 5): "
          f"max|v|={ref_max:.4f}  rms={ref_rms:.4f}")
    print(f"envelope at C{args.res} ((48/{args.res})^2={scale:.2f}, "
          f"tol x{args.tol_factor}): "
          f"max<={args.tol_factor * env_max:.4f}  "
          f"rms<={args.tol_factor * env_rms:.4f}")
    print(f"run: max|v|={got_max:.4f}  rms={got_rms:.4f}  "
          f"(ratio to envelope: max {got_max / env_max:.1f}x, "
          f"rms {got_rms / env_rms:.1f}x)")

    ok = (got_max <= args.tol_factor * env_max
          and got_rms <= args.tol_factor * env_rms)
    print("W2_DUO_ORACLE_GATE:", "PASS" if ok else
          "FAIL (distance to the duo oracle quantified above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
