#!/usr/bin/env python
"""Williamson-2 DUO-TARGET gate — score a legoESM cube W2 run against the
authoritative FV3 duo-grid reference (USER DIRECTIVE 2026-07-16: the FV3
duo-grid is the oracle/target).

Reference: Mouallem 2023 (Zenodo 8327578) ``atmos_daily.nc``, case2 (W2)
C48 duo hord6, selected at time == 5.0 d exactly, on the same 181x360
lat-lon grid the matrix exports.  Envelope: max|v| = 0.0236 m/s,
rms(v) = 0.0096 m/s (UNWEIGHTED 181x360 canvas rms, matching how both
sides are stored).

HONEST PROTOCOL NOTES (codex w2gate P1):
- This is an EXTERNAL TARGET, not a controlled comparison: the reference
  is C48 / hord6 / dt=3600 s in Fortran FV3; legoESM cube runs are C36 /
  its own calibrated solver / dt=300 s.  The gate quantifies distance to
  the duo level; it attributes nothing.
- The (48/N)^2 envelope scaling and the 1.5 tolerance factor are
  ASSUMPTIONS (second-order truncation heuristic; hord6 is formally
  higher-order in smooth flow and the imprint is corner-driven).  They
  stand until an hord6 resolution study calibrates them or a C36 duo
  reference replaces the extrapolation.

INPUT CONTRACT (codex w2gate P0 — enforced, so a day-0 / truncated /
wrong-grid npz cannot pass):
  times_days[-1] == 5.0, v shape (nt, 181, 360), all v finite,
  lat = -90..90 (181), lon 360 points.

Usage:
  w2_duo_oracle_gate.py <snapshots_latlon.npz> [--res N] [--tol-factor F]
Exit 0 within F x scaled envelope on BOTH max and rms; 1 otherwise.
The production A-L cube (~0.54 m/s at C36) is NOT expected to pass —
the gate quantifies the remaining distance for the native-duo solver.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ZENODO_BASE = ("/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/"
               "extracted/Code and simulations files")
REF_CASE = "C48.sw.case2.alpha0.duo.hord6"
REF_RES = 48
REF_DAY = 5.0


class ContractError(ValueError):
    """The run npz violates the W2 day-5 input contract."""


def reference_stats(case: str = REF_CASE) -> tuple[float, float]:
    """max|v| and unweighted rms(v) of the duo reference AT time==5.0 d
    (selected by value, not by [-1]; exact W2 v == 0 so the field IS the
    error)."""
    import netCDF4
    path = os.path.join(ZENODO_BASE, case, "rundir", "atmos_daily.nc")
    d = netCDF4.Dataset(path)
    t = np.asarray(d.variables["time"][:], dtype=np.float64)
    idx = int(np.argmin(np.abs(t - REF_DAY)))
    if abs(float(t[idx]) - REF_DAY) > 1e-6:
        d.close()
        raise ContractError(f"reference has no time == {REF_DAY} d")
    v = np.asarray(d.variables["vcomp"][idx, 0], dtype=np.float64)
    d.close()
    return float(np.abs(v).max()), float(np.sqrt((v ** 2).mean()))


def load_run_day5_v(npz_path: str) -> np.ndarray:
    """Load + CONTRACT-CHECK the run npz; return the day-5 v field."""
    z = np.load(npz_path, allow_pickle=True)
    for key in ("v", "times_days", "lat", "lon"):
        if key not in z.files:
            raise ContractError(f"npz missing '{key}'")
    t = np.asarray(z["times_days"], dtype=np.float64)
    if t.ndim != 1 or abs(float(t[-1]) - REF_DAY) > 1e-6:
        raise ContractError(
            f"times_days[-1] must be {REF_DAY} d (got "
            f"{float(t[-1]) if t.size else 'empty'}) — not a full W2 run")
    v = np.asarray(z["v"], dtype=np.float64)
    if v.ndim != 3 or v.shape[1:] != (181, 360) or v.shape[0] != t.size:
        raise ContractError(
            f"v must be (nt, 181, 360) aligned with times_days; got "
            f"{v.shape} vs nt={t.size}")
    lat = np.asarray(z["lat"], dtype=np.float64)
    lon = np.asarray(z["lon"], dtype=np.float64)
    if lat.shape != (181,) or abs(lat[0] + 90) > 1e-6 or abs(lat[-1] - 90) > 1e-6:
        raise ContractError("lat must be the canonical -90..90 (181)")
    if lon.shape != (360,):
        raise ContractError("lon must have 360 points")
    v5 = v[-1]
    if not np.isfinite(v5).all():
        raise ContractError("day-5 v contains non-finite values")
    return v5


def score(v5: np.ndarray, res: int, tol_factor: float,
          ref: tuple[float, float]) -> dict:
    """Pure scoring: envelope scaling + verdict (unit-testable without the
    Zenodo file)."""
    if res <= 0 or not np.isfinite(res):
        raise ValueError("--res must be a positive integer")
    if tol_factor <= 0 or not np.isfinite(tol_factor):
        raise ValueError("--tol-factor must be positive")
    ref_max, ref_rms = ref
    scale = (REF_RES / res) ** 2
    env_max, env_rms = ref_max * scale, ref_rms * scale
    got_max = float(np.abs(v5).max())
    got_rms = float(np.sqrt((v5 ** 2).mean()))
    ok = (got_max <= tol_factor * env_max
          and got_rms <= tol_factor * env_rms)
    return dict(ref_max=ref_max, ref_rms=ref_rms, scale=scale,
                env_max=env_max, env_rms=env_rms,
                got_max=got_max, got_rms=got_rms,
                tol_factor=tol_factor, ok=ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="legoESM W2 cube snapshots_latlon.npz")
    ap.add_argument("--res", type=int, default=36,
                    help="cube resolution N (envelope scaled by (48/N)^2 — "
                         "an ASSUMPTION, see module docstring; default C36)")
    ap.add_argument("--tol-factor", type=float, default=1.5,
                    help="acceptance = F x scaled envelope (ASSUMED default "
                         "1.5)")
    args = ap.parse_args()

    v5 = load_run_day5_v(args.npz)
    r = score(v5, args.res, args.tol_factor, reference_stats())
    print(f"duo reference ({REF_CASE}, t=5.0 d): "
          f"max|v|={r['ref_max']:.4f}  rms={r['ref_rms']:.4f} "
          f"(unweighted 181x360 canvas)")
    print(f"envelope at C{args.res} ((48/{args.res})^2={r['scale']:.2f}, "
          f"tol x{r['tol_factor']}; ASSUMED scaling): "
          f"max<={r['tol_factor'] * r['env_max']:.4f}  "
          f"rms<={r['tol_factor'] * r['env_rms']:.4f}")
    print(f"run: max|v|={r['got_max']:.4f}  rms={r['got_rms']:.4f}  "
          f"(ratio to envelope: max {r['got_max'] / r['env_max']:.1f}x, "
          f"rms {r['got_rms'] / r['env_rms']:.1f}x)")
    print("W2_DUO_TARGET_GATE:", "PASS" if r["ok"] else
          "FAIL (distance to the duo target quantified above)")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
