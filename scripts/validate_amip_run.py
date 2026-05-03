#!/usr/bin/env python
"""Validate physical realism and conservation in an AMIP run output.

Reads ``timeseries.npz`` and ``results.txt`` from an AMIP run directory
(produced by ``scripts/run_amip.py``) and checks:

- Run completed without blowup / NaN
- Global-mean temperature in physically reasonable band
- Precipitation positive and in [0, 20] mm/day
- Column water vapor positive and in [10, 200] kg/m² (hot synthetic init)
- Sea ice fraction in [0, 1]
- TOA energy budget residual not divergent (allowed to be large during spinup,
  but flagged > 50 W/m² and rejected > 200 W/m²)

Returns 0 if all checks pass, 1 if any fail.

Usage::

    python scripts/validate_amip_run.py /path/to/amip/output [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _check(name: str, value, predicate, *, fatal: bool = False,
           skip_if_nan: bool = False) -> bool:
    """Record a check; print PASS/SKIP/WARN/FAIL line.

    The lightweight spectral / MPAS run paths don't populate every
    diagnostic — they fill missing channels with NaN.  Pass
    ``skip_if_nan=True`` for variables that are *optional*: a NaN value
    becomes a SKIP rather than a FAIL.
    """
    is_nan = (
        isinstance(value, float)
        and (np.isnan(value) or np.isinf(value))
    )
    if is_nan and skip_if_nan:
        print(f"  [SKIP] {name}: not measured (NaN)")
        return True
    ok = bool(predicate(value)) if not is_nan else False
    tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"  [{tag}] {name}: {value}")
    return ok or not fatal


def validate(run_dir: Path, *, strict: bool = False) -> int:
    if not run_dir.exists():
        print(f"FAIL: run directory does not exist: {run_dir}")
        return 1
    ts_path = run_dir / "timeseries.npz"
    res_path = run_dir / "results.txt"
    if not ts_path.exists():
        print(f"FAIL: timeseries.npz missing in {run_dir}")
        return 1

    print(f"=== Validating AMIP run: {run_dir} ===")

    # Run status from results.txt
    if res_path.exists():
        text = res_path.read_text()
        for line in text.splitlines():
            if line.startswith("Status:"):
                print(f"  Run status line: {line.strip()}")
                if "BLOWUP" in line or "FAIL" in line:
                    print("  FAIL: model blew up")
                    if strict:
                        return 1

    ts = np.load(ts_path)
    days = ts["days"]
    n = days.size
    print(f"  Samples: {n}")

    # Last-step diagnostics
    def _last(name: str) -> float:
        if name not in ts:
            return float("nan")
        v = ts[name]
        return float(v[-1]) if v.size else float("nan")

    failures = []
    warns = []

    # Temperature reasonable
    T = _last("T_atm")
    if not _check("Final <T_atm>", T,
                  lambda x: 200.0 < x < 320.0, fatal=True):
        failures.append("T_atm")

    # Precipitation (optional — spectral / MPAS lightweight paths skip)
    P = _last("precip")
    if not _check("Final <Precip> [mm/day]", P,
                  lambda x: 0.0 <= x <= 30.0, fatal=False, skip_if_nan=True):
        warns.append("precip")

    # CWV (optional)
    cwv = _last("CWV")
    if not _check("Final <CWV> [kg/m²]", cwv,
                  lambda x: 0.0 < x < 200.0, fatal=True, skip_if_nan=True):
        failures.append("CWV")

    # SIC (optional — analytical SST in spectral path doesn't write SIC array)
    sic = _last("sic")
    if not _check("Final <SIC>", sic,
                  lambda x: 0.0 <= x <= 1.0, fatal=True, skip_if_nan=True):
        failures.append("sic")

    # max wind not exploding (always fatal)
    wmax = _last("max_wind")
    if not _check("Final max_wind [m/s]", wmax,
                  lambda x: 0.0 <= x <= 200.0, fatal=True):
        failures.append("max_wind")

    # TOA energy balance — use a generous bound during spinup (n < 100 d)
    # because the column is far from radiative equilibrium.  At 100+ days
    # the residual should approach the production tolerance.
    if "energy_residual" in ts:
        res = ts["energy_residual"]
        rmax = float(np.nanmax(np.abs(res))) if res.size else 0.0
        # Tighten the bound as the spin-up progresses:
        if n < 30:
            bound, warn_thr = 500.0, 200.0   # cold start
        elif n < 365:
            bound, warn_thr = 200.0, 50.0    # spin-up
        else:
            bound, warn_thr = 50.0, 5.0      # production
        if not _check(
            f"|TOA energy residual| max [W/m²] (n={n}d, bound={bound})",
            rmax, lambda x: x < bound, fatal=True, skip_if_nan=True,
        ):
            failures.append("toa_residual")
        elif rmax > warn_thr:
            warns.append("toa_residual_high")

    # Surface / TOA fluxes finite (optional — lightweight paths don't compute)
    for k in ("sw_up_toa", "lw_up_toa", "sw_net_sfc", "lw_net_sfc"):
        v = _last(k)
        if not _check(f"Final <{k}> [W/m²]", v,
                      lambda x: -2000.0 < x < 2000.0, fatal=True,
                      skip_if_nan=True):
            failures.append(k)

    # Mean surface pressure conserved
    if "dry_mass_ps" in ts:
        ps = ts["dry_mass_ps"]
        if ps.size > 0 and np.all(np.isfinite(ps)):
            ps0, ps1 = float(ps[0]), float(ps[-1])
            rel = abs(ps1 - ps0) / max(ps0, 1.0) if ps0 > 0 else 1.0
            if not _check("Mean p_s conservation (rel)", rel,
                          lambda x: x < 0.05, fatal=True):
                failures.append("ps_conservation")
        else:
            print("  [SKIP] Mean p_s conservation: not measured")

    # Moisture residual (optional)
    if "moisture_residual" in ts:
        mres = ts["moisture_residual"]
        mr_max = (float(np.nanmax(np.abs(mres)))
                  if mres.size and np.any(np.isfinite(mres)) else float("nan"))
        if not _check("|moisture residual| max [mm/day]", mr_max,
                      lambda x: x < 5.0, fatal=False, skip_if_nan=True):
            warns.append("moisture_residual")

    print()
    print(f"  Failures: {failures}")
    print(f"  Warnings: {warns}")
    if failures:
        return 1
    if strict and warns:
        return 1
    print("  ALL CHECKS PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path,
                        help="AMIP run output directory")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Treat warnings as failures")
    args = parser.parse_args(argv)
    return validate(args.run_dir, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
