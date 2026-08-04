#!/usr/bin/env python
"""Validate physical realism and conservation in an AMIP run output.

Reads ``timeseries.npz`` and ``results.txt`` from an AMIP run directory
(produced by ``scripts/run/run_amip.py``) and checks:

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


# Tripwire ceiling for the per-cell TIME-MEAN surface latent heat flux.
# Earth's strongest climatological cell means (western boundary currents,
# warm-pool) are ~250 W/m²; 500 leaves generous model headroom while still
# catching the misclassified-inland-sea potential-evaporation pathology
# (2-yr AMIP pilot: 2034 W/m² over the Caspian, 2026-07-22).
HFLS_CELL_MAX_W_M2 = 500.0

# P1.5 production-stack closure gate: convective column heating must be
# PAIRED with the latent release of the water convection removes.  The
# 2-yr AMIP pilot warm-runaway signature was this pairing broken by ~5x
# (ledger conv +322 W/m2 vs an L_v-consistent ~56 on the cold start;
# post-fix 8b47b6359: +59.9 vs ~55, rel ~0.08).  0.35 cleanly separates
# the two regimes while leaving room for legitimate sensible transport
# and CMT contributions to the convection energy row.
CONV_PAIRING_REL_MAX = 0.35
# Below this convective activity the ratio is numerical noise, not a
# physical verdict — skip rather than certify.
CONV_PAIRING_MIN_MM_DAY = 0.2


def conv_pairing_rel(run_dir: Path) -> float:
    """Relative mismatch between the run-mean convection energy row and
    -L_v x its water row from ``budget_ledger.npz``; NaN if the ledger is
    absent or convection is too weak to judge.

    Sign convention: the ledger water row is NEGATIVE when convection
    removes water from the column store, so the paired heating is
    ``-L_v * W > 0``.
    """
    from legoesm import constants

    path = run_dir / "budget_ledger.npz"
    if not path.exists():
        return float("nan")
    led = np.load(path)
    rates = np.asarray(led["rates"], dtype=np.float64)   # (t, proc, 2)
    processes = [str(p) for p in led["processes"]]
    i = processes.index("convection")
    w = float(rates[:, i, 0].mean())                     # kg/m2/s
    e = float(rates[:, i, 1].mean())                     # W/m2
    if abs(w) * 86400.0 < CONV_PAIRING_MIN_MM_DAY:
        return float("nan")
    e_from_water = -w * constants.L_v
    return abs(e - e_from_water) / max(abs(e), abs(e_from_water))


def hfls_cell_max(run_dir: Path) -> float:
    """Max over cells of the time-mean CMOR ``hfls`` [W/m²]; NaN if absent.

    Cell-resolved on purpose: a global/zonal mean dilutes a few-cell
    hotspot below any threshold (the pilot's 2034 W/m² Caspian cells left
    the global mean at a plausible 57 W/m²).
    """
    import glob

    files = sorted(glob.glob(str(run_dir / "cmor" / "Amon" / "hfls_*.nc")))
    if not files:
        return float("nan")
    try:
        import netCDF4 as nc
    except ImportError:
        return float("nan")
    with nc.Dataset(files[0]) as ds:
        hfls = np.asarray(ds.variables["hfls"][:], dtype=np.float64)
    return float(np.nanmax(hfls.mean(axis=0)))


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

    # Run status from results.txt — BLOWUP / FAIL is a fatal model-side
    # failure: even when the saved diagnostics happen to fall inside the
    # numeric bounds, a NaN-then-clamp model can leak a passing
    # validation result.  Treat the explicit status string as fatal in
    # all modes; ``--strict`` is reserved for warning-level diagnostics
    # (precip/moisture residual).
    status_failed = False
    radiation = None
    if res_path.exists():
        text = res_path.read_text()
        for line in text.splitlines():
            if line.startswith("Radiation:"):
                radiation = line.split(":", 1)[1].strip().lower()
            if line.startswith("Status:"):
                print(f"  Run status line: {line.strip()}")
                if "BLOWUP" in line or "FAIL" in line:
                    print("  FAIL: model status indicates blow-up / failure")
                    status_failed = True
                    break
    if status_failed:
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

    # TOA energy balance — gate the tolerance on **simulated days**
    # (final entry of ``ts['days']``), NOT on ``n = days.size``.  The
    # diagnostic cadence is set by ``--diag-days`` (default 5), so a
    # 365-d production run only writes ~73 samples; tying the bound to
    # ``n`` keeps a year-long run inside the cold-start band and would
    # mask a divergent imbalance.  The right gate is elapsed simulated
    # time, which is what the radiative-equilibrium argument cares about.
    # A NaN conservation residual is a MEASUREMENT FAILURE, not a pass.  It
    # used to be ``skip_if_nan=True``: because ``resid > tol`` is False for
    # NaN, a run whose residual channel was never populated printed
    # "[SKIP] not measured" and then "ALL CHECKS PASSED" with NO conservation
    # check at all.  That is exactly what a completed 365-day MPAS AMIP chain
    # did (0/155 finite samples in both residual channels).  A residual is
    # legitimately absent ONLY when the run has no radiation to close against.
    # Absence of the channel is treated exactly like an all-NaN channel: both
    # mean "not measured".  The discriminator for FAIL-vs-SKIP is whether the
    # run had radiation to close against, read from results.txt.  A file with
    # no ``Radiation:`` line is UNDETERMINABLE (not from our drivers, or a
    # minimal fixture) and skips with a message saying so -- an honest
    # "could not evaluate", never a silent pass.
    res = ts["energy_residual"] if "energy_residual" in ts else np.zeros(0)
    _finite = np.isfinite(res) if res.size else np.zeros(0, dtype=bool)
    n_finite = int(_finite.sum())
    rmax = float(np.max(np.abs(res[_finite]))) if n_finite else float("nan")
    sim_days = float(days[-1]) if days.size > 0 else 0.0
    # Tighten the bound as the simulation progresses (in days, not samples):
    if sim_days < 30.0:
        bound, warn_thr = 500.0, 200.0   # cold start
    elif sim_days < 365.0:
        bound, warn_thr = 200.0, 50.0    # spin-up
    else:
        bound, warn_thr = 50.0, 5.0      # production
    _absent = "energy_residual" not in ts
    if n_finite == 0 and radiation is None:
        print("  [SKIP] |TOA energy residual|: cannot evaluate — results.txt "
              "declares no 'Radiation:' line, so whether a closure was "
              "expected is undeterminable"
              + (" (channel ABSENT)" if _absent else ""))
    elif n_finite == 0 and radiation == "none":
        print("  [SKIP] |TOA energy residual|: run has no radiation "
              f"(Radiation: {radiation})")
    elif n_finite == 0:
        _what = ("channel ABSENT from timeseries.npz" if _absent
                 else f"0 of {res.size} samples finite")
        print(f"  [FAIL] |TOA energy residual| NEVER MEASURED: {_what} on a "
              f"run with radiation={radiation!r}. The energy conservation "
              "check is INERT.")
        failures.append("toa_residual_unmeasured")
    else:
        if n_finite < res.size:
            print(f"  [WARN] TOA energy residual: only {n_finite} of "
                  f"{res.size} samples finite")
            warns.append("toa_residual_partial")
        if not _check(
            f"|TOA energy residual| max [W/m²] "
            f"(sim_days={sim_days:.1f}, bound={bound}, n={n_finite})",
            rmax, lambda x: x < bound, fatal=True,
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

    # Moisture residual.  Same fail-loud rule as the energy residual: an
    # unmeasured closure is a FAILURE.  Legitimately absent only on a DRY run,
    # which the CWV channel identifies (no vapour -> nothing to close).
    mres = ts["moisture_residual"] if "moisture_residual" in ts else np.zeros(0)
    _mfin = np.isfinite(mres) if mres.size else np.zeros(0, dtype=bool)
    m_n = int(_mfin.sum())
    mr_max = (float(np.max(np.abs(mres[_mfin]))) if m_n else float("nan"))
    _m_absent = "moisture_residual" not in ts
    # A DRY run has nothing to close.  ``radiation is None`` additionally
    # means the file does not declare its own configuration, so the check is
    # undeterminable rather than failed (same rule as the energy residual).
    _dry = ("CWV" not in ts or ts["CWV"].size == 0
            or not np.any(np.isfinite(ts["CWV"])))
    if m_n == 0 and (_dry or radiation is None):
        _why = "dry run (no CWV measured)" if _dry else (
            "cannot evaluate — results.txt declares no 'Radiation:' line")
        print(f"  [SKIP] |moisture residual|: {_why}"
              + (" (channel ABSENT)" if _m_absent else ""))
    elif m_n == 0:
        _what = ("channel ABSENT from timeseries.npz" if _m_absent
                 else f"0 of {mres.size} samples finite")
        print(f"  [FAIL] |moisture residual| NEVER MEASURED: {_what} on a "
              "MOIST run. The moisture conservation check is INERT.")
        failures.append("moisture_residual_unmeasured")
    else:
        if m_n < mres.size:
            print(f"  [WARN] moisture residual: only {m_n} of "
                  f"{mres.size} samples finite")
            warns.append("moisture_residual_partial")
        if not _check(
            f"|moisture residual| max [mm/day] (n={m_n})",
            mr_max, lambda x: x < 5.0, fatal=False,
        ):
            warns.append("moisture_residual")

    # Per-cell hfls sanity tripwire (FATAL).  A time-mean surface latent
    # heat flux above HFLS_CELL_MAX_W_M2 in any cell is unphysical for
    # Earth (observed maxima ~250 W/m²; the 2-yr AMIP pilot's misclassified
    # Caspian/Aral ocean cells reached 2034 W/m²).  Reads the CMOR Amon
    # hfls when present (cell-resolved), else skips (the global-mean
    # timeseries cannot see a localized hotspot).
    hfls_max = hfls_cell_max(run_dir)
    if not _check("time-mean hfls cell max [W/m2]", hfls_max,
                  lambda x: x < HFLS_CELL_MAX_W_M2, fatal=True,
                  skip_if_nan=True):
        failures.append("hfls_cell_max")

    # Convective energy/water pairing (FATAL when measurable).  Reads the
    # per-process budget ledger when the run was launched with
    # --budget-ledger; skips otherwise.  Guards the assembled production
    # stack against a re-decoupling of convective heating from its water
    # sink (the mid-troposphere warm-runaway mechanism).
    pairing = conv_pairing_rel(run_dir)
    if not _check("conv energy/water pairing rel", pairing,
                  lambda x: x < CONV_PAIRING_REL_MAX, fatal=True,
                  skip_if_nan=True):
        failures.append("conv_pairing")

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
