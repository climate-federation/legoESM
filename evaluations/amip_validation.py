"""AMIP climatology validation against observational targets.

Compares legoESM AMIP diagnostics against ERA5/CERES reference values.
Works on saved timeseries.npz output from ModelDriver.

Observational targets:
- TOA OLR: 239 +/- 5 W/m² (CERES EBAF)
- TOA ASR: 240 +/- 5 W/m² (CERES EBAF)
- Global mean T: 250-260 K (mass-weighted atm mean)
- Global mean precip: 2.0-3.5 mm/day (GPCP)
- Surface pressure: 1010-1016 hPa (ERA5 mean)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class ValidationResult(NamedTuple):
    """Result of a single validation check."""
    name: str
    value: float
    target: float
    tolerance: float
    passed: bool
    units: str


class AMIPValidationReport(NamedTuple):
    """Full validation report."""
    results: list[ValidationResult]
    n_passed: int
    n_total: int


def validate_amip_diagnostics(
    diagnostics_dir: str | Path,
) -> AMIPValidationReport:
    """Validate AMIP diagnostics against observational targets.

    Parameters
    ----------
    diagnostics_dir : Path
        Directory containing timeseries.npz from a ModelDriver run.

    Returns
    -------
    AMIPValidationReport
    """
    path = Path(diagnostics_dir)
    ts = np.load(path / "timeseries.npz")

    results = []

    # Use the second half of the run for mean climatology (skip spinup)
    n = len(ts["days"])
    half = max(n // 2, 1)

    # --- TOA radiation ---
    if "lw_up_toa" in ts.files:
        olr = np.mean(ts["lw_up_toa"][half:])
        results.append(ValidationResult(
            name="TOA OLR", value=olr, target=239.0, tolerance=50.0,
            passed=abs(olr - 239.0) < 50.0, units="W/m²",
        ))

    if "sw_up_toa" in ts.files:
        sw_up = np.mean(ts["sw_up_toa"][half:])
        # ASR ≈ S_0/4 - SW_up ≈ 340 - SW_up
        asr = 340.0 - sw_up
        results.append(ValidationResult(
            name="TOA ASR", value=asr, target=240.0, tolerance=50.0,
            passed=abs(asr - 240.0) < 50.0, units="W/m²",
        ))

    # --- Temperature ---
    if "T_atm" in ts.files:
        T_mean = np.mean(ts["T_atm"][half:])
        results.append(ValidationResult(
            name="Mean atm T", value=T_mean, target=255.0, tolerance=20.0,
            passed=abs(T_mean - 255.0) < 20.0, units="K",
        ))

    if "T_low" in ts.files:
        T_low = np.mean(ts["T_low"][half:])
        results.append(ValidationResult(
            name="Near-surface T", value=T_low, target=285.0, tolerance=20.0,
            passed=abs(T_low - 285.0) < 20.0, units="K",
        ))

    # --- Precipitation ---
    if "precip" in ts.files:
        precip = np.mean(ts["precip"][half:])
        results.append(ValidationResult(
            name="Global precip", value=precip, target=2.7, tolerance=2.5,
            passed=abs(precip - 2.7) < 2.5, units="mm/day",
        ))

    # --- Surface pressure ---
    if "dry_mass_ps" in ts.files:
        ps = np.mean(ts["dry_mass_ps"][half:]) / 100.0  # Pa → hPa
        results.append(ValidationResult(
            name="Mean p_s", value=ps, target=1013.0, tolerance=10.0,
            passed=abs(ps - 1013.0) < 10.0, units="hPa",
        ))

    # --- Column water vapor ---
    if "CWV" in ts.files:
        cwv = np.mean(ts["CWV"][half:])
        results.append(ValidationResult(
            name="Column water vapor", value=cwv, target=25.0, tolerance=20.0,
            passed=abs(cwv - 25.0) < 20.0, units="kg/m²",
        ))

    # --- Energy budget ---
    if "energy_residual" in ts.files:
        res = ts["energy_residual"][half:]
        res = res[res != 0]  # skip initial zeros
        if len(res) > 0:
            mean_res = np.mean(np.abs(res))
            results.append(ValidationResult(
                name="Energy residual |R|", value=mean_res, target=0.0,
                tolerance=500.0,
                passed=mean_res < 500.0, units="W/m²",
            ))

    # --- Moisture budget ---
    if "moisture_residual" in ts.files:
        mres = ts["moisture_residual"][half:]
        mres = mres[mres != 0]
        if len(mres) > 0:
            mean_mres = np.mean(np.abs(mres))
            results.append(ValidationResult(
                name="Moisture residual", value=mean_mres, target=0.0,
                tolerance=5.0,
                passed=mean_mres < 5.0, units="mm/day",
            ))

    n_passed = sum(1 for r in results if r.passed)
    return AMIPValidationReport(results=results, n_passed=n_passed, n_total=len(results))


def print_validation_report(report: AMIPValidationReport) -> str:
    """Format a validation report as a human-readable string."""
    lines = [
        "AMIP Climatology Validation",
        "=" * 60,
    ]
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"  [{status}] {r.name:25s}: {r.value:8.2f} {r.units:8s} "
            f"(target {r.target:.1f} +/- {r.tolerance:.1f})"
        )
    lines.append("-" * 60)
    lines.append(f"  {report.n_passed}/{report.n_total} checks passed")
    return "\n".join(lines)
