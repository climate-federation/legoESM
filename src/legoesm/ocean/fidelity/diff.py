"""Pass/fail diff between a measured scalar and a tolerance window.

Each tier in the fidelity hierarchy carries a structured triage hint that
is attached to any failure record so CI logs and Layer B reports point
operators at the most likely root cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


TIER_TRIAGE_HINTS: dict[int, str] = {
    0: (
        "Machine-zero invariants broken; likely a non-symmetric operator or "
        "an unguarded NaN. Check the most recently modified file in "
        "src/legoesm/ocean/dynamics/."
    ),
    1: (
        "Wave dispersion off; likely a Coriolis or pressure-gradient "
        "regression. Run tests/ocean/unit/test_pgf_* and "
        "test_barotropic_cgrid.py."
    ),
    2: (
        "Geostrophic / thermal-wind balance not recovered; inspect the "
        "thermal_wind_residual field and re-check the EOS / hydrostatic step."
    ),
    3: (
        "Process benchmark drifted; check compute_reference_potential_energy "
        "and the advection scheme (tests/ocean/unit/test_advection_*)."
    ),
    4: (
        "Wind-driven balance broken; the sverdrup_residual map localises "
        "boundary-layer vs interior fault."
    ),
    5: (
        "Growth rate off; check CFL margin and Lambda / N stratification "
        "setup before blaming the dynamics."
    ),
    6: (
        "Channel transport / EKE slope wrong; coarse-resolution under-"
        "resolution is the most common cause — check the resolution stamp "
        "first."
    ),
    7: (
        "DINO equilibrium off; almost always a forcing or initial-condition "
        "regression, not dynamics. Diff against the cached Veros baseline."
    ),
    8: (
        "Global realism drift; check forcing fields (winds, surface flux, "
        "restoring) and initial conditions before blaming dynamics."
    ),
}


@dataclass(frozen=True)
class PassFailRecord:
    """Result of a single metric/reference diff for one case."""

    case_name: str
    tier: int
    metric: str
    measured: float
    tolerance_window: tuple[float, float]
    passed: bool
    triage_hint: Optional[str]
    notes: str = ""

    def format(self) -> str:
        """One-line human-readable summary suitable for CI logs."""
        verdict = "PASS" if self.passed else "FAIL"
        lo, hi = self.tolerance_window
        return (
            f"[T{self.tier} {self.case_name}/{self.metric}] {verdict} "
            f"measured={self.measured:.6g} window=[{lo:.6g}, {hi:.6g}]"
            + (f"  notes={self.notes}" if self.notes else "")
        )


def diff_metric(
    *,
    case_name: str,
    tier: int,
    metric: str,
    measured: float,
    tolerance_window: tuple[float, float],
) -> PassFailRecord:
    """Check ``measured`` against ``tolerance_window`` and emit a record.

    Non-finite ``measured`` fails immediately with a "measured value is
    non-finite" note. Out-of-window measurements attach the tier's triage
    hint.
    """
    lo, hi = tolerance_window
    if not (lo <= hi):
        raise ValueError(
            f"tolerance_window {tolerance_window!r} must have lo <= hi"
        )
    if not math.isfinite(measured):
        return PassFailRecord(
            case_name=case_name,
            tier=tier,
            metric=metric,
            measured=measured,
            tolerance_window=(float(lo), float(hi)),
            passed=False,
            triage_hint=TIER_TRIAGE_HINTS.get(tier),
            notes="measured value is non-finite",
        )
    passed = lo <= measured <= hi
    return PassFailRecord(
        case_name=case_name,
        tier=tier,
        metric=metric,
        measured=float(measured),
        tolerance_window=(float(lo), float(hi)),
        passed=passed,
        triage_hint=None if passed else TIER_TRIAGE_HINTS.get(tier),
    )
