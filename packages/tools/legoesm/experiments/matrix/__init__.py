"""Shared, component-agnostic test-matrix framework.

The single home for the tier taxonomy, case/result types, conservation
PASS/FAIL gates, summary reporting, and the ``MatrixRunner`` base shared by the
per-component matrix scripts in ``scripts/matrix/``.  See
``docs/validation/TESTING.md`` for the tier ladder and how to plug in a component.

Component-agnostic by contract: this subpackage imports only stdlib, numpy, and
``legoesm.diagnostics`` — never a component package — so it stays on the
``legoesm-tools`` side of the federation dependency DAG.
"""
from __future__ import annotations

from legoesm.experiments.matrix.core import (
    CaseResult,
    MatrixCase,
    ResultRecorder,
    RunMaturity,
    RunStatus,
    Tier,
)
from legoesm.experiments.matrix.gates import (
    CONS_THRESH,
    benchmark_error_gate,
    default_tol,
    drift_gate,
    energy_gate,
    finite_gate,
    heat_gate,
    mass_gate,
    salt_gate,
)
from legoesm.experiments.matrix.namelist import (
    build_namelist,
    write_case_namelist,
)
from legoesm.experiments.matrix.registry import MatrixRunner
from legoesm.experiments.matrix.report import (
    detect_regressions,
    write_summary,
)

__all__ = [
    "CaseResult",
    "MatrixCase",
    "ResultRecorder",
    "RunMaturity",
    "RunStatus",
    "Tier",
    "CONS_THRESH",
    "benchmark_error_gate",
    "default_tol",
    "drift_gate",
    "energy_gate",
    "finite_gate",
    "heat_gate",
    "mass_gate",
    "salt_gate",
    "MatrixRunner",
    "build_namelist",
    "write_case_namelist",
    "detect_regressions",
    "write_summary",
]
