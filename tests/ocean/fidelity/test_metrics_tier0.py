"""Layer A — Tier 0 (no-motion invariants).

Validates that the registry → metric → diff pipeline correctly verifies
V/H/S conservation and machine-zero rest-state drift against the committed
``tier0_invariants.json`` tolerances. Synthetic bundles built in
``tmp_path`` exercise the full pipeline without needing a real model run;
a real-artifact integration test skips automatically when
``results/ocean/rest_state*/latlon/`` is empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from legoesm.ocean.fidelity import artifacts, diff, tolerances

TIER = 0
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tier0_invariants.json"
REAL_RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results" / "ocean"


def _metric_T_drift_max_abs(bundle: artifacts.ArtifactBundle) -> float:
    T = bundle.timeseries.get("mean_T")
    if T is None or T.size < 2:
        return float("nan")
    return float(np.max(np.abs(T - T[0])))


def _metric_rel_drift(key: str) -> Callable[[artifacts.ArtifactBundle], float]:
    def fn(bundle: artifacts.ArtifactBundle) -> float:
        ts = bundle.timeseries.get(key)
        if ts is None or ts.size < 2:
            return float("nan")
        denom = max(abs(float(ts[0])), 1e-30)
        return float(abs((ts[-1] - ts[0]) / denom))
    return fn


def _metric_u_max_abs(bundle: artifacts.ArtifactBundle) -> float:
    u_max = bundle.timeseries.get("max_abs_u")
    if u_max is None or u_max.size == 0:
        return float("nan")
    return float(u_max.max())


METRIC_REGISTRY: dict[str, Callable[[artifacts.ArtifactBundle], float]] = {
    "T_drift_max_abs": _metric_T_drift_max_abs,
    "delta_V_over_V0": _metric_rel_drift("mean_eta"),
    "delta_H_over_H0": _metric_rel_drift("mean_T"),
    "delta_S_over_S0": _metric_rel_drift("mean_S"),
    "u_max_abs": _metric_u_max_abs,
}


@pytest.fixture(scope="module")
def tier_tolerances():
    return tolerances.load_tier_file(FIXTURE_PATH)


@pytest.fixture
def synthetic_passing_bundle(tmp_path):
    case_dir = tmp_path / "rest_state" / "latlon" / "36x72"
    n = 11
    artifacts.write_synthetic(
        case_dir,
        scalars={"status": "PASS", "days": 1.0, "dt": 300.0},
        timeseries={
            "step": np.arange(n),
            "time_days": np.linspace(0.0, 1.0, n),
            "mean_T": np.full(n, 10.0),
            "mean_S": np.full(n, 35.0),
            "mean_eta": np.full(n, 1.0e-3),  # nonzero baseline for relative drift
            "max_abs_u": np.full(n, 0.0),
        },
    )
    return artifacts.load(case_dir)


@pytest.fixture
def synthetic_failing_bundle(tmp_path):
    case_dir = tmp_path / "rest_state" / "latlon" / "36x72"
    n = 11
    artifacts.write_synthetic(
        case_dir,
        scalars={"status": "FAIL"},
        timeseries={
            "step": np.arange(n),
            "time_days": np.linspace(0.0, 1.0, n),
            "mean_T": np.linspace(10.0, 11.0, n),  # 1 K drift -> fails 1e-8 window
            "mean_S": np.full(n, 35.0),
            "mean_eta": np.full(n, 1.0e-3),
            "max_abs_u": np.linspace(0.0, 1.0, n),
        },
    )
    return artifacts.load(case_dir)


def test_every_tier0_case_has_a_metric_function(tier_tolerances):
    for case_name, case_tol in tier_tolerances.cases.items():
        assert case_tol.metric in METRIC_REGISTRY, (
            f"tier-0 case {case_name!r} uses metric {case_tol.metric!r} "
            f"with no METRIC_REGISTRY entry"
        )


@pytest.mark.parametrize(
    "case_name",
    [
        "rest_state_T_drift",
        "rest_state_volume_conservation",
        "rest_state_heat_conservation",
        "rest_state_salt_conservation",
        "rest_state_u_max_after_30dt",
    ],
)
def test_synthetic_passing_bundle_passes(tier_tolerances, synthetic_passing_bundle, case_name):
    case_tol = tier_tolerances.cases[case_name]
    metric_fn = METRIC_REGISTRY[case_tol.metric]
    measured = metric_fn(synthetic_passing_bundle)
    record = diff.diff_metric(
        case_name=case_name, tier=TIER, metric=case_tol.metric,
        measured=measured, tolerance_window=case_tol.tolerance_window,
    )
    assert record.passed, record.format()


def test_synthetic_failing_bundle_fails_T_drift(tier_tolerances, synthetic_failing_bundle):
    case_tol = tier_tolerances.cases["rest_state_T_drift"]
    metric_fn = METRIC_REGISTRY[case_tol.metric]
    measured = metric_fn(synthetic_failing_bundle)
    record = diff.diff_metric(
        case_name="rest_state_T_drift", tier=TIER, metric=case_tol.metric,
        measured=measured, tolerance_window=case_tol.tolerance_window,
    )
    assert not record.passed
    assert record.triage_hint is not None
    assert "non-symmetric" in record.triage_hint or "NaN" in record.triage_hint


def test_synthetic_failing_bundle_fails_u_max(tier_tolerances, synthetic_failing_bundle):
    case_tol = tier_tolerances.cases["rest_state_u_max_after_30dt"]
    metric_fn = METRIC_REGISTRY[case_tol.metric]
    measured = metric_fn(synthetic_failing_bundle)
    record = diff.diff_metric(
        case_name="rest_state_u_max_after_30dt", tier=TIER, metric=case_tol.metric,
        measured=measured, tolerance_window=case_tol.tolerance_window,
    )
    assert not record.passed


def test_missing_timeseries_yields_nan_and_fails(tier_tolerances, tmp_path):
    case_dir = tmp_path / "rest_state" / "latlon" / "36x72"
    case_dir.mkdir(parents=True)
    bundle = artifacts.load(case_dir)
    case_tol = tier_tolerances.cases["rest_state_T_drift"]
    metric_fn = METRIC_REGISTRY[case_tol.metric]
    measured = metric_fn(bundle)
    record = diff.diff_metric(
        case_name="rest_state_T_drift", tier=TIER, metric=case_tol.metric,
        measured=measured, tolerance_window=case_tol.tolerance_window,
    )
    assert not record.passed
    assert "non-finite" in record.notes


@pytest.fixture
def real_rest_state_bundle():
    candidates = list(REAL_RESULTS_ROOT.glob("rest_state*/latlon/*"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        pytest.skip(
            "no real rest_state artifacts under results/ocean/ — run "
            "`scripts/matrix/run_ocean_test_matrix.py --only rest_state --grid latlon "
            "--emit-fidelity-artifacts` to populate."
        )
    return artifacts.load(candidates[0])


def test_real_rest_state_T_drift_within_tier0_window(real_rest_state_bundle, tier_tolerances):
    case_tol = tier_tolerances.cases["rest_state_T_drift"]
    metric_fn = METRIC_REGISTRY[case_tol.metric]
    measured = metric_fn(real_rest_state_bundle)
    record = diff.diff_metric(
        case_name="rest_state_T_drift", tier=TIER, metric=case_tol.metric,
        measured=measured, tolerance_window=case_tol.tolerance_window,
    )
    if not record.passed:
        pytest.fail(record.format() + ("\n" + record.triage_hint if record.triage_hint else ""))
