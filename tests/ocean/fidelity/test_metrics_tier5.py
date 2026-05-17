"""Layer A — Tier 5 (baroclinic instability: Eady, Phillips). Nightly @slow."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from legoesm.ocean.fidelity import (
    artifacts,
    diff,
    metrics,
    references,
    tolerances,
)

pytestmark = pytest.mark.slow

TIER = 5
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures" / "tier5_baroclinic_instability.json"
)


def _metric_growth_rate_over_sigma_eady(bundle: artifacts.ArtifactBundle) -> float:
    ke_pert = bundle.timeseries.get("mean_KE_pert")
    times_days = bundle.timeseries.get("time_days")
    f = bundle.scalars.get("f_coriolis")
    Lambda = bundle.scalars.get("Lambda_shear")
    N = bundle.scalars.get("N_bv")
    t_start = bundle.scalars.get("growth_fit_t_start_days")
    t_end = bundle.scalars.get("growth_fit_t_end_days")
    if any(v is None for v in (ke_pert, times_days, f, Lambda, N, t_start, t_end)):
        return float("nan")
    sigma_meas = metrics.linear_growth_rate(
        ke_pert, times_days * 86400.0,
        t_window=(float(t_start) * 86400.0, float(t_end) * 86400.0),
    )
    sigma_eady = references.eady_growth_rate(float(f), float(Lambda), float(N))
    return sigma_meas / sigma_eady


def _metric_dominant_wavelength_over_L_d(bundle: artifacts.ArtifactBundle) -> float:
    lambda_measured = bundle.scalars.get("dominant_wavelength_m")
    N = bundle.scalars.get("N_bv")
    H = bundle.scalars.get("H_layer")
    f = bundle.scalars.get("f_coriolis")
    if any(v is None for v in (lambda_measured, N, H, f)):
        return float("nan")
    lambda_eady = references.eady_most_unstable_wavelength(
        float(N), float(H), float(f))
    return float(lambda_measured) / lambda_eady


def _metric_most_unstable_period_days(bundle: artifacts.ArtifactBundle) -> float:
    period = bundle.scalars.get("most_unstable_period_days")
    if period is None:
        return float("nan")
    return float(period)


METRIC_REGISTRY: dict[str, Callable[[artifacts.ArtifactBundle], float]] = {
    "linear_growth_rate_over_sigma_eady": _metric_growth_rate_over_sigma_eady,
    "dominant_wavelength_over_L_d": _metric_dominant_wavelength_over_L_d,
    "most_unstable_period_days": _metric_most_unstable_period_days,
}


@pytest.fixture(scope="module")
def tier_tolerances():
    return tolerances.load_tier_file(FIXTURE_PATH)


@pytest.fixture
def synthetic_passing_bundle(tmp_path):
    case_dir = tmp_path / "eady_instability" / "latlon" / "36x72"
    f = 1.0e-4
    Lambda = 1.0e-3
    N = 1.0e-2
    sigma = references.eady_growth_rate(f, Lambda, N)
    H = 4000.0
    lambda_eady = references.eady_most_unstable_wavelength(N, H, f)
    n = 200
    times_days = np.linspace(0.0, 30.0, n)
    times_s = times_days * 86400.0
    ke_pert = np.exp(2.0 * sigma * times_s) * 1.0e-6
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "status": "PASS",
            "f_coriolis": f, "Lambda_shear": Lambda, "N_bv": N, "H_layer": H,
            "growth_fit_t_start_days": 5.0,
            "growth_fit_t_end_days": 20.0,
            "dominant_wavelength_m": lambda_eady,
            "most_unstable_period_days": 25.0,
        },
        timeseries={
            "step": np.arange(n),
            "time_days": times_days,
            "mean_KE_pert": ke_pert,
        },
    )
    return artifacts.load(case_dir)


def test_every_tier5_case_has_a_metric_function(tier_tolerances):
    for case_name, case_tol in tier_tolerances.cases.items():
        assert case_tol.metric in METRIC_REGISTRY, (
            f"tier-5 case {case_name!r} uses metric {case_tol.metric!r} "
            f"with no METRIC_REGISTRY entry"
        )


@pytest.mark.parametrize(
    "case_name",
    [
        "eady_instability_growth_rate_over_sigma_eady",
        "eady_uniform_dominant_wavelength_over_L_d",
        "phillips_two_layer_period_days",
    ],
)
def test_synthetic_passing_bundle_passes(tier_tolerances, synthetic_passing_bundle, case_name):
    case_tol = tier_tolerances.cases[case_name]
    measured = METRIC_REGISTRY[case_tol.metric](synthetic_passing_bundle)
    record = diff.diff_metric(
        case_name=case_name, tier=TIER, metric=case_tol.metric,
        measured=measured, tolerance_window=case_tol.tolerance_window,
    )
    assert record.passed, record.format()
