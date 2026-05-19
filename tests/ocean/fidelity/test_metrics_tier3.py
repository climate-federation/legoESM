"""Layer A — Tier 3 (process benchmarks: lock-exchange RPE, overflow plume)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from legoesm.ocean.fidelity import (
    artifacts,
    diff,
    metrics,
    tolerances,
)

TIER = 3
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures" / "tier3_process_benchmarks.json"
)


def _metric_rpe_rate_over_petersen2015(bundle: artifacts.ArtifactBundle) -> float:
    rpe = bundle.timeseries.get("rpe")
    times_days = bundle.timeseries.get("time_days")
    petersen_rate = bundle.scalars.get("petersen_rpe_rate")
    if rpe is None or times_days is None or petersen_rate is None or rpe.size < 2:
        return float("nan")
    slope = float(np.polyfit(times_days * 86400.0, rpe, 1)[0])
    return slope / float(petersen_rate)


def _metric_nose_descent_time_over_petersen2015(bundle: artifacts.ArtifactBundle) -> float:
    rho_anom = bundle.snapshots.get("overflow/rho_anom_t_z_xy")
    z_centers = bundle.snapshots.get("overflow/z_centers")
    times_s = bundle.snapshots.get("overflow/times_s")
    threshold = bundle.scalars.get("nose_threshold_kg_m3")
    sill_depth = bundle.scalars.get("sill_depth_m")
    petersen_t = bundle.scalars.get("petersen_descent_time_s")
    if (rho_anom is None or z_centers is None or times_s is None
            or threshold is None or sill_depth is None or petersen_t is None):
        return float("nan")
    nose = metrics.overflow_nose_descent(rho_anom, z_centers, float(threshold))
    above = np.where(nose >= float(sill_depth))[0]
    if above.size == 0:
        return float("nan")
    measured_t = float(times_s[above[0]])
    return measured_t / float(petersen_t)


def _metric_vertical_mass_flux_sign_score(bundle: artifacts.ArtifactBundle) -> float:
    w = bundle.snapshots.get("overflow/w_at_sill")
    if w is None or w.size == 0:
        return float("nan")
    # +1 if mean flux negative (downward dense plume), -1 otherwise.
    return -1.0 if float(np.mean(w)) > 0 else 1.0


METRIC_REGISTRY: dict[str, Callable[[artifacts.ArtifactBundle], float]] = {
    "rpe_rate_over_petersen2015": _metric_rpe_rate_over_petersen2015,
    "nose_descent_time_over_petersen2015": _metric_nose_descent_time_over_petersen2015,
    "vertical_mass_flux_sign_score": _metric_vertical_mass_flux_sign_score,
}


@pytest.fixture(scope="module")
def tier_tolerances():
    return tolerances.load_tier_file(FIXTURE_PATH)


@pytest.fixture
def synthetic_passing_bundle(tmp_path):
    case_dir = tmp_path / "lock_exchange" / "latlon" / "36x72"
    n = 21
    times_days = np.linspace(0.0, 1.0, n)
    petersen_rate = 1.0e-4  # arbitrary reference (units irrelevant for ratio)
    rpe = petersen_rate * (times_days * 86400.0)  # exact match
    n_t, n_z, n_y, n_x = 12, 8, 3, 4
    z_centers = np.linspace(50.0, 1800.0, n_z)
    rho_anom = np.zeros((n_t, n_z, n_y, n_x))
    sill_idx = n_z - 2
    for it in range(n_t):
        depth_idx = min(int(it / (n_t - 1) * sill_idx) + 1, n_z - 1)
        rho_anom[it, :depth_idx, :, :] = 1.0
    times_s = times_days * 86400.0
    sill_time_s = times_s[int(n_t * (sill_idx - 1) / (n_z - 1))]
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "status": "PASS",
            "petersen_rpe_rate": petersen_rate,
            "nose_threshold_kg_m3": 0.5,
            "sill_depth_m": float(z_centers[sill_idx - 1]),
            "petersen_descent_time_s": float(sill_time_s),
        },
        timeseries={
            "step": np.arange(n),
            "time_days": times_days,
            "rpe": rpe,
        },
        snapshots={
            "overflow": {
                "rho_anom_t_z_xy": rho_anom,
                "z_centers": z_centers,
                "times_s": times_s[:n_t],
                "w_at_sill": -np.abs(np.arange(n_t) + 1.0),  # downward
            },
        },
    )
    return artifacts.load(case_dir)


def test_every_tier3_case_has_a_metric_function(tier_tolerances):
    for case_name, case_tol in tier_tolerances.cases.items():
        assert case_tol.metric in METRIC_REGISTRY, (
            f"tier-3 case {case_name!r} uses metric {case_tol.metric!r} "
            f"with no METRIC_REGISTRY entry"
        )


@pytest.mark.parametrize(
    "case_name",
    [
        "lock_exchange_rpe_rate",
        "overflow_nose_descent_time",
        "overflow_vertical_mass_flux_sign",
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


def test_upward_vertical_flux_flips_sign_and_fails(tier_tolerances, tmp_path):
    case_dir = tmp_path / "overflow" / "latlon" / "36x72"
    artifacts.write_synthetic(
        case_dir,
        scalars={},
        snapshots={"overflow": {"w_at_sill": np.array([1.0, 2.0, 3.0])}},
    )
    bundle = artifacts.load(case_dir)
    case_tol = tier_tolerances.cases["overflow_vertical_mass_flux_sign"]
    measured = METRIC_REGISTRY[case_tol.metric](bundle)
    record = diff.diff_metric(
        case_name="overflow_vertical_mass_flux_sign", tier=TIER,
        metric=case_tol.metric, measured=measured,
        tolerance_window=case_tol.tolerance_window,
    )
    assert not record.passed
