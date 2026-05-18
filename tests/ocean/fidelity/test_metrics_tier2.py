"""Layer A — Tier 2 (geostrophic / thermal-wind adjustment)."""

from __future__ import annotations

import math
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

TIER = 2
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures" / "tier2_geostrophic_thermalwind.json"
)


def _metric_adjustment_over_inverse_f(bundle: artifacts.ArtifactBundle) -> float:
    ke = bundle.timeseries.get("mean_KE")
    times_days = bundle.timeseries.get("time_days")
    f = bundle.scalars.get("f_coriolis")
    if ke is None or times_days is None or f is None:
        return float("nan")
    times_s = times_days * 86400.0
    t_eq = metrics.adjustment_timescale(ke, times_s, plateau_frac=0.9)
    return float(t_eq * abs(float(f)))


def _metric_front_width_over_L_d(bundle: artifacts.ArtifactBundle) -> float:
    width = bundle.scalars.get("front_width_m")
    L_d = bundle.scalars.get("L_d_m")
    if width is None or L_d is None or L_d == 0:
        return float("nan")
    return float(width) / float(L_d)


def _metric_thermal_wind_residual_relative(bundle: artifacts.ArtifactBundle) -> float:
    rho = bundle.snapshots.get("final/rho")
    u = bundle.snapshots.get("final/u")
    f = bundle.scalars.get("f_coriolis")
    dy = bundle.scalars.get("dy_m")
    dz = bundle.scalars.get("dz_m")
    f_Lambda_scale = bundle.scalars.get("f_Lambda_scale")
    if rho is None or u is None or f is None or dy is None or dz is None:
        return float("nan")
    res = metrics.thermal_wind_residual(rho, u, f=float(f),
                                        dy=float(dy), dz=float(dz))
    scale = float(f_Lambda_scale) if f_Lambda_scale is not None else 1.0
    return float(np.mean(np.abs(res)) / max(abs(scale), 1e-30))


METRIC_REGISTRY: dict[str, Callable[[artifacts.ArtifactBundle], float]] = {
    "adjustment_timescale_over_inverse_f": _metric_adjustment_over_inverse_f,
    "front_width_over_L_d": _metric_front_width_over_L_d,
    "thermal_wind_residual_relative": _metric_thermal_wind_residual_relative,
}


@pytest.fixture(scope="module")
def tier_tolerances():
    return tolerances.load_tier_file(FIXTURE_PATH)


@pytest.fixture
def synthetic_passing_bundle(tmp_path):
    case_dir = tmp_path / "geostrophic_adjustment" / "latlon" / "36x72"
    f = 1.0e-4
    n = 51
    times_days = np.linspace(0.0, 2.0, n)
    times_s = times_days * 86400.0
    # Linear ramp to plateau at t = 1/f, then flat. With plateau_frac=0.9
    # the crossing falls at t = 0.9 / f, so t_eq * f = 0.9 (inside window).
    t_target = 1.0 / f
    ke = np.minimum(times_s / t_target, 1.0)
    # Thermal-wind-balanced fields (rho linear in y, u linear in z)
    from legoesm import constants
    nz, ny, nx = 10, 10, 4
    dy, dz = 1.0e4, 100.0
    rho_y_grad = 0.01
    du_dz = -constants.g * rho_y_grad / (constants.rho_water * f)
    rho_field = np.zeros((nz, ny, nx))
    u_field = np.zeros((nz, ny, nx))
    for j in range(ny):
        rho_field[:, j, :] = rho_y_grad * (j * dy)
    for k in range(nz):
        u_field[k, :, :] = du_dz * (k * dz)
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "status": "PASS",
            "f_coriolis": f,
            "front_width_m": 5.0e4,
            "L_d_m": 5.0e4,
            "dy_m": dy,
            "dz_m": dz,
            "f_Lambda_scale": f * abs(du_dz),
        },
        timeseries={
            "step": np.arange(n),
            "time_days": times_days,
            "mean_KE": ke,
        },
        snapshots={"final": {"rho": rho_field, "u": u_field}},
    )
    return artifacts.load(case_dir)


def test_every_tier2_case_has_a_metric_function(tier_tolerances):
    for case_name, case_tol in tier_tolerances.cases.items():
        assert case_tol.metric in METRIC_REGISTRY, (
            f"tier-2 case {case_name!r} uses metric {case_tol.metric!r} "
            f"with no METRIC_REGISTRY entry"
        )


@pytest.mark.parametrize(
    "case_name",
    [
        "geostrophic_adjustment_timescale",
        "geostrophic_adjustment_front_width",
        "baroclinic_thermal_wind_residual",
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


def test_unbalanced_state_fails_thermal_wind(tier_tolerances, tmp_path):
    case_dir = tmp_path / "baroclinic" / "latlon" / "36x72"
    rng = np.random.default_rng(0)
    rho = rng.standard_normal((10, 10, 4))
    u = np.zeros((10, 10, 4))
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "f_coriolis": 1.0e-4,
            "dy_m": 1.0e4, "dz_m": 100.0,
            "f_Lambda_scale": 1.0e-7,
        },
        snapshots={"final": {"rho": rho, "u": u}},
    )
    bundle = artifacts.load(case_dir)
    case_tol = tier_tolerances.cases["baroclinic_thermal_wind_residual"]
    measured = METRIC_REGISTRY[case_tol.metric](bundle)
    record = diff.diff_metric(
        case_name="baroclinic_thermal_wind_residual", tier=TIER,
        metric=case_tol.metric, measured=measured,
        tolerance_window=case_tol.tolerance_window,
    )
    assert not record.passed
