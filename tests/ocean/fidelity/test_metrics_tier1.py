"""Layer A — Tier 1 (linear wave dispersion).

Cases compare measured phase speed / dispersion from a Hovmöller snapshot
against analytical IGW (Bishnu 2024), Kelvin (c=sqrt(gH)), and barotropic
Rossby formulas in ``references.py``. Synthetic fixtures store an
``(x, t)`` plane wave whose recovered phase speed should match.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.fidelity import (
    artifacts,
    diff,
    metrics,
    references,
    tolerances,
)

TIER = 1
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tier1_linear_waves.json"
REAL_RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results" / "ocean"


def _metric_phase_speed_relative_error(bundle: artifacts.ArtifactBundle) -> float:
    """|c_meas - c_ref| / c_ref using a stored Hovmöller and reference speed."""
    field = bundle.snapshots.get("hovmoller/eta_x_t")
    c_ref = bundle.scalars.get("c_reference_m_s")
    dx = bundle.scalars.get("dx_m")
    dt_s = bundle.scalars.get("snapshot_dt_s")
    if field is None or c_ref is None or dx is None or dt_s is None:
        return float("nan")
    c_meas = metrics.measured_phase_speed_2d(field, float(dx), float(dt_s))
    return abs(c_meas - float(c_ref)) / max(abs(float(c_ref)), 1e-30)


def _metric_dispersion_rms_error(bundle: artifacts.ArtifactBundle) -> float:
    """Relative RMS error of IGW frequency at a few sampled wavenumbers."""
    f = bundle.scalars.get("f_coriolis")
    H = bundle.scalars.get("layer_depth_m")
    k_samples = bundle.snapshots.get("dispersion/k_samples")
    omega_meas = bundle.snapshots.get("dispersion/omega_measured")
    if f is None or H is None or k_samples is None or omega_meas is None:
        return float("nan")
    omega_ref = references.igw_omega(k_samples, np.zeros_like(k_samples),
                                     H=float(H), f=float(f))
    return float(np.sqrt(np.mean(((omega_meas - omega_ref) / omega_ref) ** 2)))


METRIC_REGISTRY: dict[str, Callable[[artifacts.ArtifactBundle], float]] = {
    "phase_speed_relative_error": _metric_phase_speed_relative_error,
    "dispersion_rms_error": _metric_dispersion_rms_error,
}


@pytest.fixture(scope="module")
def tier_tolerances():
    return tolerances.load_tier_file(FIXTURE_PATH)


def _build_hovmoller_passing(n_x=256, n_t=1024, dx=1.0e4, dt=60.0, c=234.0):
    k = 2 * math.pi / (dx * 16)
    omega = c * k
    X, T = np.meshgrid(np.arange(n_x) * dx, np.arange(n_t) * dt, indexing="ij")
    return np.cos(k * X - omega * T)


@pytest.fixture
def synthetic_passing_bundle(tmp_path):
    case_dir = tmp_path / "barotropic_wave" / "latlon" / "48x72"
    H = 4000.0
    c_ref = references.kelvin_wave_speed(H)
    field = _build_hovmoller_passing(c=c_ref)
    # IGW dispersion samples that exactly match analytical omega.
    f = 1.0e-4
    k_samples = np.linspace(1.0e-6, 1.0e-4, 8)
    omega_ref = references.igw_omega(k_samples, np.zeros_like(k_samples), H=H, f=f)
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "status": "PASS",
            "c_reference_m_s": c_ref,
            "dx_m": 1.0e4,
            "snapshot_dt_s": 60.0,
            "f_coriolis": f,
            "layer_depth_m": H,
        },
        snapshots={
            "hovmoller": {"eta_x_t": field},
            "dispersion": {
                "k_samples": k_samples,
                "omega_measured": omega_ref,  # exact match -> rms = 0
            },
        },
    )
    return artifacts.load(case_dir)


@pytest.fixture
def synthetic_failing_bundle(tmp_path):
    """Hovmöller built with the wrong phase speed (2x reference)."""
    case_dir = tmp_path / "barotropic_wave" / "latlon" / "48x72"
    H = 4000.0
    c_ref = references.kelvin_wave_speed(H)
    field = _build_hovmoller_passing(c=2.0 * c_ref)
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "status": "FAIL",
            "c_reference_m_s": c_ref,
            "dx_m": 1.0e4,
            "snapshot_dt_s": 60.0,
        },
        snapshots={"hovmoller": {"eta_x_t": field}},
    )
    return artifacts.load(case_dir)


def test_every_tier1_case_has_a_metric_function(tier_tolerances):
    for case_name, case_tol in tier_tolerances.cases.items():
        assert case_tol.metric in METRIC_REGISTRY, (
            f"tier-1 case {case_name!r} uses metric {case_tol.metric!r} "
            f"with no METRIC_REGISTRY entry"
        )


def test_barotropic_wave_phase_speed_passes(tier_tolerances, synthetic_passing_bundle):
    case_tol = tier_tolerances.cases["barotropic_wave_phase_speed"]
    measured = METRIC_REGISTRY[case_tol.metric](synthetic_passing_bundle)
    record = diff.diff_metric(
        case_name="barotropic_wave_phase_speed", tier=TIER,
        metric=case_tol.metric, measured=measured,
        tolerance_window=case_tol.tolerance_window,
    )
    assert record.passed, record.format()


def test_igw_dispersion_rms_zero_for_exact_match(tier_tolerances, synthetic_passing_bundle):
    case_tol = tier_tolerances.cases["inertia_gravity_wave"]
    measured = METRIC_REGISTRY[case_tol.metric](synthetic_passing_bundle)
    record = diff.diff_metric(
        case_name="inertia_gravity_wave", tier=TIER,
        metric=case_tol.metric, measured=measured,
        tolerance_window=case_tol.tolerance_window,
    )
    assert record.passed, record.format()
    assert measured == pytest.approx(0.0, abs=1e-12)


def test_wrong_phase_speed_fails(tier_tolerances, synthetic_failing_bundle):
    case_tol = tier_tolerances.cases["barotropic_wave_phase_speed"]
    measured = METRIC_REGISTRY[case_tol.metric](synthetic_failing_bundle)
    record = diff.diff_metric(
        case_name="barotropic_wave_phase_speed", tier=TIER,
        metric=case_tol.metric, measured=measured,
        tolerance_window=case_tol.tolerance_window,
    )
    assert not record.passed
    assert record.triage_hint is not None
