"""Layer A — Tier 4 (wind-driven gyres: Munk / Stommel / Sverdrup)."""

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

TIER = 4
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures" / "tier4_wind_driven_gyres.json"
)


def _metric_boundary_width_over_munk(bundle: artifacts.ArtifactBundle) -> float:
    u_midline = bundle.snapshots.get("gyre/u_midline")
    x = bundle.snapshots.get("gyre/x_midline_m")
    A_h = bundle.scalars.get("A_h")
    beta = bundle.scalars.get("beta")
    if u_midline is None or x is None or A_h is None or beta is None:
        return float("nan")
    width = metrics.boundary_layer_width(u_midline, x)
    return width / references.munk_width(float(A_h), float(beta))


def _metric_peak_psi_over_sverdrup(bundle: artifacts.ArtifactBundle) -> float:
    psi_bt = bundle.snapshots.get("gyre/psi_bt")
    curl_tau_peak = bundle.scalars.get("curl_tau_peak")
    beta = bundle.scalars.get("beta")
    if psi_bt is None or curl_tau_peak is None or beta is None:
        return float("nan")
    v_sv = references.sverdrup_transport(np.asarray([float(curl_tau_peak)]),
                                         beta=float(beta))[0]
    return float(np.max(np.abs(psi_bt))) / abs(v_sv)


def _metric_sverdrup_residual_relative(bundle: artifacts.ArtifactBundle) -> float:
    v_int = bundle.snapshots.get("gyre/v_integrated_interior")
    curl_tau = bundle.snapshots.get("gyre/curl_tau_interior")
    beta = bundle.scalars.get("beta")
    if v_int is None or curl_tau is None or beta is None:
        return float("nan")
    res = metrics.sverdrup_residual(v_int, curl_tau, beta=float(beta))
    scale = float(np.mean(np.abs(curl_tau)))
    return float(np.mean(res) / max(scale, 1e-30))


METRIC_REGISTRY: dict[str, Callable[[artifacts.ArtifactBundle], float]] = {
    "boundary_width_over_munk": _metric_boundary_width_over_munk,
    "peak_psi_bt_over_sverdrup": _metric_peak_psi_over_sverdrup,
    "sverdrup_residual_relative": _metric_sverdrup_residual_relative,
}


@pytest.fixture(scope="module")
def tier_tolerances():
    return tolerances.load_tier_file(FIXTURE_PATH)


@pytest.fixture
def synthetic_passing_bundle(tmp_path):
    case_dir = tmp_path / "regional_gyre" / "latlon" / "24x48"
    A_h = 1.0e3
    beta = 2.0e-11
    munk = references.munk_width(A_h, beta)
    # Gaussian u-midline whose FWHM matches munk_width
    sigma = munk / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = np.linspace(0.0, 4.0e6, 4001)
    u_mid = np.exp(-((x - 2.0e6) ** 2) / (2.0 * sigma ** 2))
    # Sverdrup balance: v_int = curl_tau / (rho * beta), peak ψ_bt magnitude ≈ V_sv
    curl_tau_peak = 1.0e-7
    v_sv = float(references.sverdrup_transport(
        np.asarray([curl_tau_peak]), beta=beta)[0])
    psi_bt = np.full((20, 20), v_sv)
    # Interior balance: v_int exactly matches curl_tau / (rho*beta)
    curl_interior = np.full((15, 15), curl_tau_peak)
    v_int_interior = curl_interior / (references.constants.rho_water * beta)
    artifacts.write_synthetic(
        case_dir,
        scalars={
            "status": "PASS",
            "A_h": A_h,
            "beta": beta,
            "curl_tau_peak": curl_tau_peak,
        },
        snapshots={
            "gyre": {
                "u_midline": u_mid,
                "x_midline_m": x,
                "psi_bt": psi_bt,
                "v_integrated_interior": v_int_interior,
                "curl_tau_interior": curl_interior,
            },
        },
    )
    return artifacts.load(case_dir)


def test_every_tier4_case_has_a_metric_function(tier_tolerances):
    for case_name, case_tol in tier_tolerances.cases.items():
        assert case_tol.metric in METRIC_REGISTRY, (
            f"tier-4 case {case_name!r} uses metric {case_tol.metric!r} "
            f"with no METRIC_REGISTRY entry"
        )


@pytest.mark.parametrize(
    "case_name",
    [
        "regional_gyre_boundary_width_over_munk",
        "regional_gyre_peak_psi_over_sverdrup",
        "global_barotropic_wind_sverdrup_residual",
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
