"""Tests for the LES intercomparison gate (D7 / gate-0)."""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.atmosphere.les_suite.intercomparison import (
    CBL_ENVELOPE,
    cbl_diagnostics,
    evaluate_cbl_gate,
    gate_from_cbl_profiles,
)

from legoesm import constants

G = float(constants.g)
Q0 = 0.06
THETA0 = 300.0
ZI = 800.0


def _canonical_cbl(nz=64, Lz=1600.0, zi=ZI, sigw_ratio=0.6,
                   surface_ratio=1.0, entrain_ratio=-0.2, gamma_ml=0.0):
    """Build a synthetic dry-CBL mean profile matching the convective envelope.

    Returns (z, theta, ww, wth). ``gamma_ml`` sets the mixed-layer lapse [K/m]
    (0 = well-mixed); the flux is linear from surface_ratio*Q0 at the ground to
    entrain_ratio*Q0 at z_i, then decays to 0 above.
    """
    z = np.linspace(Lz / nz * 0.5, Lz, nz)  # surface-first
    w_star = (G / THETA0 * Q0 * zi) ** (1.0 / 3.0)
    # theta: mixed layer (lapse gamma_ml) below z_i + a LOCALIZED capping inversion
    # (a saturating tanh jump whose gradient peaks AT z_i, so the max-∂θ/∂z z_i
    # diagnosis lands on z_i and the free atmosphere above is quiescent).
    theta = THETA0 + gamma_ml * np.minimum(z, zi) + 2.0 * np.tanh((z - zi) / 100.0)
    theta = np.where(z > zi, theta, THETA0 + gamma_ml * z)
    # w variance: parabola peaking mid-CBL at (sigw_ratio * w_star)^2
    zeta = np.clip(z / zi, 0.0, 1.0)
    shape = np.clip(4.0 * zeta * (1.0 - zeta), 0.0, 1.0)  # 0 at 0 and z_i, 1 mid
    ww = (sigw_ratio * w_star) ** 2 * shape
    # heat flux: linear surface_ratio -> entrain_ratio through the CBL, then the
    # negative entrainment flux relaxes to 0 over a thin inversion layer above z_i.
    dz_ent = 150.0
    in_cbl = Q0 * (surface_ratio + (entrain_ratio - surface_ratio) * (z / zi))
    in_ent = Q0 * entrain_ratio * np.clip(1.0 - (z - zi) / dz_ent, 0.0, 1.0)
    wth = np.where(z <= zi, in_cbl, in_ent)
    return z, theta, ww, wth


def test_canonical_profile_passes_gate():
    z, theta, ww, wth = _canonical_cbl()
    gate = gate_from_cbl_profiles(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    assert gate.passed, gate.report()
    # z_i recovered near the imposed inversion
    assert abs(gate.diagnostics.z_i_m - ZI) < 100.0


def test_diagnostics_values_reasonable():
    z, theta, ww, wth = _canonical_cbl()
    diag = cbl_diagnostics(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    assert 0.5 < diag.sigma_w_over_wstar_max < 0.72
    assert abs(diag.mixed_layer_dtheta_dz_mK_m) < 0.6
    assert 0.85 < diag.surface_flux_ratio < 1.08
    assert -0.4 < diag.entrainment_flux_ratio < -0.05


def test_not_well_mixed_fails():
    # a strongly stratified "mixed layer" (0.01 K/m = 10 mK/m) must fail
    z, theta, ww, wth = _canonical_cbl(gamma_ml=0.01)
    gate = gate_from_cbl_profiles(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    assert not gate.passed
    ml = next(r for r in gate.results if r.band.name == "mixed_layer_dtheta_dz_mK_m")
    assert not ml.passed


def test_wrong_sigma_w_fails():
    # variance too weak (0.3 w_*) is outside the band
    z, theta, ww, wth = _canonical_cbl(sigw_ratio=0.3)
    gate = gate_from_cbl_profiles(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    assert not gate.passed
    r = next(x for x in gate.results if x.band.name == "sigma_w_over_wstar_max")
    assert not r.passed


def test_no_entrainment_fails():
    # zero entrainment flux (ratio 0 at z_i) is outside the [-0.4,-0.05] band
    z, theta, ww, wth = _canonical_cbl(entrain_ratio=0.0)
    gate = gate_from_cbl_profiles(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    r = next(x for x in gate.results if x.band.name == "entrainment_flux_ratio")
    assert not r.passed


def test_entrainment_uses_flux_minimum_not_gradient_max():
    # Regression (gate-0): in a real CBL the entrainment flux MINIMUM sits at the
    # base of the inversion, BELOW the max-∂θ/∂z height where the flux has already
    # recovered toward 0. The metric must report the minimum (~-0.15), not the
    # near-zero flux at the θ-gradient peak.
    nz = 64
    z = np.linspace(1600.0 / nz * 0.5, 1600.0, nz)
    zi_grad = 1008.0          # θ-gradient max (mid-inversion)
    z_fluxmin = 942.0         # entrainment flux minimum (inversion base, below)
    # localized inversion peaking at zi_grad
    theta = 300.0 + 2.0 * np.tanh((z - zi_grad) / 60.0)
    theta = np.where(z > zi_grad, theta, 300.0)
    w_star = (G / THETA0 * Q0 * zi_grad) ** (1.0 / 3.0)
    zeta = np.clip(z / zi_grad, 0.0, 1.0)
    ww = (0.6 * w_star) ** 2 * np.clip(4.0 * zeta * (1.0 - zeta), 0.0, 1.0)
    # flux: linear 1 -> -0.15 down to z_fluxmin, then recovers to ~0 by/above zi_grad
    r = np.where(
        z <= z_fluxmin,
        1.0 - 1.15 * (z / z_fluxmin),
        -0.15 * np.clip(1.0 - (z - z_fluxmin) / (zi_grad - z_fluxmin), 0.0, 1.0),
    )
    wth = Q0 * r
    diag = cbl_diagnostics(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    # z_i diagnosed at the gradient max ...
    assert abs(diag.z_i_m - zi_grad) < 40.0
    # ... but the entrainment ratio comes from the flux minimum (~-0.15), NOT the
    # near-zero flux at the gradient max.
    assert -0.25 < diag.entrainment_flux_ratio < -0.10
    assert evaluate_cbl_gate(diag).passed


def test_negative_Q0_rejected():
    z, theta, ww, wth = _canonical_cbl()
    with pytest.raises(ValueError):
        cbl_diagnostics(z, theta, ww, wth, Q0=-0.06, theta0=THETA0)


def test_shape_mismatch_rejected():
    z, theta, ww, wth = _canonical_cbl()
    with pytest.raises(ValueError):
        cbl_diagnostics(z[:-1], theta, ww, wth, Q0=Q0, theta0=THETA0)


def test_envelope_bands_ordered():
    # every band must have lo <= target <= hi (a well-formed accept band)
    for b in CBL_ENVELOPE:
        assert b.lo <= b.target <= b.hi, b.name


def test_report_contains_verdict():
    z, theta, ww, wth = _canonical_cbl()
    gate = gate_from_cbl_profiles(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    text = gate.report()
    assert "PASS" in text or "FAIL" in text
    assert "sigma_w_over_wstar_max" in text


def test_gate_reevaluation_consistent():
    z, theta, ww, wth = _canonical_cbl()
    diag = cbl_diagnostics(z, theta, ww, wth, Q0=Q0, theta0=THETA0)
    assert evaluate_cbl_gate(diag).passed
