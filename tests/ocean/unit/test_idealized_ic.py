"""Direct tests for the shared analytic-IC primitives.

Covers :mod:`legoesm.ocean.experiments.idealized_ic` and pins each helper to
the exact hand-rolled expression it replaced (so the migrated experiments keep
byte-identical ICs).
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.experiments.idealized_ic import (
    coriolis_f,
    coriolis_f_safe,
    gaussian_lat_envelope,
    integrate_thermal_wind_bottom_up,
    thermal_wind_dudz_coeff,
)

from legoesm import constants


def test_coriolis_f_matches_formula():
    lat = np.radians(np.array([-45.0, 0.0, 25.0, 60.0]))
    np.testing.assert_array_equal(
        coriolis_f(lat), 2.0 * constants.Omega * np.sin(lat))


def test_coriolis_f_safe_matches_silvestri_clamp():
    """The exact clamp the Silvestri jet hand-rolled."""
    lat = np.radians(np.array([-30.0, -1e-9, 0.0, 1e-9, 30.0]))
    f = 2.0 * constants.Omega * np.sin(lat)
    expected = np.where(np.abs(f) < 1e-12, np.sign(f + 1e-30) * 1e-12, f)
    np.testing.assert_array_equal(coriolis_f_safe(lat), expected)
    # Equator is floored, not zero ⇒ 1/f stays finite.
    assert np.all(np.isfinite(1.0 / coriolis_f_safe(lat)))


def test_gaussian_envelope_matches_formula_and_peaks_at_center():
    lat = np.array([10.0, 25.0, 40.0])
    center, width = 25.0, 10.0
    expected = np.exp(-((lat - center) / width) ** 2)
    got = gaussian_lat_envelope(lat, center, width)
    np.testing.assert_array_equal(got, expected)
    assert got[1] == pytest.approx(1.0)            # peak at the centre
    assert got[0] < 1.0 and got[2] < 1.0


def test_thermal_wind_coeff_sign_and_value():
    f0 = 2.0 * constants.Omega * np.sin(np.radians(45.0))
    alpha_T = 2.0e-4
    coeff = thermal_wind_dudz_coeff(f0, alpha_T)
    assert coeff == pytest.approx(-constants.g * alpha_T / f0)
    assert coeff < 0.0


def test_integrate_thermal_wind_matches_reference_loop():
    """Reproduce the eady_instability bottom-up loop + depth-mean removal."""
    rng = np.random.default_rng(0)
    nlat, nlev = 5, 6
    dTdy = rng.standard_normal((nlat, nlev))
    dz = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    coeff = -3.0e-3

    # Reference: the exact hand-rolled loop.
    U = np.zeros((nlat, nlev))
    for k in range(nlev - 2, -1, -1):
        dz_half = 0.5 * (dz[k] + dz[k + 1]) if k + 1 < nlev else dz[k]
        U[:, k] = U[:, k + 1] + coeff * dTdy[:, k] * dz_half
    H_col = np.sum(dz)
    U_bar_ref = np.sum(U * dz[np.newaxis, :], axis=1) / H_col
    U_ref = U - U_bar_ref[:, np.newaxis]

    U_got, U_bar_got = integrate_thermal_wind_bottom_up(dTdy, dz, coeff)
    np.testing.assert_array_equal(U_got, U_ref)
    np.testing.assert_array_equal(U_bar_got, U_bar_ref)
    # Baroclinic ⇒ depth mean exactly zero.
    np.testing.assert_allclose(np.sum(U_got * dz, axis=-1), 0.0, atol=1e-12)


def test_integrate_thermal_wind_zero_dtdy_is_zero():
    dz = np.array([10.0, 20.0, 30.0])
    U, U_bar = integrate_thermal_wind_bottom_up(np.zeros((4, 3)), dz, -1.0)
    np.testing.assert_array_equal(U, np.zeros((4, 3)))
    np.testing.assert_array_equal(U_bar, np.zeros(4))


def test_integrate_thermal_wind_no_forced_cast_fp32():
    """The helper must NOT force float64 on the inputs: a float32 ∂T/∂y must
    reproduce the old loop bit-for-bit (only the U accumulator is float64)."""
    rng = np.random.default_rng(3)
    nlat, nlev = 4, 5
    dTdy = rng.standard_normal((nlat, nlev)).astype(np.float32)
    dz = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    coeff = -3.0e-3

    # Exact old loop: U accumulator float64, dTdy/dz at their native dtype.
    U = np.zeros((nlat, nlev), dtype=np.float64)
    for k in range(nlev - 2, -1, -1):
        dz_half = 0.5 * (dz[k] + dz[k + 1]) if k + 1 < nlev else dz[k]
        U[:, k] = U[:, k + 1] + coeff * dTdy[:, k] * dz_half
    H_col = np.sum(dz)
    U_bar_ref = np.sum(U * dz[np.newaxis, :], axis=1) / H_col
    U_ref = U - U_bar_ref[:, np.newaxis]

    U_got, U_bar_got = integrate_thermal_wind_bottom_up(dTdy, dz, coeff)
    np.testing.assert_array_equal(U_got, U_ref)
    np.testing.assert_array_equal(U_bar_got, U_bar_ref)


def test_eady_latlon_thermal_wind_callsite_byte_identical():
    """The Eady-latlon migration (zonal-mean ∂T/∂y → helper) reproduces the
    old inline bottom-up loop exactly."""
    rng = np.random.default_rng(5)
    nlat_faces, nlev = 7, 6
    dTdy_zonal = rng.standard_normal((nlat_faces, nlev))
    dz = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0])
    coeff = thermal_wind_dudz_coeff(
        coriolis_f(np.radians(45.0)), 2.0e-4)

    # Old inline body.
    U_zonal = np.zeros((nlat_faces, nlev), dtype=np.float64)
    for k in range(nlev - 2, -1, -1):
        dz_half = 0.5 * (dz[k] + dz[k + 1]) if k + 1 < nlev else dz[k]
        U_zonal[:, k] = U_zonal[:, k + 1] + coeff * dTdy_zonal[:, k] * dz_half
    H_col = np.sum(dz)
    U_bar_old = np.sum(U_zonal * dz[np.newaxis, :], axis=1) / H_col
    U_old = U_zonal - U_bar_old[:, np.newaxis]

    U_new, U_bar_new = integrate_thermal_wind_bottom_up(dTdy_zonal, dz, coeff)
    np.testing.assert_array_equal(U_new, U_old)
    np.testing.assert_array_equal(U_bar_new, U_bar_old)
