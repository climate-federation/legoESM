"""Unit tests for legoesm.ocean.fidelity.references (analytical formulas)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.fidelity import references


def test_kelvin_wave_speed_matches_sqrt_gH():
    H = 4000.0
    assert references.kelvin_wave_speed(H) == pytest.approx(math.sqrt(constants.g * H))


@pytest.mark.parametrize("bad_H", [0.0, -1.0])
def test_kelvin_wave_speed_rejects_nonpositive(bad_H):
    with pytest.raises(ValueError):
        references.kelvin_wave_speed(bad_H)


def test_igw_omega_zero_wavenumber_reduces_to_f():
    f = 1.0e-4
    H = 4000.0
    omega = references.igw_omega(0.0, 0.0, H, f)
    assert float(omega) == pytest.approx(f)


def test_igw_omega_high_k_dominant_term():
    f = 1.0e-4
    H = 4000.0
    k = 1.0e-4  # large enough that gH k^2 >> f^2
    omega = float(references.igw_omega(k, 0.0, H, f))
    assert omega == pytest.approx(math.sqrt(constants.g * H) * k, rel=1e-2)


def test_igw_omega_array_broadcast_monotone_in_k():
    k = np.array([0.0, 1e-5, 2e-5, 5e-5])
    l = np.zeros_like(k)
    out = references.igw_omega(k, l, H=4000.0, f=1.0e-4)
    assert out.shape == k.shape
    assert np.all(np.diff(out) > 0)


def test_igw_omega_rejects_nonpositive_H():
    with pytest.raises(ValueError):
        references.igw_omega(1e-5, 0.0, H=0.0, f=1e-4)


def test_rossby_dispersion_sign_convention():
    omega = references.rossby_dispersion(k=1e-6, l=0.0, beta=2e-11, L_d=5.0e4)
    assert float(omega) < 0


def test_rossby_dispersion_rejects_nonpositive_L_d():
    with pytest.raises(ValueError):
        references.rossby_dispersion(k=1e-6, l=0.0, beta=2e-11, L_d=0.0)


def test_eady_growth_rate_matches_textbook_coefficient():
    f = 1.0e-4
    Lambda = 1.0e-3
    N = 1.0e-2
    expected = 0.31 * f * Lambda / N
    assert references.eady_growth_rate(f, Lambda, N) == pytest.approx(expected)


def test_eady_growth_rate_uses_absolute_values():
    out_pos = references.eady_growth_rate(f=1e-4, Lambda=1e-3, N=1e-2)
    out_neg = references.eady_growth_rate(f=-1e-4, Lambda=-1e-3, N=1e-2)
    assert out_pos == pytest.approx(out_neg)
    assert out_pos > 0


def test_eady_growth_rate_rejects_nonpositive_N():
    with pytest.raises(ValueError):
        references.eady_growth_rate(f=1e-4, Lambda=1e-3, N=0.0)


def test_eady_wavelength_formula():
    N, H, f = 1e-2, 4000.0, 1e-4
    expected = 2.0 * math.pi * N * H / (1.61 * f)
    assert references.eady_most_unstable_wavelength(N, H, f) == pytest.approx(expected)


def test_eady_wavelength_rejects_zero_f():
    with pytest.raises(ValueError, match="f"):
        references.eady_most_unstable_wavelength(N=1e-2, H=4000.0, f=0.0)


def test_stommel_munk_widths_match_definitions():
    r, beta, A_h = 1.0e-6, 2.0e-11, 1.0e3
    assert references.stommel_width(r, beta) == pytest.approx(r / beta)
    assert references.munk_width(A_h, beta) == pytest.approx((A_h / beta) ** (1.0 / 3.0))


@pytest.mark.parametrize(
    "fn,args",
    [
        (references.stommel_width, (0.0, 2e-11)),
        (references.stommel_width, (1e-6, 0.0)),
        (references.munk_width, (0.0, 2e-11)),
        (references.munk_width, (1e3, 0.0)),
    ],
)
def test_widths_reject_nonpositive_inputs(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_sverdrup_transport_uses_density_and_beta():
    curl = np.array([1.0e-7, -1.0e-7, 0.0])
    out = references.sverdrup_transport(curl, beta=2.0e-11)
    np.testing.assert_allclose(out, curl / (constants.rho_water * 2.0e-11))


def test_sverdrup_transport_accepts_custom_density():
    curl = np.array([1.0e-7])
    out = references.sverdrup_transport(curl, beta=2.0e-11, rho_0=1025.0)
    np.testing.assert_allclose(out, curl / (1025.0 * 2.0e-11))


def test_sverdrup_transport_rejects_nonpositive_beta():
    with pytest.raises(ValueError):
        references.sverdrup_transport(np.array([1e-7]), beta=0.0)


def test_held_larichev_slope_returns_minus_three():
    assert references.held_larichev_slope() == -3.0


def test_coriolis_equator_zero():
    assert float(references.coriolis(0.0)) == pytest.approx(0.0)


def test_coriolis_pole_full_omega():
    assert float(references.coriolis(math.pi / 2)) == pytest.approx(2.0 * constants.Omega)


def test_coriolis_southern_hemisphere_negative():
    assert float(references.coriolis(-math.pi / 4)) < 0


def test_beta_plane_equator_uses_R_earth():
    expected = 2.0 * constants.Omega / constants.R_earth
    assert float(references.beta_plane(0.0)) == pytest.approx(expected)


def test_beta_plane_pole_zero():
    assert float(references.beta_plane(math.pi / 2)) == pytest.approx(0.0)


def test_thermal_wind_shear_sign_for_dense_north():
    out = references.thermal_wind_shear(rho_y=1.0, f=1e-4)
    assert float(out) < 0


def test_thermal_wind_zero_f_raises():
    with pytest.raises(ValueError, match="f"):
        references.thermal_wind_shear(rho_y=1.0, f=0.0)


def test_thermal_wind_rejects_nonpositive_rho_0():
    with pytest.raises(ValueError):
        references.thermal_wind_shear(rho_y=1.0, f=1e-4, rho_0=0.0)
