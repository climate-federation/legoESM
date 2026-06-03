"""Unit tests for the metric helper functions in legoesm.ocean.fidelity.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.fidelity import metrics


# ---------------------------------------------------------------------------
# measured_phase_speed_2d
# ---------------------------------------------------------------------------

def test_phase_speed_recovers_known_traveling_wave():
    n_x, n_t = 256, 1024
    dx = 1.0e4   # 10 km
    dt = 60.0    # 1 min
    c_true = 100.0  # m/s
    k = 2 * math.pi / (dx * 16)  # 16 cells per wavelength
    omega = c_true * k
    x = np.arange(n_x) * dx
    t = np.arange(n_t) * dt
    X, T = np.meshgrid(x, t, indexing="ij")
    field = np.cos(k * X - omega * T)
    c_meas = metrics.measured_phase_speed_2d(field, dx, dt)
    assert c_meas == pytest.approx(c_true, rel=0.03)


def test_phase_speed_sign_flips_with_westward_wave():
    n_x, n_t = 256, 1024
    dx, dt = 1.0e4, 60.0
    k = 2 * math.pi / (dx * 16)
    omega = -100.0 * k  # westward (negative phase speed)
    X, T = np.meshgrid(np.arange(n_x) * dx, np.arange(n_t) * dt, indexing="ij")
    field = np.cos(k * X - omega * T)
    c = metrics.measured_phase_speed_2d(field, dx, dt)
    assert c < 0
    assert c == pytest.approx(-100.0, rel=0.03)


def test_phase_speed_rejects_bad_inputs():
    with pytest.raises(ValueError):
        metrics.measured_phase_speed_2d(np.zeros((5,)), 1.0, 1.0)
    with pytest.raises(ValueError):
        metrics.measured_phase_speed_2d(np.zeros((5, 5)), 0.0, 1.0)
    with pytest.raises(ValueError):
        metrics.measured_phase_speed_2d(np.zeros((5, 5)), 1.0, -1.0)


# ---------------------------------------------------------------------------
# adjustment_timescale
# ---------------------------------------------------------------------------

def test_adjustment_timescale_monotone_ramp():
    t = np.linspace(0.0, 100.0, 101)
    ke = np.minimum(t / 50.0, 1.0)
    # threshold 0.9 -> at t=45
    t_eq = metrics.adjustment_timescale(ke, t, plateau_frac=0.9)
    assert t_eq == pytest.approx(45.0, abs=1.0)


def test_adjustment_timescale_returns_last_if_never_crossed():
    t = np.linspace(0.0, 10.0, 11)
    ke = np.full_like(t, 1.0)
    # threshold = 1.0, never reaches 0.9 * peak strictly... actually 1.0 >= 0.9 always, so first idx 0
    # use plateau_frac that requires growth
    ke = np.linspace(0.0, 0.5, 11)
    t_eq = metrics.adjustment_timescale(ke, t, plateau_frac=0.9)
    # threshold = 0.9 * 0.5 = 0.45; crossed at index 9 (t = 9.0)
    assert t_eq == pytest.approx(9.0, abs=1.0)


def test_adjustment_timescale_rejects_bad_plateau_frac():
    t = np.linspace(0, 1, 5)
    ke = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        metrics.adjustment_timescale(ke, t, plateau_frac=0.0)
    with pytest.raises(ValueError):
        metrics.adjustment_timescale(ke, t, plateau_frac=1.0)


def test_adjustment_timescale_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.adjustment_timescale(np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------------
# thermal_wind_residual
# ---------------------------------------------------------------------------

def test_thermal_wind_residual_zero_for_balanced_state():
    nz, ny, nx = 20, 20, 4
    dz, dy = 100.0, 1.0e4
    f = 1.0e-4
    rho_0 = constants.rho_water
    # rho linear in y: rho = rho_y * y -> dρ/dy = rho_y
    rho_y_grad = 0.01  # kg/m^4
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    rho_profile = np.zeros((nz, ny, nx))
    for j in range(ny):
        rho_profile[:, j, :] = rho_y_grad * y[j]
    # u satisfying du/dz = -g/(rho_0 f) * dρ/dy = const
    du_dz = -constants.g * rho_y_grad / (rho_0 * f)
    u = np.zeros((nz, ny, nx))
    for k in range(nz):
        u[k, :, :] = du_dz * z[k]
    residual = metrics.thermal_wind_residual(rho_profile, u, f=f, dy=dy, dz=dz)
    assert np.max(np.abs(residual)) < 1e-12


def test_thermal_wind_residual_nonzero_when_unbalanced():
    nz, ny, nx = 8, 8, 2
    rho = np.random.default_rng(0).standard_normal((nz, ny, nx))
    u = np.zeros((nz, ny, nx))
    res = metrics.thermal_wind_residual(rho, u, f=1e-4, dy=1e4, dz=100.0)
    assert np.max(res) > 0


def test_thermal_wind_residual_validation():
    with pytest.raises(ValueError, match="f"):
        metrics.thermal_wind_residual(np.zeros((3, 3)), np.zeros((3, 3)), f=0.0, dy=1.0, dz=1.0)
    with pytest.raises(ValueError):
        metrics.thermal_wind_residual(np.zeros((3, 3)), np.zeros((3, 4)), f=1e-4, dy=1.0, dz=1.0)
    with pytest.raises(ValueError):
        metrics.thermal_wind_residual(np.zeros((3,)), np.zeros((3,)), f=1e-4, dy=1.0, dz=1.0)


# ---------------------------------------------------------------------------
# overflow_nose_descent
# ---------------------------------------------------------------------------

def test_overflow_nose_descent_recovers_known_front():
    n_t, n_z, n_y, n_x = 4, 10, 3, 5
    z = np.array([10.0, 30.0, 50.0, 80.0, 120.0, 200.0, 350.0, 600.0, 1000.0, 2000.0])
    rho = np.zeros((n_t, n_z, n_y, n_x))
    # Anomaly above threshold at deepening levels per timestep
    deepest_idx = [2, 4, 6, 8]
    for it, idx in enumerate(deepest_idx):
        rho[it, : idx + 1, :, :] = 1.0
    nose = metrics.overflow_nose_descent(rho, z, threshold=0.5)
    expected = np.array([z[i] for i in deepest_idx])
    np.testing.assert_allclose(nose, expected)


def test_overflow_nose_zero_when_no_threshold_crossed():
    z = np.array([10.0, 20.0, 30.0])
    rho = np.zeros((3, 3, 4, 4))
    nose = metrics.overflow_nose_descent(rho, z, threshold=1.0)
    np.testing.assert_array_equal(nose, np.zeros(3))


def test_overflow_nose_validates_shapes():
    with pytest.raises(ValueError):
        metrics.overflow_nose_descent(np.zeros((3,)), np.array([1.0]), threshold=0.1)
    with pytest.raises(ValueError):
        metrics.overflow_nose_descent(np.zeros((3, 4)), np.array([1.0, 2.0]), threshold=0.1)


# ---------------------------------------------------------------------------
# boundary_layer_width
# ---------------------------------------------------------------------------

def test_boundary_layer_width_gaussian_recovers_fwhm():
    x = np.linspace(0.0, 1.0e6, 1001)
    sigma = 5.0e4
    u = np.exp(-((x - 5.0e5) ** 2) / (2.0 * sigma ** 2))
    fwhm = metrics.boundary_layer_width(u, x)
    expected = 2.0 * sigma * math.sqrt(2.0 * math.log(2.0))
    assert fwhm == pytest.approx(expected, rel=0.02)


def test_boundary_layer_width_zero_profile():
    x = np.linspace(0.0, 1.0, 10)
    u = np.zeros_like(x)
    assert metrics.boundary_layer_width(u, x) == 0.0


def test_boundary_layer_width_validates():
    with pytest.raises(ValueError):
        metrics.boundary_layer_width(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        metrics.boundary_layer_width(np.zeros(5), np.zeros(4))


# ---------------------------------------------------------------------------
# sverdrup_residual
# ---------------------------------------------------------------------------

def test_sverdrup_residual_zero_for_balanced_flow():
    beta = 2.0e-11
    rho_0 = constants.rho_water
    curl_tau = np.array([1.0e-7, -2.0e-7, 0.0])
    v_int = curl_tau / (rho_0 * beta)
    res = metrics.sverdrup_residual(v_int, curl_tau, beta)
    np.testing.assert_allclose(res, 0.0, atol=1e-20)


def test_sverdrup_residual_validates():
    with pytest.raises(ValueError):
        metrics.sverdrup_residual(np.zeros(3), np.zeros(3), beta=0.0)
    with pytest.raises(ValueError):
        metrics.sverdrup_residual(np.zeros(3), np.zeros(3), beta=1e-11, rho_0=0.0)
    with pytest.raises(ValueError):
        metrics.sverdrup_residual(np.zeros(3), np.zeros(4), beta=1e-11)


# ---------------------------------------------------------------------------
# linear_growth_rate
# ---------------------------------------------------------------------------

def test_linear_growth_rate_recovers_known_sigma():
    sigma = 1.0e-5  # s^-1
    t = np.linspace(0.0, 1.0e6, 200)
    ke = np.exp(2.0 * sigma * t)
    measured = metrics.linear_growth_rate(ke, t, t_window=(2.0e5, 9.0e5))
    assert measured == pytest.approx(sigma, rel=1e-6)


def test_linear_growth_rate_window_validation():
    t = np.linspace(0.0, 10.0, 11)
    ke = np.exp(0.1 * t)
    with pytest.raises(ValueError):
        metrics.linear_growth_rate(ke, t, t_window=(5.0, 5.0))
    with pytest.raises(ValueError):
        metrics.linear_growth_rate(ke, t, t_window=(0.0, 0.5))   # < 3 samples


def test_linear_growth_rate_rejects_nonpositive_ke():
    t = np.linspace(0.0, 10.0, 11)
    ke = np.linspace(-1.0, 1.0, 11)
    with pytest.raises(ValueError, match="strictly positive"):
        metrics.linear_growth_rate(ke, t, t_window=(0.0, 5.0))


# ---------------------------------------------------------------------------
# bottom_form_stress
# ---------------------------------------------------------------------------

def test_bottom_form_stress_uniform_weights_mean():
    p_b = np.array([[1.0, 2.0], [3.0, 4.0]])
    dh = np.array([[1.0, 1.0], [1.0, 1.0]])
    out = metrics.bottom_form_stress(p_b, dh)
    assert out == pytest.approx(2.5)


def test_bottom_form_stress_with_weights():
    p_b = np.array([1.0, 2.0, 3.0])
    dh = np.array([1.0, 1.0, 1.0])
    w = np.array([0.0, 0.0, 1.0])
    assert metrics.bottom_form_stress(p_b, dh, weights=w) == pytest.approx(3.0)


def test_bottom_form_stress_validates_shapes():
    with pytest.raises(ValueError):
        metrics.bottom_form_stress(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        metrics.bottom_form_stress(np.zeros((2,)), np.zeros((2,)), weights=np.zeros((3,)))
