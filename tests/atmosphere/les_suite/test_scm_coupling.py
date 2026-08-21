"""Tests for the SCM↔LES coupling primitives (regrid + θ↔T)."""
from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.atmosphere.les_suite.bridge import LESTruth
from legoesm.atmosphere.les_suite.scm_coupling import (
    T_from_theta,
    interp_profile,
    liquid_water_theta,
    regrid_truth,
    saturation_adjust,
    theta_from_temperature,
)
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants


def test_saturation_adjust_clear_column_is_identity():
    # A subsaturated column carries no cloud: q_c=0, θ=θ_l, q_v=q_t.
    p = jnp.array([90000.0, 85000.0, 80000.0])
    exner = exner_function(p)
    theta_l = jnp.array([300.0, 301.0, 302.0])
    q_t = jnp.array([0.002, 0.002, 0.002])
    theta, q_v, q_c = saturation_adjust(theta_l, q_t, exner, p)
    assert jnp.allclose(q_c, 0.0, atol=1e-12)
    assert jnp.allclose(theta, theta_l, atol=1e-9)
    assert jnp.allclose(q_v, q_t, atol=1e-12)


def test_saturation_adjust_saturated_conserves_qt_and_inverts_theta_l():
    # A saturated column splits q_t into q_v(+q_c) and warms θ above θ_l; the
    # result is the exact inverse of liquid_water_theta, and q_t is conserved.
    p = jnp.array([90000.0, 85000.0, 80000.0])
    exner = exner_function(p)
    theta_l = jnp.array([289.0, 289.5, 290.0])           # cold marine-Sc mixed layer
    q_t = jnp.array([0.020, 0.022, 0.024])               # well above saturation
    theta, q_v, q_c = saturation_adjust(theta_l, q_t, exner, p)
    assert bool(jnp.all(q_c > 0.0))                      # cloud forms
    assert jnp.allclose(q_v + q_c, q_t, atol=1e-12)      # total water conserved
    assert bool(jnp.all(theta > theta_l))               # latent heating raises θ
    # round-trip: liquid_water_theta(θ, q_c) recovers θ_l exactly.
    assert jnp.allclose(liquid_water_theta(theta, q_c, exner), theta_l, atol=1e-6)


def test_theta_T_roundtrip():
    # θ→T→θ must be the identity (exact inverse via the Exner factor)
    p = jnp.array([1.0e5, 9.0e4, 8.0e4])
    theta = jnp.array([300.0, 302.0, 305.0])
    T = T_from_theta(theta, p)
    theta_back = theta_from_temperature(T, p)
    assert jnp.allclose(theta_back, theta, atol=1e-6)


def test_theta_equals_T_at_reference_pressure():
    # at p = p_ref, θ == T (Exner factor = 1)
    p = jnp.full((4,), constants.p_ref)
    T = jnp.array([280.0, 290.0, 300.0, 310.0])
    assert jnp.allclose(theta_from_temperature(T, p), T, atol=1e-6)


def test_theta_exceeds_T_above_surface():
    # above the surface (p < p_ref) θ > T
    p = jnp.array([8.0e4])
    T = jnp.array([280.0])
    assert float(theta_from_temperature(T, p)[0]) > float(T[0])


# --- interp_profile -----------------------------------------------------------
def test_interp_identity_same_grid():
    z = jnp.linspace(10.0, 1000.0, 11)
    v = 300.0 + 0.01 * z
    assert jnp.allclose(interp_profile(v, z, z), v, atol=1e-6)


def test_interp_linear_midpoints():
    z_src = jnp.array([0.0, 100.0, 200.0])
    v = jnp.array([10.0, 20.0, 30.0])
    z_dst = jnp.array([50.0, 150.0])
    out = interp_profile(v, z_src, z_dst)
    assert jnp.allclose(out, jnp.array([15.0, 25.0]))


def test_interp_clamps_out_of_range():
    z_src = jnp.array([100.0, 200.0, 300.0])
    v = jnp.array([1.0, 2.0, 3.0])
    z_dst = jnp.array([50.0, 400.0])  # below and above source
    out = interp_profile(v, z_src, z_dst)
    assert jnp.allclose(out, jnp.array([1.0, 3.0]))  # clamped to endpoints


def test_interp_time_series():
    z_src = jnp.array([0.0, 100.0, 200.0])
    v = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])  # (nt=2, nz=3)
    z_dst = jnp.array([50.0, 150.0])
    out = interp_profile(v, z_src, z_dst)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[15.0, 25.0], [45.0, 55.0]]))


def test_interp_rejects_non_increasing_src():
    z_src = jnp.array([0.0, 200.0, 100.0])
    with pytest.raises(ValueError):
        interp_profile(jnp.array([1.0, 2.0, 3.0]), z_src, jnp.array([50.0]))


def test_interp_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        interp_profile(jnp.array([1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]),
                       jnp.array([0.5]))


# --- regrid_truth -------------------------------------------------------------
def _truth(nz=6, nt=None):
    z = jnp.linspace(10.0, 1600.0, nz)
    if nt is None:
        return LESTruth(
            case_name="cbl", heights_m=z, times_s=jnp.array([3600.0]),
            theta=300.0 + 0.003 * z, u=jnp.full((nz,), 1.0),
            v=jnp.zeros((nz,)), wtheta=jnp.full((nz,), 0.05),
        )
    return LESTruth(
        case_name="cbl", heights_m=z, times_s=jnp.linspace(0.0, 3600.0, nt),
        theta=jnp.broadcast_to(300.0 + 0.003 * z, (nt, nz)),
        u=jnp.full((nt, nz), 1.0), v=jnp.zeros((nt, nz)),
        wtheta=jnp.full((nt, nz), 0.05),
    )


def test_regrid_truth_changes_grid_single_time():
    truth = _truth(nz=6)
    z_dst = jnp.linspace(10.0, 1600.0, 20)
    rg = regrid_truth(truth, z_dst)
    assert rg.heights_m.shape == (20,)
    assert rg.theta.shape == (20,)
    assert rg.qt is None
    # linear θ profile is preserved by linear interpolation
    assert jnp.allclose(rg.theta, 300.0 + 0.003 * z_dst, atol=1e-4)


def test_regrid_truth_time_series():
    truth = _truth(nz=6, nt=4)
    z_dst = jnp.linspace(10.0, 1600.0, 12)
    rg = regrid_truth(truth, z_dst)
    assert rg.theta.shape == (4, 12)
    assert rg.wtheta.shape == (4, 12)


def test_regrid_truth_moist_carries_qt():
    z = jnp.linspace(10.0, 1500.0, 6)
    truth = LESTruth(
        case_name="bomex", heights_m=z, times_s=jnp.array([3600.0]),
        theta=300.0 + 0.003 * z, u=jnp.full((6,), -8.0), v=jnp.zeros((6,)),
        wtheta=jnp.full((6,), 0.02), qt=0.016 - 1e-6 * z, wqt=jnp.full((6,), 1e-4),
    )
    rg = regrid_truth(truth, jnp.linspace(10.0, 1500.0, 15))
    assert rg.qt is not None and rg.qt.shape == (15,)
    assert rg.wqt is not None and rg.wqt.shape == (15,)
