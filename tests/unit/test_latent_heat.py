"""The one latent-heat family: values, slopes, closure, phase weighting, AD, jit."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import (
    latent_heat_fusion,
    latent_heat_sublimation,
    latent_heat_vaporization,
    surface_latent_heat,
)

T0 = constants.T_freeze


def test_constants_close_identically():
    assert constants.L_s == constants.L_v + constants.L_f


@pytest.mark.parametrize("fn, ref", [
    (latent_heat_vaporization, constants.L_v),
    (latent_heat_sublimation, constants.L_s),
    (latent_heat_fusion, constants.L_f),
])
def test_recovers_the_constant_at_the_freezing_point(fn, ref):
    assert float(fn(jnp.asarray(T0))) == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("fn, slope", [
    (latent_heat_vaporization, -(constants.c_pw - constants.c_pv)),   # -2372
    (latent_heat_sublimation, constants.c_pv - constants.c_pi),        # -260
    (latent_heat_fusion, constants.c_pw - constants.c_pi),             # +2112
])
def test_slopes_are_the_kirchhoff_specific_heat_differences(fn, slope):
    g = jax.grad(lambda T: fn(T))(jnp.asarray(T0 + 12.0))
    assert float(g) == pytest.approx(slope, rel=1e-12)


def test_sublimation_decreases_with_temperature():
    assert float(latent_heat_sublimation(jnp.asarray(T0 - 10.0))) > constants.L_s
    assert float(latent_heat_sublimation(jnp.asarray(T0 + 10.0))) < constants.L_s


def test_fusion_is_sublimation_minus_vaporization_everywhere():
    T = jnp.linspace(T0 - 40.0, T0 + 40.0, 17)
    np.testing.assert_allclose(
        np.asarray(latent_heat_fusion(T)),
        np.asarray(latent_heat_sublimation(T) - latent_heat_vaporization(T)),
        rtol=1e-14, atol=0.0)


def test_warm_sst_value_is_about_three_percent_below_the_constant():
    L30 = float(latent_heat_vaporization(jnp.asarray(T0 + 30.0)))
    assert 0.965 < L30 / constants.L_v < 0.975


def test_emanuel_liquid_heat_capacity_kwarg_changes_only_the_slope():
    T = jnp.asarray(T0 + 20.0)
    a = float(latent_heat_vaporization(T)); b = float(latent_heat_vaporization(T, c_liquid=4190.0))
    assert b - a == pytest.approx((constants.c_pw - 4190.0) * 20.0, rel=1e-9)
    assert float(latent_heat_vaporization(jnp.asarray(T0), c_liquid=4190.0)) == constants.L_v


def test_surface_latent_heat_is_a_fractional_blend_with_a_finite_nonzero_gradient():
    T = jnp.asarray(T0 - 3.0)
    for f in (0.0, 0.3, 1.0):
        want = (1 - f) * latent_heat_vaporization(T) + f * latent_heat_sublimation(T)
        assert float(surface_latent_heat(T, f)) == pytest.approx(float(want), rel=1e-14)
    g_f = jax.grad(lambda f: surface_latent_heat(T, f))(0.3)
    assert float(g_f) == pytest.approx(
        float(latent_heat_sublimation(T) - latent_heat_vaporization(T)), rel=1e-12)
    g_T = jax.grad(lambda t: surface_latent_heat(t, 0.3))(T)
    assert np.isfinite(float(g_T)) and float(g_T) != 0.0


def test_dtype_preserving_and_jit_parity():
    T32 = jnp.asarray([T0 - 5.0, T0 + 25.0], dtype=jnp.float32)
    for fn in (latent_heat_vaporization, latent_heat_sublimation, latent_heat_fusion):
        out = fn(T32)
        assert out.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(jax.jit(fn)(T32)), np.asarray(out))
    f = jnp.asarray([0.0, 0.5], dtype=jnp.float32)
    assert surface_latent_heat(T32, f).dtype == jnp.float32
