"""Stage-program tests for NEMO's key_RK3 active tracers."""

import jax.numpy as jnp
import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module


def test_nemo_ws_tracer_stage_polynomial(monkeypatch):
    rate = 0.2

    def linear_flux_pair(a, b, *args, **kwargs):
        zeros_a = jnp.zeros_like(a)
        zeros_b = jnp.zeros_like(b)
        return (rate * a, zeros_a), (rate * b, zeros_b)

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", linear_flux_pair)
    a0 = jnp.array([[[2.0]]], dtype=jnp.float64)
    b0 = jnp.array([[[3.0]]], dtype=jnp.float64)
    ones = jnp.ones_like(a0)
    a, b = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", ones, ones, jnp.ones((1, 1, 2)),
        ones, ones, ones, ones, object(), 0.5, ones,
    )
    z = -0.5 * rate
    amplification = 1.0 + z + z * z / 2.0 + z * z * z / 6.0
    np.testing.assert_allclose(np.asarray(a), 2.0 * amplification, rtol=0, atol=2e-16)
    np.testing.assert_allclose(np.asarray(b), 3.0 * amplification, rtol=0, atol=2e-16)


def test_nemo_ws_tracer_zero_flux_is_exact_identity(monkeypatch):
    def zero_flux_pair(a, b, *args, **kwargs):
        return (jnp.zeros_like(a), jnp.zeros_like(a)), (
            jnp.zeros_like(b), jnp.zeros_like(b))

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", zero_flux_pair)
    tracer = jnp.array([[[1.25, -2.0]]], dtype=jnp.float64)
    h = jnp.ones_like(tracer)
    got_a, got_b = model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", h, h, jnp.ones((1, 1, 3)),
        h, h, h, h, object(), 2.0, h,
    )
    np.testing.assert_array_equal(np.asarray(got_a), np.asarray(tracer))
    np.testing.assert_array_equal(np.asarray(got_b), np.asarray(tracer))
