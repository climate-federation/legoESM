"""Direct tests for NEMO's live dynadv_up3 vertical momentum flux."""

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.vertical import nemo_up3_vertical_momentum_advection

jax.config.update("jax_enable_x64", True)


def _literal_reference(u, w, h, active):
    """Scalar-loop transcription of dynadv_up3.F90:271-365."""
    n = len(u)
    flux = np.zeros(n + 1, dtype=np.float64)
    previous_lap = 0.0
    for k in range(n - 1):
        d_here = (u[k] - u[k + 1]) * active[k] * active[k + 1]
        d_next = 0.0
        if k + 2 < n:
            d_next = (u[k + 1] - u[k + 2]) * active[k + 1] * active[k + 2]
        lap = d_here - d_next
        correction = lap if w[k + 1] > 0.0 else previous_lap
        flux[k + 1] = 0.5 * w[k + 1] * (
            u[k] + u[k + 1] - correction / 3.0)
        previous_lap = lap
    out = -(flux[:-1] - flux[1:]) / np.maximum(h, 1.0e-10)
    return out * active


def test_nemo_up3_matches_independent_literal_recurrence():
    u = np.array([0.2, -0.4, 0.8, 0.1, -0.2])
    w = np.array([0.0, 0.3, -0.2, 0.5, -0.1, 0.0])
    h = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    active = np.ones_like(u)
    got = nemo_up3_vertical_momentum_advection(
        jnp.asarray(u), jnp.asarray(w), jnp.asarray(h), jnp.asarray(active))
    np.testing.assert_allclose(
        np.asarray(got), _literal_reference(u, w, h, active), rtol=0.0,
        atol=2.0e-16)


def test_nemo_up3_rest_and_column_flux_controls():
    u = jnp.full((6,), 0.75, dtype=jnp.float64)
    w = jnp.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0])
    h = jnp.arange(1.0, 7.0)
    tendency = nemo_up3_vertical_momentum_advection(u, w, h)
    # A uniform velocity has nonzero local flux divergence when w varies, but
    # the closed-column momentum flux telescopes exactly.
    assert abs(float(jnp.sum(tendency * h))) < 2.0e-16
    np.testing.assert_array_equal(
        np.asarray(nemo_up3_vertical_momentum_advection(u, jnp.zeros_like(w), h)),
        np.zeros(6),
    )


def test_nemo_up3_is_differentiable():
    w = jnp.array([0.0, 0.2, -0.1, 0.3, 0.0])
    h = jnp.ones(4)
    grad = jax.grad(
        lambda u: jnp.sum(nemo_up3_vertical_momentum_advection(u, w, h) ** 2)
    )(jnp.array([0.1, 0.2, -0.3, 0.4]))
    assert np.all(np.isfinite(np.asarray(grad)))
