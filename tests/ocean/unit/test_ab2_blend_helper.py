"""Direct unit tests for the #517 item 8 shared AB2 time-blend helper.

``ab2_blend(f_new, f_old, eps)`` factors the open-coded Adams-Bashforth-2
extrapolation ``(1.5 + eps)·F^n − (0.5 + eps)·F^{n-1}`` out of the
rigid-lid streamfunction update and the split / unsplit baroclinic
tracer + momentum predictors in ocean_model_latlon_cgrid.py.

Every test asserts BIT-IDENTITY against the exact open-coded expression
(both the literal ``(1.5 + eps)`` form and the ``a_n = 1.5 + eps`` /
``a_p = 0.5 + eps`` named form), so the refactor is a pure no-op.  AB2
sits in the end-to-end ``jax.grad`` path, so differentiability is checked
too.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.ocean_tendency_common import ab2_blend


def _rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("eps", [0.0, 0.001, 0.01, 0.1])
def test_ab2_blend_matches_open_coded_literal(eps):
    # rigid_lid_latlon_cgrid form: (1.5 + eps)*f_new - (0.5 + eps)*f_old.
    g = _rng(12)
    f_new = jnp.asarray(g.normal(size=(5, 6)))
    f_old = jnp.asarray(g.normal(size=(5, 6)))
    ref = (1.5 + eps) * f_new - (0.5 + eps) * f_old
    out = ab2_blend(f_new, f_old, eps)
    assert jnp.array_equal(out, ref)


def test_ab2_blend_matches_named_an_ap_form():
    # ocean_model_latlon_cgrid form: a_n = 1.5 + eps, a_p = 0.5 + eps.
    g = _rng(13)
    f_new = jnp.asarray(g.normal(size=(4, 7, 3)))
    f_old = jnp.asarray(g.normal(size=(4, 7, 3)))
    eps = 0.013
    a_n, a_p = 1.5 + eps, 0.5 + eps
    ref = a_n * f_new - a_p * f_old
    out = ab2_blend(f_new, f_old, eps)
    assert jnp.array_equal(out, ref)


def test_ab2_blend_zero_eps_is_classic_ab2():
    f_new = jnp.asarray([2.0])
    f_old = jnp.asarray([1.0])
    out = ab2_blend(f_new, f_old, 0.0)
    assert jnp.allclose(out, 1.5 * 2.0 - 0.5 * 1.0)


def test_ab2_blend_broadcasts_and_traced_eps():
    # eps may be a traced scalar array (config.ab2_epsilon under a loss).
    g = _rng(15)
    f_new = jnp.asarray(g.normal(size=(3, 4)))
    f_old = jnp.asarray(g.normal(size=(3, 4)))
    eps = jnp.asarray(0.02)
    ref = (1.5 + eps) * f_new - (0.5 + eps) * f_old
    out = ab2_blend(f_new, f_old, eps)
    assert jnp.array_equal(out, ref)


def test_ab2_blend_differentiable():
    g = _rng(14)
    f_new = jnp.asarray(g.normal(size=(6,)))
    f_old = jnp.asarray(g.normal(size=(6,)))

    def loss(a, b):
        return jnp.sum(ab2_blend(a, b, 0.01) ** 2)

    ga, gb = jax.grad(loss, argnums=(0, 1))(f_new, f_old)
    assert np.all(np.isfinite(np.asarray(ga)))
    assert np.all(np.isfinite(np.asarray(gb)))
    assert np.any(np.asarray(ga) != 0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
