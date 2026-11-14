"""Gradient-only fix of the MG2 conservation limiters (factor = available /
sink): ``div_gated_tangent`` keeps the forward division bitwise and drops the
factor's derivative where the conserved hydrometeor is absent.

Measured motivation (.physics-validator/mg2_absent_gradient_gate/README.md): at
stocks and sinks of 1e-223 .. 1e-291 inside a macmic sub-step the factor's
local derivative reached 1e28 per operation and the physics-call adjoint
1e75 against a finite difference of 1e3.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.test_util  # noqa: F401  (check_grads)
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics._warm_rain import (  # noqa: E402
    MG2_QSMALL,
    div_gated_tangent,
    mg2_qi_conservation_ratio,
)

DT = 1800.0 / 3.0
OMSM = 1.0 - 1.0e-5


def _plain_ratio(q_avail, src, snk, width):
    """mg2_qi_conservation_ratio with a plain division (the pre-fix form)."""
    has = snk > 1.0e-30
    r = jnp.where(has, (q_avail / DT + src) / jnp.where(has, snk, 1.0) * OMSM,
                  1.0e30)
    from legoesm.core.smooth import smooth_min
    if width > 0.0:
        return smooth_min(jnp.ones_like(r), r, 1.0 / width)
    return jnp.where((snk - src) * DT > q_avail, r, 1.0)


def test_forward_is_bitwise_the_division():
    rng = np.random.default_rng(0)
    x = jnp.asarray(10.0 ** rng.uniform(-300, 2, 64))
    y = jnp.asarray(10.0 ** rng.uniform(-300, 2, 64))
    live = jnp.asarray(rng.uniform(size=64) > 0.5)
    assert bool(jnp.all(div_gated_tangent(x, y, live) == x / y))


@pytest.mark.parametrize("width", [0.0, 0.01])
def test_limiter_forward_unchanged(width):
    rng = np.random.default_rng(1)
    q = jnp.asarray(10.0 ** rng.uniform(-300, -3, 128))
    src = jnp.asarray(10.0 ** rng.uniform(-300, -8, 128))
    snk = jnp.asarray(10.0 ** rng.uniform(-300, -6, 128))
    new = mg2_qi_conservation_ratio(q, src, snk, DT, OMSM, width=width)
    old = _plain_ratio(q, src, snk, width)
    assert bool(jnp.all(new == old))


def test_absent_stock_has_no_factor_derivative_and_the_plain_one_explodes():
    """Planted tiny stock and sink (the measured regime): the pre-fix factor's
    derivative w.r.t. the stock is 1/(dt sink) ~ 6e25; after the fix it is 0,
    and the limited sink's derivative w.r.t. the sink is the factor alone."""
    q = jnp.asarray([2.6e-223])
    src = jnp.asarray([0.0])
    snk = jnp.asarray([3.0e-29])          # above the 1e-30 sentinel, active
    assert float(q[0]) < MG2_QSMALL

    def factor_new(qq):
        return jnp.sum(mg2_qi_conservation_ratio(qq, src, snk, DT, OMSM, width=0.0))

    def factor_old(qq):
        return jnp.sum(_plain_ratio(qq, src, snk, 0.0))
    assert abs(float(jax.grad(factor_old)(q)[0])) > 1e25
    assert float(jax.grad(factor_new)(q)[0]) == 0.0
    g_sink = jax.grad(lambda s: jnp.sum(
        s * mg2_qi_conservation_ratio(q, src, s, DT, OMSM, width=0.0)))(snk)
    factor = mg2_qi_conservation_ratio(q, src, snk, DT, OMSM, width=0.0)
    np.testing.assert_allclose(np.asarray(g_sink), np.asarray(factor), rtol=1e-14)


@pytest.mark.parametrize("width", [0.0, 0.01])
def test_present_stock_gradient_is_the_plain_gradient(width):
    """Where the stock is present the gated tangent is the exact derivative of
    the division: gradients agree with the plain form to roundoff."""
    q = jnp.asarray([2.0e-6, 5.0e-5, 1.0e-4])
    src = jnp.asarray([1.0e-10, 0.0, 3.0e-9])
    snk = jnp.asarray([4.0e-9, 1.0e-7, 2.0e-7])
    assert bool(jnp.all(q >= MG2_QSMALL))
    for argnum in (0, 1, 2):
        def f_new(*a):
            return jnp.sum(mg2_qi_conservation_ratio(*a, DT, OMSM, width=width))

        def f_old(*a):
            return jnp.sum(_plain_ratio(*a, width))
        g_new = jax.grad(f_new, argnums=argnum)(q, src, snk)
        g_old = jax.grad(f_old, argnums=argnum)(q, src, snk)
        np.testing.assert_allclose(np.asarray(g_new), np.asarray(g_old),
                                   rtol=1e-12, atol=0.0)


def test_gated_tangent_passes_check_grads_where_live():
    x = jnp.asarray([3.0, 1.0])      # O(1): check_grads steps by 1e-4 absolute
    y = jnp.asarray([2.0, 5.0])
    live = jnp.asarray([True, True])
    jax.test_util.check_grads(lambda a, b: div_gated_tangent(a, b, live),
                              (x, y), order=1, modes=("fwd", "rev"))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_second_order_is_finite_at_a_gated_tiny_denominator(dtype):
    """Hessian through the gated division: the gated cell (y = 2.6e-223 in
    fp64, 2e-30 in fp32) must give finite second derivatives (0), and the
    live cell its exact ones; the tangent must not divide by the gated y."""
    tiny = 2.6e-223 if dtype == jnp.float64 else 2.0e-30
    x = jnp.asarray([1.0e-3, 3.0], dtype)
    y = jnp.asarray([tiny, 2.0], dtype)
    live = jnp.asarray([False, True])

    def f(xy):
        return jnp.sum(div_gated_tangent(xy[:2], xy[2:], live))
    h = jax.hessian(f)(jnp.concatenate([x, y]))
    assert bool(jnp.all(jnp.isfinite(h)))
    # live cell: d2(x/y)/dy2 = 2x/y^3 = 0.75, d2/dxdy = -1/y^2 = -0.25
    np.testing.assert_allclose(float(h[3, 3]), 0.75, rtol=1e-6)
    np.testing.assert_allclose(float(h[1, 3]), -0.25, rtol=1e-6)
    assert float(jnp.max(jnp.abs(h[0]))) == 0.0 and float(jnp.max(jnp.abs(h[2]))) == 0.0
