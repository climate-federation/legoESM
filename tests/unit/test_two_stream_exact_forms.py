"""Two-stream layer quantities against the exact solution in 60-digit arithmetic.

These regimes are where RTE's own double-precision formulas lose digits, so the
reference here is the analytic Meador-Weaver / Toon solution, not the Fortran.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.radiation.rrtmgp.rte import monochromatic_two_stream as mts

getcontext().prec = 60
_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
_D = Decimal(repr(mts._LW_DIFFUSIVE_FACTOR))


def _exact_lw(tau, w, g, s_top, s_bot):
    tau, w, g, s_top, s_bot = (Decimal(float(x)) for x in (tau, w, g, s_top, s_bot))
    g1, g2 = _D * (1 - w * (1 + g) / 2), _D * w * (1 - g) / 2
    k2 = (g1 - g2) * (g1 + g2)
    if k2 <= 0:
        den = 1 + g1 * tau
        r, t = g2 * tau / den, 1 / den
    else:
        k = k2.sqrt()
        e = (-k * tau).exp()
        den = k * (1 + e * e) + g1 * (1 - e * e)
        r, t = g2 * (1 - e * e) / den, 2 * k * e / den
    z = (s_bot - s_top) / (tau * (g1 + g2))
    return (float(r), float(t),
            float(_PI * ((z + s_top) - r * (-z + s_top) - t * (z + s_bot))),
            float(_PI * ((-z + s_bot) - r * (z + s_bot) - t * (-z + s_top))))


def _col(a, dtype=jnp.float64):
    return jnp.asarray(a, dtype).reshape(-1, 1, 1)


def test_lw_layer_matches_exact_solution():
    r = np.random.default_rng(1)
    n = 400
    tau = 10.0 ** r.uniform(-8, 3, n)
    w = np.where(r.random(n) < 0.3, 1.0, 1 - 10.0 ** r.uniform(-8, 0, n))
    g = r.uniform(-0.9, 0.99, n)
    st = 10.0 ** r.uniform(-6, 1, n)
    sb = st + 10.0 ** r.uniform(-6, 1, n)
    o = mts.lw_cell_source_and_properties(_col(tau), _col(w), _col(sb), _col(st), _col(g))
    ref = np.array([_exact_lw(*a) for a in zip(tau, w, g, st, sb)]).T
    for i, key in enumerate(("r_diff", "t_diff", "src_up", "src_down")):
        assert np.abs(np.asarray(o[key]).ravel() - ref[i]).max() < 1e-10, key


@pytest.mark.parametrize("tau", [1e-9, 1e-6, 1e-4, 1e-2])
def test_thin_absorbing_layer_emits_its_linear_limit(tau):
    """An isothermal thin absorber emits pi S (1 - exp(-D tau)); no cutoff."""
    s = 2.0
    o = mts.lw_cell_source_and_properties(_col([tau]), _col([0.0]), _col([s]), _col([s]),
                                          _col([0.0]))
    want = math.pi * s * -math.expm1(-mts._LW_DIFFUSIVE_FACTOR * tau)
    assert float(o["src_up"].ravel()[0]) == pytest.approx(want, rel=1e-12)


def _sw(mu, tau, ssa, g, dtype=jnp.float64):
    return mts.sw_cell_properties(jnp.arccos(jnp.asarray(mu, dtype)), _col(tau, dtype),
                                  _col(ssa, dtype), _col(g, dtype))


def test_direct_beam_is_exactly_zero_without_scattering():
    o = _sw(0.5, [1e-6, 1.0, 50.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.9])
    assert np.all(np.asarray(o["r_dir"]) == 0.0) and np.all(np.asarray(o["t_dir"]) == 0.0)


@pytest.mark.parametrize("dtype,rtol", [(jnp.float64, 1e-9), (jnp.float32, 2e-3)])
def test_direct_beam_is_continuous_through_resonance(dtype, rtol):
    """k mu0 = 1 is a removable singularity."""
    ssa, g, tau = 0.2, 0.5, 1.0
    g1, g2 = (8 - ssa * (5 + 3 * g)) / 4, 3 * ssa * (1 - g) / 4
    mu = 1 / math.sqrt((g1 - g2) * (g1 + g2))
    at = _sw(mu, [tau], [ssa], [g], dtype)
    off = _sw(mu * (1 + 1e-3), [tau], [ssa], [g], dtype)
    for key in ("r_dir", "t_dir"):
        a, b = float(at[key].ravel()[0]), float(off[key].ravel()[0])
        assert np.isfinite(a) and a == pytest.approx(b, rel=max(rtol, 5e-3))


def test_level_source_keeps_cold_values_and_boundary_layers():
    top = jnp.asarray([0.0, 3e-6, 5e-6, 7e-6, 0.0]).reshape(1, 1, 5)
    bot = jnp.asarray([0.0, 2e-6, 4e-6, 6e-6, 0.0]).reshape(1, 1, 5)
    c = mts.lw_combine_sources({"planck_src_top": top, "planck_src_bottom": bot})
    t = np.asarray(c["planck_src_top"]).ravel()
    assert t[0] == 2e-6  # surface: lowest layer's own source
    assert t[1] == pytest.approx(math.sqrt(3e-6 * 4e-6))  # far below the old 1e-3 floor
    assert t[3] == 7e-6  # top: top layer's own source, no halo fraction


def test_level_source_gradient_finite_at_zero():
    def f(x):
        c = mts.lw_combine_sources({"planck_src_top": x, "planck_src_bottom": x})
        return jnp.sum(c["planck_src_top"])
    assert np.isfinite(np.asarray(jax.grad(f)(jnp.zeros((1, 1, 5))))).all()


@pytest.mark.parametrize("tau", [0.0, 1e-9, 1e-3, 0.5, 1.0 / 1.66, 30.0])
def test_lw_gradients_through_every_branch(tau):
    # check_grads' finite-difference step exceeds a 1e-9 layer: finiteness only there.
    f = lambda t, w: jnp.sum(mts.lw_cell_source_and_properties(  # noqa: E731
        _col([t]), _col([w]), _col([3.0]), _col([2.0]), _col([0.3]))["src_up"])
    if tau < 1e-6:
        assert np.isfinite(float(jax.grad(f)(tau, 0.5)))
        return
    check_grads(f, (tau, 0.4), order=1, modes=("rev", "fwd"), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("ssa", [0.0, 0.2, 1.0])
def test_sw_gradients_finite(ssa):
    f = lambda t, w: jnp.sum(jnp.stack([v.sum() for v in _sw(0.4, [t], [w], [0.5]).values()]))  # noqa: E731
    gr = jax.grad(f, argnums=(0, 1))(0.7, ssa)
    assert all(np.isfinite(float(x)) for x in gr)


def test_sw_gradients_agree_with_finite_differences():
    f = lambda t, w: jnp.sum(_sw(0.4, [t], [w], [0.5])["t_diff"])  # noqa: E731
    check_grads(f, (0.7, 0.6), order=1, modes=("rev",), rtol=1e-4, atol=1e-6)
