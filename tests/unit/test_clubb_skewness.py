"""Unit tests for CLUBB skewness diagnostics (``clubb_skewness.py``).

Part of the fuller CLUBB port — see ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_skewness import (  # noqa: E402
    LG_2005_ansatz,
    Skx_func,
    compute_gamma_Skw,
    xp3_LG_2005_ansatz,
)

# CAM-default gamma coefficients (CLUBBParams).
_GC, _GB, _GCF = 0.308, 0.32, 5.0


# ---------------------------------------------------------------------------
# Skx_func
# ---------------------------------------------------------------------------

def test_skx_func_zero_denom_coef_is_normalized_third_moment():
    xp2 = jnp.asarray([0.5, 1.0, 2.0])
    xp3 = jnp.asarray([0.1, -0.2, 0.3])
    out = Skx_func(xp2, xp3, x_tol=2e-2, Skw_denom_coef=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(xp3 * xp2 ** -1.5), rtol=1e-12)


def test_skx_func_sign_follows_third_moment():
    xp2 = jnp.asarray([1.0, 1.0])
    xp3 = jnp.asarray([0.5, -0.5])
    out = Skx_func(xp2, xp3, x_tol=2e-2, Skw_denom_coef=4.0)
    assert float(out[0]) > 0 and float(out[1]) < 0
    # Odd symmetry in xp3.
    out_neg = Skx_func(xp2, -xp3, x_tol=2e-2, Skw_denom_coef=4.0)
    np.testing.assert_allclose(np.asarray(out_neg), -np.asarray(out), rtol=1e-12)


# ---------------------------------------------------------------------------
# compute_gamma_Skw
# ---------------------------------------------------------------------------

def test_gamma_skw_constant_when_flag_off():
    Skw = jnp.linspace(-3.0, 3.0, 20)
    out = compute_gamma_Skw(Skw, _GC, _GB, _GCF, l_gamma_Skw=False)
    np.testing.assert_allclose(np.asarray(out), _GC, rtol=1e-12)


def test_gamma_skw_limits():
    """At Skw=0 -> gamma_coef; at large |Skw| -> gamma_coefb; bounded between."""
    out0 = compute_gamma_Skw(jnp.asarray([0.0]), _GC, _GB, _GCF)
    assert float(out0[0]) == pytest.approx(_GC, rel=1e-12)
    out_big = compute_gamma_Skw(jnp.asarray([50.0]), _GC, _GB, _GCF)
    assert float(out_big[0]) == pytest.approx(_GB, rel=1e-6)
    Skw = jnp.linspace(-10.0, 10.0, 100)
    out = compute_gamma_Skw(Skw, _GC, _GB, _GCF)
    lo, hi = min(_GC, _GB), max(_GC, _GB)
    assert jnp.all(out >= lo - 1e-12) and jnp.all(out <= hi + 1e-12)


def test_gamma_skw_degenerate_coefficients_constant():
    Skw = jnp.linspace(-3.0, 3.0, 10)
    out = compute_gamma_Skw(Skw, 0.3, 0.3, _GCF)  # gc == gb -> constant
    np.testing.assert_allclose(np.asarray(out), 0.3, rtol=1e-12)


def test_gamma_skw_golden():
    out = compute_gamma_Skw(jnp.asarray([1.0]), _GC, _GB, _GCF)
    expected = _GB + (_GC - _GB) * np.exp(-0.5 * (1.0 / _GCF) ** 2)
    assert float(out[0]) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# LG_2005_ansatz + xp3 inverse
# ---------------------------------------------------------------------------

def test_lg2005_zero_flux_zero_skewness():
    Skw = jnp.asarray([0.5, -0.5])
    out = LG_2005_ansatz(
        Skw, wpxp=jnp.zeros(2), wp2=jnp.full(2, 0.5), xp2=jnp.full(2, 0.1),
        sigma_sqd_w=jnp.full(2, 0.3), beta=2.4, x_tol=1e-2, w_tol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_xp3_ansatz_inverts_skx_func():
    """Skx_func(xp2, xp3_LG_2005_ansatz(...)) == LG_2005_ansatz(...)."""
    Skw = jnp.asarray([0.6, -0.3, 0.0])
    wpxp = jnp.asarray([0.05, -0.02, 0.0])
    wp2 = jnp.full(3, 0.5)
    xp2 = jnp.full(3, 0.1)
    ssw = jnp.full(3, 0.3)
    beta, x_tol, w_tol, denom = 2.4, 1e-2, 2e-2, 4.0
    skx_direct = LG_2005_ansatz(Skw, wpxp, wp2, xp2, ssw, beta, x_tol, w_tol)
    xp3 = xp3_LG_2005_ansatz(Skw, wpxp, wp2, xp2, ssw, beta, x_tol, w_tol, denom)
    skx_round = Skx_func(xp2, xp3, x_tol, denom)
    np.testing.assert_allclose(np.asarray(skx_round), np.asarray(skx_direct), rtol=1e-10)


# ---------------------------------------------------------------------------
# AD / JIT
# ---------------------------------------------------------------------------

def test_gamma_skw_differentiable_wrt_coef():
    def loss(gc):
        return jnp.sum(compute_gamma_Skw(jnp.linspace(-2.0, 2.0, 10), gc, _GB, _GCF))

    g = jax.grad(loss)(_GC)
    assert jnp.isfinite(g)


def test_skewness_jit_clean():
    Skw = jnp.asarray([0.5, -0.5])
    out = jax.jit(lambda s: compute_gamma_Skw(s, _GC, _GB, _GCF))(Skw)
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
