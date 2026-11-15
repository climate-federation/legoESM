"""MOST revisit J3 stable form in ``legoesm.core.bulk_flux`` (``"most_revisit_j3"``).

phi = 1 + a_inf z + (a0 - a_inf) z / (1 + z/zc), heat = momentum; psi is its
closed-form integral.  Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.core import bulk_flux as bf

jax.config.update("jax_enable_x64", True)
S = "most_revisit_j3"


def _phi(z):
    return 1.0 + bf._J3_A_INF * z + (bf._J3_A0 - bf._J3_A_INF) * z / (1.0 + z / bf._J3_ZC)


def test_psi_is_the_integral_of_phi():
    z = jnp.linspace(1e-3, 7.9, 400)  # below the |zeta| cap onset
    for f in (bf.psi_m, bf.psi_h):
        np.testing.assert_allclose(bf.phi_from_psi(lambda x: f(x, S), z), _phi(z),
                                   rtol=1e-12, atol=0)


def test_heat_equals_momentum_and_psi_zero_at_neutral():
    z = jnp.linspace(-5.0, 30.0, 701)
    np.testing.assert_array_equal(bf.psi_h(z, S), bf.psi_m(z, S))
    assert float(bf.psi_m(jnp.asarray(0.0), S)) == 0.0


def test_asymptotes():
    st = lambda z: bf._stable_psi_m(z, S)  # noqa: E731  (uncapped stable branch)
    dphi = jax.grad(lambda z: 1.0 - z * jax.grad(st)(z))
    assert float(bf.phi_from_psi(st, jnp.asarray([1e-9]))[0]) == pytest.approx(1.0, abs=1e-8)
    assert float(dphi(1e-9)) == pytest.approx(bf._J3_A0, rel=1e-6)
    assert float(dphi(1e5)) == pytest.approx(bf._J3_A_INF, rel=1e-6)


def test_gradients_through_both_branches_and_the_cap():
    for z0 in (-2.0, -1e-3, 0.0, 1e-3, 0.5, 3.0, 9.5, 12.0):
        for f in (bf.psi_m, bf.psi_h, bf.psi_m_coare, bf.psi_h_coare):
            check_grads(lambda z: f(z, S), (jnp.asarray(z0),), order=2, modes=["rev"])


def test_unknown_name_raises():
    for bad in ("j3", "most_revisit"):
        with pytest.raises(ValueError, match="stability_scheme"):
            bf.psi_m(jnp.asarray(0.5), bad)
