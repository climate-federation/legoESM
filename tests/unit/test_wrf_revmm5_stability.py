"""WRF/MPAS revised MM5 surface-layer stability functions (Jimenez et al. 2012).

Stable branch is selectable as ``stability_scheme="wrf_revmm5"``; the unstable
branch is ``unstable_scheme="wrf_revmm5"`` (``bulk_flux.UNSTABLE_PSI_FUNCTIONS``); the
independent selector wires it.  Oracle: MMM-physics sf_sfclayrev.F90
``psi{m,h}_{stable,unstable}_full`` compiled with gfortran -fdefault-real-8, kind_phys = 8
(scripts/validate/oracle/wrf_sfclayrev/oracle.f90).

Run: JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/test_wrf_revmm5_stability.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.core.bulk_flux import UNSTABLE_PSI_FUNCTIONS, psi_h, psi_h_coare, psi_m, psi_m_coare

jax.config.update("jax_enable_x64", True)

# zeta, psim_stable_full, psih_stable_full, psim_unstable_full(-zeta), psih_unstable_full(-zeta)
_ORACLE = np.array([
    [1.0000000000000000E-03, -6.0970291142994999E-03, -7.7091468019279798E-03, 3.9801577656217732E-03, 7.9524253796472519E-03],
    [1.0000000000000000E-02, -6.0721176499854618E-02, -8.2742716075782236E-02, 3.8145301125664104E-02, 7.5589038097590086E-02],
    [1.0000000000000001E-01, -5.8839593792915434E-01, -8.4098277137607291E-01, 2.8333482038818331E-01, 5.3505743614603740E-01],
    [5.0000000000000000E-01, -2.7409768101751864E+00, -3.4472326922564496E+00, 7.8911774008745872E-01, 1.3931649685088074E+00],
    [1.0000000000000000E+00, -5.1322658401041616E+00, -5.6023522548976716E+00, 1.1154023585717361E+00, 1.8756023239920849E+00],
    [2.0000000000000000E+00, -8.6582181554661855E+00, -8.3496436760908566E+00, 1.5159516570770686E+00, 2.3793278582866115E+00],
    [5.0000000000000000E+00, -1.4067438540647672E+01, -1.2596013404534858E+01, 2.1381635056664621E+00, 3.0852815032267391E+00],
    [1.0000000000000000E+01, -1.8277819976407347E+01, -1.6064719899140247E+01, 2.6585186662659122E+00, 3.6557208446329805E+00],
])
# Oracle built with -fdefault-real-8 (literals 1./2.5, 0.33, ATAN(1.) promoted,
# as in a double-precision WRF/MPAS build), so the port matches to round-off.
# A default single-precision build differs from these rows by up to 4e-5
# relative (at zeta = 1e-3, fp32 cancellation), not tested.
_ORACLE_RTOL = 1e-12
_ORACLE_ATOL = 1e-15

_UNSTABLE_M, _UNSTABLE_H = UNSTABLE_PSI_FUNCTIONS["wrf_revmm5"]


def _phi(psi):
    dpsi = jax.grad(psi)
    return lambda z: 1.0 - z * dpsi(z)


def test_stable_matches_fortran():
    z = jnp.asarray(_ORACLE[:, 0])
    np.testing.assert_allclose(psi_m(z, "wrf_revmm5", zeta_cap_width=0.0), _ORACLE[:, 1], rtol=_ORACLE_RTOL, atol=_ORACLE_ATOL)
    np.testing.assert_allclose(psi_h(z, "wrf_revmm5", zeta_cap_width=0.0), _ORACLE[:, 2], rtol=_ORACLE_RTOL, atol=_ORACLE_ATOL)


def test_unstable_table_matches_fortran():
    z = -jnp.asarray(_ORACLE[:, 0])
    np.testing.assert_allclose(_UNSTABLE_M(z), _ORACLE[:, 3], rtol=_ORACLE_RTOL, atol=_ORACLE_ATOL)
    np.testing.assert_allclose(_UNSTABLE_H(z), _ORACLE[:, 4], rtol=_ORACLE_RTOL, atol=_ORACLE_ATOL)


def test_unstable_side_of_psi_is_still_businger_dyer():
    z = jnp.asarray([-5.0, -1.0, -0.1])
    np.testing.assert_array_equal(psi_m(z, "wrf_revmm5"), psi_m(z, "dyer1974"))
    np.testing.assert_array_equal(psi_h(z, "wrf_revmm5"), psi_h(z, "dyer1974"))


@pytest.mark.parametrize("f", [lambda z: psi_m(z, "wrf_revmm5"), lambda z: psi_h(z, "wrf_revmm5"),
                               _UNSTABLE_M, _UNSTABLE_H])
def test_neutral_limit(f):
    assert float(f(0.0)) == 0.0
    assert abs(float(_phi(f)(0.0)) - 1.0) < 1e-12


@pytest.mark.parametrize("stable,unstable", [
    (lambda z: psi_m(z, "wrf_revmm5"), _UNSTABLE_M),
    (lambda z: psi_h(z, "wrf_revmm5"), _UNSTABLE_H),
])
def test_phi_continuous_at_zero_and_c1_elsewhere(stable, unstable):
    """phi continuous at 0 (slope may jump there only); C1 everywhere else.

    The C1 claim rests on the algebra: each branch is ONE analytic expression
    on its open half-line, so there is no branch point besides 0.  The grid is
    a supplementary guard: every step change of the autodiff slope phi' must
    be explained by the local curvature, |dphi'| <= h max|phi''| (1 + 10%);
    log spacing keeps the steps small where psi_h'' ~ zeta^-0.9.
    """
    eps = 1e-9
    assert abs(float(_phi(stable)(eps)) - float(_phi(unstable)(-eps))) < 1e-6
    g = np.geomspace(1e-4, 9.99, 20001)
    for f, grid in ((stable, g), (unstable, -g)):
        dphi_f = jax.grad(_phi(f))
        z = jnp.asarray(grid)
        dphi = np.asarray(jax.vmap(dphi_f)(z))
        d2phi = np.abs(np.asarray(jax.vmap(jax.grad(dphi_f))(z)))
        assert np.all(np.isfinite(dphi)) and np.all(np.isfinite(d2phi))
        h = np.abs(np.diff(grid))
        bound = 1.1 * h * np.maximum(d2phi[1:], d2phi[:-1]) + 1e-9
        assert np.all(np.abs(np.diff(dphi)) <= bound)


def test_one_sided_slopes_at_zero_recorded():
    """Slope of phi at 0 from each side (informational, the allowed jump)."""
    for stable, unstable, want in (
        (lambda z: psi_m(z, "wrf_revmm5"), _UNSTABLE_M, (6.1, 4.0)),
        (lambda z: psi_h(z, "wrf_revmm5"), _UNSTABLE_H, (5.3, 8.0)),
    ):
        # psi_h's zeta^1.1 term adds 5.83 zeta^0.1 to the stable slope, so the
        # limit is approached slowly; 1e-40 puts that term at ~1e-4 relative.
        right = float(jax.grad(_phi(stable))(1e-40))
        left = float(jax.grad(_phi(unstable))(-1e-40))
        np.testing.assert_allclose((right, left), want, rtol=2e-3)


@pytest.mark.parametrize("z0", [-3.0, -0.2, 0.2, 3.0])
def test_gradients(z0):
    for f in (lambda z: psi_m(z, "wrf_revmm5"), lambda z: psi_h(z, "wrf_revmm5")):
        check_grads(f, (jnp.asarray(z0),), order=2, modes=["rev"])
    if z0 < 0:
        for f in (_UNSTABLE_M, _UNSTABLE_H):
            check_grads(f, (jnp.asarray(z0),), order=2, modes=["rev"])


@pytest.mark.parametrize("z0", [-0.5, -5e-4, 0.0, 5e-4])
def test_second_derivatives_finite_both_modes_incl_coare(z0):
    """Masked stable branch never feeds inf/NaN, in reverse and forward mode."""
    for f in (psi_m, psi_h, psi_m_coare, psi_h_coare):
        g = lambda z, f=f: f(z, "wrf_revmm5")
        for d2 in (jax.grad(jax.grad(g)), jax.jacfwd(jax.jacfwd(g))):
            assert np.isfinite(float(d2(jnp.asarray(z0)))), (f.__name__, z0)
