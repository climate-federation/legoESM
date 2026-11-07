"""MYNN surface-layer (psi_opt = 0) stability functions in ``legoesm.core.bulk_flux``.

Oracle: the four full functions of CCPP ``module_sf_mynnsfc_driver.F90``
(``psim/psih_{stable,unstable}_full``), compiled verbatim by
``scripts/validate/oracle/mynn_psi/oracle.f90``.  ``_ORACLE_R8`` is that program
built with ``-fdefault-real-8`` (the source's default-real literals ``sqrt3``,
``atan1`` and ``p333`` promoted to double); the port must match it to round-off.
Built without the flag, those literals are single precision and the oracle
differs from the port by the amounts pinned in ``test_single_precision_literal_gap``.

Known difference (recorded, not a defect): MYNN evaluates these functions through
0.01-spaced linear lookup tables for |zeta| < 10; legoESM evaluates the full
functions, which the tables interpolate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.core import bulk_flux
from legoesm.core.bulk_flux import (
    UNSTABLE_PSI_FUNCTIONS,
    mynn_psi_h_unstable,
    mynn_psi_m_unstable,
    psi_h,
    psi_m,
)

jax.config.update("jax_enable_x64", True)

# zeta, psim_stable_full(zeta), psih_stable_full(zeta)  (oracle_r8)
_ORACLE_R8_STABLE = [[0.001, -0.0060970291142995, -0.00770914680192798], [0.01, -0.06072117649985462, -0.08274271607578224], [0.1, -0.5883959379291543, -0.8409827713760729], [0.37, -2.0634166275095995, -2.7191555869081814], [1.0, -5.132265840104162, -5.602352254897672], [2.5, -9.936305516970481, -9.33240833848595], [7.0, -16.107631349118783, -14.261611225914109], [9.9, -18.216610839914466, -16.013491854483487], [1e-06, -6.0999969522091485e-06, -6.510268540785502e-06], [0.5, -2.7409768101751864, -3.4472326922564496], [3.0, -11.006059581878334, -10.163190428850523], [10.0, -18.277819976407347, -16.064719899140247]]
# zeta (<0), psim_unstable_full(zeta), psih_unstable_full(zeta)  (oracle_r8)
_ORACLE_R8_UNSTABLE = [[-0.001, 0.003980157798002412, 0.007952425492334535], [-0.01, 0.03814533390096245, 0.07558914525110617], [-0.1, 0.2833631086910307, 0.5351312987306333], [-0.37, 0.6697363682855327, 1.2004926873174397], [-1.0, 1.122645667958451, 1.8882809525330408], [-2.5, 1.67847261851383, 2.5761022163214027], [-7.0, 2.41827440854416, 3.403767080408865], [-9.9, 2.6867804992570585, 3.6966786953769377], [-1e-06, 3.999979103274541e-06, 7.999952000223759e-06], [-0.5, 0.7910478548235621, 1.3969421911358133], [-3.0, 1.8022044535392634, 2.7177980373875723], [-10.0, 2.694691672166397, 3.7052661432895944]]
# Same, default-real literals left single precision (plain gfortran build).
_ORACLE_R4LIT_UNSTABLE = [[-0.001, 0.003980201510392349, 0.007952425492481846], [-0.01, 0.038145377623700134, 0.07558914526624007], [-0.1, 0.2833631535403856, 0.5351313004737878], [-0.37, 0.6697364281333629, 1.2004927123263185], [-1.0, 1.1226457904555036, 1.8882810717801408], [-2.5, 1.6784728215459572, 2.5761024488306927], [-7.0, 2.4182746671172493, 3.4037673803962822], [-9.9, 2.6867807719695533, 3.696679010602716], [-1e-06, 4.043691389953855e-06, 7.999952000223906e-06], [-0.5, 0.7910479266064044, 1.3969422344446167], [-3.0, 1.8022046688668953, 2.717798285875154], [-10.0, 2.6946919452742577, 3.7052664589326354]]


def _a(x):
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def test_stable_matches_fortran_oracle():
    o = np.asarray(_ORACLE_R8_STABLE)
    np.testing.assert_allclose(psi_m(_a(o[:, 0]), "mynn_sfc"), o[:, 1], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(psi_h(_a(o[:, 0]), "mynn_sfc"), o[:, 2], rtol=1e-13, atol=1e-15)


def test_unstable_matches_fortran_oracle():
    """atol 2e-12: the source's ``atan1 = 0.785398163397`` is pi/4 truncated by
    4.8e-13; the port uses exact pi/4 so that psi(0) = 0 exactly."""
    o = np.asarray(_ORACLE_R8_UNSTABLE)
    np.testing.assert_allclose(mynn_psi_m_unstable(_a(o[:, 0])), o[:, 1], rtol=1e-13, atol=2e-12)
    np.testing.assert_allclose(mynn_psi_h_unstable(_a(o[:, 0])), o[:, 2], rtol=1e-13, atol=2e-12)


def test_single_precision_literal_gap():
    """Port vs the oracle with single-precision literals: recorded per point."""
    o = np.asarray(_ORACLE_R4LIT_UNSTABLE)
    gap_m = np.asarray(mynn_psi_m_unstable(_a(o[:, 0]))) - o[:, 1]
    gap_h = np.asarray(mynn_psi_h_unstable(_a(o[:, 0]))) - o[:, 2]
    np.testing.assert_allclose(gap_m, _R4LIT_GAP_M, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(gap_h, _R4LIT_GAP_H, rtol=1e-6, atol=1e-12)


_R4LIT_GAP_M = [-4.371149332077495e-08, -4.372184106427035e-08, -4.484845689267658e-08, -5.984691675564591e-08, -1.2249608660752642e-07, -2.030311110345906e-07, -2.5857205621093726e-07, -2.727114609690773e-07, -4.3711390063198953e-08, -7.178191807621914e-08, -2.153266100979323e-07, -2.7310682737535785e-07]
_R4LIT_GAP_H = [-1.4731098285647448e-13, -1.5133783115572896e-11, -1.743144295573984e-09, -2.5008754178657e-08, -1.1924658260831222e-07, -2.3250839698008008e-07, -2.9998640194150994e-07, -3.15224754299237e-07, -1.4738373282224826e-19, -4.330859626477945e-08, -2.484866494789628e-07, -3.1564201607992004e-07]


def test_registered_in_both_tables():
    bulk_flux.validate_stability_scheme("mynn_sfc")
    assert UNSTABLE_PSI_FUNCTIONS["mynn_sfc"] == (mynn_psi_m_unstable, mynn_psi_h_unstable)


def test_neutral_limit_and_phi_continuity():
    """psi(0) = 0 on both branches; phi = 1 - zeta dpsi/dzeta -> 1 from both sides."""
    for f in (mynn_psi_m_unstable, mynn_psi_h_unstable):
        assert abs(float(f(_a(0.0)))) < 1e-15
    for f in (lambda z: psi_m(z, "mynn_sfc"), lambda z: psi_h(z, "mynn_sfc")):
        assert abs(float(f(_a(0.0)))) < 1e-15
    eps = 1e-7
    phi = lambda f, z: 1.0 - z * jax.grad(f)(z)  # noqa: E731
    for f in (lambda z: psi_m(z, "mynn_sfc"), lambda z: psi_h(z, "mynn_sfc")):
        assert abs(float(phi(f, _a(eps))) - 1.0) < 1e-5
    for f in (mynn_psi_m_unstable, mynn_psi_h_unstable):
        assert abs(float(phi(f, _a(-eps))) - 1.0) < 1e-5


def _dphi(f):
    phi = lambda z: 1.0 - z * jax.grad(f)(z)  # noqa: E731
    return jax.grad(phi)


@pytest.mark.parametrize("f,side,slope", [
    (lambda z: psi_m(z, "mynn_sfc"), 1.0, 6.1),
    (lambda z: psi_h(z, "mynn_sfc"), 1.0, 5.3),
    (mynn_psi_m_unstable, -1.0, 4.0),
    (mynn_psi_h_unstable, -1.0, 8.0),
])
def test_phi_slope_at_neutral_each_side(f, side, slope):
    """dphi/dzeta at 0+/0-: 6.1 / 5.3 stable, gamma/4 = 4 / gamma/2 = 8 unstable.

    The only slope discontinuity is at zeta = 0 (allowed); there are no other
    branch points in either branch, so each side is C1 on its own domain.  The
    stable heat slope approaches 5.3 only as 5.3 (1 + 1.1 zeta^0.1), hence the
    evaluation at |zeta| = 1e-40.
    """
    assert float(_dphi(f)(_a(side * 1e-40))) == pytest.approx(slope, rel=2e-3)


@pytest.mark.parametrize("f,side", [
    (lambda z: psi_m(z, "mynn_sfc"), 1.0), (lambda z: psi_h(z, "mynn_sfc"), 1.0),
    (mynn_psi_m_unstable, -1.0), (mynn_psi_h_unstable, -1.0),
])
def test_phi_is_c1_within_each_branch(f, side):
    """No hidden kink inside a branch: dphi/dzeta changes smoothly on a fine grid.

    Stops below |zeta| = 10, where psi_m/psi_h clip zeta for every scheme.
    """
    z = side * np.geomspace(1e-5, 9.9, 4000)
    d = np.asarray(jax.vmap(_dphi(f))(_a(z)))
    assert np.all(np.isfinite(d))
    jumps = np.abs(np.diff(d)) / (np.abs(d[:-1]) + 1.0)
    assert jumps.max() < 0.05


@pytest.mark.parametrize("z", [-3.0, -0.2, 0.2, 3.0])
def test_gradients_check(z):
    check_grads(lambda x: psi_m(x, "mynn_sfc"), (_a(z),), order=2, modes=["rev"])
    check_grads(lambda x: psi_h(x, "mynn_sfc"), (_a(z),), order=2, modes=["rev"])
    if z < 0:
        check_grads(mynn_psi_m_unstable, (_a(z),), order=2, modes=["rev"])
        check_grads(mynn_psi_h_unstable, (_a(z),), order=2, modes=["rev"])


def test_masked_stable_lane_has_finite_gradient_at_zero():
    """At zeta <= 0 the stable branch is masked; its zeta^1.1 must not leak NaN."""
    for f in (lambda z: psi_m(z, "mynn_sfc"), lambda z: psi_h(z, "mynn_sfc")):
        for z in (-1.0, 0.0):
            assert np.isfinite(float(jax.grad(f)(_a(z))))
            assert np.isfinite(float(jax.grad(jax.grad(f))(_a(z))))


def test_other_schemes_unchanged():
    """Adding mynn_sfc leaves every existing scheme's psi bit-identical (pinned)."""
    z = _a(np.linspace(-5.0, 5.0, 41))
    for sc, (sm, sh) in _EXISTING_SUMS.items():
        assert float(jnp.sum(psi_m(z, sc))) == sm
        assert float(jnp.sum(psi_h(z, sc))) == sh


_EXISTING_SUMS = {'dyer1974': (-231.50203653706794, -212.6539984861407), 'beljaars_holtslag1991': (-137.6237751959356, -141.72229375670454), 'grachev2007_sheba': (-140.11451007361222, -50.287853151144816), 'gryanik2020': (-174.39566071089558, -116.55191198587116)}


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        psi_m(_a(0.5), "mynn")
    with pytest.raises(ValueError):
        psi_h(_a(0.5), "mynn")
