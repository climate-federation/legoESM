"""FV3_3D iter 644: p_var_core port.

Faithful JAX port of FV3 ``p_var`` core algorithm
(tools/init_hydro.F90:41-145).  Derives (pe, peln, pk, pkz, ps)
from (delp, ptop).

Tests
-----

1. ``test_p_var_core_shapes``.
2. ``test_p_var_core_pe_cumsum``.
3. ``test_p_var_core_ps_equals_pe_bottom``.
4. ``test_p_var_core_pk_power``.
5. ``test_p_var_core_top_edge_normal_branch``.
6. ``test_p_var_core_top_edge_small_ptop_branch``.
7. ``test_p_var_core_pkz_log_mean``.
8. ``test_p_var_core_batched``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.vertical import p_var_core


def test_p_var_core_shapes():
    """pe, peln, pk shape (km+1,); pkz shape (km,); ps scalar."""
    km = 20
    delp = jnp.full((km,), 5000.0)
    pe, peln, pk, pkz, ps = p_var_core(delp, ptop=1.0e3)
    assert pe.shape == (km + 1,)
    assert peln.shape == (km + 1,)
    assert pk.shape == (km + 1,)
    assert pkz.shape == (km,)
    assert ps.shape == ()


def test_p_var_core_pe_cumsum():
    """pe[k] = ptop + Σ delp[0..k-1]."""
    km = 8
    delp = jnp.asarray([1000.0, 2000.0, 1500.0, 3000.0, 5000.0, 8000.0, 10000.0, 20000.0])
    ptop = 1.0e3
    pe, _, _, _, _ = p_var_core(delp, ptop=ptop)
    expected_pe = ptop + jnp.concatenate([jnp.asarray([0.0]), jnp.cumsum(delp)])
    assert jnp.allclose(pe, expected_pe, atol=1e-10)


def test_p_var_core_ps_equals_pe_bottom():
    """ps = pe[km]."""
    km = 10
    delp = jnp.full((km,), 5000.0)
    pe, _, _, _, ps = p_var_core(delp, ptop=1.0e3)
    assert abs(float(ps) - float(pe[km])) < 1e-10


def test_p_var_core_pk_power():
    """pk[k] = pe[k]^cappa."""
    km = 8
    delp = jnp.full((km,), 5000.0)
    cappa = constants.kappa
    pe, _, pk, _, _ = p_var_core(delp, ptop=1.0e3, cappa=cappa)
    assert jnp.allclose(pk, pe ** cappa, atol=1e-8)


def test_p_var_core_top_edge_normal_branch():
    """For ptop > ptop_min, peln[0] = log(ptop)."""
    km = 6
    delp = jnp.full((km,), 5000.0)
    ptop = 1000.0
    _, peln, _, _, _ = p_var_core(delp, ptop=ptop, ptop_min=1.0e-8)
    assert abs(float(peln[0]) - float(jnp.log(ptop))) < 1e-12


def test_p_var_core_top_edge_small_ptop_branch():
    """For ptop < ptop_min, peln[0] = peln[1] - (cappa+1)/cappa."""
    km = 6
    delp = jnp.full((km,), 5000.0)
    cappa = constants.kappa
    ptop = 1.0e-9  # below ptop_min
    _, peln, _, _, _ = p_var_core(
        delp, ptop=ptop, cappa=cappa, ptop_min=1.0e-8,
    )
    ak1 = (cappa + 1.0) / cappa
    expected = float(peln[1]) - ak1
    assert abs(float(peln[0]) - expected) < 1e-10


def test_p_var_core_pkz_log_mean():
    """Hydrostatic pkz[k] = (pk[k+1] - pk[k]) / (cappa · (peln[k+1] - peln[k]))."""
    km = 8
    delp = jnp.full((km,), 5000.0)
    cappa = constants.kappa
    pe, peln, pk, pkz, _ = p_var_core(delp, ptop=1.0e3, cappa=cappa)
    for k in range(km):
        expected = float((pk[k + 1] - pk[k]) / (cappa * (peln[k + 1] - peln[k])))
        assert abs(float(pkz[k]) - expected) / max(abs(expected), 1.0) < 1e-10


def test_p_var_core_batched():
    """Batched delp shape (..., km) → outputs broadcast on leading axes."""
    km = 6
    rng = np.random.default_rng(seed=644)
    delp = jnp.asarray(rng.uniform(1000.0, 5000.0, size=(3, 4, km)))
    pe, peln, pk, pkz, ps = p_var_core(delp, ptop=1.0e3)
    assert pe.shape == (3, 4, km + 1)
    assert ps.shape == (3, 4)
    assert jnp.all(jnp.isfinite(pe))
    assert jnp.all(jnp.isfinite(peln))
