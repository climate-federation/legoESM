"""FV3_3D iter 646: hydro_eq IC builder port.

Faithful JAX port of FV3 ``hydro_eq`` (tools/init_hydro.F90:
277-456, hybrid sigma-p hydrostatic branch).

Tests
-----

1. ``test_hydro_eq_shapes``.
2. ``test_hydro_eq_no_mountain_uniform_ps``.
3. ``test_hydro_eq_delp_positive``.
4. ``test_hydro_eq_pt_bounded_below_t1``.
5. ``test_hydro_eq_delp_sums_to_ps_minus_ptop``.
6. ``test_hydro_eq_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import hydro_eq


def _build_ak_bk(km):
    """Simple hybrid coord for testing."""
    sigma_half = jnp.linspace(0.0, 1.0, km + 1)
    p_top = 100.0  # 1 hPa
    p_ref = 1.0e5
    bk = sigma_half ** 2
    ak = p_top * (1.0 - sigma_half) + (1.0 - bk) * 0.0
    return ak, bk


def test_hydro_eq_shapes():
    """ps shape (...,); delp, pt shape (..., km)."""
    km = 32
    ak, bk = _build_ak_bk(km)
    hs = jnp.zeros((6, 8, 8))
    ps, delp, pt = hydro_eq(ak, bk, hs, drym=1.0e5)
    assert ps.shape == (6, 8, 8)
    assert delp.shape == (6, 8, 8, km)
    assert pt.shape == (6, 8, 8, km)


def test_hydro_eq_no_mountain_uniform_ps():
    """mountain=False → ps uniform = drym."""
    km = 32
    ak, bk = _build_ak_bk(km)
    hs = jnp.zeros((4, 4))
    drym = 99500.0
    ps, _, _ = hydro_eq(ak, bk, hs, drym=drym, mountain=False)
    assert jnp.allclose(ps, drym, atol=1e-8)


def test_hydro_eq_delp_positive():
    """All delp > 0."""
    km = 32
    ak, bk = _build_ak_bk(km)
    hs = jnp.zeros((4, 4))
    _, delp, _ = hydro_eq(ak, bk, hs, drym=1.0e5)
    assert jnp.all(delp > 0)


def test_hydro_eq_pt_bounded_below_t1():
    """pt ≥ T1 = 200 K (FV3 lower bound)."""
    km = 32
    ak, bk = _build_ak_bk(km)
    hs = jnp.zeros((4, 4))
    _, _, pt = hydro_eq(ak, bk, hs, drym=1.0e5)
    assert jnp.all(pt >= 200.0 - 1e-10)


def test_hydro_eq_delp_sums_to_ps_minus_ptop():
    """Σ delp = ps - ptop = ps - ak[0]."""
    km = 32
    ak, bk = _build_ak_bk(km)
    hs = jnp.zeros((4, 4))
    drym = 1.0e5
    ps, delp, _ = hydro_eq(ak, bk, hs, drym=drym)
    total_delp = jnp.sum(delp, axis=-1)
    expected = ps - ak[0]
    assert jnp.allclose(total_delp, expected, atol=1e-6)


def test_hydro_eq_finite():
    """No NaN/Inf in outputs."""
    km = 24
    ak, bk = _build_ak_bk(km)
    rng = np.random.default_rng(seed=646)
    hs = jnp.asarray(rng.uniform(0.0, 1000.0, size=(3, 4))) * 9.81
    ps, delp, pt = hydro_eq(ak, bk, hs, drym=1.0e5, mountain=False)
    assert jnp.all(jnp.isfinite(ps))
    assert jnp.all(jnp.isfinite(delp))
    assert jnp.all(jnp.isfinite(pt))
